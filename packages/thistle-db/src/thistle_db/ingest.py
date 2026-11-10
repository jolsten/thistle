"""The delta engine — validate, buffer, flush to every sink in order.

One pass: ingest validates records, buffers them grouped by object, and at
flush time (end of scan, or every ``flush_records``) hands the same delta
to every sink in a fixed order:

    1. store.add(delta)        -- chunked dialect upsert, dedup on
                                  UNIQUE(epoch, record_hash); then the OMM
                                  metadata insert/merge pass
    2. output.apply(delta) ×N  -- tail-guarded appends / merge rewrites /
                                  date upserts (outputs.py)
    3. commit ingest_files     -- only for files fully consumed by this
                                  flush

That ordering rule is the entire crash-safety story: a crash replays at
most the files whose state was never recorded, and every sink is
idempotent against the replay (the store dedups; object outputs dedup on
merge and guard appends by tail epoch; date outputs upsert on
``epoch >=``).

Validation is tolerant: one bad record never aborts a file (it lands in
the ``rejects`` table) and one bad file never aborts a scan.
"""

import dataclasses
import datetime
import enum
import hashlib
import json
import pathlib
from collections import Counter
from typing import Iterable, Iterator, Optional, cast

from loguru import logger
from sqlalchemy import Table, select, update
from sqlalchemy.orm import Session

from thistle_db.config import IngestSource, Settings
from thistle_db.model import (
    IngestFile,
    OmmMetadata,
    Reject,
    SIDECAR_FIELDS,
    TLE,
    metadata_row_from_record,
    needs_metadata,
    tle_row_from_record,
    utcnow,
)
from thistle_db.outputs import OutputApplier
from thistle_db.progress import NO_PROGRESS, ProgressReporter
from thistle_db.reader import (
    detect_format,
    read_omm_csv,
    read_omm_json,
    read_omm_xml,
    read_tle,
)
from thistle_db.record import (
    GPRecord,
    MalformedRecordError,
    checksum_ok,
    from_omm,
    from_tle,
    to_csv_dict,
)
from thistle_db.record import merge as merge_records
from thistle_db.rejects import (
    DbRejectWriter,
    NO_REJECTS,
    ReasonCode,
    RejectSink,
    iter_readmission_candidates,
    payload_for_lines,
    payload_for_record,
    payload_hash,
    upsert_rejects,
)
from thistle_db.runs import IngestReport
from thistle_db.writer import WritePool

CHUNK_SIZE = 5000

# Dedup index columns for ON CONFLICT dialects (SQLite/PostgreSQL); MariaDB
# uses INSERT IGNORE, which keys off the same unique constraints.
_TLE_CONFLICT = ["epoch", "record_hash"]
_OMM_CONFLICT = ["tle_id"]

_TLE_TABLE = cast(Table, TLE.__table__)
_OMM_TABLE = cast(Table, OmmMetadata.__table__)

_OMM_READERS = {
    "omm_json": read_omm_json,
    "omm_csv": read_omm_csv,
    "omm_xml": read_omm_xml,
}


def _insert_executor(session: Session, table: Table, index_elements: list[str]):
    """Build the dialect-appropriate insert-ignore call for `table`.

    Returns a callable taking a list of record dicts and returning the number
    of rows inserted. The statement is dialect-specific but chunk-independent,
    so it is built once per call rather than per chunk.
    """
    dialect = session.bind.dialect.name  # type: ignore[union-attr]

    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        stmt = sqlite_insert(table).on_conflict_do_nothing(
            index_elements=index_elements
        )

        def execute(rows: list[dict]) -> int:
            return session.execute(stmt, rows).rowcount  # type: ignore[attr-defined]

    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg_stmt = pg_insert(table).on_conflict_do_nothing(
            index_elements=index_elements
        )
        # psycopg3 doesn't populate rowcount for bulk inserts; use RETURNING.
        returning = pg_stmt.returning(table.c.id)

        def execute(rows: list[dict]) -> int:
            return len(session.execute(returning, rows).all())

    else:
        # mysql / mariadb (and any other INSERT IGNORE dialect)
        from sqlalchemy import insert

        my_stmt = insert(table).prefix_with("IGNORE")

        def execute(rows: list[dict]) -> int:
            return session.execute(my_stmt, rows).rowcount  # type: ignore[attr-defined]

    return execute


# --- validation ---------------------------------------------------------------


def _epoch_limit(max_epoch_ahead_days: int) -> Optional[datetime.datetime]:
    """Latest admissible epoch, or None when the guard is disabled (0).

    Computed once per file rather than per record; the horizon is measured
    in days, so a run's worth of clock drift is irrelevant.
    """
    if max_epoch_ahead_days <= 0:
        return None
    return utcnow() + datetime.timedelta(days=max_epoch_ahead_days)


def _malformed_code(exc: MalformedRecordError) -> ReasonCode:
    if "NORAD" in str(exc):
        return ReasonCode.NO_NORAD_ID
    return ReasonCode.MALFORMED_ELEMENT


def validate_tle_pairs(
    pairs: Iterable[tuple[str, str]],
    rejects: RejectSink = NO_REJECTS,
    *,
    verify_checksums: bool = True,
    epoch_limit: Optional[datetime.datetime] = None,
    max_epoch_ahead_days: int = 0,
) -> Iterator[GPRecord]:
    """Parse and validate TLE line pairs, yielding storable GPRecords.

    Everything yielded is safe to store; everything else is reported to
    `rejects` with a reason code. Lazy, so arbitrarily large inputs never
    materialize.
    """
    for line1, line2 in pairs:
        rejects.seen()
        if verify_checksums and not (checksum_ok(line1) and checksum_ok(line2)):
            rejects.reject(
                ReasonCode.CHECKSUM,
                "TLE checksum mismatch (column 69)",
                line1=line1,
                line2=line2,
            )
            continue
        try:
            record = from_tle(line1, line2)
        except MalformedRecordError as exc:
            logger.debug(f"Rejecting unusable TLE: {exc}")
            rejects.reject(_malformed_code(exc), str(exc), line1=line1, line2=line2)
        except Exception as exc:
            logger.debug(f"Rejecting unparseable TLE: {exc}")
            rejects.reject(
                ReasonCode.UNPARSEABLE,
                f"unparseable: {exc}",
                line1=line1,
                line2=line2,
            )
        else:
            if epoch_limit is not None and record.epoch > epoch_limit:
                rejects.reject(
                    ReasonCode.FUTURE_EPOCH,
                    f"epoch more than {max_epoch_ahead_days} days in the future",
                    line1=line1,
                    line2=line2,
                )
            else:
                yield record


def validate_omm_records(
    omm_records: Iterable[dict],
    rejects: RejectSink = NO_REJECTS,
    *,
    verify_checksums: bool = True,
    epoch_limit: Optional[datetime.datetime] = None,
    max_epoch_ahead_days: int = 0,
) -> Iterator[GPRecord]:
    """OMM twin of `validate_tle_pairs`.

    Rejected records quarantine as OMM JSON — the original record, not
    just its lines. Embedded TLE lines are checksum-verified when present
    (they are stored verbatim and passed through to TLE outputs).
    """
    for omm in omm_records:
        rejects.seen()
        line1 = (omm.get("TLE_LINE1") or "").strip() or None
        line2 = (omm.get("TLE_LINE2") or "").strip() or None
        if (
            verify_checksums
            and line1
            and line2
            and not (checksum_ok(line1) and checksum_ok(line2))
        ):
            rejects.reject(
                ReasonCode.CHECKSUM,
                "TLE checksum mismatch (column 69)",
                line1=line1,
                line2=line2,
                record=omm,
            )
            continue
        try:
            record = from_omm(omm)
        except MalformedRecordError as exc:
            logger.debug(f"Rejecting unusable OMM record: {exc}")
            rejects.reject(
                _malformed_code(exc), str(exc), line1=line1, line2=line2, record=omm
            )
        except Exception as exc:
            logger.debug(f"Rejecting unparseable OMM record: {exc}")
            rejects.reject(
                ReasonCode.UNPARSEABLE,
                f"unparseable: {exc}",
                line1=line1,
                line2=line2,
                record=omm,
            )
        else:
            if epoch_limit is not None and record.epoch > epoch_limit:
                rejects.reject(
                    ReasonCode.FUTURE_EPOCH,
                    f"epoch more than {max_epoch_ahead_days} days in the future",
                    line1=line1,
                    line2=line2,
                    record=omm,
                )
            else:
                yield record


# --- the store sink -------------------------------------------------------------


def store_add(session: Session, records: list[GPRecord]) -> int:
    """Insert records into the store; silently skip duplicates.

    Chunked dialect upsert on UNIQUE(epoch, record_hash); a chunk the
    database refuses is retried row by row so only the offending records
    are lost — and those are quarantined (`db_refused`). Rows that carry
    sidecar data then insert/merge their omm_metadata row. Returns the
    count of newly inserted element-set rows.
    """
    if not records:
        return 0
    now = utcnow()
    rows = [tle_row_from_record(record, now) for record in records]
    executor = _insert_executor(session, _TLE_TABLE, _TLE_CONFLICT)

    inserted = 0
    for i in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[i : i + CHUNK_SIZE]
        chunk_records = records[i : i + CHUNK_SIZE]
        try:
            inserted += executor(chunk)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning(
                f"Insert of {len(chunk)} records failed "
                f"({type(exc).__name__}: {exc}); retrying row by row"
            )
            inserted += _insert_individually(session, executor, chunk, chunk_records)

    sidecar = [record for record in records if needs_metadata(record)]
    if sidecar:
        _apply_sidecar(session, sidecar, now)
    return inserted


def _insert_individually(
    session: Session,
    executor,
    chunk: list[dict],
    records: list[GPRecord],
) -> int:
    """Re-insert a failed chunk one row at a time, quarantining culprits.

    A chunk fails as a unit — some drivers (pymysql) reject a value while
    building the statement, before any SQL is sent — so a single bad record
    would otherwise cost every good record batched with it. Slow by design:
    this runs only after a chunk has already failed.
    """
    inserted = 0
    for row, record in zip(chunk, records):
        try:
            inserted += executor([row])
            session.commit()
        except Exception as exc:
            session.rollback()
            reason = f"database rejected the row: {type(exc).__name__}: {exc}"
            logger.warning(f"{reason} (norad={record.norad_cat_id})")
            _quarantine_db_refused(session, record, reason)
    return inserted


def _quarantine_db_refused(session: Session, record: GPRecord, reason: str) -> None:
    """Retain a row the database itself refused. Contained: a quarantine
    failure must not fail the ingest."""
    try:
        if record.tle_line1 and record.tle_line2:
            payload = payload_for_lines(record.tle_line1, record.tle_line2)
            fmt = "tle"
        else:
            payload = payload_for_record(to_csv_dict(record))
            fmt = "omm"
        now = utcnow()
        upsert_rejects(
            session,
            [
                {
                    "record_hash": payload_hash(payload),
                    "format": fmt,
                    "payload": payload,
                    "truncated": False,
                    "reason_code": str(ReasonCode.DB_REFUSED),
                    "reason": reason,
                    "source_path": None,
                    "times_seen": 1,
                    "norad_cat_id": record.norad_cat_id,
                    "epoch": record.epoch,
                    "first_seen": now,
                    "last_seen": now,
                }
            ],
        )
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning(f"Could not quarantine refused row: {exc}")


def _apply_sidecar(
    session: Session, records: list[GPRecord], now: datetime.datetime
) -> None:
    """Insert missing omm_metadata rows; merge-update existing ones.

    The FK lookup constrains on epoch AND record_hash: the only index
    containing the hash is UNIQUE(epoch, record_hash), and epoch is its
    leading column — hash alone would be a full-table scan. The epoch
    IN-list can only exclude rows, never wrongly include: a matching hash
    implies the row's epoch equals its own record's epoch.
    """
    by_hash = {bytes.fromhex(record.record_hash): record for record in records}
    pairs = [(record.epoch, bytes.fromhex(record.record_hash)) for record in records]

    for i in range(0, len(pairs), CHUNK_SIZE):
        chunk = pairs[i : i + CHUNK_SIZE]
        epochs = sorted({epoch for epoch, _ in chunk})
        hashes = [digest for _, digest in chunk]
        stmt = (
            select(TLE.id, TLE.record_hash, OmmMetadata)
            .outerjoin(OmmMetadata, OmmMetadata.tle_id == TLE.id)
            .where(TLE.epoch.in_(epochs), TLE.record_hash.in_(hashes))
        )
        missing: list[dict] = []
        for tle_id, digest, metadata in session.execute(stmt):
            record = by_hash.get(digest)
            if record is None:
                continue
            if metadata is None:
                missing.append(metadata_row_from_record(record, tle_id, now))
            else:
                _merge_sidecar(metadata, record)

        if missing:
            executor = _insert_executor(session, _OMM_TABLE, _OMM_CONFLICT)
            try:
                executor(missing)
            except Exception as exc:
                session.rollback()
                logger.warning(f"omm_metadata insert failed: {exc}")
        session.commit()


def _merge_sidecar(metadata: OmmMetadata, record: GPRecord) -> bool:
    """Apply the enrichment-merge precedence to an existing sidecar row —
    the one sanctioned UPDATE (PLAN.md §3.2):

    1. Non-null beats null.
    2. Both non-null and unequal: newest CREATION_DATE wins field-wise;
       null CREATION_DATE loses to non-null; full tie → last-ingested wins.
    """
    candidate = metadata_row_from_record(record, metadata.tle_id, metadata.created)
    if record.creation_date is None:
        prefer_new = metadata.creation_date is None
    elif metadata.creation_date is None:
        prefer_new = True
    else:
        prefer_new = record.creation_date >= metadata.creation_date

    changed = False
    for name in SIDECAR_FIELDS:
        new_value = candidate[name]
        old_value = getattr(metadata, name)
        if old_value is None and new_value is not None:
            setattr(metadata, name, new_value)
            changed = True
        elif (
            old_value is not None
            and new_value is not None
            and new_value != old_value
            and prefer_new
        ):
            setattr(metadata, name, new_value)
            changed = True
    return changed


# --- source-file processing ------------------------------------------------------


class FileStatus(enum.StrEnum):
    INGESTED = "ingested"
    """File was parsed and buffered (possibly 0 new rows via dedup)."""
    SKIPPED = "skipped"
    """Size and mtime match recorded state; file not opened."""
    REFRESHED = "refreshed"
    """Stat changed but content hash identical; state refreshed, no parse."""
    FAILED = "failed"
    """Stat/read/parse raised; logged, state not recorded (retried next scan)."""


@dataclasses.dataclass
class PendingFileState:
    """A successfully parsed file whose state commits with the flush that
    carries its last record — never before its records reach every sink."""

    path: str
    path_hash: str
    size: int
    mtime_ns: int
    sha256: str


def _path_key(path: pathlib.Path) -> tuple[str, str]:
    """Return (absolute resolved path, sha256 hex of that path string)."""
    resolved = str(path.resolve())
    return resolved, hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class DeltaEngine:
    """Buffers validated records by object and flushes them to every sink.

    Use as a context manager; exit flushes the tail of the buffer, stamps
    re-admissions, and emits the once-per-run warnings.
    """

    def __init__(
        self,
        session: Session,
        settings: Settings,
        report: IngestReport,
        *,
        run_id: Optional[int] = None,
    ):
        self.session = session
        self.settings = settings
        self.report = report
        self.run_id = run_id
        self.flush_records = settings.ingest.flush_records
        self._pool = WritePool(settings.output.write_workers)
        self._applier: Optional[OutputApplier] = None
        self._buffer: dict[int, dict[bytes, GPRecord]] = {}
        self._count = 0
        self._pending_files: list[PendingFileState] = []
        self._readmitted_ids: list[int] = []

    def __enter__(self) -> "DeltaEngine":
        self._pool.__enter__()
        self._applier = OutputApplier(
            self.settings.output.files, self._pool, self.report
        )
        return self

    def __exit__(self, exc_type, *exc_info) -> None:
        try:
            if exc_type is None:
                self.close()
        finally:
            self._pool.__exit__(exc_type, *exc_info)

    # -- buffering ---------------------------------------------------------

    def add(self, record: GPRecord) -> None:
        digest = bytes.fromhex(record.record_hash)
        per_object = self._buffer.setdefault(record.norad_cat_id, {})
        existing = per_object.get(digest)
        if existing is not None:
            # Same identity delivered twice in one run ("TLE first,
            # identical-lined OMM later"): merge in the buffer so the
            # richer record reaches every sink.
            per_object[digest] = merge_records(existing, record)
            return
        per_object[digest] = record
        self._count += 1
        if self._count >= self.flush_records:
            self.flush()

    def file_done(self, state: PendingFileState) -> None:
        self._pending_files.append(state)

    def readmitted(self, reject_id: int) -> None:
        self._readmitted_ids.append(reject_id)

    # -- flushing ----------------------------------------------------------

    def flush(self) -> None:
        """Store, then outputs, then file state — the one ordering rule."""
        if self._count == 0 and not self._pending_files:
            return
        delta = {
            norad: list(per_object.values())
            for norad, per_object in self._buffer.items()
        }
        records = [record for group in delta.values() for record in group]
        self.report.records_inserted += store_add(self.session, records)
        assert self._applier is not None, "DeltaEngine used outside its context"
        self._applier.apply(delta)
        self._commit_file_states()
        self._buffer = {}
        self._count = 0

    def close(self) -> None:
        self.flush()
        self._stamp_readmissions()
        if self.report.lineless_skipped:
            logger.warning(
                f"{self.report.lineless_skipped} line-less record(s) skipped "
                "by tle outputs (TLE outputs are verbatim passthrough)"
            )
        if self._applier is not None and self._applier.alpha5_overflow:
            ids = sorted(self._applier.alpha5_overflow)
            logger.warning(
                f"{len(ids)} object(s) beyond alpha-5 range named with "
                f"integer stems (e.g. {ids[0]}) in alpha5 outputs"
            )

    def _commit_file_states(self) -> None:
        if not self._pending_files:
            return
        for state in self._pending_files:
            row = self.session.execute(
                select(IngestFile).where(IngestFile.path_hash == state.path_hash)
            ).scalar_one_or_none()
            if row is None:
                self.session.add(
                    IngestFile(
                        path=state.path,
                        path_hash=state.path_hash,
                        size=state.size,
                        mtime_ns=state.mtime_ns,
                        sha256=state.sha256,
                        last_run_id=self.run_id,
                    )
                )
            else:
                row.size = state.size
                row.mtime_ns = state.mtime_ns
                row.sha256 = state.sha256
                row.last_run_id = self.run_id
        self.session.commit()
        self._pending_files = []

    def _stamp_readmissions(self) -> None:
        if not self._readmitted_ids:
            return
        self.session.execute(
            update(Reject)
            .where(Reject.id.in_(self._readmitted_ids))
            .values(resolved_at=utcnow())
        )
        self.session.commit()
        self._readmitted_ids = []


def process_file(
    session: Session,
    engine: DeltaEngine,
    path: pathlib.Path,
    *,
    force: bool = False,
) -> FileStatus:
    """Validate one file into the engine, with state tracking.

    Unchanged files (per the ingest_files state table) are skipped without
    being opened. Never raises for per-file problems — logs a warning and
    returns FAILED, leaving the file's state unrecorded so it is retried
    on the next scan.
    """
    report = engine.report
    settings = engine.settings
    try:
        resolved, key = _path_key(path)
        # Stat before reading: a file modified mid-ingest looks changed on
        # the next scan; the redundant re-ingest is absorbed by dedup.
        stat = path.stat()

        row = session.execute(
            select(IngestFile).where(IngestFile.path_hash == key)
        ).scalar_one_or_none()

        if (
            not force
            and row is not None
            and row.size == stat.st_size
            and row.mtime_ns == stat.st_mtime_ns
        ):
            logger.debug(f"Unchanged (size/mtime), skipping: {path}")
            return FileStatus.SKIPPED

        digest = _file_sha256(path)

        if not force and row is not None and row.sha256 == digest:
            row.size = stat.st_size
            row.mtime_ns = stat.st_mtime_ns
            row.last_run_id = engine.run_id
            session.commit()
            logger.debug(f"Content unchanged, refreshed state: {path}")
            return FileStatus.REFRESHED

        fmt = detect_format(path)
        logger.info(f"Ingesting {path} (format: {fmt})")
        ahead = settings.ingest.max_epoch_ahead_days
        checksums = settings.ingest.verify_checksums
        epoch_limit = _epoch_limit(ahead)
        with DbRejectWriter(session, source_path=path, fmt=fmt) as sink:
            if fmt == "tle":
                records = validate_tle_pairs(
                    read_tle(path),
                    sink,
                    verify_checksums=checksums,
                    epoch_limit=epoch_limit,
                    max_epoch_ahead_days=ahead,
                )
            else:
                records = validate_omm_records(
                    _OMM_READERS[fmt](path),
                    sink,
                    verify_checksums=checksums,
                    epoch_limit=epoch_limit,
                    max_epoch_ahead_days=ahead,
                )
            for record in records:
                engine.add(record)
        report.records_seen += sink.total
        report.records_rejected += sink.rejected
        report.add_reject_reasons(sink.reasons)
        summary = sink.summary()
        if summary:
            # Reported as a fraction: the ratio is what separates a few bad
            # lines in a good file from an entirely wrong file.
            logger.warning(f"  {path.name}: {summary}")

        engine.file_done(
            PendingFileState(resolved, key, stat.st_size, stat.st_mtime_ns, digest)
        )
        return FileStatus.INGESTED
    except Exception as exc:
        session.rollback()
        logger.warning(f"Failed to ingest {path}: {exc}")
        return FileStatus.FAILED


# --- re-admission -------------------------------------------------------------


def readmit_matured_rejects(session: Session, engine: DeltaEngine) -> int:
    """Re-validate unresolved future-epoch rejects whose epoch has come
    inside the horizon and promote survivors into the delta.

    Their source file is state-tracked and will never be re-read, so this
    is the only path back in. A candidate that now fails for a *different*
    reason has its reject row's reason updated, which removes it from
    future candidate scans.
    """
    days = engine.settings.ingest.max_epoch_ahead_days
    epoch_limit = _epoch_limit(days)
    horizon = epoch_limit or (utcnow() + datetime.timedelta(days=36500))
    candidates = list(iter_readmission_candidates(session, horizon))
    if not candidates:
        return 0

    promoted = 0
    checksums = engine.settings.ingest.verify_checksums
    with DbRejectWriter(session, source_path=None, fmt="tle") as sink:
        for reject in candidates:
            if reject.format == "tle":
                lines = [
                    line for line in reject.payload.split("\n") if line.strip()
                ]
                if len(lines) != 2:
                    continue
                records = validate_tle_pairs(
                    [(lines[0], lines[1])],
                    sink,
                    verify_checksums=checksums,
                    epoch_limit=epoch_limit,
                    max_epoch_ahead_days=days,
                )
            else:
                try:
                    omm = json.loads(reject.payload)
                except json.JSONDecodeError:
                    continue
                records = validate_omm_records(
                    [omm],
                    sink,
                    verify_checksums=checksums,
                    epoch_limit=epoch_limit,
                    max_epoch_ahead_days=days,
                )
            for record in records:
                engine.add(record)
                engine.readmitted(reject.id)
                promoted += 1
    if promoted:
        logger.info(
            f"Re-admitted {promoted} previously future-epoch record(s) now "
            "inside the horizon"
        )
        engine.report.records_readmitted += promoted
    return promoted


# --- scan orchestration ---------------------------------------------------------


def ingest_paths(
    session: Session,
    engine: DeltaEngine,
    files: list[pathlib.Path],
    *,
    progress: ProgressReporter = NO_PROGRESS,
) -> None:
    """Ingest explicitly named files (always re-read, state still recorded)."""
    report = engine.report
    task = progress.task("Ingesting files", total=len(files))
    for path in files:
        status = process_file(session, engine, path, force=True)
        report.files_scanned += 1
        _count_status(report, status)
        progress.advance(task)
    progress.finish(task)


def ingest_sources(
    session: Session,
    engine: DeltaEngine,
    sources: list[IngestSource],
    *,
    force: bool = False,
    progress: ProgressReporter = NO_PROGRESS,
) -> None:
    """Scan configured source directories into the engine.

    Files are processed in name order — delivery files are date-named, so
    historical backfills arrive chronologically and hit the hot edge of
    the dedup index and the outputs' append path. A bad file is logged and
    never aborts the scan; a missing directory is a warning.
    """
    report = engine.report
    for source in sources:
        source_path = pathlib.Path(source.path)
        if not source_path.is_dir():
            logger.warning(f"Source directory does not exist: {source_path}")
            continue

        files = sorted(source_path.glob(source.pattern))
        logger.info(
            f"Found {len(files)} files in {source_path} matching {source.pattern}"
        )

        task = progress.task(f"Ingesting {source_path.name}", total=len(files))
        outcomes: Counter = Counter()
        for path in files:
            status = process_file(session, engine, path, force=force)
            level = "DEBUG" if status == FileStatus.SKIPPED else "INFO"
            logger.log(level, f"  {path.name}: {status}")
            outcomes[status] += 1
            report.files_scanned += 1
            _count_status(report, status)
            progress.advance(task)
        progress.finish(task)

        logger.info(
            f"{source_path}: "
            f"{outcomes[FileStatus.INGESTED]} ingested, "
            f"{outcomes[FileStatus.SKIPPED]} skipped, "
            f"{outcomes[FileStatus.REFRESHED]} refreshed, "
            f"{outcomes[FileStatus.FAILED]} failed"
        )


def _count_status(report: IngestReport, status: FileStatus) -> None:
    if status == FileStatus.INGESTED:
        report.files_ingested += 1
    elif status == FileStatus.SKIPPED:
        report.files_skipped += 1
    elif status == FileStatus.REFRESHED:
        report.files_refreshed += 1
    else:
        report.files_failed += 1
