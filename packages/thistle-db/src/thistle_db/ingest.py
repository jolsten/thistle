import contextlib
import datetime
import enum
import hashlib
import pathlib
from collections import Counter
from typing import Iterable, Optional, cast

from loguru import logger
from sqlalchemy import Table, select
from sqlalchemy.orm import Session

from thistle_db.config import DEFAULT_MAX_EPOCH_AHEAD_DAYS, IngestSource
from thistle_db.model import (
    EpochDateState,
    IngestFile,
    MalformedElsetError,
    OmmMetadata,
    TLE,
    utcnow,
)
from thistle_db.progress import NO_PROGRESS, ProgressReporter
from thistle_db.rejects import (
    NO_REJECTS,
    RejectSink,
    RejectWriter,
    namespaces,
    reject_path,
)
from thistle_db.reader import (
    TLETuple,
    detect_format,
    read_omm_csv,
    read_omm_json,
    read_omm_xml,
    read_tle,
)

CHUNK_SIZE = 5000

# Dedup index columns for ON CONFLICT dialects (SQLite/PostgreSQL); MariaDB
# uses INSERT IGNORE, which keys off the same unique constraints.
_TLE_CONFLICT = ["epoch", "line_hash"]
_OMM_CONFLICT = ["tle_id"]

_TLE_TABLE = cast(Table, TLE.__table__)
_OMM_TABLE = cast(Table, OmmMetadata.__table__)
_DATE_STATE_TABLE = cast(Table, EpochDateState.__table__)


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


def _touch_epoch_dates(session: Session, records: list[dict]) -> None:
    """Bump the epoch-date watermarks covered by `records`.

    Called inside the caller's transaction, never committing on its own: the
    watermark and the rows it describes must land together, or that date's
    output files would never regenerate (see `model.EpochDateState`).

    All records in a batch share one `created` (set by `_tle_to_record`), so
    the watermark is taken from the batch rather than read from the clock
    again. Dedup means some of these rows may already have existed — the
    watermark can therefore run ahead of reality, costing at most one
    redundant file rewrite. `INSERT IGNORE` cannot report which rows were
    new per date on MariaDB, and a redundant write is cheaper than finding
    out.
    """
    if not records:
        return
    watermarks: dict[datetime.date, datetime.datetime] = {}
    for record in records:
        epoch_date = record["epoch"].date()
        created = record["created"]
        if watermarks.get(epoch_date, created) <= created:
            watermarks[epoch_date] = created
    rows = [
        {"epoch_date": date, "last_created": created}
        for date, created in sorted(watermarks.items())
    ]

    dialect = session.bind.dialect.name  # type: ignore[union-attr]
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        stmt = sqlite_insert(_DATE_STATE_TABLE)
        stmt = stmt.on_conflict_do_update(
            index_elements=["epoch_date"],
            set_={"last_created": stmt.excluded.last_created},
        )
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg_stmt = pg_insert(_DATE_STATE_TABLE)
        stmt = pg_stmt.on_conflict_do_update(
            index_elements=["epoch_date"],
            set_={"last_created": pg_stmt.excluded.last_created},
        )
    else:
        from sqlalchemy.dialects.mysql import insert as mysql_insert

        my_stmt = mysql_insert(_DATE_STATE_TABLE)
        stmt = my_stmt.on_duplicate_key_update(
            last_created=my_stmt.inserted.last_created
        )
    session.execute(stmt, rows)


def _insert_individually(
    session: Session,
    execute,
    chunk: list[dict],
    rejects: RejectSink,
) -> int:
    """Re-insert a failed chunk one row at a time, quarantining the culprits.

    A chunk fails as a unit — some drivers (pymysql) reject a value while
    building the statement, before any SQL is sent — so a single bad record
    would otherwise cost every good record batched with it, and, since the
    exception propagates, the rest of the file. Slow by design: this runs
    only after a chunk has already failed.
    """
    inserted = 0
    for record in chunk:
        try:
            inserted += execute([record])
            if "epoch" in record:
                _touch_epoch_dates(session, [record])
            session.commit()
        except Exception as exc:
            session.rollback()
            reason = f"database rejected the row: {type(exc).__name__}: {exc}"
            logger.warning(f"{reason} ({_record_ident(record)})")
            rejects.reject(
                reason,
                line1=record.get("line1"),
                line2=record.get("line2"),
            )
    return inserted


def _record_ident(record: dict) -> str:
    """Short identifier for a record dict in a log line."""
    line1 = record.get("line1")
    if line1:
        return str(line1)[:24]
    return f"tle_id={record.get('tle_id')}"


def _bulk_insert_ignore(
    session: Session,
    table: Table,
    records: list[dict],
    index_elements: list[str],
    rejects: RejectSink = NO_REJECTS,
) -> int:
    """Insert records, silently skipping unique-key duplicates.

    Dialect-aware: ON CONFLICT DO NOTHING (SQLite/PostgreSQL), INSERT IGNORE
    (MariaDB/MySQL). Returns count of newly inserted rows.

    A chunk the database refuses is retried row by row so that only the
    offending records are lost (and quarantined); see `_insert_individually`.
    """
    if not records:
        return 0

    execute = _insert_executor(session, table, index_elements)
    total_inserted = 0

    for i in range(0, len(records), CHUNK_SIZE):
        chunk = records[i : i + CHUNK_SIZE]
        try:
            total_inserted += execute(chunk)
            if table is _TLE_TABLE:
                _touch_epoch_dates(session, chunk)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning(
                f"Insert of {len(chunk)} records failed "
                f"({type(exc).__name__}: {exc}); retrying row by row"
            )
            total_inserted += _insert_individually(session, execute, chunk, rejects)

    return total_inserted


def ingest_tles(
    session: Session,
    tles: Iterable[TLETuple],
    rejects: RejectSink = NO_REJECTS,
    *,
    max_epoch_ahead_days: int = DEFAULT_MAX_EPOCH_AHEAD_DAYS,
) -> int:
    """Ingest TLE tuples into the database.

    Consumes the iterable lazily in CHUNK_SIZE batches, so arbitrarily large
    inputs (full-catalog restores) never materialize in memory. Skips
    malformed records and duplicates. Returns count of newly inserted rows.

    Records that cannot be stored are reported to `rejects` — quarantined
    for inspection when the caller supplies a writer, dropped otherwise.
    That includes elsets whose epoch runs more than `max_epoch_ahead_days`
    into the future (see `config.IngestConfig.max_epoch_ahead_days`; 0
    disables) — a corrupted epoch parses cleanly into 2000-2056, and a
    stored one permanently degrades that object's incremental output.
    """
    table = _TLE_TABLE
    total = 0
    records: list[dict] = []
    epoch_limit = _epoch_limit(max_epoch_ahead_days)
    for line1, line2 in tles:
        rejects.seen()
        try:
            tle = TLE.from_twoline(line1, line2)
        except MalformedElsetError as exc:
            # Parsed, but there is no orbit here: storing it would put a row
            # with no usable elements into date/object output files.
            logger.debug(f"Rejecting unusable TLE: {exc}")
            rejects.reject(str(exc), line1=line1, line2=line2)
        except Exception as exc:
            logger.debug(f"Rejecting unparseable TLE: {exc}")
            rejects.reject(f"unparseable: {exc}", line1=line1, line2=line2)
        else:
            if epoch_limit is not None and tle.epoch > epoch_limit:
                logger.debug(
                    f"Rejecting future-epoch TLE: {tle.epoch.isoformat()} is "
                    f"past the {max_epoch_ahead_days}-day horizon"
                )
                # Generic reason (no per-record epoch): the quarantine log
                # pairs each reason with line1, which carries the epoch, and
                # the truncation marker's histogram groups by reason string.
                rejects.reject(
                    f"epoch more than {max_epoch_ahead_days} days in the future",
                    line1=line1,
                    line2=line2,
                )
            else:
                records.append(_tle_to_record(tle))
        if len(records) >= CHUNK_SIZE:
            total += _bulk_insert_ignore(
                session, table, records, _TLE_CONFLICT, rejects
            )
            records = []

    total += _bulk_insert_ignore(session, table, records, _TLE_CONFLICT, rejects)
    return total


def _epoch_limit(max_epoch_ahead_days: int) -> Optional[datetime.datetime]:
    """Latest admissible epoch, or None when the guard is disabled (0).

    Computed once per file rather than per record; the horizon is measured
    in days, so a run's worth of clock drift is irrelevant.
    """
    if max_epoch_ahead_days <= 0:
        return None
    return utcnow() + datetime.timedelta(days=max_epoch_ahead_days)


def _tle_to_record(tle: TLE, now=None) -> dict:
    """Convert a TLE ORM object to a dict for Core-level INSERT."""
    if now is None:
        now = utcnow()
    record = {
        col.name: getattr(tle, col.name)
        for col in TLE.__table__.columns
        if col.name != "id"
    }
    record["created"] = now
    return record


def ingest_omm(
    session: Session,
    omm_records: list[dict],
    rejects: RejectSink = NO_REJECTS,
    *,
    max_epoch_ahead_days: int = DEFAULT_MAX_EPOCH_AHEAD_DAYS,
) -> int:
    """Ingest OMM records into the database.

    Extracts TLE_LINE1/TLE_LINE2 from each record, inserts into TLE table,
    then populates OmmMetadata for rows that don't have it yet.
    Returns count of newly inserted TLE rows.

    Records that cannot be stored are reported to `rejects`, which
    quarantines them as OMM JSON — the original record, not just its lines.
    """
    # Build TLE records from OMM data
    tle_records = []
    omm_by_hash: dict[bytes, dict] = {}
    epoch_limit = _epoch_limit(max_epoch_ahead_days)

    for omm in omm_records:
        rejects.seen()
        line1 = omm.get("TLE_LINE1", "").strip()
        line2 = omm.get("TLE_LINE2", "").strip()
        if not line1 or not line2:
            logger.debug(
                f"OMM record missing TLE lines: {omm.get('OBJECT_NAME', 'unknown')}"
            )
            rejects.reject("OMM record has no TLE_LINE1/TLE_LINE2", record=omm)
            continue

        try:
            tle = TLE.from_twoline(line1, line2)
        except MalformedElsetError as exc:
            logger.debug(f"Rejecting unusable OMM record: {exc}")
            rejects.reject(str(exc), line1=line1, line2=line2, record=omm)
        except Exception as exc:
            logger.debug(f"Rejecting unparseable OMM record: {exc}")
            rejects.reject(f"unparseable: {exc}", line1=line1, line2=line2, record=omm)
        else:
            if epoch_limit is not None and tle.epoch > epoch_limit:
                logger.debug(
                    f"Rejecting future-epoch OMM record: "
                    f"{tle.epoch.isoformat()} is past the "
                    f"{max_epoch_ahead_days}-day horizon"
                )
                rejects.reject(
                    f"epoch more than {max_epoch_ahead_days} days in the future",
                    line1=line1,
                    line2=line2,
                    record=omm,
                )
            else:
                record = _tle_to_record(tle)
                tle_records.append(record)
                omm_by_hash[record["line_hash"]] = omm

    # Bulk insert TLE rows
    inserted = _bulk_insert_ignore(
        session, _TLE_TABLE, tle_records, _TLE_CONFLICT, rejects
    )

    if not omm_by_hash:
        return inserted

    # FK resolution: SELECT back the TLE ids that still lack metadata.
    # The lookup must constrain on epoch AND line_hash: the only index
    # containing line_hash is UNIQUE(epoch, line_hash), and epoch is its
    # leading column — line_hash alone would be a full-table scan. The
    # epoch IN-list can only exclude rows, never wrongly include: a
    # matching hash implies the row's epoch equals its own record's epoch
    # (the epoch is encoded in line1).
    pairs = [(rec["epoch"], rec["line_hash"]) for rec in tle_records]
    for i in range(0, len(pairs), CHUNK_SIZE):
        chunk = pairs[i : i + CHUNK_SIZE]
        epochs = sorted({e for e, _ in chunk})
        hashes = [h for _, h in chunk]
        stmt = (
            select(TLE.id, TLE.line_hash)
            .where(TLE.epoch.in_(epochs), TLE.line_hash.in_(hashes))
            .outerjoin(OmmMetadata, TLE.id == OmmMetadata.tle_id)
            .where(OmmMetadata.id.is_(None))
        )
        rows = session.execute(stmt).all()

        now = utcnow()
        metadata_records = []
        for tle_id, line_hash in rows:
            omm = omm_by_hash.get(line_hash)
            if omm is None:
                continue
            metadata_records.append(
                dict(
                    tle_id=tle_id,
                    object_name=omm.get("OBJECT_NAME"),
                    object_type=omm.get("OBJECT_TYPE"),
                    country_code=omm.get("COUNTRY_CODE"),
                    rcs_size=omm.get("RCS_SIZE"),
                    launch_date=omm.get("LAUNCH_DATE"),
                    site=omm.get("SITE"),
                    decay_date=omm.get("DECAY_DATE"),
                    originator=omm.get("ORIGINATOR"),
                    gp_id=int(omm["GP_ID"]) if omm.get("GP_ID") else None,
                    created=now,
                )
            )

        _bulk_insert_ignore(session, _OMM_TABLE, metadata_records, _OMM_CONFLICT)

    return inserted


def ingest_file(
    session: Session,
    path: pathlib.Path,
    *,
    reject_dir: Optional[pathlib.Path] = None,
    reject_root: Optional[pathlib.Path] = None,
    reject_namespace: Optional[str] = None,
    max_epoch_ahead_days: int = DEFAULT_MAX_EPOCH_AHEAD_DAYS,
) -> int:
    """Detect format and ingest a single file. Returns count of newly inserted rows.

    With `reject_dir` set, records that cannot be stored are quarantined
    under it, mirroring the file's path below `reject_root` (the configured
    source directory) so provenance stays visible.
    """
    fmt = detect_format(path)
    logger.info(f"Ingesting {path} (format: {fmt})")

    if fmt not in ("tle", "omm_json", "omm_csv", "omm_xml"):
        logger.warning(f"Unknown format for {path}, skipping")
        return 0

    with _reject_writer(
        path, fmt, reject_dir, reject_root, reject_namespace
    ) as rejects:
        ahead = max_epoch_ahead_days
        if fmt == "tle":
            count = ingest_tles(
                session, read_tle(path), rejects, max_epoch_ahead_days=ahead
            )
        elif fmt == "omm_json":
            count = ingest_omm(
                session, read_omm_json(path), rejects, max_epoch_ahead_days=ahead
            )
        elif fmt == "omm_csv":
            count = ingest_omm(
                session, read_omm_csv(path), rejects, max_epoch_ahead_days=ahead
            )
        else:
            count = ingest_omm(
                session, read_omm_xml(path), rejects, max_epoch_ahead_days=ahead
            )

        summary = rejects.summary() if isinstance(rejects, RejectWriter) else None
    if summary:
        # Reported as a fraction: the ratio is what separates a few bad
        # lines in a good file from an entirely wrong file.
        logger.warning(f"  {path.name}: {summary}")
    return count


@contextlib.contextmanager
def _reject_writer(
    path: pathlib.Path,
    fmt: str,
    reject_dir: Optional[pathlib.Path],
    reject_root: Optional[pathlib.Path],
    namespace: Optional[str],
):
    """A RejectWriter for `path`, or the inert sink when quarantine is off."""
    if reject_dir is None:
        yield NO_REJECTS
        return
    target = reject_path(
        path, reject_dir, root=reject_root, namespace=namespace
    )
    with RejectWriter(path, target, omm=fmt != "tle") as writer:
        yield writer


class FileStatus(enum.StrEnum):
    INGESTED = "ingested"
    """File was parsed and inserted (possibly 0 new rows via dedup)."""
    SKIPPED = "skipped"
    """Size and mtime match recorded state; file not opened."""
    REFRESHED = "refreshed"
    """Stat changed but content hash identical; state refreshed, no parse."""
    FAILED = "failed"
    """Stat/read/parse raised; logged, state not recorded (retried next scan)."""


def _exclude_under(
    files: list[pathlib.Path], directory: pathlib.Path
) -> tuple[list[pathlib.Path], int]:
    """Split `files` into those outside `directory` and a count of those in it."""
    root = directory.resolve()
    kept = [f for f in files if not f.resolve().is_relative_to(root)]
    return kept, len(files) - len(kept)


def _path_key(path: pathlib.Path) -> tuple[str, str]:
    """Return (absolute resolved path, sha256 hex of that path string)."""
    resolved = str(path.resolve())
    return resolved, hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ingest_source_file(
    session: Session,
    path: pathlib.Path,
    *,
    force: bool = False,
    reject_dir: Optional[pathlib.Path] = None,
    reject_root: Optional[pathlib.Path] = None,
    reject_namespace: Optional[str] = None,
    max_epoch_ahead_days: int = DEFAULT_MAX_EPOCH_AHEAD_DAYS,
) -> tuple[FileStatus, int]:
    """Ingest one file with state tracking and error isolation.

    Unchanged files (per the ingest_files state table) are skipped without
    being opened. Returns (status, newly_inserted_row_count). Never raises for
    per-file problems — logs a warning and returns (FAILED, 0) instead, leaving
    the file's state unrecorded so it is retried on the next scan.
    """
    try:
        resolved, key = _path_key(path)
        # Stat before reading: a file modified mid-ingest looks changed on the
        # next scan; the redundant re-ingest is absorbed by DB dedup.
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
            return FileStatus.SKIPPED, 0

        digest = _file_sha256(path)

        if not force and row is not None and row.sha256 == digest:
            row.size = stat.st_size
            row.mtime_ns = stat.st_mtime_ns
            session.commit()
            logger.debug(f"Content unchanged, refreshed state: {path}")
            return FileStatus.REFRESHED, 0

        count = ingest_file(
            session,
            path,
            reject_dir=reject_dir,
            reject_root=reject_root,
            reject_namespace=reject_namespace,
            max_epoch_ahead_days=max_epoch_ahead_days,
        )

        if row is None:
            session.add(
                IngestFile(
                    path=resolved,
                    path_hash=key,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=digest,
                )
            )
        else:
            row.size = stat.st_size
            row.mtime_ns = stat.st_mtime_ns
            row.sha256 = digest
        session.commit()

        return FileStatus.INGESTED, count
    except Exception as exc:
        session.rollback()
        logger.warning(f"Failed to ingest {path}: {exc}")
        return FileStatus.FAILED, 0


def ingest_sources(
    session: Session,
    sources: list[IngestSource],
    *,
    force: bool = False,
    reject_dir: Optional[pathlib.Path] = None,
    max_epoch_ahead_days: int = DEFAULT_MAX_EPOCH_AHEAD_DAYS,
    progress: ProgressReporter = NO_PROGRESS,
) -> int:
    """Scan configured source directories and ingest all matching files.

    Unchanged files are skipped via the ingest_files state table; a bad file
    is logged and never aborts the scan. Pass force=True to re-ingest
    regardless of recorded state.

    Per-file outcomes log at INFO, except routine skips (unchanged files),
    which log at DEBUG — on a steady-state scan they are the overwhelming
    majority and pure noise. Each directory gets an INFO summary line.

    With `reject_dir` set, unstorable records are quarantined there under a
    namespace per source directory (see `rejects.namespaces`).
    """
    total = 0
    roots = [pathlib.Path(source.path) for source in sources]
    # Resolved once for the whole scan: two configured sources with the same
    # basename must not share a quarantine namespace.
    namespace_by_root = namespaces(roots)
    for source in sources:
        source_path = pathlib.Path(source.path)
        if not source_path.is_dir():
            logger.warning(f"Source directory does not exist: {source_path}")
            continue

        files = sorted(source_path.glob(source.pattern))
        if reject_dir is not None:
            # A reject_dir inside a source directory would re-ingest its own
            # quarantine every run, rejecting the same records again and
            # nesting the tree one level deeper each time.
            files, quarantined = _exclude_under(files, reject_dir)
            if quarantined:
                logger.warning(
                    f"Ignoring {quarantined} file(s) under the quarantine "
                    f"directory {reject_dir} — move reject_dir outside "
                    f"{source_path} to silence this"
                )
        logger.info(
            f"Found {len(files)} files in {source_path} matching {source.pattern}"
        )

        task = progress.task(f"Ingesting {source_path.name}", total=len(files))
        outcomes = Counter()
        source_total = 0
        for file in files:
            status, count = ingest_source_file(
                session,
                file,
                force=force,
                reject_dir=reject_dir,
                reject_root=source_path,
                reject_namespace=namespace_by_root.get(str(source_path)),
                max_epoch_ahead_days=max_epoch_ahead_days,
            )
            level = "DEBUG" if status == FileStatus.SKIPPED else "INFO"
            logger.log(level, f"  {file.name}: {status}, {count} new records")
            outcomes[status] += 1
            source_total += count
            progress.advance(task)
        progress.finish(task)

        logger.info(
            f"{source_path}: {source_total} new records "
            f"({outcomes[FileStatus.INGESTED]} ingested, "
            f"{outcomes[FileStatus.SKIPPED]} skipped, "
            f"{outcomes[FileStatus.REFRESHED]} refreshed, "
            f"{outcomes[FileStatus.FAILED]} failed)"
        )
        total += source_total

    return total
