"""Output rendering — cursor-scoped updates and full rebuilds, one write path.

Outputs are lossy projections of the store (TLE text has no metadata; the
STK CSV has no provenance), always rebuildable, never the reverse. They
are not written during ingest: ``update_outputs`` runs afterwards
(``thistle-db update-outputs``, or ``ingest --update-outputs``) and
``rebuild_outputs`` re-derives everything.

**One write path.** A file is always rendered *whole* from the store and
replaced atomically (sibling ``.tmp`` + ``os.replace``) — there are no
appends, no merges with on-disk content, and therefore no torn tails and
no half-written files at any kill point. Every file either path touches
is byte-identical to what a full rebuild would produce, so touching is
healing. Stray ``.tmp`` files from a killed render are swept at the start
of the next run (safe under the writer lock).

**Scoping.** Each ``[[output.files]]`` entry records the highest
``tle.id`` it reflects (``output_cursor``). ``update_outputs`` takes one
PK-range scan over the rows past the cursor to find the dirty object ids
and dirty epoch dates, re-renders only those files, and advances every
cursor — only after *all* dirty files rendered, so a cancelled run
re-renders the same dirty set next time, idempotently. ``rebuild``
initializes/resets the cursors; a new or renamed entry has none and must
be rebuilt first.

Enrichment caveat: an UPDATE of ``omm_metadata`` creates no new ``tle``
row, so a redelivery with identical lines but richer metadata dirties
nothing — the refresh lands when that object/day next dirties, or at the
next rebuild.

All filesystem work runs on the write pool; each task owns a distinct
path and receives plain data. Sessions and queries stay on the calling
thread.
"""

import csv
import dataclasses
import datetime
import hashlib
import io
import os
import pathlib
import struct
from typing import Iterable, Optional, Sequence

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from thistle_db.config import MAX_SATNUM, OutputConfig, OutputFile
from thistle_db.model import (
    OmmMetadata,
    OutputCursor,
    TLE,
    date_from_sql,
    record_from_row,
    utcnow,
)
from thistle_db.progress import NO_PROGRESS
from thistle_db.reader import OMM_CSV_FIELDS
from thistle_db.record import GPRecord, STORE_COLUMNS, to_csv_dict, to_omm_dict
from thistle_db.writer import WritePool

__all__ = [
    "MissingCursorError",
    "OutputRun",
    "output_key",
    "output_lag",
    "rebuild_outputs",
    "update_outputs",
]


class MissingCursorError(RuntimeError):
    """An output entry has no cursor row — `rebuild` initializes them.

    Raised by ``update_outputs``: an entry without a cursor has never been
    rendered (new, or renamed — the key embeds the directory), and its
    catch-up set is the whole store, which is exactly what ``rebuild``
    does with better streaming (and it sets the cursor when done).
    """

    def __init__(self, keys: Sequence[str]):
        self.keys = list(keys)
        super().__init__(
            "no output cursor for: "
            + ", ".join(self.keys)
            + " — run `thistle-db rebuild` to initialize new or renamed outputs"
        )


def output_key(entry: OutputFile) -> str:
    """Stable identifier for an output entry in reports/logs/cursors."""
    return f"{entry.type}:{entry.format}:{entry.dir}"


def _fieldnames(entry: OutputFile) -> list[str]:
    return STORE_COLUMNS if entry.columns == "full" else OMM_CSV_FIELDS


# Element fields stored as single precision (model.Float32). CSV outputs
# render at *stored* precision so every render of the same store rows is
# byte-identical, whichever command produced it.
_F32_FIELDS = (
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
)


def _f32(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _stored_precision(record: GPRecord) -> GPRecord:
    changes = {
        name: rounded
        for name in _F32_FIELDS
        if (rounded := _f32(getattr(record, name))) != getattr(record, name)
    }
    return dataclasses.replace(record, **changes) if changes else record


def _row_for(entry: OutputFile, record: GPRecord) -> dict:
    record = _stored_precision(record)
    return to_csv_dict(record) if entry.columns == "full" else to_omm_dict(record)


def _atomic_write(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _render_csv(fieldnames: Sequence[str], rows: Iterable[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _render_tle(pairs: Iterable[tuple[str, str]]) -> str:
    return "".join(f"{line1}\n{line2}\n" for line1, line2 in pairs)


# --- output-file readers (verify's counting half) -----------------------------


def read_tle_pairs(path: pathlib.Path) -> list[tuple[str, str]]:
    """All (line1, line2) pairs in an output file."""
    pairs: list[tuple[str, str]] = []
    line1: Optional[str] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if line.startswith("1 "):
                line1 = line
            elif line.startswith("2 ") and line1 is not None:
                pairs.append((line1, line))
                line1 = None
    return pairs


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


# --- cursors -------------------------------------------------------------------


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_cursors(
    session: Session, entries: Sequence[OutputFile]
) -> dict[str, Optional[int]]:
    """Each entry's last_tle_id, keyed by output_key; None = no cursor yet."""
    keys = {output_key(entry): _key_hash(output_key(entry)) for entry in entries}
    rows = session.execute(
        select(OutputCursor).where(OutputCursor.key_hash.in_(list(keys.values())))
    ).scalars()
    by_hash = {row.key_hash: row.last_tle_id for row in rows}
    return {key: by_hash.get(digest) for key, digest in keys.items()}


def set_cursors(
    session: Session, entries: Sequence[OutputFile], target: int
) -> None:
    """Upsert every entry's cursor to `target` and commit.

    Plain select-then-write: the single-writer lock serializes all access,
    so no dialect upsert is needed (same pattern as ingest_files).
    """
    now = utcnow()
    for entry in entries:
        key = output_key(entry)
        digest = _key_hash(key)
        row = session.execute(
            select(OutputCursor).where(OutputCursor.key_hash == digest)
        ).scalar_one_or_none()
        if row is None:
            session.add(
                OutputCursor(
                    output_key=key, key_hash=digest, last_tle_id=target, updated=now
                )
            )
        else:
            row.last_tle_id = target
            row.updated = now
    session.commit()


def output_lag(
    session: Session, entries: Sequence[OutputFile]
) -> tuple[int, list[str]]:
    """(store ids past the slowest cursor, entries with no cursor).

    For `status`. The id count is an upper bound on pending rows (dialect
    insert-ignore burns ids on skipped duplicates).
    """
    cursors = load_cursors(session, entries)
    missing = sorted(key for key, value in cursors.items() if value is None)
    present = [value for value in cursors.values() if value is not None]
    if not present:
        return 0, missing
    max_id = session.execute(select(func.max(TLE.id))).scalar_one() or 0
    return max(0, max_id - min(present)), missing


def _sweep_tmp_files(entries: Sequence[OutputFile]) -> None:
    """Delete stray *.tmp files left by a killed render.

    Safe under the writer lock — no render is in flight. The worst a kill
    can leave behind is a complete old-or-new file plus one of these.
    """
    for directory in {pathlib.Path(entry.dir) for entry in entries}:
        if not directory.is_dir():
            continue
        for tmp in directory.glob("*.tmp"):
            try:
                tmp.unlink()
            except OSError:
                pass


# --- rendering (pure filesystem; runs on the pool) ----------------------------


def _render_object_file(
    entry: OutputFile, norad: int, records: list[GPRecord]
) -> tuple[int, int]:
    """Render one object's whole file from its store records.

    Returns (files_written, lineless_skipped). Records arrive in
    (epoch, id) order from the query. TLE outputs are verbatim
    passthrough: line-less records are skipped and counted.
    """
    path = entry.object_path(norad)
    if entry.format == "tle":
        pairs = [
            (r.tle_line1, r.tle_line2)
            for r in records
            if r.tle_line1 and r.tle_line2
        ]
        lineless = len(records) - len(pairs)
        if not pairs:
            return (0, lineless)
        _atomic_write(path, _render_tle(pairs))  # type: ignore[arg-type]
        return (1, lineless)
    rows = [_row_for(entry, r) for r in records]
    _atomic_write(path, _render_csv(_fieldnames(entry), rows))
    return (1, 0)


def _render_date_file(
    entry: OutputFile, day: datetime.date, records: list[GPRecord]
) -> tuple[int, int]:
    """Render one date's whole file (the day's winner per object).

    Returns (files_written, lineless_skipped)."""
    path = entry.date_path(day)
    ordered = sorted(records, key=lambda r: r.norad_cat_id)
    if entry.format == "tle":
        pairs = [
            (r.tle_line1, r.tle_line2)
            for r in ordered
            if r.tle_line1 and r.tle_line2
        ]
        lineless = len(ordered) - len(pairs)
        if not pairs:
            return (0, lineless)
        _atomic_write(path, _render_tle(pairs))  # type: ignore[arg-type]
        return (1, lineless)
    rows = [_row_for(entry, r) for r in ordered]
    _atomic_write(path, _render_csv(_fieldnames(entry), rows))
    return (1, 0)


# --- the render core -----------------------------------------------------------

# Objects per pool batch: bounds the rows materialized at once while
# keeping the write pool busy.
_CHUNK_OBJECTS = 200


@dataclasses.dataclass
class _RenderStats:
    written: dict[str, int] = dataclasses.field(default_factory=dict)
    lineless_skipped: int = 0
    alpha5_overflow: set[int] = dataclasses.field(default_factory=set)


def _flush_object_batch(
    entries: Sequence[OutputFile],
    batch: dict[int, list[GPRecord]],
    pool: WritePool,
    stats: _RenderStats,
    progress,
    task,
) -> None:
    if not batch:
        return
    for entry in entries:
        key = output_key(entry)
        if entry.object_id == "alpha5":
            stats.alpha5_overflow.update(n for n in batch if n > MAX_SATNUM)

        def write_one(
            item: tuple[int, list[GPRecord]], entry: OutputFile = entry
        ) -> tuple[int, int]:
            norad, records = item
            return _render_object_file(entry, norad, records)

        results = pool.map(write_one, list(batch.items()))
        stats.written[key] = stats.written.get(key, 0) + sum(w for w, _ in results)
        stats.lineless_skipped += sum(skipped for _, skipped in results)
        progress.advance(task, len(batch))


def _iter_records(rows) -> Iterable[tuple[int, GPRecord]]:
    for tle, metadata in rows:
        yield tle.norad_cat_id, record_from_row(tle, metadata)


def _render_objects_full(session, entries, pool, stats, progress) -> None:
    """Every object, from one linear store scan (rebuild)."""
    count = session.execute(
        select(func.count(func.distinct(TLE.norad_cat_id)))
    ).scalar_one()
    task = progress.task("Rebuilding object files", total=count * len(entries))

    stmt = (
        select(TLE, OmmMetadata)
        .outerjoin(OmmMetadata, OmmMetadata.tle_id == TLE.id)
        .order_by(TLE.norad_cat_id, TLE.epoch, TLE.id)
        .execution_options(yield_per=5000)
    )

    batch: dict[int, list[GPRecord]] = {}
    current_norad: Optional[int] = None
    for norad, record in _iter_records(session.execute(stmt)):
        if norad != current_norad and len(batch) >= _CHUNK_OBJECTS:
            _flush_object_batch(entries, batch, pool, stats, progress, task)
            batch = {}
        current_norad = norad
        batch.setdefault(norad, []).append(record)
    _flush_object_batch(entries, batch, pool, stats, progress, task)
    progress.finish(task)


def _render_objects_scoped(
    session, entries, pool, stats, progress, norads: list[int]
) -> None:
    """Only the dirty objects, each rendered from its full history."""
    task = progress.task(
        "Updating object files", total=len(norads) * len(entries)
    )
    for i in range(0, len(norads), _CHUNK_OBJECTS):
        chunk = norads[i : i + _CHUNK_OBJECTS]
        stmt = (
            select(TLE, OmmMetadata)
            .outerjoin(OmmMetadata, OmmMetadata.tle_id == TLE.id)
            .where(TLE.norad_cat_id.in_(chunk))
            .order_by(TLE.norad_cat_id, TLE.epoch, TLE.id)
            .execution_options(yield_per=5000)
        )
        batch: dict[int, list[GPRecord]] = {}
        for norad, record in _iter_records(session.execute(stmt)):
            batch.setdefault(norad, []).append(record)
        _flush_object_batch(entries, batch, pool, stats, progress, task)
    progress.finish(task)


def _render_dates(
    session,
    entries,
    stats: _RenderStats,
    progress,
    *,
    days: Optional[list[datetime.date]] = None,
) -> None:
    """Date files from per-(date, object) winners computed in SQL.

    ``days=None`` renders every date (rebuild); otherwise only the given
    dirty days. Scoping filters by an epoch range covering the dirty days
    — the UNIQUE(epoch, record_hash) index is epoch-first, so this is a
    range scan — plus an exact day check in the loop (gap days inside the
    hull cost their rows' window computation, nothing else).
    """
    date_expr = func.date(TLE.epoch).label("epoch_date")
    rn = (
        func.row_number()
        .over(
            partition_by=(date_expr, TLE.norad_cat_id),
            order_by=(TLE.epoch.desc(), TLE.id.desc()),
        )
        .label("rn")
    )
    winners_sel = select(TLE.id.label("tle_id"), date_expr, rn)
    if days:
        start = datetime.datetime.combine(min(days), datetime.time.min)
        end = datetime.datetime.combine(
            max(days) + datetime.timedelta(days=1), datetime.time.min
        )
        winners_sel = winners_sel.where(TLE.epoch >= start, TLE.epoch < end)
    winners = winners_sel.subquery()
    stmt = (
        select(TLE, OmmMetadata, winners.c.epoch_date)
        .join(winners, TLE.id == winners.c.tle_id)
        .outerjoin(OmmMetadata, OmmMetadata.tle_id == TLE.id)
        .where(winners.c.rn == 1)
        .order_by(winners.c.epoch_date, TLE.norad_cat_id)
        .execution_options(yield_per=5000)
    )

    days_set = set(days) if days else None
    task = progress.task(
        "Updating date files" if days else "Rebuilding date files",
        total=len(days) if days else None,
    )
    current_date: Optional[datetime.date] = None
    day_records: list[GPRecord] = []

    def flush_day() -> None:
        if current_date is None or not day_records:
            return
        if days_set is not None and current_date not in days_set:
            return
        for entry in entries:
            key = output_key(entry)
            wrote, lineless = _render_date_file(entry, current_date, day_records)
            stats.written[key] = stats.written.get(key, 0) + wrote
            stats.lineless_skipped += lineless
        progress.advance(task)

    for tle, metadata, raw_date in session.execute(stmt):
        day = date_from_sql(raw_date)
        if day != current_date:
            flush_day()
            current_date = day
            day_records = []
        day_records.append(record_from_row(tle, metadata))
    flush_day()
    progress.finish(task)


def _log_render_warnings(stats: _RenderStats) -> None:
    if stats.lineless_skipped:
        logger.warning(
            f"{stats.lineless_skipped} line-less record(s) have no TLE "
            "rendering and are absent from tle outputs"
        )
    if stats.alpha5_overflow:
        ids = sorted(stats.alpha5_overflow)
        logger.warning(
            f"{len(ids)} object(s) beyond alpha-5 range named with "
            f"integer stems (e.g. {ids[0]}) in alpha5 outputs"
        )


# --- entry points ----------------------------------------------------------------


@dataclasses.dataclass
class OutputRun:
    """Structured result of one ``update_outputs`` run."""

    cursor: int = 0
    """Where the slowest entry stood before this run."""
    target: int = 0
    """max(tle.id) this run caught every entry up to."""
    dirty_objects: int = 0
    dirty_days: int = 0
    written: dict[str, int] = dataclasses.field(default_factory=dict)
    """Files written per output key."""
    lineless_skipped: int = 0
    alpha5_overflow: set[int] = dataclasses.field(default_factory=set)

    @property
    def up_to_date(self) -> bool:
        return self.target <= self.cursor


def update_outputs(
    session: Session,
    output: OutputConfig,
    *,
    progress=None,
) -> OutputRun:
    """Re-render exactly the output files dirtied since the last update.

    One PK-range scan over ``(cursor, target]`` finds the dirty object ids
    and dirty epoch dates; each dirty file is rendered whole from the
    store (byte-identical to a rebuild of that file) and every entry's
    cursor advances to ``target`` — only after all renders complete, so a
    cancelled run redoes the same dirty set next time. Raises
    ``MissingCursorError`` for entries ``rebuild`` has never initialized.
    """
    progress = progress or NO_PROGRESS
    entries = output.files
    cursors = load_cursors(session, entries)
    missing = sorted(key for key, value in cursors.items() if value is None)
    if missing:
        raise MissingCursorError(missing)

    cursor = min(value for value in cursors.values() if value is not None)
    target = session.execute(select(func.max(TLE.id))).scalar_one() or 0
    run = OutputRun(cursor=cursor, target=target)
    if run.up_to_date:
        logger.info(f"Outputs are up to date (cursor at store id {cursor})")
        return run

    for entry in entries:
        pathlib.Path(entry.dir).mkdir(parents=True, exist_ok=True)
    _sweep_tmp_files(entries)

    norads = sorted(
        session.execute(
            select(TLE.norad_cat_id)
            .where(TLE.id > cursor, TLE.id <= target)
            .distinct()
        ).scalars()
    )
    days = sorted(
        {
            date_from_sql(raw)
            for raw in session.execute(
                select(func.date(TLE.epoch))
                .where(TLE.id > cursor, TLE.id <= target)
                .distinct()
            ).scalars()
            if raw is not None
        }
    )
    run.dirty_objects = len(norads)
    run.dirty_days = len(days)

    stats = _RenderStats()
    object_entries = [entry for entry in entries if entry.type == "object"]
    date_entries = [entry for entry in entries if entry.type == "date"]
    with WritePool(output.write_workers) as pool:
        if object_entries and norads:
            _render_objects_scoped(
                session, object_entries, pool, stats, progress, norads
            )
        if date_entries and days:
            _render_dates(session, date_entries, stats, progress, days=days)

    set_cursors(session, entries, target)
    _log_render_warnings(stats)
    run.written = stats.written
    run.lineless_skipped = stats.lineless_skipped
    run.alpha5_overflow = stats.alpha5_overflow
    logger.info(
        f"Updated {sum(stats.written.values())} output file(s): "
        f"{run.dirty_objects} dirty object(s), {run.dirty_days} dirty day(s)"
    )
    return run


def rebuild_outputs(
    session: Session,
    output: OutputConfig,
    *,
    progress=None,
) -> dict[str, int]:
    """Re-derive every output from a linear store scan; reset the cursors.

    First run, disaster recovery, or a new/renamed output entry. The
    cursor target is captured before the scan (stable under the writer
    lock) and committed only at the end, so an interrupted rebuild leaves
    the cursors untouched. Returns files written per output key.
    """
    progress = progress or NO_PROGRESS
    object_entries = [entry for entry in output.files if entry.type == "object"]
    date_entries = [entry for entry in output.files if entry.type == "date"]
    for entry in output.files:
        pathlib.Path(entry.dir).mkdir(parents=True, exist_ok=True)
    _sweep_tmp_files(output.files)

    target = session.execute(select(func.max(TLE.id))).scalar_one() or 0
    stats = _RenderStats()
    with WritePool(output.write_workers) as pool:
        if object_entries:
            _render_objects_full(session, object_entries, pool, stats, progress)
        if date_entries:
            _render_dates(session, date_entries, stats, progress)

    set_cursors(session, output.files, target)
    _log_render_warnings(stats)
    return stats.written
