"""Output file generation.

Outputs are declared as ``[[output.files]]`` entries — each names a type
(``date`` or ``object``), a format (``tle`` or ``omm``), a destination
directory, and a filename scheme (see ``OutputFile``). Steady-state cost is
O(new rows), independent of catalog size, with no persistent generator
state:

- **Date files** (latest elset per object that day) are rewritten when the
  database holds rows newer than the file. Scope comes from
  ``epoch_date_state``, a one-row-per-date watermark table that ingest
  maintains in the same transaction as the rows it describes: each date's
  watermark is compared against that date's file mtime, so a run costs one
  ``stat`` per date per output and rewrites only what is genuinely stale.
  Because the table grows by a row a day rather than with the catalog,
  *every* date file is checked every run — there is no trailing window and
  no lookback for dates, so a generation gap of any length, a restore that
  resets ``created``, and a delivery for a date years old all self-correct
  on the next ordinary run. Silent corruption of a file whose watermark is
  older than it is caught by ``--verify``, not by the normal run.
- **Object files** (full epoch-ordered history per object) are appended to:
  rows created within ``output.lookback_days`` are streamed in keyset
  batches ordered by (norad_cat_id, epoch, id) and appended to each object's
  file in every output. A tail guard makes this idempotent — only rows
  strictly newer than the file's last epoch are appended, so overlapping
  lookbacks across cron runs never duplicate. Each output decides
  independently; anything irregular (late delivery, epoch tie at the
  boundary, torn last line from a crash, missing/odd file) falls back to a
  full rewrite of that object's file from the database, which is
  authoritative. The lookback must comfortably exceed the ingest cadence;
  if generation hasn't run for longer than the lookback, run ``--all``.

Rows are read as Core column tuples rather than ORM entities, and the
filesystem work runs on a thread pool (``writer.WritePool``) — see those
modules for why. Queries always run on the calling thread; the pool only
ever receives data that has already been fetched.
"""

import csv
import datetime
import io
import os
import pathlib
from typing import Iterator, Optional, Protocol, Sequence, cast

from loguru import logger
from sgp4.api import Satrec
from sgp4.exporter import export_omm
from sqlalchemy import and_, func, null, or_, select
from sqlalchemy.orm import Session

from thistle_db.config import OutputConfig, OutputFile
from thistle_db.model import (
    EpochDateState,
    OmmMetadata,
    TLE,
    date_from_sql,
    epoch_from_lines,
    utcnow,
)
from thistle_db.progress import NO_PROGRESS, ProgressReporter
from thistle_db.reader import OMM_CSV_FIELDS, render_tle
from thistle_db.writer import WritePool

BATCH_SIZE = 50_000

# Objects processed between thread-pool barriers. Large enough that pool
# startup and the batched rewrite query amortize, small enough that only a
# bounded slice of the catalog's rows is in memory at once.
OBJECT_CHUNK = 256

# Read at most this many bytes from the end of a file to find its last
# record. Two TLE lines are ~140 bytes; OMM CSV rows are a few hundred.
_TAIL_BYTES = 8192


class Elset(Protocol):
    """The columns generation reads for one element set.

    A structural view of the Core row tuples the queries below return —
    materializing ORM entities instead cost ~7.6s of a 38s full rebuild and
    held the whole batch as Python objects.
    """

    id: int
    norad_cat_id: int
    epoch: datetime.datetime
    created: datetime.datetime
    line1: str
    line2: str
    object_id: str
    object_name: Optional[str]


def _elset_columns(with_names: bool):
    """Columns for an elset query, joined to OMM metadata only when needed.

    ``object_name`` is only consulted when an omm output exists; without one
    the outer join would be a per-row index probe bought for nothing, so a
    NULL literal stands in and keeps the row shape identical.
    """
    return (
        TLE.id,
        TLE.norad_cat_id,
        TLE.epoch,
        TLE.created,
        TLE.line1,
        TLE.line2,
        TLE.object_id,
        OmmMetadata.object_name if with_names else null().label("object_name"),
    )


def _needs_names(outputs: Sequence[OutputFile]) -> bool:
    return any(out.format == "omm" for out in outputs)


def _join_names(stmt, with_names: bool):
    if with_names:
        return stmt.outerjoin(OmmMetadata, TLE.id == OmmMetadata.tle_id)
    return stmt


def _reconstruct_satrec(elset: Elset) -> Satrec:
    """Reconstruct a Satrec object from stored TLE lines."""
    return Satrec.twoline2rv(elset.line1, elset.line2)


def _safe_export_omm(sat: Satrec, name: str) -> dict | None:
    """Export OMM dict from Satrec, returning None if export fails."""
    try:
        return export_omm(sat, name)
    except (ValueError, AttributeError):
        return None


def _get_object_name(elset: Elset) -> str:
    """Object name from OMM metadata if available, else the object_id."""
    return elset.object_name or elset.object_id


def _omm_rows(elsets: Sequence[Elset]) -> list[list]:
    """OMM CSV rows, in OMM_CSV_FIELDS order, for elsets sgp4 can export.

    Rows are built as lists rather than dicts so the writer can use
    `csv.writer`: `DictWriter` re-reads every field from the dict per row,
    which showed up as 7.3M dict lookups in the profile.
    """
    rows = []
    for elset in elsets:
        sat = _reconstruct_satrec(elset)
        record = _safe_export_omm(sat, _get_object_name(elset))
        if record is not None:
            rows.append([record[field] for field in OMM_CSV_FIELDS])
    return rows


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_tle_full(path: pathlib.Path, elsets: Sequence[Elset]) -> None:
    with open(path, "w") as f:
        f.write(render_tle((e.line1, e.line2) for e in elsets))


def _write_omm_full(path: pathlib.Path, rows: Sequence[Sequence]) -> None:
    """Write an OMM CSV with header. No rows → no file (and any stale file
    removed, since a rewrite is authoritative)."""
    if not rows:
        path.unlink(missing_ok=True)
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OMM_CSV_FIELDS)
        writer.writerows(rows)


def _append_omm(path: pathlib.Path, rows: Sequence[Sequence]) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(OMM_CSV_FIELDS)
        writer.writerows(rows)


def _append_tle(path: pathlib.Path, elsets: Sequence[Elset]) -> None:
    with open(path, "a") as f:
        f.write(render_tle((e.line1, e.line2) for e in elsets))


def _write_outputs(
    outputs: Sequence[OutputFile],
    paths: Sequence[pathlib.Path],
    elsets: Sequence[Elset],
) -> None:
    """Write the same elsets to one path per output, in each output's format.

    The OMM export is shared across omm outputs — it is the expensive part
    (a Satrec reconstruction plus a field export per record).
    """
    omm_rows: Optional[list[list]] = None
    for out, path in zip(outputs, paths):
        if out.format == "tle":
            _write_tle_full(path, elsets)
        else:
            if omm_rows is None:
                omm_rows = _omm_rows(elsets)
            _write_omm_full(path, omm_rows)


# ---------------------------------------------------------------------------
# Date files
# ---------------------------------------------------------------------------


def _latest_per_object_for_date(
    session: Session, date_val: datetime.date, with_names: bool
) -> list[Elset]:
    """Latest elset per object with an epoch on `date_val` (UTC).

    Half-open epoch range (sargable — uses the epoch index) + ROW_NUMBER
    per object; supported on SQLite 3.25+, MariaDB 10.2+, PostgreSQL.
    """
    start = datetime.datetime.combine(date_val, datetime.time.min)
    end = start + datetime.timedelta(days=1)

    rn = (
        func.row_number()
        .over(
            partition_by=TLE.norad_cat_id,
            order_by=(TLE.epoch.desc(), TLE.id.desc()),
        )
        .label("rn")
    )
    subq = (
        select(TLE.id.label("tle_id"), rn)
        .where(TLE.epoch >= start, TLE.epoch < end, TLE.norad_cat_id.is_not(None))
        .subquery()
    )
    stmt = select(*_elset_columns(with_names)).join(subq, TLE.id == subq.c.tle_id)
    stmt = _join_names(stmt, with_names).where(subq.c.rn == 1).order_by(TLE.norad_cat_id)
    return cast(list[Elset], session.execute(stmt).all())


def _mtime(path: pathlib.Path) -> datetime.datetime:
    """File mtime as a naive UTC datetime, comparable to stored timestamps."""
    return datetime.datetime.fromtimestamp(
        os.stat(path).st_mtime, datetime.timezone.utc
    ).replace(tzinfo=None)


def _dates_to_write(
    session: Session,
    outputs: Sequence[OutputFile],
    pool: Optional[WritePool],
    rebuild_all: bool,
) -> set[datetime.date]:
    """Dates whose files are missing, or older than the rows they should hold.

    Scope comes from `epoch_date_state`, which ingest maintains in the same
    transaction as the rows (see `model.EpochDateState`). That table holds
    one row per *date*, so it grows by a row a day rather than with the
    catalog — which is what makes it affordable to check **every** date file
    on every run instead of guessing with a trailing window.

    The comparison is each date's watermark against that date's file mtime,
    so it is exact and needs no tunable: a generation gap of any length, a
    restore that resets `created`, or a delivery for a date years old all
    self-correct on the next ordinary run. It carries the same requirement
    as the object pass — ingest and generate must not run concurrently, or a
    row committed mid-run can look older than the file that missed it.
    """
    rows = [
        (epoch_date, last_created)
        for epoch_date, last_created in session.execute(
            select(EpochDateState.epoch_date, EpochDateState.last_created)
        )
    ]
    if not rows:
        if session.execute(select(TLE.id).limit(1)).first() is not None:
            logger.warning(
                "epoch_date_state is empty but the catalog is not — no date "
                "files can be generated. Run `thistle-db init-db` to rebuild "
                "it from the element sets."
            )
        return set()
    if rebuild_all:
        return {epoch_date for epoch_date, _ in rows}

    def stale(entry: tuple[datetime.date, datetime.datetime]):
        epoch_date, last_created = entry
        for out in outputs:
            try:
                mtime = _mtime(out.date_path(epoch_date))
            except OSError:
                return epoch_date  # missing (or unreadable): rebuild it
            if last_created >= mtime:
                return epoch_date  # rows arrived after this file was written
        return None

    run = pool.map if pool is not None else lambda fn, items: [fn(i) for i in items]
    return {date for date in run(stale, rows) if date is not None}


def generate_date_files(
    session: Session,
    outputs: Sequence[OutputFile],
    dates: set[datetime.date],
    progress: ProgressReporter = NO_PROGRESS,
    pool: Optional[WritePool] = None,
) -> None:
    """(Re)generate every date output for `dates` (latest elset per object)."""
    if not dates:
        logger.info("Date files: no dates changed")
        return
    logger.info(f"Generating date files for {len(dates)} dates")
    with_names = _needs_names(outputs)

    task = progress.task("Date files", total=len(dates))
    for date_val in sorted(dates):
        progress.advance(task)
        elsets = _latest_per_object_for_date(session, date_val, with_names)
        if not elsets:
            continue
        paths = [out.date_path(date_val) for out in outputs]
        _write_date_outputs(outputs, paths, elsets, pool)

    progress.finish(task)


def _write_date_outputs(
    outputs: Sequence[OutputFile],
    paths: Sequence[pathlib.Path],
    elsets: Sequence[Elset],
    pool: Optional[WritePool],
) -> None:
    """One task per output for a single date's files.

    Date files are few and large, so the parallelism here is across an
    output's formats rather than across dates — which would mean holding
    several dates' worth of rows in memory at once.
    """
    if pool is None:
        _write_outputs(outputs, paths, elsets)
        return
    omm_outputs = [(o, p) for o, p in zip(outputs, paths) if o.format == "omm"]
    tle_outputs = [(o, p) for o, p in zip(outputs, paths) if o.format == "tle"]
    tasks: list = [lambda o=o, p=p: _write_tle_full(p, elsets) for o, p in tle_outputs]
    if omm_outputs:
        # Built once and shared, as in the serial path: the export dominates.
        rows = _omm_rows(elsets)
        tasks += [lambda p=p: _write_omm_full(p, rows) for _, p in omm_outputs]
    pool.map(lambda task: task(), tasks)


# ---------------------------------------------------------------------------
# Object files — tail-guarded append with compact fallback
# ---------------------------------------------------------------------------


def _keyset_batches(
    session: Session, cutoff: Optional[datetime.datetime], with_names: bool
) -> Iterator[Elset]:
    """Stream elsets ordered by (norad_cat_id, epoch, id) in keyset batches.

    Each batch is a short standalone query — no long-lived server cursor to
    time out, memory bounded by BATCH_SIZE. The strictly-increasing unique
    sort key (id as tiebreaker) makes the seek predicate exact: no skipped or
    duplicated rows at batch seams. With `cutoff`, only rows created on/after
    it are returned (incremental mode); with None, the whole table streams
    (--all rebuilds).
    """
    last: Optional[tuple[int, datetime.datetime, int]] = None
    while True:
        stmt = select(*_elset_columns(with_names))
        stmt = _join_names(stmt, with_names)
        stmt = (
            stmt.where(TLE.norad_cat_id.is_not(None))
            .order_by(TLE.norad_cat_id, TLE.epoch, TLE.id)
            .limit(BATCH_SIZE)
        )
        if cutoff is not None:
            stmt = stmt.where(TLE.created >= cutoff)
        if last is not None:
            n, e, i = last
            # Expanded OR form of (norad, epoch, id) > (n, e, i): MariaDB's
            # optimizer handles this as an index range more reliably than the
            # row-constructor syntax.
            stmt = stmt.where(
                or_(
                    TLE.norad_cat_id > n,
                    and_(TLE.norad_cat_id == n, TLE.epoch > e),
                    and_(TLE.norad_cat_id == n, TLE.epoch == e, TLE.id > i),
                )
            )

        rows = cast(list[Elset], session.execute(stmt).all())
        if not rows:
            return
        yield from rows
        if len(rows) < BATCH_SIZE:
            return
        tail = rows[-1]
        last = (tail.norad_cat_id, tail.epoch, tail.id)


def _tail_record(
    path: pathlib.Path, size: Optional[int] = None
) -> tuple[list[str], bool]:
    """Return (last non-empty lines of the file, file ends with a newline).

    Reads only the final _TAIL_BYTES. A missing trailing newline means a torn
    write (crash mid-append) — callers must treat it as damage. `size` lets a
    caller that already stat'ed the file pass it in: the incremental object
    pass probes every output of every changed object, so a redundant stat
    there is one wasted syscall per probe across the whole catalog.
    """
    if size is None:
        size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - _TAIL_BYTES))
        data = f.read()
    ends_clean = data.endswith(b"\n")
    lines = [ln for ln in data.decode("utf-8", errors="replace").splitlines() if ln]
    return lines, ends_clean


def _tle_tail_epoch(
    path: pathlib.Path, size: Optional[int] = None
) -> Optional[datetime.datetime]:
    """Epoch of the last TLE in a .tle file, or None if the tail is damaged.

    Uses the same sgp4 epoch computation as the database column, so the
    comparison against row epochs is exact, not approximate.
    """
    try:
        lines, ends_clean = _tail_record(path, size)
        if not ends_clean or len(lines) < 2:
            return None
        line1, line2 = lines[-2], lines[-1]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            return None
        return epoch_from_lines(line1, line2)
    except Exception:
        return None


def _omm_tail_epoch(
    path: pathlib.Path, size: Optional[int] = None
) -> Optional[datetime.datetime]:
    """Epoch of the last row in an .omm CSV, or None if damaged/header-only.

    export_omm formats EPOCH with the same sat_epoch_datetime used for the
    database epoch column, so this comparison is also exact.
    """
    try:
        lines, ends_clean = _tail_record(path, size)
        if not ends_clean or len(lines) < 2:  # header only, or torn
            return None
        row = next(csv.reader(io.StringIO(lines[-1])))
        epoch_idx = OMM_CSV_FIELDS.index("EPOCH")
        # fromisoformat: tolerant of an epoch with zero microseconds
        # (isoformat omits ".000000"), which strptime with .%f would
        # reject and misreport as a damaged tail.
        return datetime.datetime.fromisoformat(row[epoch_idx])
    except Exception:
        return None


def _fetch_objects(
    session: Session, norad_ids: Sequence[int], with_names: bool
) -> dict[int, list[Elset]]:
    """All rows for several objects in file order — the authoritative content.

    One query for a whole chunk of objects rather than one per object: a
    rewrite is the fallback path, but on a full sweep it is *every* object,
    and per-object round trips dominate there.
    """
    if not norad_ids:
        return {}
    stmt = select(*_elset_columns(with_names))
    stmt = _join_names(stmt, with_names)
    stmt = stmt.where(TLE.norad_cat_id.in_(norad_ids)).order_by(
        TLE.norad_cat_id, TLE.epoch, TLE.id
    )
    grouped: dict[int, list[Elset]] = {norad_id: [] for norad_id in norad_ids}
    for row in cast(list[Elset], session.execute(stmt).all()):
        grouped[row.norad_cat_id].append(row)
    return grouped


def _output_action(
    out: OutputFile, norad_id: int, new_rows: Sequence[Elset]
) -> tuple[str, list[Elset]]:
    """Decide how `new_rows` reach one output's file for this object.

    Returns ("rewrite" | "append" | "unchanged", pending_rows). Two guards
    decide, both stateless:

    - **Tail epoch**: rows with epoch strictly greater than the file's last
      epoch are provably not on disk — appending them keeps the file sorted.
    - **File mtime as watermark**: a row created *before* the file was last
      written was seen by the run that wrote it (runs process everything in
      their lookback, and the cadence is faster than the lookback), so a
      row at/below the tail epoch with created < mtime is already on disk —
      the routine overlapping-lookback case, skipped. A row at/below the
      tail but created *after* the last write is a late delivery (or an
      epoch tie): it belongs in the middle of the file, so rewrite from the
      database. The mtime comparison assumes ingest and generate run on the
      same host (the cron deployment model), sharing one clock, and that
      they never run concurrently: a row committed while generate is
      between its query and its file write would get created < mtime and
      be misclassified as already on disk. Serialize the two (cron
      chaining or flock — see the README); `--verify` remains the backstop.

    An omm file only holds rows sgp4 could export — for some objects a
    subset — so its tail can legitimately lag the database. Re-appending a
    previously seen but unexportable row exports nothing, so it stays
    harmless; an object with no exportable rows at all keeps no file and
    rewrites as a no-op whenever it receives rows.
    """
    path = out.object_path(norad_id)
    # One stat answers all three questions below — existence, the tail's
    # seek offset, and the mtime watermark. Asking separately tripled the
    # syscalls on the hottest path in the incremental run.
    try:
        stat = os.stat(path)
    except OSError:
        # New object (or deleted file): a full write is the append.
        return "rewrite", []
    tail_fn = _tle_tail_epoch if out.format == "tle" else _omm_tail_epoch
    tail_epoch = tail_fn(path, stat.st_size)
    if tail_epoch is None:  # torn/damaged/header-only tail: self-heal
        return "rewrite", []

    watermark = datetime.datetime.fromtimestamp(
        stat.st_mtime, datetime.timezone.utc
    ).replace(tzinfo=None)
    if any(t.epoch <= tail_epoch and t.created >= watermark for t in new_rows):
        return "rewrite", []

    pending = [t for t in new_rows if t.epoch > tail_epoch]
    if not pending:
        return "unchanged", []
    return "append", pending


def _probe_object(
    args: tuple[Sequence[OutputFile], int, list[Elset]],
) -> list[tuple[str, list[Elset]]]:
    """Filesystem half of the decision for one object, one entry per output.

    Pure stat/read work, so it runs on the write pool: the tail probes were
    2.3s of a 13.5s incremental run, all of it blocked on syscalls.
    """
    outputs, norad_id, new_rows = args
    return [_output_action(out, norad_id, new_rows) for out in outputs]


def _emit_object(
    args: tuple[
        Sequence[OutputFile],
        int,
        Sequence[tuple[str, list[Elset]]],
        Optional[list[Elset]],
    ],
) -> str:
    """Land one object's new rows in its file in every output.

    Runs on the write pool with everything it needs already in hand: the
    per-output decisions from `_probe_object` and, when any output needs a
    rewrite, that object's full history from the batched fetch. Outputs
    needing a rewrite share one OMM export.

    Returns the strongest action taken: "rewritten" > "appended" >
    "unchanged".
    """
    outputs, norad_id, actions, full_history = args

    rewrite_outs = [out for out, (action, _) in zip(outputs, actions) if action == "rewrite"]
    if rewrite_outs and full_history is not None:
        paths = [out.object_path(norad_id) for out in rewrite_outs]
        _write_outputs(rewrite_outs, paths, full_history)

    appended = False
    omm_rows: Optional[list[list]] = None
    for out, (action, pending) in zip(outputs, actions):
        if action != "append":
            continue
        path = out.object_path(norad_id)
        if out.format == "tle":
            _append_tle(path, pending)
        else:
            if omm_rows is None:
                omm_rows = _omm_rows(pending)
            _append_omm(path, omm_rows)
        appended = True

    if rewrite_outs:
        return "rewritten"
    return "appended" if appended else "unchanged"


def _object_chunks(
    session: Session, cutoff: Optional[datetime.datetime], with_names: bool
) -> Iterator[list[tuple[int, list[Elset]]]]:
    """Group the keyset stream into chunks of whole objects.

    An object's rows never straddle a chunk, so each chunk can be handed to
    the pool as a unit and every task owns its object's files outright.
    """
    chunk: list[tuple[int, list[Elset]]] = []
    current_id: Optional[int] = None
    buffer: list[Elset] = []

    for elset in _keyset_batches(session, cutoff, with_names):
        if elset.norad_cat_id != current_id:
            if current_id is not None:
                chunk.append((current_id, buffer))
                if len(chunk) >= OBJECT_CHUNK:
                    yield chunk
                    chunk = []
            current_id = elset.norad_cat_id
            buffer = []
        buffer.append(elset)

    if current_id is not None:
        chunk.append((current_id, buffer))
    if chunk:
        yield chunk


def generate_object_files(
    session: Session,
    outputs: Sequence[OutputFile],
    cutoff: Optional[datetime.datetime],
    progress: ProgressReporter = NO_PROGRESS,
    pool: Optional[WritePool] = None,
) -> None:
    """Generate/refresh one file per satellite in every object output.

    Incremental (cutoff set): stream rows created since cutoff, grouped by
    object, and append each object's new rows behind the tail guard.
    Full (cutoff None): stream everything and rewrite every object's files.
    """
    # The object count isn't known until the keyset stream is exhausted, so
    # this bar is indeterminate (running count instead of a percentage).
    task = progress.task("Object files", total=None)
    counts = {"appended": 0, "rewritten": 0, "unchanged": 0}
    with_names = _needs_names(outputs)
    run = pool.map if pool is not None else lambda fn, items: [fn(i) for i in items]

    for chunk in _object_chunks(session, cutoff, with_names):
        if cutoff is None:
            # Full rebuild: every object is a rewrite from the rows already
            # streamed, so there is nothing to probe and nothing to re-fetch.
            run(
                _rewrite_from_stream,
                [(outputs, norad_id, rows) for norad_id, rows in chunk],
            )
            counts["rewritten"] += len(chunk)
        else:
            probes = run(
                _probe_object,
                [(outputs, norad_id, rows) for norad_id, rows in chunk],
            )
            needs_fetch = [
                norad_id
                for (norad_id, _), actions in zip(chunk, probes)
                if any(action == "rewrite" for action, _ in actions)
            ]
            histories = _fetch_objects(session, needs_fetch, with_names)
            results = run(
                _emit_object,
                [
                    (outputs, norad_id, actions, histories.get(norad_id))
                    for (norad_id, _), actions in zip(chunk, probes)
                ],
            )
            for result in results:
                counts[result] += 1
        progress.advance(task, len(chunk))
    progress.finish(task)

    mode = "full rewrite" if cutoff is None else "incremental"
    logger.info(
        f"Object files ({mode}): {counts['appended']} appended, "
        f"{counts['rewritten']} rewritten, {counts['unchanged']} unchanged"
    )


def _rewrite_from_stream(
    args: tuple[Sequence[OutputFile], int, Sequence[Elset]],
) -> None:
    outputs, norad_id, elsets = args
    paths = [out.object_path(norad_id) for out in outputs]
    _write_outputs(outputs, paths, elsets)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _count_lines(path: pathlib.Path) -> int:
    """Count newlines in a file without parsing it (binary chunked read)."""
    count = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            count += chunk.count(b"\n")
    return count


def _check_object(
    args: tuple[Sequence[OutputFile], Sequence[OutputFile], int, int],
) -> bool:
    """Whether one object's files agree with its database row count."""
    tle_outputs, omm_outputs, norad_id, count = args
    for out in tle_outputs:
        path = out.object_path(norad_id)
        if not path.exists() or _count_lines(path) != 2 * count:
            return False
        # Line count alone misses a torn (newline-less) final line;
        # the tail check is cheap.
        if not _tail_record(path)[1]:
            return False
    # An omm file may legitimately be absent or hold a subset, but a
    # present one must at least end cleanly.
    for out in omm_outputs:
        path = out.object_path(norad_id)
        if path.exists() and not _tail_record(path)[1]:
            return False
    return True


def verify_object_files(
    session: Session,
    outputs: Sequence[OutputFile],
    progress: ProgressReporter = NO_PROGRESS,
    pool: Optional[WritePool] = None,
) -> None:
    """Reconcile every object's tle files against the database and rewrite
    objects that disagree.

    The check is deliberately cheap: one aggregated index-only query for
    per-object row counts, then a newline count per file (2 lines per elset,
    intact trailing newline) — no parsing. It catches damage the incremental
    tail guard cannot see, such as truncation or edits in the middle of a
    file. Cost is O(total output size), so this is meant for a periodic
    (e.g. weekly) cron flag, not every run.

    The tle outputs are the verification authority (they hold every row
    verbatim); a failed check rewrites the object's file in *every* output,
    omm included. Without a tle object output there is no derivable expected
    count (sgp4 cannot export every elset), so verification is skipped with
    a warning.
    """
    tle_outputs = [o for o in outputs if o.format == "tle"]
    omm_outputs = [o for o in outputs if o.format == "omm"]
    if not tle_outputs:
        logger.warning(
            "--verify requires a tle object output (the omm row count is "
            "not derivable); skipping verification"
        )
        return

    stmt = (
        select(TLE.norad_cat_id, func.count())
        .where(TLE.norad_cat_id.is_not(None))
        .group_by(TLE.norad_cat_id)
    )
    expected = {norad: count for norad, count in session.execute(stmt)}
    with_names = _needs_names(outputs)

    task = progress.task("Verifying object files", total=len(expected))
    run = pool.map if pool is not None else lambda fn, items: [fn(i) for i in items]
    repaired = 0
    items = list(expected.items())
    for start in range(0, len(items), OBJECT_CHUNK):
        chunk = items[start : start + OBJECT_CHUNK]
        # Reading every output file is the whole cost of --verify, and it is
        # pure I/O — the same reason the write pool exists.
        ok = run(
            _check_object,
            [(tle_outputs, omm_outputs, norad_id, count) for norad_id, count in chunk],
        )
        bad = [norad_id for (norad_id, _), good in zip(chunk, ok) if not good]
        if bad:
            histories = _fetch_objects(session, bad, with_names)
            run(
                _rewrite_from_stream,
                [(outputs, norad_id, histories.get(norad_id, [])) for norad_id in bad],
            )
            repaired += len(bad)
        progress.advance(task, len(chunk))
    progress.finish(task)

    for out in tle_outputs:
        orphans = [
            p.name
            for p in pathlib.Path(out.dir).glob(out.object_glob())
            if (nid := out.parse_object_name(p.name)) is not None
            and nid not in expected
        ]
        if orphans:
            logger.warning(
                f"Verify: {len(orphans)} object file(s) in {out.dir} have no "
                f"database rows (left untouched): {', '.join(sorted(orphans)[:10])}"
                + ("…" if len(orphans) > 10 else "")
            )

    logger.info(f"Verified {len(expected)} objects: {repaired} repaired")


def verify_date_files(
    session: Session,
    outputs: Sequence[OutputFile],
    progress: ProgressReporter = NO_PROGRESS,
    pool: Optional[WritePool] = None,
) -> None:
    """Reconcile every date file against the database.

    The normal run only rewrites dates whose watermark is newer than their
    file, so silent corruption of an untouched date file — mid-file
    truncation, a hand edit — is caught here instead. One aggregated query
    gives the expected object count per date; each tle date file is checked
    by newline count (2 lines per object) and a clean tail, and any
    mismatch rewrites that date in every date output.

    As with object files, the tle outputs are the authority: an omm date
    file may hold a subset (sgp4 cannot export every elset), so a present
    one is only checked for a clean tail.
    """
    tle_outputs = [o for o in outputs if o.format == "tle"]
    if not tle_outputs:
        logger.warning(
            "--verify requires a tle date output to check date files (the "
            "omm row count is not derivable); skipping date verification"
        )
        return

    date_expr = func.date(TLE.epoch)
    stmt = (
        select(date_expr, func.count(func.distinct(TLE.norad_cat_id)))
        .where(TLE.norad_cat_id.is_not(None))
        .group_by(date_expr)
    )
    expected = {date_from_sql(raw): count for raw, count in session.execute(stmt) if raw}

    task = progress.task("Verifying date files", total=len(expected))
    with_names = _needs_names(outputs)
    repaired = 0
    for date_val, count in sorted(expected.items()):
        progress.advance(task)
        ok = True
        for out in tle_outputs:
            path = out.date_path(date_val)
            if not path.exists() or _count_lines(path) != 2 * count:
                ok = False
                break
            if not _tail_record(path)[1]:
                ok = False
                break
        if ok:
            continue
        elsets = _latest_per_object_for_date(session, date_val, with_names)
        if elsets:
            paths = [out.date_path(date_val) for out in outputs]
            _write_date_outputs(outputs, paths, elsets, pool)
            repaired += 1
    progress.finish(task)

    logger.info(f"Verified {len(expected)} date files: {repaired} repaired")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(
    session: Session,
    config: OutputConfig,
    *,
    rebuild_all: bool = False,
    lookback_days: Optional[int] = None,
    verify: bool = False,
    progress: ProgressReporter = NO_PROGRESS,
) -> None:
    """Generate all configured output files (``config.files``).

    Incremental by default; ``rebuild_all=True`` regenerates everything from
    scratch (first run, disaster recovery, or after a gap in cron coverage
    longer than the lookback). ``verify=True`` follows the normal run with a
    full reconciliation sweep of both date and object files (see
    ``verify_date_files`` / ``verify_object_files``) — intended for a
    periodic cron flag.

    Raises ValueError when no outputs are configured: generating nothing is
    a misconfiguration, not a no-op (the CLI checks first and exits 2).
    """
    if not config.files:
        raise ValueError(
            "no outputs configured — add at least one [[output.files]] entry"
        )

    date_outputs = [f for f in config.files if f.type == "date"]
    object_outputs = [f for f in config.files if f.type == "object"]
    for out in config.files:
        pathlib.Path(out.dir).mkdir(parents=True, exist_ok=True)

    lookback = lookback_days if lookback_days is not None else config.lookback_days
    cutoff = utcnow() - datetime.timedelta(days=lookback)

    with WritePool(config.write_workers) as pool:
        if date_outputs:
            dates = _dates_to_write(session, date_outputs, pool, rebuild_all)
            generate_date_files(session, date_outputs, dates, progress, pool)

        if object_outputs:
            generate_object_files(
                session,
                object_outputs,
                None if rebuild_all else cutoff,
                progress,
                pool,
            )

        if verify and not rebuild_all:  # --all just rewrote everything
            if date_outputs:
                verify_date_files(session, date_outputs, progress, pool)
            if object_outputs:
                verify_object_files(session, object_outputs, progress, pool)

    dirs = ", ".join(sorted({out.dir for out in config.files}))
    logger.info(f"Output generation complete → {dirs}")

