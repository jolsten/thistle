"""Output file generation.

Outputs are declared as ``[[output.files]]`` entries — each names a type
(``date`` or ``object``), a format (``tle`` or ``omm``), a destination
directory, and a filename scheme (integer vs alpha-5 object IDs, zero
padding, date format, extension). Steady-state cost is O(new rows),
independent of catalog size, with no persistent generator state:

- **Date files** (latest elset per object that day) are rewritten for a
  trailing epoch window (``output.window_days``) plus any date that actually
  received new rows in the ingest lookback — so a TLE delivered arbitrarily
  late still lands in the proper old date file.
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
"""

import csv
import datetime
import io
import pathlib
from typing import Iterator, Optional, Sequence

from loguru import logger
from sgp4.api import Satrec
from sgp4.exporter import export_omm
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, selectinload

from thistle_db.config import OutputConfig, OutputFile
from thistle_db.model import TLE, epoch_from_lines, utcnow
from thistle_db.progress import NO_PROGRESS, ProgressReporter
from thistle_db.reader import OMM_CSV_FIELDS, TLETuple, write_tle

BATCH_SIZE = 50_000

# Read at most this many bytes from the end of a file to find its last
# record. Two TLE lines are ~140 bytes; OMM CSV rows are a few hundred.
_TAIL_BYTES = 8192


def _reconstruct_satrec(tle: TLE) -> Satrec:
    """Reconstruct a Satrec object from stored TLE lines."""
    return Satrec.twoline2rv(tle.line1, tle.line2)


def _safe_export_omm(sat: Satrec, name: str) -> dict | None:
    """Export OMM dict from Satrec, returning None if export fails."""
    try:
        return export_omm(sat, name)
    except (ValueError, AttributeError):
        return None


def _get_object_name(tle: TLE) -> str:
    """Get object name from OmmMetadata if available, else fall back to object_id."""
    if tle.omm_metadata and tle.omm_metadata.object_name:
        return tle.omm_metadata.object_name
    return tle.object_id


def _omm_records(tles: Sequence[TLE]) -> list[dict]:
    records = []
    for tle in tles:
        sat = _reconstruct_satrec(tle)
        record = _safe_export_omm(sat, _get_object_name(tle))
        if record is not None:
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Date files
# ---------------------------------------------------------------------------


def _latest_per_object_for_date(session: Session, date_val: datetime.date) -> list[TLE]:
    """Latest TLE per object with an epoch on `date_val` (UTC).

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
    stmt = (
        select(TLE)
        .join(subq, TLE.id == subq.c.tle_id)
        .where(subq.c.rn == 1)
        .order_by(TLE.norad_cat_id)
        .options(selectinload(TLE.omm_metadata))
    )
    return list(session.execute(stmt).scalars().all())


def _dates_with_new_rows(
    session: Session, cutoff: datetime.datetime
) -> set[datetime.date]:
    """Epoch dates of rows created since `cutoff` (uses the created index)."""
    stmt = (
        select(func.date(TLE.epoch))
        .distinct()
        .where(TLE.created >= cutoff, TLE.norad_cat_id.is_not(None))
    )
    return {_coerce_date(d) for d in session.execute(stmt).scalars() if d is not None}


def _dates_in_window(
    session: Session, window_start: datetime.date
) -> set[datetime.date]:
    """Dates on/after `window_start` that actually have rows (sargable range)."""
    start = datetime.datetime.combine(window_start, datetime.time.min)
    stmt = select(func.date(TLE.epoch)).distinct().where(TLE.epoch >= start)
    return {_coerce_date(d) for d in session.execute(stmt).scalars() if d is not None}


def _all_dates(session: Session) -> set[datetime.date]:
    """Every distinct epoch date — full scan, used only by --all rebuilds."""
    stmt = select(func.date(TLE.epoch)).distinct()
    return {_coerce_date(d) for d in session.execute(stmt).scalars() if d is not None}


def _coerce_date(raw) -> datetime.date:
    # func.date() returns str on SQLite and datetime.date on MariaDB/PostgreSQL.
    if isinstance(raw, datetime.date):
        return raw
    return datetime.date.fromisoformat(str(raw))


def generate_date_files(
    session: Session,
    outputs: Sequence[OutputFile],
    dates: set[datetime.date],
    progress: ProgressReporter = NO_PROGRESS,
) -> None:
    """(Re)generate every date output for `dates` (latest TLE per object)."""
    logger.info(f"Generating date files for {len(dates)} dates")

    task = progress.task("Date files", total=len(dates))
    for date_val in sorted(dates):
        progress.advance(task)
        tles = _latest_per_object_for_date(session, date_val)
        if not tles:
            continue

        omm_records: Optional[list[dict]] = None  # shared across omm outputs
        for out in outputs:
            path = out.date_path(date_val)
            if out.format == "tle":
                tle_tuples: list[TLETuple] = [(t.line1, t.line2) for t in tles]
                write_tle(path, tle_tuples)
            else:
                if omm_records is None:
                    omm_records = _omm_records(tles)
                _write_omm_full(path, omm_records)

    progress.finish(task)


# ---------------------------------------------------------------------------
# Object files — tail-guarded append with compact fallback
# ---------------------------------------------------------------------------


def _keyset_batches(
    session: Session, cutoff: Optional[datetime.datetime]
) -> Iterator[TLE]:
    """Stream TLE rows ordered by (norad_cat_id, epoch, id) in keyset batches.

    Each batch is a short standalone query — no long-lived server cursor to
    time out, memory bounded by BATCH_SIZE. The strictly-increasing unique
    sort key (id as tiebreaker) makes the seek predicate exact: no skipped or
    duplicated rows at batch seams. With `cutoff`, only rows created on/after
    it are returned (incremental mode); with None, the whole table streams
    (--all rebuilds).
    """
    last: Optional[tuple[int, datetime.datetime, int]] = None
    while True:
        stmt = (
            select(TLE)
            .where(TLE.norad_cat_id.is_not(None))
            .order_by(TLE.norad_cat_id, TLE.epoch, TLE.id)
            .limit(BATCH_SIZE)
            .options(selectinload(TLE.omm_metadata))
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

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return
        yield from rows
        if len(rows) < BATCH_SIZE:
            return
        tail = rows[-1]
        assert tail.norad_cat_id is not None
        last = (tail.norad_cat_id, tail.epoch, tail.id)


def _tail_record(path: pathlib.Path) -> tuple[list[str], bool]:
    """Return (last non-empty lines of the file, file ends with a newline).

    Reads only the final _TAIL_BYTES. A missing trailing newline means a torn
    write (crash mid-append) — callers must treat it as damage.
    """
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - _TAIL_BYTES))
        data = f.read()
    ends_clean = data.endswith(b"\n")
    lines = [ln for ln in data.decode("utf-8", errors="replace").splitlines() if ln]
    return lines, ends_clean


def _tle_tail_epoch(path: pathlib.Path) -> Optional[datetime.datetime]:
    """Epoch of the last TLE in a .tle file, or None if the tail is damaged.

    Uses the same sgp4 epoch computation as the database column, so the
    comparison against row epochs is exact, not approximate.
    """
    try:
        lines, ends_clean = _tail_record(path)
        if not ends_clean or len(lines) < 2:
            return None
        line1, line2 = lines[-2], lines[-1]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            return None
        return epoch_from_lines(line1, line2)
    except Exception:
        return None


def _omm_tail_epoch(path: pathlib.Path) -> Optional[datetime.datetime]:
    """Epoch of the last row in an .omm CSV, or None if damaged/header-only.

    export_omm formats EPOCH with the same sat_epoch_datetime used for the
    database epoch column, so this comparison is also exact.
    """
    try:
        lines, ends_clean = _tail_record(path)
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


def _write_omm_full(path: pathlib.Path, records: list[dict]) -> None:
    """Write an OMM CSV with header. Empty records → no file (and any stale
    file removed, since a rewrite is authoritative)."""
    if not records:
        path.unlink(missing_ok=True)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OMM_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _append_omm(path: pathlib.Path, records: list[dict]) -> None:
    if not records:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OMM_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def _append_tle(path: pathlib.Path, tles: Sequence[TLE]) -> None:
    with open(path, "a") as f:
        for tle in tles:
            f.write(f"{tle.line1}\n{tle.line2}\n")


def _fetch_object(session: Session, norad_id: int) -> list[TLE]:
    """All rows for one object in file order — the authoritative content."""
    stmt = (
        select(TLE)
        .where(TLE.norad_cat_id == norad_id)
        .order_by(TLE.epoch, TLE.id)
        .options(selectinload(TLE.omm_metadata))
    )
    return list(session.execute(stmt).scalars().all())


def _rewrite_object(
    session: Session,
    outputs: Sequence[OutputFile],
    norad_id: int,
) -> None:
    """Full authoritative rewrite of one object's file in every output."""
    _write_object(outputs, norad_id, _fetch_object(session, norad_id))


def _write_object(
    outputs: Sequence[OutputFile],
    norad_id: int,
    tles: Sequence[TLE],
) -> None:
    omm_records: Optional[list[dict]] = None  # shared across omm outputs
    for out in outputs:
        path = out.object_path(norad_id)
        if out.format == "tle":
            write_tle(path, [(t.line1, t.line2) for t in tles])
        else:
            if omm_records is None:
                omm_records = _omm_records(tles)
            _write_omm_full(path, omm_records)


def _output_action(
    out: OutputFile, norad_id: int, new_rows: list[TLE]
) -> tuple[str, list[TLE]]:
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
    if not path.exists():
        # New object (or deleted file): a full write is the append.
        return "rewrite", []
    tail_fn = _tle_tail_epoch if out.format == "tle" else _omm_tail_epoch
    tail_epoch = tail_fn(path)
    if tail_epoch is None:  # torn/damaged/header-only tail: self-heal
        return "rewrite", []

    watermark = datetime.datetime.fromtimestamp(
        path.stat().st_mtime, datetime.timezone.utc
    ).replace(tzinfo=None)
    if any(t.epoch <= tail_epoch and t.created >= watermark for t in new_rows):
        return "rewrite", []

    pending = [t for t in new_rows if t.epoch > tail_epoch]
    if not pending:
        return "unchanged", []
    return "append", pending


def _emit_object(
    session: Session,
    outputs: Sequence[OutputFile],
    norad_id: int,
    new_rows: list[TLE],
) -> str:
    """Land `new_rows` in one object's file in every output.

    Each output decides independently against its own tail and mtime
    watermark (see `_output_action`); outputs needing a rewrite are rebuilt
    together from a single authoritative database fetch.

    Returns the strongest action taken across outputs: "rewritten" >
    "appended" > "unchanged".
    """
    actions = [(out, _output_action(out, norad_id, new_rows)) for out in outputs]

    rewrite_outs = [out for out, (action, _) in actions if action == "rewrite"]
    if rewrite_outs:
        _rewrite_object(session, rewrite_outs, norad_id)

    appended = False
    for out, (action, pending) in actions:
        if action != "append":
            continue
        path = out.object_path(norad_id)
        if out.format == "tle":
            _append_tle(path, pending)
        else:
            _append_omm(path, _omm_records(pending))
        appended = True

    if rewrite_outs:
        return "rewritten"
    return "appended" if appended else "unchanged"


def generate_object_files(
    session: Session,
    outputs: Sequence[OutputFile],
    cutoff: Optional[datetime.datetime],
    progress: ProgressReporter = NO_PROGRESS,
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
    current_id: Optional[int] = None
    buffer: list[TLE] = []

    def flush() -> None:
        if current_id is None:
            return
        if cutoff is None:
            _write_object(outputs, current_id, buffer)
            counts["rewritten"] += 1
        else:
            counts[_emit_object(session, outputs, current_id, buffer)] += 1
        progress.advance(task)

    for tle in _keyset_batches(session, cutoff):
        if tle.norad_cat_id != current_id:
            flush()
            current_id = tle.norad_cat_id
            buffer = []
        buffer.append(tle)
    flush()
    progress.finish(task)

    mode = "full rewrite" if cutoff is None else "incremental"
    logger.info(
        f"Object files ({mode}): {counts['appended']} appended, "
        f"{counts['rewritten']} rewritten, {counts['unchanged']} unchanged"
    )


def _count_lines(path: pathlib.Path) -> int:
    """Count newlines in a file without parsing it (binary chunked read)."""
    count = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            count += chunk.count(b"\n")
    return count


def verify_object_files(
    session: Session,
    outputs: Sequence[OutputFile],
    progress: ProgressReporter = NO_PROGRESS,
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

    task = progress.task("Verifying object files", total=len(expected))
    repaired = 0
    for norad_id, count in expected.items():
        progress.advance(task)
        ok = True
        for out in tle_outputs:
            path = out.object_path(norad_id)
            ok = path.exists() and _count_lines(path) == 2 * count
            if ok:
                # Line count alone misses a torn (newline-less) final line;
                # the tail check is cheap.
                _, ends_clean = _tail_record(path)
                ok = ends_clean
            if not ok:
                break
        if ok:
            # An omm file may legitimately be absent or hold a subset, but a
            # present one must at least end cleanly.
            for out in omm_outputs:
                omm_path = out.object_path(norad_id)
                if omm_path.exists():
                    _, ends_clean = _tail_record(omm_path)
                    ok = ends_clean
                    if not ok:
                        break
        if not ok:
            _rewrite_object(session, outputs, norad_id)
            repaired += 1
    progress.finish(task)

    for out in tle_outputs:
        orphans = [
            p.name
            for p in pathlib.Path(out.dir).glob(f"*{out.ext}")
            if (nid := out.parse_object_stem(p.stem)) is not None
            and nid not in expected
        ]
        if orphans:
            logger.warning(
                f"Verify: {len(orphans)} object file(s) in {out.dir} have no "
                f"database rows (left untouched): {', '.join(sorted(orphans)[:10])}"
                + ("…" if len(orphans) > 10 else "")
            )

    logger.info(f"Verified {len(expected)} objects: {repaired} repaired")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(
    session: Session,
    config: OutputConfig,
    *,
    rebuild_all: bool = False,
    window_days: Optional[int] = None,
    lookback_days: Optional[int] = None,
    verify: bool = False,
    progress: ProgressReporter = NO_PROGRESS,
) -> None:
    """Generate all configured output files (``config.files``).

    Incremental by default; ``rebuild_all=True`` regenerates everything from
    scratch (first run, disaster recovery, or after a gap in cron coverage
    longer than the lookback). ``verify=True`` follows the normal run with a
    full count-reconciliation sweep of object files (see
    ``verify_object_files``) — intended for a periodic cron flag.

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

    window = window_days if window_days is not None else config.window_days
    lookback = lookback_days if lookback_days is not None else config.lookback_days
    cutoff = utcnow() - datetime.timedelta(days=lookback)

    if date_outputs:
        if rebuild_all:
            dates = _all_dates(session)
        else:
            window_start = utcnow().date() - datetime.timedelta(days=window)
            # Trailing window (self-healing for recent files) plus every date
            # that received new rows — catches arbitrarily late deliveries.
            dates = _dates_in_window(session, window_start)
            dates |= _dates_with_new_rows(session, cutoff)
        generate_date_files(session, date_outputs, dates, progress)

    if object_outputs:
        generate_object_files(
            session, object_outputs, None if rebuild_all else cutoff, progress
        )
        if verify and not rebuild_all:  # --all just rewrote everything
            verify_object_files(session, object_outputs, progress)

    dirs = ", ".join(sorted({out.dir for out in config.files}))
    logger.info(f"Output generation complete → {dirs}")
