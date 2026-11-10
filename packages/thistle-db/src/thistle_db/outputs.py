"""Output maintenance — the second sink of the delta flush, plus rebuild.

Outputs are lossy projections of the store (TLE text has no metadata; the
STK CSV has no provenance), always rebuildable (`rebuild`), never the
reverse. A damaged output is an inconvenience, not data loss — the
incremental machinery here (tail-epoch append guard, merge-dedup rewrite,
date upsert, atomic tmp+rename) is an efficiency layer on top of that
guarantee, and every path is idempotent against replay of the same delta.

Per flush, each ``[[output.files]]`` entry consumes the *validated,
pre-dedup* delta (grouped by object) — outputs never ask the store which
rows were new:

- **object** files: append when every buffered epoch for the object is
  strictly newer than the file's tail; anything else (late delivery, tie,
  re-delivery, torn tail) merges: read, dedup, stable-sort by epoch with
  existing records before new, rewrite atomically.
- **date** files: upsert the delta's winners ("latest per object that
  day" can only change when a newer-epoch record for that (object, date)
  arrives); replace on ``epoch >=`` so newest-ingested wins ties.

TLE outputs are verbatim passthrough: line-less records are always
skipped and counted (one WARNING per run, emitted by the engine) — a TLE
output never contains a byte thistle synthesized.

All filesystem work runs on the write pool; each task owns a distinct
path and receives plain data. Sessions and queries stay on the calling
thread.
"""

import csv
import dataclasses
import datetime
import io
import os
import pathlib
import struct
from typing import Iterable, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from thistle_db.config import MAX_SATNUM, OutputConfig, OutputFile
from thistle_db.model import OmmMetadata, TLE, record_from_row
from thistle_db.reader import OMM_CSV_FIELDS
from thistle_db.record import (
    GPRecord,
    STORE_COLUMNS,
    epoch_from_lines,
    to_csv_dict,
    to_omm_dict,
)
from thistle_db.runs import IngestReport
from thistle_db.writer import WritePool

__all__ = ["OutputApplier", "rebuild_outputs", "output_key"]

# How much of a file's tail to read when probing the last record. A TLE
# pair is ~142 bytes and a store-CSV row ~600; 8 KB comfortably holds the
# last record of either, and a probe that finds nothing parseable simply
# falls back to the merge path (which reads the whole file and heals it).
_TAIL_PROBE_BYTES = 8192


def output_key(entry: OutputFile) -> str:
    """Stable identifier for an output entry in reports/logs."""
    return f"{entry.type}:{entry.format}:{entry.dir}"


def _fieldnames(entry: OutputFile) -> list[str]:
    return STORE_COLUMNS if entry.columns == "full" else OMM_CSV_FIELDS


# Element fields stored as single precision (model.Float32). CSV outputs
# render at *stored* precision so an incremental write and a later rebuild
# (which reads the rounded columns back) produce byte-identical files.
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


def _parse_iso(text: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None


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


# --- existing-file readers ---------------------------------------------------


def _read_tle_pairs(path: pathlib.Path) -> list[tuple[str, str]]:
    """All (line1, line2) pairs in an output file. A torn tail (crash mid
    append) simply yields no pair for the torn record — healed on rewrite."""
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


def _read_csv_rows(path: pathlib.Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _tail_epoch_tle(path: pathlib.Path) -> Optional[datetime.datetime]:
    """Exact epoch of the file's last complete record, or None (merge)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - _TAIL_PROBE_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln.rstrip("\r") for ln in tail.split("\n") if ln.strip()]
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].startswith("2 ") and lines[i - 1].startswith("1 "):
            try:
                return epoch_from_lines(lines[i - 1], lines[i])
            except Exception:
                return None
    return None


def _tail_epoch_csv(path: pathlib.Path, entry: OutputFile) -> Optional[datetime.datetime]:
    """Epoch of the last complete CSV row, parsed from a comma split that
    only touches comma-free fields (hash/ids before EPOCH in full columns;
    the trailing numeric fields in the STK projection). Any doubt returns
    None — the merge path is always safe."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - _TAIL_PROBE_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in tail.split("\n") if ln.strip()]
    if len(lines) < 1:
        return None
    last = lines[-1]
    parts = last.split(",")
    try:
        if entry.columns == "full":
            # RECORD_HASH, NORAD_CAT_ID, OBJECT_ID, EPOCH, ... — nothing
            # before EPOCH can contain a comma.
            return _parse_iso(parts[3])
        # STK: OBJECT_NAME may contain commas, but the 15 fields from EPOCH
        # to the end (numbers, one letter, an ISO date) cannot — count from
        # the end.
        return _parse_iso(parts[-15])
    except IndexError:
        return None


def _norad_from_line1(line1: str) -> Optional[int]:
    try:
        from sgp4.alpha5 import from_alpha5

        return from_alpha5(line1[2:7].replace(" ", "0"))
    except Exception:
        return None


# --- the applier --------------------------------------------------------------


class OutputApplier:
    """Applies validated deltas to every configured output."""

    def __init__(
        self,
        outputs: list[OutputFile],
        pool: WritePool,
        report: IngestReport,
    ):
        self.outputs = outputs
        self.pool = pool
        self.report = report
        self.alpha5_overflow: set[int] = set()
        for entry in outputs:
            pathlib.Path(entry.dir).mkdir(parents=True, exist_ok=True)

    def apply(self, delta: dict[int, list[GPRecord]]) -> None:
        """Hand the flush's delta to every output entry.

        `delta` maps norad_cat_id -> records, already deduplicated by
        record_hash, in ingest order.
        """
        if not delta:
            return
        for entry in self.outputs:
            if entry.type == "object":
                self._apply_object(entry, delta)
            else:
                self._apply_date(entry, delta)

    # -- object outputs ---------------------------------------------------

    def _apply_object(self, entry: OutputFile, delta: dict[int, list[GPRecord]]) -> None:
        key = output_key(entry)
        tasks = []
        for norad, records in delta.items():
            if entry.object_id == "alpha5" and norad > MAX_SATNUM:
                self.alpha5_overflow.add(norad)
            path = entry.object_path(norad)
            tasks.append((path, records))

        def task(item: tuple[pathlib.Path, list[GPRecord]]) -> tuple[int, int]:
            path, records = item
            return _apply_object_file(entry, path, records)

        written = 0
        for wrote, lineless in self.pool.map(task, tasks):
            written += wrote
            self.report.lineless_skipped += lineless
        self.report.output_writes[key] = (
            self.report.output_writes.get(key, 0) + written
        )

    # -- date outputs -------------------------------------------------------

    def _apply_date(self, entry: OutputFile, delta: dict[int, list[GPRecord]]) -> None:
        key = output_key(entry)
        # Winners per (date, object) within this delta; later records win
        # ties (newest-ingested, matching the on-disk replace rule).
        winners: dict[datetime.date, dict[int, GPRecord]] = {}
        for norad, records in delta.items():
            for record in records:
                day = record.epoch.date()
                best = winners.setdefault(day, {}).get(norad)
                if best is None or record.epoch >= best.epoch:
                    winners[day][norad] = record

        items = [
            (entry.date_path(day), day_winners)
            for day, day_winners in sorted(winners.items())
        ]

        def task(item: tuple[pathlib.Path, dict[int, GPRecord]]) -> tuple[int, int]:
            path, day_winners = item
            return _apply_date_file(entry, path, day_winners)

        written = 0
        for wrote, lineless in self.pool.map(task, items):
            written += wrote
            self.report.lineless_skipped += lineless
        self.report.output_writes[key] = (
            self.report.output_writes.get(key, 0) + written
        )


# --- object-file application (pure filesystem; runs on the pool) -------------


def _apply_object_file(
    entry: OutputFile, path: pathlib.Path, records: list[GPRecord]
) -> tuple[int, int]:
    """Returns (files_written, lineless_skipped)."""
    records = sorted(records, key=lambda r: r.epoch)
    if entry.format == "tle":
        lineless = sum(1 for r in records if not (r.tle_line1 and r.tle_line2))
        usable = [r for r in records if r.tle_line1 and r.tle_line2]
        if not usable:
            return (0, lineless)
        if not path.exists():
            _atomic_write(
                path, _render_tle((r.tle_line1, r.tle_line2) for r in usable)  # type: ignore[arg-type]
            )
            return (1, lineless)
        tail = _tail_epoch_tle(path)
        if tail is not None and usable[0].epoch > tail:
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write(_render_tle((r.tle_line1, r.tle_line2) for r in usable))  # type: ignore[arg-type]
            return (1, lineless)
        _merge_object_tle(path, usable)
        return (1, lineless)

    # omm-csv
    rows = [(r.epoch, r.record_hash, _row_for(entry, r)) for r in records]
    if not path.exists():
        _atomic_write(
            path, _render_csv(_fieldnames(entry), [row for _, _, row in rows])
        )
        return (1, 0)
    tail = _tail_epoch_csv(path, entry)
    if tail is not None and rows[0][0] > tail:
        text = _render_csv(_fieldnames(entry), [row for _, _, row in rows])
        body = text.split("\n", 1)[1]  # drop the header for the append
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(body)
        return (1, 0)
    _merge_object_csv(entry, path, rows)
    return (1, 0)


def _merge_object_tle(path: pathlib.Path, records: list[GPRecord]) -> None:
    """Merge-rewrite: dedup on the exact line text, stable epoch order with
    existing records before new, tmp + atomic rename."""
    existing = _read_tle_pairs(path)
    seen = set(existing)
    merged: list[tuple[datetime.datetime, int, tuple[str, str]]] = []
    for order, (line1, line2) in enumerate(existing):
        try:
            epoch = epoch_from_lines(line1, line2)
        except Exception:
            continue  # torn/corrupt record: dropped, healed by this rewrite
        merged.append((epoch, order, (line1, line2)))
    base = len(merged)
    for i, record in enumerate(records):
        pair = (record.tle_line1, record.tle_line2)
        if pair in seen:
            continue
        seen.add(pair)  # type: ignore[arg-type]
        merged.append((record.epoch, base + i, pair))  # type: ignore[arg-type]
    merged.sort(key=lambda item: (item[0], item[1]))
    _atomic_write(path, _render_tle(pair for _, _, pair in merged))


def _merge_object_csv(
    entry: OutputFile,
    path: pathlib.Path,
    rows: list[tuple[datetime.datetime, str, dict]],
) -> None:
    """Merge-rewrite for CSV object files. Dedup: RECORD_HASH for full
    columns; the entire rendered row for the STK projection (it carries no
    hash). An enrichment that changes a projected field can therefore
    leave both renderings of one record in an STK object file until the
    next `rebuild` — accepted; rebuild is the heal-all."""
    fieldnames = _fieldnames(entry)
    existing = _read_csv_rows(path)
    merged: list[tuple[datetime.datetime, int, dict]] = []
    seen_keys: set[object]
    if entry.columns == "full":
        seen_keys = {row.get("RECORD_HASH", "") for row in existing}
    else:
        seen_keys = {
            tuple(row.get(f, "") for f in fieldnames) for row in existing
        }
    for order, row in enumerate(existing):
        epoch = _parse_iso(row.get("EPOCH", "") or "")
        if epoch is None:
            continue  # torn tail row: dropped, healed by this rewrite
        merged.append((epoch, order, row))
    base = len(merged)
    for i, (epoch, hash_hex, row) in enumerate(rows):
        key = (
            hash_hex
            if entry.columns == "full"
            else tuple(row.get(f, "") for f in fieldnames)
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append((epoch, base + i, row))
    merged.sort(key=lambda item: (item[0], item[1]))
    _atomic_write(path, _render_csv(fieldnames, (row for _, _, row in merged)))


# --- date-file application (pure filesystem; runs on the pool) ---------------


def _apply_date_file(
    entry: OutputFile, path: pathlib.Path, winners: dict[int, GPRecord]
) -> tuple[int, int]:
    """Upsert the delta's winners into one date file.

    Replace on ``epoch >=`` so newest-ingested wins ties. Returns
    (files_written, lineless_skipped).
    """
    lineless = 0
    if entry.format == "tle":
        current: dict[int, tuple[datetime.datetime, tuple[str, str]]] = {}
        if path.exists():
            for line1, line2 in _read_tle_pairs(path):
                norad = _norad_from_line1(line1)
                if norad is None:
                    continue
                try:
                    epoch = epoch_from_lines(line1, line2)
                except Exception:
                    continue
                best = current.get(norad)
                if best is None or epoch >= best[0]:
                    current[norad] = (epoch, (line1, line2))
        changed = not path.exists()
        for norad, record in winners.items():
            if not (record.tle_line1 and record.tle_line2):
                lineless += 1
                continue
            best = current.get(norad)
            if best is None or record.epoch >= best[0]:
                pair = (record.tle_line1, record.tle_line2)
                if best is None or best[1] != pair:
                    current[norad] = (record.epoch, pair)
                    changed = True
        if not changed:
            return (0, lineless)
        _atomic_write(
            path,
            _render_tle(current[n][1] for n in sorted(current)),
        )
        return (1, lineless)

    # omm-csv
    fieldnames = _fieldnames(entry)
    current_rows: dict[int, tuple[datetime.datetime, dict]] = {}
    if path.exists():
        for row in _read_csv_rows(path):
            try:
                norad = int(str(row.get("NORAD_CAT_ID", "")).strip())
            except ValueError:
                continue
            epoch = _parse_iso(row.get("EPOCH", "") or "")
            if epoch is None:
                continue
            best = current_rows.get(norad)
            if best is None or epoch >= best[0]:
                current_rows[norad] = (epoch, row)
    changed = not path.exists()
    for norad, record in winners.items():
        best = current_rows.get(norad)
        if best is None or record.epoch >= best[0]:
            row = _row_for(entry, record)
            if best is None or best[1] != row:
                current_rows[norad] = (record.epoch, row)
                changed = True
    if not changed:
        return (0, 0)
    _atomic_write(
        path,
        _render_csv(fieldnames, (current_rows[n][1] for n in sorted(current_rows))),
    )
    return (1, 0)


# --- rebuild -------------------------------------------------------------------

# Objects per pool batch during rebuild: bounds the rows materialized at
# once while keeping the write pool busy.
_REBUILD_CHUNK_OBJECTS = 200


def rebuild_outputs(
    session: Session,
    output: OutputConfig,
    *,
    progress=None,
) -> dict[str, int]:
    """Re-derive every output from a linear store scan.

    First run, disaster recovery, or a new output entry. Object outputs
    stream grouped by object; date outputs stream the per-(date, object)
    winners via a window function. Returns files written per output key.
    """
    from thistle_db.progress import NO_PROGRESS

    progress = progress or NO_PROGRESS
    written: dict[str, int] = {}
    object_entries = [e for e in output.files if e.type == "object"]
    date_entries = [e for e in output.files if e.type == "date"]
    for entry in output.files:
        pathlib.Path(entry.dir).mkdir(parents=True, exist_ok=True)

    with WritePool(output.write_workers) as pool:
        if object_entries:
            _rebuild_objects(session, object_entries, pool, written, progress)
        if date_entries:
            _rebuild_dates(session, date_entries, pool, written, progress)
    return written


def _iter_records(rows) -> Iterable[tuple[int, GPRecord]]:
    for tle, metadata in rows:
        yield tle.norad_cat_id, record_from_row(tle, metadata)


def _rebuild_objects(session, entries, pool, written, progress) -> None:
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

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        for entry in entries:
            key = output_key(entry)

            def write_one(item: tuple[int, list[GPRecord]]) -> int:
                norad, records = item
                return _rebuild_object_file(entry, norad, records)

            results = pool.map(write_one, list(batch.items()))
            written[key] = written.get(key, 0) + sum(results)
            progress.advance(task, len(batch))
        batch = {}

    current_norad: Optional[int] = None
    for norad, record in _iter_records(session.execute(stmt)):
        if norad != current_norad and len(batch) >= _REBUILD_CHUNK_OBJECTS:
            flush_batch()
        current_norad = norad
        batch.setdefault(norad, []).append(record)
    flush_batch()
    progress.finish(task)


def _rebuild_object_file(
    entry: OutputFile, norad: int, records: list[GPRecord]
) -> int:
    path = entry.object_path(norad)
    if entry.format == "tle":
        pairs = [
            (r.tle_line1, r.tle_line2)
            for r in records
            if r.tle_line1 and r.tle_line2
        ]
        if not pairs:
            return 0
        _atomic_write(path, _render_tle(pairs))  # type: ignore[arg-type]
        return 1
    rows = [_row_for(entry, r) for r in records]
    _atomic_write(path, _render_csv(_fieldnames(entry), rows))
    return 1


def _rebuild_dates(session, entries, pool, written, progress) -> None:
    # Per-(epoch date, object) winners in SQL: one candidate row each, so
    # memory stays O(one date) regardless of history depth.
    date_expr = func.date(TLE.epoch).label("epoch_date")
    rn = (
        func.row_number()
        .over(
            partition_by=(date_expr, TLE.norad_cat_id),
            order_by=(TLE.epoch.desc(), TLE.id.desc()),
        )
        .label("rn")
    )
    winners = select(TLE.id.label("tle_id"), date_expr, rn).subquery()
    stmt = (
        select(TLE, OmmMetadata, winners.c.epoch_date)
        .join(winners, TLE.id == winners.c.tle_id)
        .outerjoin(OmmMetadata, OmmMetadata.tle_id == TLE.id)
        .where(winners.c.rn == 1)
        .order_by(winners.c.epoch_date, TLE.norad_cat_id)
        .execution_options(yield_per=5000)
    )

    task = progress.task("Rebuilding date files", total=None)
    current_date: Optional[datetime.date] = None
    day_records: list[GPRecord] = []

    def flush_day() -> None:
        if current_date is None or not day_records:
            return
        for entry in entries:
            key = output_key(entry)
            written[key] = written.get(key, 0) + _rebuild_date_file(
                entry, current_date, day_records
            )
        progress.advance(task)

    from thistle_db.model import date_from_sql

    for tle, metadata, raw_date in session.execute(stmt):
        day = date_from_sql(raw_date)
        if day != current_date:
            flush_day()
            current_date = day
            day_records = []
        day_records.append(record_from_row(tle, metadata))
    flush_day()
    progress.finish(task)


def _rebuild_date_file(
    entry: OutputFile, day: datetime.date, records: list[GPRecord]
) -> int:
    path = entry.date_path(day)
    if entry.format == "tle":
        pairs = [
            (r.tle_line1, r.tle_line2)
            for r in sorted(records, key=lambda r: r.norad_cat_id)
            if r.tle_line1 and r.tle_line2
        ]
        if not pairs:
            return 0
        _atomic_write(path, _render_tle(pairs))  # type: ignore[arg-type]
        return 1
    rows = [
        _row_for(entry, r) for r in sorted(records, key=lambda r: r.norad_cat_id)
    ]
    _atomic_write(path, _render_csv(_fieldnames(entry), rows))
    return 1
