"""File-backed element-set store — the ``files`` storage backend.

With ``[storage] backend = "files"`` there is no element-set database: the
first ``type="object", format="tle"`` entry in ``[[output.files]]`` is the
canonical **store** — one epoch-ordered, deduplicated file of verbatim TLE
text per satellite — and every other configured output (date files, extra
object trees) is derived from the same in-memory delta during ingest. The
ingest/generate cycle collapses to a single pass: at flush time the ingest
buffer *is* the delta, so there are no ``created`` timestamps, no watermark
tables, no lookback window, and no ingest/generate serialization rule.

Why per-file dedup is enough: the dedup key is the exact text of
``(line1, line2)``, and both the NORAD ID and the epoch are encoded in
those lines — every record deterministically maps to one object file, so
deduplicating within each file is globally equivalent to the database's
``UNIQUE(epoch, line_hash)`` index.

Flush mechanics per touched object, per object output (mirroring the
generator's tail guard):

- every new epoch strictly greater than the file's tail epoch → **append**
  (provably not on disk, keeps the file sorted);
- anything else — late delivery, epoch tie, re-ingest of an old file,
  torn/damaged tail → **merge-rewrite**: read the file, drop records whose
  line hash is already present, stable-sort by epoch, write via tmp +
  atomic rename. The merge is also the self-heal for torn appends.

Tie rule (replaces the database's ``id DESC``): merges are stable with
existing records sorting before new ones among equal epochs, so file order
among equal epochs *is* ingestion order; date files treat the last record
in that order as "latest" (the incremental upsert replaces on
``epoch >=``). Incremental updates and ``rebuild_derived`` therefore always
agree, and both match the database backend's newest-ingested-wins behavior.

Crash safety without transactions: appends are guarded by the torn-tail
check, rewrites are atomic renames, and source-file state is committed only
*after* the flush that covers it — a crash at any point re-ingests at most
the files whose state was never recorded, and dedup absorbs the replay.
"""

import datetime
import os
import pathlib
import shutil
from typing import Iterable, NamedTuple, Optional, Sequence, cast

from loguru import logger
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from sqlalchemy import Engine, Table, create_engine, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from thistle_db.config import OutputConfig, OutputFile, Settings
from thistle_db.generator import OBJECT_CHUNK, _tle_tail_epoch
from thistle_db.model import IngestFile, compute_line_hash
from thistle_db.progress import NO_PROGRESS, ProgressReporter
from thistle_db.reader import TLETuple, read_tle, render_tle, tle_epoch
from thistle_db.writer import WritePool


class StoredElset(NamedTuple):
    """One validated record on its way into the file store.

    ``epoch`` comes from the same sgp4 parse as the database column
    (``sat_epoch_datetime``), so comparisons against epochs re-derived from
    file text are exact.
    """

    norad_cat_id: int
    epoch: datetime.datetime
    line1: str
    line2: str
    line_hash: bytes


def store_entry(output: OutputConfig) -> OutputFile:
    """The canonical store: the first object/tle output entry.

    Config validation guarantees one exists when the files backend is
    selected (``Settings._check_files_backend``).
    """
    for entry in output.files:
        if entry.type == "object" and entry.format == "tle":
            return entry
    raise ValueError(
        "files backend requires a type='object', format='tle' output entry"
    )


def open_state_session(config: Settings) -> tuple[Session, Engine]:
    """Open (creating if needed) the files backend's state database.

    Holds only ``ingest_files`` — the unchanged-file skip cache. Unlike the
    element-set database it is created on first use rather than by
    ``init-db``: it is a cache, and losing it only costs re-reading source
    files, which dedup absorbs.
    """
    path = pathlib.Path(config.storage.state)
    if path.parent != pathlib.Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(URL.create("sqlite", database=str(path)))
    cast(Table, IngestFile.__table__).create(engine, checkfirst=True)
    return sessionmaker(bind=engine)(), engine


def _ident_from_lines(line1: str, line2: str) -> tuple[int, datetime.datetime]:
    """(norad_cat_id, epoch) of a TLE, matching ingest-time values exactly."""
    sat = Satrec.twoline2rv(line1, line2)
    return sat.satnum, sat_epoch_datetime(sat).replace(tzinfo=None)


def _read_pairs_lenient(path: pathlib.Path) -> tuple[list[TLETuple], bool]:
    """All (line1, line2) pairs in a file, tolerating a torn tail.

    Returns (pairs, clean). A file not ending in a newline had a torn write
    (crash mid-append): the incomplete final line is dropped from the pairs
    and ``clean`` is False so the caller rewrites even when nothing new
    arrived — the rewrite is the heal. A dangling unpaired line1 is simply
    dropped, exactly as ``read_tle`` would drop it.
    """
    raw = path.read_bytes()
    clean = len(raw) == 0 or raw.endswith(b"\n")
    text = raw.decode("utf-8", errors="replace")
    if not clean:
        cut = text.rfind("\n")
        text = text[: cut + 1] if cut >= 0 else ""

    pairs: list[TLETuple] = []
    line1: Optional[str] = None
    for rawline in text.splitlines():
        line = rawline.rstrip()
        if not line:
            continue
        if line.startswith("1 "):
            line1 = line
        elif line.startswith("2 ") and line1 is not None:
            pairs.append((line1, line))
            line1 = None
    return pairs, clean


def _atomic_write(path: pathlib.Path, content: str) -> None:
    """Write via tmp + rename so a crash never leaves a half-written file.

    Ingest runs are serialized by deployment (same assumption as the
    database backend's generate), so a fixed tmp name per target is safe.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Per-object flush work (pure filesystem; runs on the write pool)
# ---------------------------------------------------------------------------


def _emit_object(
    args: tuple[Sequence[OutputFile], int, list[StoredElset]],
) -> tuple[str, int]:
    """Land one object's buffered records in its file in every output.

    Returns (strongest action, records newly written). Decision per output:
    append when every record's epoch is strictly greater than the file's
    tail epoch (provably new, keeps the file sorted), merge otherwise. The
    action/count reported is the store entry's (always the first output) —
    derived outputs land the same records, so separate counts would be
    noise. A *derived* output whose file is missing (an entry added to the
    config later, or a deleted file) is healed by copying the store's file,
    which at that point already holds full history plus this delta.
    """
    outputs, _norad, recs = args
    recs = sorted(recs, key=lambda r: r.epoch)  # stable: arrival order on ties
    store_path = outputs[0].object_path(recs[0].norad_cat_id)

    action, written = "unchanged", 0
    for i, out in enumerate(outputs):
        path = out.object_path(recs[0].norad_cat_id)
        try:
            stat = os.stat(path)
        except OSError:
            stat = None

        if i > 0 and stat is None and path != store_path:
            shutil.copyfile(store_path, path)
            continue

        if stat is not None:
            tail = _tle_tail_epoch(path, stat.st_size)
        else:
            tail = None

        if stat is None or (tail is not None and recs[0].epoch > tail):
            # Missing file (new object), or all-new epochs: append.
            with open(path, "a") as f:
                f.write(render_tle((r.line1, r.line2) for r in recs))
            out_action, out_written = "appended", len(recs)
        else:
            out_written = _merge_object(path, recs)
            out_action = "merged" if out_written or tail is None else "unchanged"

        if i == 0:
            action, written = out_action, out_written
    return action, written


def _merge_object(path: pathlib.Path, recs: Sequence[StoredElset]) -> int:
    """Merge records into an existing object file, returning the new count.

    Reads the file, drops already-present records by line hash, and
    rewrites atomically with a stable epoch sort — existing records keep
    their relative order and sort before new ones on epoch ties, so file
    order among ties remains ingestion order (the tie rule). Nothing new
    and an intact file → no write at all. A torn tail rewrites regardless:
    the rewrite is the self-heal.
    """
    pairs, clean = _read_pairs_lenient(path)
    seen = {compute_line_hash(l1, l2) for l1, l2 in pairs}
    new = [r for r in recs if r.line_hash not in seen]
    if not new and clean:
        return 0

    existing = [(_ident_from_lines(l1, l2)[1], l1, l2) for l1, l2 in pairs]
    combined = existing + [(r.epoch, r.line1, r.line2) for r in new]
    combined.sort(key=lambda t: t[0])  # stable
    _atomic_write(path, render_tle((l1, l2) for _, l1, l2 in combined))
    return len(new)


def _upsert_date(
    args: tuple[OutputFile, datetime.date, dict[int, StoredElset]],
) -> bool:
    """Fold new latest-per-object candidates into one date output's file.

    Correct without consulting object files: "latest elset per object that
    day" can only change when a newer-epoch record for that (object, date)
    arrives — exactly what the delta holds. Replacement on an equal epoch
    with different text implements newest-ingested-wins (the database's
    ``id DESC``); an equal epoch with identical text is a re-delivery and
    changes nothing, so an untouched file is never rewritten.
    """
    out, date_val, best = args
    path = out.date_path(date_val)

    existing: dict[int, tuple[datetime.datetime, str, str]] = {}
    changed = False
    if path.exists():
        pairs, clean = _read_pairs_lenient(path)
        for l1, l2 in pairs:
            norad, epoch = _ident_from_lines(l1, l2)
            existing[norad] = (epoch, l1, l2)
        changed = not clean  # torn file: rewrite even without new winners

    for norad, rec in best.items():
        cur = existing.get(norad)
        if (
            cur is None
            or rec.epoch > cur[0]
            or (rec.epoch == cur[0] and (rec.line1, rec.line2) != (cur[1], cur[2]))
        ):
            existing[norad] = (rec.epoch, rec.line1, rec.line2)
            changed = True

    if changed:
        ordered = ((l1, l2) for _, (_, l1, l2) in sorted(existing.items()))
        _atomic_write(path, render_tle(ordered))
    return changed


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class FileStore:
    """In-memory ingest buffer over the output-file tree.

    ``add`` deduplicates within the run and self-flushes past the memory
    budget; ``close`` flushes the remainder and logs a summary. Source-file
    state handed to ``defer_file_state`` is committed (via the state-db
    session) only after the flush that covers it, so a crash can never mark
    a file ingested whose records did not reach disk — the file-backend
    analog of the watermark-in-same-transaction rule.

    Records buffered from a file that later *fails* mid-parse may still be
    flushed; that is harmless (the file's state stays unrecorded, the next
    scan retries it, and dedup absorbs the replay).
    """

    def __init__(
        self,
        session: Session,
        output: OutputConfig,
        *,
        flush_records: int = 1_000_000,
        progress: ProgressReporter = NO_PROGRESS,
    ):
        self.store = store_entry(output)
        # The store decides first (its action/count is the one reported).
        self.object_outputs = [self.store] + [
            f for f in output.files if f.type == "object" and f is not self.store
        ]
        self.date_outputs = [f for f in output.files if f.type == "date"]
        for out in output.files:
            pathlib.Path(out.dir).mkdir(parents=True, exist_ok=True)

        self._session = session
        self._flush_records = flush_records
        self._workers = output.write_workers
        self._progress = progress

        self._buffers: dict[int, list[StoredElset]] = {}
        self._hashes: set[bytes] = set()
        self._count = 0
        self._pending_state: list[dict] = []
        self._totals = {"appended": 0, "merged": 0, "unchanged": 0, "records": 0}
        self._flushes = 0

    def add(self, records: Iterable[StoredElset]) -> int:
        """Buffer records, deduplicating within the run. Returns the number
        newly buffered; on-disk duplicates are discovered (and dropped) at
        flush time."""
        added = 0
        for rec in records:
            if rec.line_hash in self._hashes:
                continue
            self._hashes.add(rec.line_hash)
            self._buffers.setdefault(rec.norad_cat_id, []).append(rec)
            added += 1
        self._count += added
        if self._count >= self._flush_records:
            logger.info(
                f"File store: buffer reached {self._count} records, flushing early"
            )
            self.flush()
        return added

    def defer_file_state(
        self, path: str, path_hash: str, size: int, mtime_ns: int, sha256: str
    ) -> None:
        """Record a source file's ingest state at the next flush (not before:
        its records are still only in memory)."""
        self._pending_state.append(
            dict(
                path=path,
                path_hash=path_hash,
                size=size,
                mtime_ns=mtime_ns,
                sha256=sha256,
            )
        )

    def flush(self) -> None:
        """Write buffered records to every output, then commit file states."""
        if self._count:
            with WritePool(self._workers) as pool:
                self._flush_objects(pool)
                self._flush_dates(pool)
        self._commit_states()
        # Cross-flush duplicates within one run fall to the merge path's
        # on-disk hash check, so the dedup set need not outlive the buffer
        # (keeping it would grow memory with the run, defeating the budget).
        self._buffers.clear()
        self._hashes.clear()
        self._totals["records"] += self._count
        self._count = 0
        self._flushes += 1

    def close(self) -> None:
        """Final flush plus the run summary."""
        self.flush()
        t = self._totals
        written = t["appended"] + t["merged"]
        logger.info(
            f"File store: {t['records']} records buffered across "
            f"{self._flushes} flush(es); objects: {t['appended']} appended, "
            f"{t['merged']} merged, {t['unchanged']} unchanged"
        )
        if not written and t["records"]:
            logger.info("File store: all buffered records were already on disk")

    def _flush_objects(self, pool: WritePool) -> None:
        items = sorted(self._buffers.items())
        task = self._progress.task("Writing object files", total=len(items))
        for start in range(0, len(items), OBJECT_CHUNK):
            chunk = items[start : start + OBJECT_CHUNK]
            results = pool.map(
                _emit_object,
                [(self.object_outputs, norad, recs) for norad, recs in chunk],
            )
            for action, _written in results:
                self._totals[action] += 1
            self._progress.advance(task, len(chunk))
        self._progress.finish(task)

    def _flush_dates(self, pool: WritePool) -> None:
        if not self.date_outputs:
            return
        # Best new candidate per (date, object): max epoch, with arrival
        # order breaking ties (>=), so the last-ingested wins — consistent
        # with the merge tie rule and the database's id DESC.
        updates: dict[datetime.date, dict[int, StoredElset]] = {}
        for norad, recs in self._buffers.items():
            for rec in recs:
                per_date = updates.setdefault(rec.epoch.date(), {})
                cur = per_date.get(norad)
                if cur is None or rec.epoch >= cur.epoch:
                    per_date[norad] = rec

        tasks = [
            (out, date_val, best)
            for date_val, best in sorted(updates.items())
            for out in self.date_outputs
        ]
        results = pool.map(_upsert_date, tasks)
        logger.info(
            f"Date files: {sum(results)} of {len(tasks)} updated "
            f"({len(updates)} dates)"
        )

    def _commit_states(self) -> None:
        if not self._pending_state:
            return
        for state in self._pending_state:
            row = self._session.execute(
                select(IngestFile).where(IngestFile.path_hash == state["path_hash"])
            ).scalar_one_or_none()
            if row is None:
                self._session.add(IngestFile(**state))
            else:
                row.size = state["size"]
                row.mtime_ns = state["mtime_ns"]
                row.sha256 = state["sha256"]
        self._session.commit()
        self._pending_state = []


# ---------------------------------------------------------------------------
# Reads (get-tle, dump, thistle's fallback)
# ---------------------------------------------------------------------------


def read_object_tles(output: OutputConfig, satnum: int) -> list[TLETuple]:
    """All (line1, line2) pairs for one satellite from the store, epoch order
    (the store file's own order). Empty list when the object has no file."""
    path = store_entry(output).object_path(satnum)
    if not path.exists():
        return []
    return list(read_tle(path))


def _epoch_datetime(tle: TLETuple) -> datetime.datetime:
    """Approximate epoch datetime from the fast float parse.

    Sub-millisecond differences from the exact sgp4 value don't matter for
    nearest-neighbor selection.
    """
    epoch = tle_epoch(tle)
    year, doy = divmod(epoch, 1000)
    return datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=doy - 1)


def nearest_tles_for_date(
    output: OutputConfig, date: datetime.date, days: float
) -> list[TLETuple]:
    """Nearest TLE per satellite to 12:00 UTC on `date`, within +/- `days`,
    from the configured date files.

    Candidates are each day's latest-per-object records (what date files
    hold), so a superseded same-day elset nearer to noon than that day's
    latest is not considered — a bounded approximation the database backend
    doesn't make. Raises ValueError without a tle date output.
    """
    date_outs = [f for f in output.files if f.type == "date" and f.format == "tle"]
    if not date_outs:
        raise ValueError(
            "date queries on the files backend need a type='date', "
            "format='tle' [[output.files]] entry"
        )
    out = date_outs[0]
    center = datetime.datetime.combine(date, datetime.time(12))
    window = datetime.timedelta(days=days)

    best: dict[int, tuple[datetime.timedelta, TLETuple]] = {}
    day = (center - window).date()
    last = (center + window).date()
    while day <= last:
        path = out.date_path(day)
        if path.exists():
            for pair in read_tle(path):
                delta = abs(_epoch_datetime(pair) - center)
                if delta > window:
                    continue
                norad, _ = _ident_from_lines(*pair)
                cur = best.get(norad)
                if cur is None or delta < cur[0]:
                    best[norad] = (delta, pair)
        day += datetime.timedelta(days=1)
    return [best[norad][1] for norad in sorted(best)]


def _store_files(store: OutputFile) -> list[tuple[int, pathlib.Path]]:
    """The store's object files as (norad_id, path), sorted by id."""
    files = []
    for path in pathlib.Path(store.dir).glob(store.object_glob()):
        norad = store.parse_object_name(path.name)
        if norad is not None:
            files.append((norad, path))
    return sorted(files)


def dump_store(output: OutputConfig, tle_path: pathlib.Path) -> int:
    """Concatenate the store into one re-ingestable TLE file.

    The files-backend counterpart of ``api.dump_db`` — object order, epoch
    order within each object, verbatim lines. Returns the record count.
    """
    count = 0
    with open(tle_path, "w") as f:
        for _norad, path in _store_files(store_entry(output)):
            pairs = list(read_tle(path))
            f.write(render_tle(pairs))
            count += len(pairs)
    return count


# ---------------------------------------------------------------------------
# Rebuild (generate --all)
# ---------------------------------------------------------------------------


def rebuild_derived(
    output: OutputConfig,
    *,
    flush_records: int = 1_000_000,
    progress: ProgressReporter = NO_PROGRESS,
) -> None:
    """Rebuild every derived output from the store (disaster recovery, or
    backfill after adding an output entry to the config).

    One linear scan of the store, memory bounded by ``flush_records``: date
    candidates accumulate per (date, object) and spill to the date files in
    batches — the first batch to touch a date overwrites its file (dropping
    any stale content), later batches upsert into it. An object's records
    never split across batches, so each (date, object) winner is decided
    wholly in one batch and batches never contend. The store itself is the
    source and is never touched.
    """
    store = store_entry(output)
    object_outs = [
        f
        for f in output.files
        if f.type == "object" and f is not store
    ]
    date_outs = [f for f in output.files if f.type == "date"]
    if not object_outs and not date_outs:
        logger.info("Rebuild: no derived outputs configured; store left as is")
        return
    for out in object_outs + date_outs:
        pathlib.Path(out.dir).mkdir(parents=True, exist_ok=True)

    files = _store_files(store)
    logger.info(f"Rebuilding derived outputs from {len(files)} store files")
    task = progress.task("Rebuilding outputs", total=len(files))

    # date -> norad -> (float epoch, line1, line2); float epochs (fast text
    # parse) are mutually consistent within the rebuild, which is all the
    # comparisons need.
    accum: dict[datetime.date, dict[int, tuple[float, str, str]]] = {}
    count = 0
    overwritten: set[tuple[int, datetime.date]] = set()

    with WritePool(output.write_workers) as pool:
        for start in range(0, len(files), OBJECT_CHUNK):
            chunk = files[start : start + OBJECT_CHUNK]
            contents = pool.map(lambda item: list(read_tle(item[1])), chunk)

            if object_outs:
                pool.map(
                    _rewrite_object,
                    [
                        (object_outs, norad, pairs)
                        for (norad, _), pairs in zip(chunk, contents)
                        if pairs
                    ],
                )

            for (norad, _), pairs in zip(chunk, contents):
                for pair in pairs:
                    epoch = tle_epoch(pair)
                    date_val = _epoch_datetime(pair).date()
                    per_date = accum.setdefault(date_val, {})
                    cur = per_date.get(norad)
                    # >=: later file position wins ties (ingestion order).
                    if cur is None or epoch >= cur[0]:
                        per_date[norad] = (epoch, pair[0], pair[1])
                count += len(pairs)

            # Whole object files only ever land in one batch, so flushing
            # here (between chunks) keeps (date, object) winners intact.
            if count >= flush_records:
                _spill_dates(date_outs, accum, overwritten, pool)
                accum, count = {}, 0
            progress.advance(task, len(chunk))

        _spill_dates(date_outs, accum, overwritten, pool)
    progress.finish(task)
    logger.info(
        f"Rebuilt {len(object_outs)} object output(s) and "
        f"{len(date_outs)} date output(s) from the store"
    )


def _rewrite_object(
    args: tuple[Sequence[OutputFile], int, list[TLETuple]],
) -> None:
    outputs, norad, pairs = args
    for out in outputs:
        _atomic_write(out.object_path(norad), render_tle(pairs))


def _spill_dates(
    date_outs: Sequence[OutputFile],
    accum: dict[datetime.date, dict[int, tuple[float, str, str]]],
    overwritten: set[tuple[int, datetime.date]],
    pool: WritePool,
) -> None:
    """Write one accumulation batch to the date outputs.

    First touch of a date overwrites the file — the rebuild is
    authoritative, so stale content from a previous run must not survive —
    and later batches merge into what this run wrote.
    """
    tasks = []
    for date_val, per_date in sorted(accum.items()):
        for i, out in enumerate(date_outs):
            tasks.append((out, date_val, per_date, (i, date_val) in overwritten))
            overwritten.add((i, date_val))
    pool.map(_spill_one_date, tasks)


def _spill_one_date(
    args: tuple[OutputFile, datetime.date, dict[int, tuple[float, str, str]], bool],
) -> None:
    out, date_val, per_date, merge = args
    path = out.date_path(date_val)
    if merge and path.exists():
        pairs, _clean = _read_pairs_lenient(path)
        for l1, l2 in pairs:
            epoch = tle_epoch((l1, l2))
            norad, _ = _ident_from_lines(l1, l2)
            cur = per_date.get(norad)
            # Existing entries lose only to a strictly newer epoch: they
            # were written by an earlier batch of the same rebuild, and no
            # (date, object) pair spans batches, so equal epochs here mean
            # an unrelated object collision cannot occur — keep the merge
            # conservative anyway.
            if cur is None or epoch > cur[0]:
                per_date[norad] = (epoch, l1, l2)
    ordered = ((l1, l2) for _, (_, l1, l2) in sorted(per_date.items()))
    _atomic_write(path, render_tle(ordered))
