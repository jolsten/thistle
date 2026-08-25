"""Benchmark output generation against a synthetic catalog.

Generation performance is dominated by per-file syscall latency and by how
much work the incremental path can skip, neither of which the unit tests
measure. This builds a throwaway SQLite catalog and times the two shapes
that matter: a full rebuild and a steady-state incremental run.

    uv run python packages/thistle-db/scripts/bench_generate.py
    uv run python packages/thistle-db/scripts/bench_generate.py --objects 30000
    uv run python packages/thistle-db/scripts/bench_generate.py --profile

Rows are dated one elset per object per day, with `created` spread across
the same span so an incremental run sees a realistic lookback slice.
"""

import argparse
import cProfile
import datetime
import io
import pathlib
import pstats
import shutil
import tempfile
import time

from loguru import logger
from sgp4.alpha5 import to_alpha5
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from thistle_db.config import OutputConfig, OutputFile
from thistle_db.generator import generate
from thistle_db.ingest import _bulk_insert_ignore, _TLE_CONFLICT, _TLE_TABLE, _tle_to_record
from thistle_db.model import Base, TLE
from thistle_db.writer import resolve_workers

TEMPLATE = (
    "1 {satnum}U 98067A   {epoch}  .00016717  00000-0  10270-3 0  {elnum:04d}",
    "2 {satnum}  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)


def _checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in line[:68]) % 10


def _lines(satnum: int, epoch: str, elnum: int) -> tuple[str, str]:
    alpha5 = to_alpha5(satnum)
    pair = tuple(t.format(satnum=alpha5, epoch=epoch, elnum=elnum) for t in TEMPLATE)
    return tuple(line[:68] + str(_checksum(line)) for line in pair)  # type: ignore[return-value]


def build_catalog(db_path: pathlib.Path, objects: int, days: int) -> int:
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    # `now` is fixed so `created` timestamps are deterministic across runs.
    now = datetime.datetime(2026, 8, 24, 12, 0, 0)
    total = 0
    for day_index in range(days):
        day = now.date() - datetime.timedelta(days=days - 1 - day_index)
        stamp = f"{day.year % 100:02d}{day.timetuple().tm_yday:03d}"
        records = []
        for i in range(objects):
            epoch = f"{stamp}.{(i * 7919) % 100_000_000:08d}"
            line1, line2 = _lines(900 + i, epoch, (day_index + 1) % 10_000)
            record = _tle_to_record(
                TLE.from_twoline(line1, line2),
                now=now - datetime.timedelta(days=days - 1 - day_index),
            )
            records.append(record)
        total += _bulk_insert_ignore(session, _TLE_TABLE, records, _TLE_CONFLICT)
    session.close()
    engine.dispose()
    return total


def _config(out_dir: pathlib.Path, workers: int) -> OutputConfig:
    return OutputConfig(
        files=[
            OutputFile(type=kind, format=fmt, dir=str(out_dir))
            for kind in ("date", "object")
            for fmt in ("tle", "omm")
        ],
        write_workers=workers,
    )


def _run(session, config, *, rebuild_all: bool, profile: bool) -> float:
    profiler = cProfile.Profile() if profile else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    generate(session, config, rebuild_all=rebuild_all)
    if profiler:
        profiler.disable()
    elapsed = time.perf_counter() - start
    if profiler:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(15)
        print(stream.getvalue())
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=5000)
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="output.write_workers (0 = auto, 1 = serial)",
    )
    parser.add_argument("--profile", action="store_true", help="print a cProfile table")
    parser.add_argument(
        "--keep", action="store_true", help="keep the scratch directory"
    )
    args = parser.parse_args()

    logger.remove()  # the bench prints its own numbers
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="thistle-bench-"))
    try:
        db_path = workdir / "bench.db"
        out_dir = workdir / "out"

        start = time.perf_counter()
        rows = build_catalog(db_path, args.objects, args.days)
        build = time.perf_counter() - start
        print(
            f"catalog: {rows:,} rows "
            f"({args.objects:,} objects x {args.days} days) in {build:.1f}s"
        )

        engine = create_engine(f"sqlite:///{db_path}")
        session = sessionmaker(bind=engine)()
        config = _config(out_dir, args.workers)
        resolved = resolve_workers(config.write_workers)
        print(f"outputs: {len(config.files)}  write threads: {resolved}")

        full = _run(session, config, rebuild_all=True, profile=args.profile)
        print(f"generate --all:   {full:6.2f}s")

        # Second run with everything already on disk: the steady-state shape,
        # where only the lookback slice is reconsidered.
        incremental = _run(session, config, rebuild_all=False, profile=args.profile)
        print(f"generate (incr):  {incremental:6.2f}s")

        session.close()
        engine.dispose()
    finally:
        if args.keep:
            print(f"scratch kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
