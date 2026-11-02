"""Bench: file-store vs SQLite ingest cost for a historical rebuild.

Feeds the same synthetic catalog through the shipping FileStore (v1, TLE
lines) and through the real `tle` schema in SQLite with chunked INSERT OR
IGNORE — in chronological delivery order (the normal historical-restore
case) and reverse-chronological (the merge-heavy worst case).
"""

import datetime
import shutil
import sys
import tempfile
import time
from pathlib import Path

from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from thistle_db.config import Settings
from thistle_db.filestore import FileStore, StoredElset, open_state_session
from thistle_db.model import Base, TLE, compute_line_hash

N_OBJ = 2000
DAYS = 60
PER_DAY = 2
FLUSH = 60_000
CHUNK = 5000  # ingest.CHUNK_SIZE

BASE = datetime.datetime(2025, 1, 1)


def synth_records() -> list[StoredElset]:
    recs = []
    for day in range(DAYS):
        date = BASE + datetime.timedelta(days=day)
        yy = date.year % 100
        doy = date.timetuple().tm_yday
        for k in range(PER_DAY):
            frac = 0.25 + 0.4 * k  # two elsets per day per object
            for obj in range(N_OBJ):
                satnum = 10000 + obj
                epoch_field = f"{yy:02d}{doy + frac:012.8f}"
                ma = (day * 7.13 + k * 3.7 + obj * 0.01) % 360
                l1 = (
                    f"1 {satnum:05d}U 59001A   {epoch_field}"
                    f"  .00003193  00000-0  16873-2 0  9997"
                )
                l2 = (
                    f"2 {satnum:05d}  32.8658 246.7104 1448126 168.7229 "
                    f"{ma:8.4f} 15.18021433 1552"
                )
                epoch = date + datetime.timedelta(days=frac)
                recs.append(
                    StoredElset(satnum, epoch, l1, l2, compute_line_hash(l1, l2))
                )
    return recs


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def bench_files(recs, tmp: Path) -> tuple[float, int]:
    settings = Settings(
        storage={"backend": "files", "state": str(tmp / "state.sqlite")},
        output={
            "files": [
                {"type": "object", "format": "tle", "dir": str(tmp / "objects")}
            ]
        },
        ingest={"sources": [{"path": str(tmp), "pattern": "*.none"}]},
    )
    session, engine = open_state_session(settings)
    t0 = time.perf_counter()
    store = FileStore(session, settings.output, flush_records=FLUSH)
    store.add(recs)
    store.close()
    elapsed = time.perf_counter() - t0
    session.close()
    engine.dispose()
    return elapsed, tree_bytes(tmp / "objects")


def bench_sqlite(recs, tmp: Path) -> tuple[float, int]:
    db = tmp / "tle.sqlite"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    stmt = insert(TLE.__table__).prefix_with("OR IGNORE")
    t0 = time.perf_counter()
    for start in range(0, len(recs), CHUNK):
        rows = [
            dict(
                line1=r.line1,
                line2=r.line2,
                line_hash=r.line_hash,
                object_id="1959-001A",
                norad_cat_id=r.norad_cat_id,
                classification="U",
                epoch=r.epoch,
                mean_motion=15.18021433,
                eccentricity=0.1448126,
                inclination=32.8658,
                ra_of_asc_node=246.7104,
                arg_of_pericenter=168.7229,
                mean_anomaly=194.9782,
                ephemeris_type=0,
                element_set_no=999,
                rev_at_epoch=1552,
                bstar=0.0016873,
                mean_motion_dot=3.193e-05,
                mean_motion_ddot=0.0,
                created=datetime.datetime.now(),
            )
            for r in recs[start : start + CHUNK]
        ]
        session.execute(stmt, rows)
        session.commit()
    elapsed = time.perf_counter() - t0
    session.close()
    engine.dispose()
    return elapsed, db.stat().st_size


def main():
    recs = synth_records()
    total = len(recs)
    print(f"catalog: {N_OBJ} objects x {DAYS} days x {PER_DAY}/day = {total} records")
    chrono = sorted(recs, key=lambda r: r.epoch)
    reverse = list(reversed(chrono))

    results = {}
    for name, data, fn in [
        ("files chrono ", chrono, bench_files),
        ("files reverse", reverse, bench_files),
        ("sqlite chrono ", chrono, bench_sqlite),
        ("sqlite reverse", reverse, bench_sqlite),
    ]:
        tmp = Path(tempfile.mkdtemp(prefix="bench_", dir=Path(__file__).parent))
        try:
            elapsed, size = fn(data, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rate = total / elapsed
        results[name] = elapsed
        print(
            f"{name}: {elapsed:7.2f}s  {rate:9.0f} rec/s  "
            f"{elapsed / total * 1e6:6.1f} us/rec  on-disk {size / 1e6:6.1f} MB"
        )
    print(
        f"\nfiles/sqlite ratio: chrono {results['files chrono '] / results['sqlite chrono ']:.2f}x"
        f"  reverse {results['files reverse'] / results['sqlite reverse']:.2f}x"
    )


if __name__ == "__main__":
    sys.exit(main())
