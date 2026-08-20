"""Incremental generation: tail-guarded appends, rewrites, self-healing.

These tests use synthetic TLEs (epoch field edited in line1) so one object
can have many elsets. sgp4 does not validate checksums, and dedup is on the
exact text, so edited lines are distinct valid records.
"""

import pathlib

import pytest
from sqlalchemy.orm import Session

from thistle_db.config import OutputConfig, OutputFile
from thistle_db.generator import generate
from thistle_db.ingest import ingest_tles
from thistle_db.model import epoch_from_lines

# Object 900: epochs on 2025 day 100 (Apr 10) at .10, .20, .30, .40 of a day.
_L2 = "2 00900  50.2761  24.4877 0093242 224.4518 134.8929 15.18021433735521"


def _tle(day: str) -> tuple[str, str]:
    line1 = f"1 00900U 59009A   25{day}  .00013443  00000-0  59558-3 0  9999"
    return (line1, _L2)


TLE_A = _tle("100.10000000")
TLE_B = _tle("100.20000000")
TLE_C = _tle("100.30000000")
TLE_D = _tle("100.40000000")

TLE_OTHER = (
    "1 81069U          25108.62144655 +.00005722 +00000+0 +78011-2 0  9997",
    "2 81069 100.3732 355.8061 0063114 282.4083  76.9996 13.58632624483575",
)


def _config(tmpdir, *, omm: bool = True) -> OutputConfig:
    files = [OutputFile(type="object", format="tle", dir=str(tmpdir))]
    if omm:
        files.append(OutputFile(type="object", format="omm", dir=str(tmpdir)))
    return OutputConfig(files=files)


def _object_lines(tmpdir) -> list[str]:
    return (pathlib.Path(tmpdir) / "900.tle").read_text().splitlines()


@pytest.fixture
def outdir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


def test_new_object_file_written(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, _config(outdir))
    assert _object_lines(outdir) == [*TLE_A, *TLE_B]
    assert (outdir / "900.omm").exists()


def test_generate_is_idempotent(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, _config(outdir))
    before_tle = (outdir / "900.tle").read_bytes()
    before_omm = (outdir / "900.omm").read_bytes()

    generate(db_session, _config(outdir))

    assert (outdir / "900.tle").read_bytes() == before_tle
    assert (outdir / "900.omm").read_bytes() == before_omm


def test_newer_rows_appended(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, _config(outdir))

    ingest_tles(db_session, [TLE_C, TLE_D])
    generate(db_session, _config(outdir))

    assert _object_lines(outdir) == [*TLE_A, *TLE_B, *TLE_C, *TLE_D]
    # OMM CSV: header + 4 rows, epochs ascending
    omm_lines = (outdir / "900.omm").read_text().splitlines()
    assert len(omm_lines) == 5


def test_late_delivery_triggers_rewrite(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_C])
    generate(db_session, _config(outdir))
    assert _object_lines(outdir) == [*TLE_A, *TLE_C]

    # TLE_B's epoch is between A and C: it must land in the middle, sorted.
    ingest_tles(db_session, [TLE_B])
    generate(db_session, _config(outdir))
    assert _object_lines(outdir) == [*TLE_A, *TLE_B, *TLE_C]


def test_torn_tle_file_self_heals(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, _config(outdir))

    # Simulate a crash mid-append: truncate the last line partway.
    path = outdir / "900.tle"
    data = path.read_bytes()
    path.write_bytes(data[:-30])

    ingest_tles(db_session, [TLE_C])
    generate(db_session, _config(outdir))
    assert _object_lines(outdir) == [*TLE_A, *TLE_B, *TLE_C]


def test_deleted_file_regenerated(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, _config(outdir))
    (outdir / "900.tle").unlink()

    ingest_tles(db_session, [TLE_C])
    generate(db_session, _config(outdir))
    assert _object_lines(outdir) == [*TLE_A, *TLE_B, *TLE_C]


def test_untouched_object_not_modified(db_session: Session, outdir):
    """New rows for one object must not touch another object's files."""
    ingest_tles(db_session, [TLE_A, TLE_OTHER])
    generate(db_session, _config(outdir))
    other = outdir / "81069.tle"
    before_mtime = other.stat().st_mtime_ns

    ingest_tles(db_session, [TLE_B])
    generate(db_session, _config(outdir))

    assert other.stat().st_mtime_ns == before_mtime
    assert _object_lines(outdir) == [*TLE_A, *TLE_B]


def test_rebuild_all(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_B, TLE_OTHER])
    generate(db_session, _config(outdir))

    # Corrupt a file in a way the tail guard can't see (middle of the file);
    # --all must restore it even with no new rows.
    path = outdir / "900.tle"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[2:]) + "\n")

    generate(db_session, _config(outdir), rebuild_all=True)
    assert _object_lines(outdir) == [*TLE_A, *TLE_B]
    assert (outdir / "81069.tle").exists()


def test_verify_repairs_middle_truncation(db_session: Session, outdir):
    """Damage the tail guard can't see: --verify must catch and repair it,
    even with no new rows in the lookback."""
    ingest_tles(db_session, [TLE_A, TLE_B, TLE_C])
    generate(db_session, _config(outdir))

    path = outdir / "900.tle"
    lines = path.read_text().splitlines()
    del lines[2:4]  # remove TLE_B, keeping the file well-formed
    path.write_text("\n".join(lines) + "\n")

    generate(db_session, _config(outdir), verify=True)
    assert _object_lines(outdir) == [*TLE_A, *TLE_B, *TLE_C]


def test_verify_leaves_intact_files_alone(db_session: Session, outdir):
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, _config(outdir))
    before = (outdir / "900.tle").stat().st_mtime_ns

    generate(db_session, _config(outdir), verify=True)
    assert (outdir / "900.tle").stat().st_mtime_ns == before


def test_verify_repairs_missing_quiet_object(db_session: Session, outdir):
    """A deleted file for an object with no new rows: the incremental pass
    never visits it, but --verify restores it."""
    ingest_tles(db_session, [TLE_A, TLE_OTHER])
    generate(db_session, _config(outdir))
    (outdir / "81069.tle").unlink()

    generate(db_session, _config(outdir), lookback_days=0)  # nothing pending
    assert not (outdir / "81069.tle").exists()

    generate(db_session, _config(outdir), lookback_days=0, verify=True)
    assert (outdir / "81069.tle").exists()
    assert (outdir / "81069.tle").read_text().splitlines() == [*TLE_OTHER]


def test_date_files_pick_up_late_rows(db_session: Session, outdir):
    """Epochs far outside the trailing window still get their date files,
    because the rows were created within the lookback."""
    config = OutputConfig(
        files=[OutputFile(type="date", format="tle", dir=str(outdir))]
    )
    ingest_tles(db_session, [TLE_A, TLE_B, TLE_C])
    generate(db_session, config)

    # 2025 day 100 = 2025-04-10; latest elset per object for that date.
    date_file = outdir / "20250410.tle"
    assert date_file.exists()
    assert date_file.read_text().splitlines() == [*TLE_C]


def test_omm_tail_epoch_handles_whole_second_epochs(tmp_path):
    # An epoch with zero microseconds is formatted without ".000000";
    # the tail parser must read it rather than reporting a damaged tail.
    import datetime

    from thistle_db.generator import _omm_tail_epoch
    from thistle_db.reader import OMM_CSV_FIELDS

    row = {field: "" for field in OMM_CSV_FIELDS}
    row["EPOCH"] = "2025-04-10T02:24:00"
    path = tmp_path / "900.omm"
    path.write_text(
        ",".join(OMM_CSV_FIELDS)
        + "\n"
        + ",".join(row[field] for field in OMM_CSV_FIELDS)
        + "\n"
    )
    assert _omm_tail_epoch(path) == datetime.datetime(2025, 4, 10, 2, 24, 0)


def test_epoch_from_lines_matches_db(db_session: Session):
    """The tail guard's epoch computation must equal the stored column."""
    from sqlalchemy import select

    from thistle_db.model import TLE

    ingest_tles(db_session, [TLE_A])
    row = db_session.execute(select(TLE)).scalar_one()
    assert row.epoch == epoch_from_lines(*TLE_A)
