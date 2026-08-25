"""Incremental generation: tail-guarded appends, rewrites, self-healing.

These tests use synthetic TLEs (epoch field edited in line1) so one object
can have many elsets. sgp4 does not validate checksums, and dedup is on the
exact text, so edited lines are distinct valid records.
"""

import os
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


# ---------------------------------------------------------------------------
# Date files: rewrite only what changed
# ---------------------------------------------------------------------------


def _date_config(outdir, *, omm: bool = False) -> OutputConfig:
    files = [OutputFile(type="date", format="tle", dir=str(outdir))]
    if omm:
        files.append(OutputFile(type="date", format="omm", dir=str(outdir)))
    return OutputConfig(files=files)


DATE_FILE = "20250410.tle"


def test_unchanged_date_file_is_not_rewritten(db_session: Session, outdir):
    """The steady-state win: a date whose watermark predates its file costs
    one stat, not a query and a rewrite."""
    config = _date_config(outdir)
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, config)

    path = outdir / DATE_FILE
    before = path.stat().st_mtime_ns

    generate(db_session, config)  # nothing ingested in between

    assert path.stat().st_mtime_ns == before  # untouched
    assert path.read_text().splitlines() == [*TLE_B]


def test_stale_date_file_is_rewritten(db_session: Session, outdir):
    """A file older than its watermark is stale however it got that way."""
    config = _date_config(outdir)
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, config)

    path = outdir / DATE_FILE
    path.write_text("clobbered\n")
    old = path.stat().st_mtime_ns - 10_000_000_000
    os.utime(path, ns=(old, old))

    generate(db_session, config)

    assert path.read_text().splitlines() == [*TLE_B]


def test_changed_date_file_is_rewritten(db_session: Session, outdir):
    config = _date_config(outdir)
    ingest_tles(db_session, [TLE_A])
    generate(db_session, config)
    assert (outdir / DATE_FILE).read_text().splitlines() == [*TLE_A]

    # A later elset for the same date must replace it as that day's latest.
    ingest_tles(db_session, [TLE_C])
    generate(db_session, config)
    assert (outdir / DATE_FILE).read_text().splitlines() == [*TLE_C]


def test_deleted_date_file_is_restored(db_session: Session, outdir):
    """Self-healing with no window and no lookback: a missing file is stale
    by definition, however old its date."""
    config = _date_config(outdir)
    ingest_tles(db_session, [TLE_A, TLE_B])
    generate(db_session, config)
    (outdir / DATE_FILE).unlink()

    # lookback_days=0: nothing qualifies as recently created, and the epoch
    # date is years old. Neither bounds the date pass any more.
    generate(db_session, config, lookback_days=0)

    assert (outdir / DATE_FILE).read_text().splitlines() == [*TLE_B]


def test_verify_repairs_a_corrupted_date_file(db_session: Session, outdir):
    """The compensating control for no longer rewriting unchanged dates."""
    config = _date_config(outdir, omm=True)
    ingest_tles(db_session, [TLE_A, TLE_OTHER])
    generate(db_session, config)

    path = outdir / DATE_FILE
    path.write_text("1 garbage\n")  # truncated in place, still present

    generate(db_session, config, lookback_days=0, verify=True)

    assert path.read_text().splitlines() == [*TLE_A]


def test_verify_leaves_intact_date_files_alone(db_session: Session, outdir):
    config = _date_config(outdir)
    ingest_tles(db_session, [TLE_A])
    generate(db_session, config)

    path = outdir / DATE_FILE
    before = path.stat().st_mtime_ns
    generate(db_session, config, lookback_days=0, verify=True)

    assert path.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Write pool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 4])
def test_output_is_identical_regardless_of_worker_count(
    db_session: Session, tmp_path, workers
):
    """Threading must not change a single byte of the output."""
    outdir = tmp_path / f"out{workers}"
    outdir.mkdir()
    config = OutputConfig(
        files=[
            OutputFile(type=kind, format=fmt, dir=str(outdir))
            for kind in ("date", "object")
            for fmt in ("tle", "omm")
        ],
        write_workers=workers,
    )
    ingest_tles(db_session, [TLE_A, TLE_B, TLE_C, TLE_OTHER])
    generate(db_session, config, rebuild_all=True)

    written = {p.name: p.read_bytes() for p in outdir.iterdir()}
    # TLE_OTHER has a blank international designator, which sgp4 cannot
    # export as OMM — so its date and object omm files are legitimately
    # absent, exactly as in the serial path.
    assert set(written) == {
        "20250410.tle",
        "20250410.omm",
        "20250418.tle",
        "900.tle",
        "900.omm",
        "81069.tle",
    }
    assert written["900.tle"].decode().splitlines() == [*TLE_A, *TLE_B, *TLE_C]


def test_future_dated_file_is_restored(db_session: Session, outdir):
    """Feeds carry elsets with epochs ahead of now; their date files must
    still be inside the self-healing sweep, which stops at the catalog's
    newest epoch rather than at today."""
    import datetime

    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=2)
    day_of_year = future.timetuple().tm_yday
    line1 = (
        f"1 00900U 59009A   {future.year % 100:02d}{day_of_year:03d}.50000000"
        "  .00013443  00000-0  59558-3 0  9999"
    )
    config = _date_config(outdir)
    ingest_tles(db_session, [(line1, _L2)])
    generate(db_session, config)

    path = outdir / f"{future.strftime('%Y%m%d')}.tle"
    assert path.exists()
    path.unlink()

    # No date qualifies as "changed": only the missing-file sweep can
    # restore it, and the epoch is two days past today.
    generate(db_session, config, lookback_days=0)
    assert path.exists()


def test_far_future_epoch_costs_one_row_and_one_stat(db_session: Session, outdir):
    """A malformed elset can carry an epoch decades out (a two-digit year
    maps 00-56 to 2000-2056). Ingest's max_epoch_ahead_days guard rejects
    those by default, but rows can predate the guard or arrive out of band —
    and even then, scope comes from the per-date watermark table, so a 2056
    epoch costs exactly one extra row and one extra stat, not a calendar
    loop between now and 2056."""
    from sqlalchemy import select

    from thistle_db.model import EpochDateState

    far_future = (
        "1 00900U 59009A   56365.50000000  .00013443  00000-0  59558-3 0  9999",
        _L2,
    )
    config = _date_config(outdir)
    # Guard disabled: simulating a row that got in before the guard existed.
    ingest_tles(db_session, [TLE_A, far_future], max_epoch_ahead_days=0)
    generate(db_session, config)

    dates = set(db_session.execute(select(EpochDateState.epoch_date)).scalars())
    assert len(dates) == 2  # one row per real date, nothing in between
    # The bogus elset still gets its own date file: it is in the database, and
    # generation reflects the database. Keeping it out was ingest's job.
    assert (outdir / "20561230.tle").exists()


def test_watermarks_track_ingest(db_session: Session, outdir):
    """The scope table is maintained by ingest, in the same transaction."""
    from sqlalchemy import select

    from thistle_db.model import EpochDateState

    ingest_tles(db_session, [TLE_A])
    first = db_session.execute(select(EpochDateState.last_created)).scalar_one()

    ingest_tles(db_session, [TLE_C])
    second = db_session.execute(select(EpochDateState.last_created)).scalar_one()

    assert second >= first  # same date, bumped by the later delivery


def test_rebuild_recovers_from_out_of_band_inserts(db_session: Session, outdir):
    """Rows inserted outside ingest leave no watermark, so their date files
    would never regenerate — the rebuild is the repair."""
    from sqlalchemy import delete

    from thistle_db.api import rebuild_epoch_date_state
    from thistle_db.model import EpochDateState

    config = _date_config(outdir)
    ingest_tles(db_session, [TLE_A])
    # Simulate a bulk load that bypassed ingest: rows present, watermark gone.
    db_session.execute(delete(EpochDateState))
    db_session.commit()

    generate(db_session, config)
    assert not (outdir / DATE_FILE).exists()  # nothing looks stale

    rebuild_epoch_date_state(db_session.get_bind())
    # The rebuild ran on its own connection. Under MariaDB's REPEATABLE READ
    # this session is still on the snapshot it opened above, so end that
    # transaction before reading — in real use init-db and generate are
    # separate processes and the question never arises.
    db_session.commit()
    generate(db_session, config)
    assert (outdir / DATE_FILE).read_text().splitlines() == [*TLE_A]
