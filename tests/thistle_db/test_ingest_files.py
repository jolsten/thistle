import hashlib
import os
import pathlib
import shutil

from sqlalchemy.orm import Session

from .conftest import TLES
from thistle_db.config import IngestSource
from thistle_db.ingest import FileStatus, ingest_source_file, ingest_sources
from thistle_db.model import IngestFile, TLE

DATA = pathlib.Path("tests/thistle_db/data")


def _copy(tmp_path: pathlib.Path, name: str = "25544.txt") -> pathlib.Path:
    return pathlib.Path(shutil.copy(DATA / name, tmp_path / name))


def _sources(tmp_path: pathlib.Path, pattern: str = "*") -> list[IngestSource]:
    return [IngestSource(path=str(tmp_path), pattern=pattern)]


def _append_tles(file: pathlib.Path) -> int:
    with open(file, "a") as f:
        for line1, line2 in TLES:
            f.write(f"{line1}\n{line2}\n")
    return len(TLES)


def test_scan_records_file_state(db_session: Session, tmp_path: pathlib.Path):
    file = _copy(tmp_path)
    count = ingest_sources(db_session, _sources(tmp_path))
    assert count > 0

    rows = db_session.query(IngestFile).all()
    assert len(rows) == 1
    row = rows[0]
    stat = file.stat()
    assert row.path == str(file.resolve())
    assert row.size == stat.st_size
    assert row.mtime_ns == stat.st_mtime_ns
    assert row.sha256 == hashlib.sha256(file.read_bytes()).hexdigest()


def test_second_scan_skips_unchanged_file(db_session: Session, tmp_path: pathlib.Path):
    file = _copy(tmp_path)
    first = ingest_sources(db_session, _sources(tmp_path))
    tle_count = db_session.query(TLE).count()
    assert first == tle_count > 0

    assert ingest_sources(db_session, _sources(tmp_path)) == 0
    assert ingest_source_file(db_session, file) == (FileStatus.SKIPPED, 0)
    assert db_session.query(IngestFile).count() == 1
    assert db_session.query(TLE).count() == tle_count


def test_appended_file_reingested_only_new_rows(
    db_session: Session, tmp_path: pathlib.Path
):
    file = _copy(tmp_path)
    first = ingest_sources(db_session, _sources(tmp_path))

    added = _append_tles(file)
    second = ingest_sources(db_session, _sources(tmp_path))
    assert second == added
    assert db_session.query(TLE).count() == first + added

    row = db_session.query(IngestFile).one()
    assert row.size == file.stat().st_size


def test_rewritten_identical_content_refreshes_state(
    db_session: Session, tmp_path: pathlib.Path
):
    file = _copy(tmp_path)
    ingest_sources(db_session, _sources(tmp_path))
    tle_count = db_session.query(TLE).count()

    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert ingest_source_file(db_session, file) == (FileStatus.REFRESHED, 0)
    row = db_session.query(IngestFile).one()
    assert row.mtime_ns == file.stat().st_mtime_ns
    assert db_session.query(TLE).count() == tle_count


def test_force_reingests_unchanged_file(db_session: Session, tmp_path: pathlib.Path):
    file = _copy(tmp_path)
    ingest_sources(db_session, _sources(tmp_path))
    tle_count = db_session.query(TLE).count()

    assert ingest_source_file(db_session, file, force=True) == (FileStatus.INGESTED, 0)
    assert ingest_sources(db_session, _sources(tmp_path), force=True) == 0
    assert db_session.query(TLE).count() == tle_count


def test_corrupt_file_does_not_abort_scan(db_session: Session, tmp_path: pathlib.Path):
    _copy(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")

    count = ingest_sources(db_session, _sources(tmp_path))
    assert count == db_session.query(TLE).count() > 0

    rows = db_session.query(IngestFile).all()
    assert len(rows) == 1
    assert rows[0].path != str(bad.resolve())

    # Not skipped next run: failure left no state, so the file is retried.
    assert ingest_source_file(db_session, bad) == (FileStatus.FAILED, 0)


def test_explicit_ingest_bypasses_skip_and_records_state(
    db_session: Session, tmp_path: pathlib.Path
):
    file = _copy(tmp_path)

    status1, count1 = ingest_source_file(db_session, file, force=True)
    assert status1 == FileStatus.INGESTED
    assert count1 > 0

    status2, count2 = ingest_source_file(db_session, file, force=True)
    assert status2 == FileStatus.INGESTED
    assert count2 == 0

    assert db_session.query(IngestFile).count() == 1


def test_missing_explicit_file_is_isolated(db_session: Session, tmp_path: pathlib.Path):
    missing = tmp_path / "nope.txt"
    assert ingest_source_file(db_session, missing, force=True) == (FileStatus.FAILED, 0)
    assert db_session.query(IngestFile).count() == 0
