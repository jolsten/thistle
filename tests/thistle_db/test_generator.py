import pathlib
import tempfile

import pytest
from sqlalchemy.orm import Session

from .conftest import TLES
from thistle_db.config import OutputConfig, OutputFile
from thistle_db.generator import generate
from thistle_db.ingest import ingest_tles


@pytest.fixture
def populated_db(db_session: Session) -> Session:
    ingest_tles(db_session, TLES)
    return db_session


def test_generate_tle_date_files(populated_db: Session):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OutputConfig(
            files=[OutputFile(type="date", format="tle", dir=tmpdir)]
        )
        generate(populated_db, config)

        tle_files = list(pathlib.Path(tmpdir).glob("*.tle"))
        assert len(tle_files) > 0
        for f in tle_files:
            # Filename should be YYYYMMDD.tle
            stem = f.stem
            assert len(stem) == 8
            assert stem.isdigit()


def test_generate_tle_object_files(populated_db: Session):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OutputConfig(
            files=[OutputFile(type="object", format="tle", dir=tmpdir)]
        )
        generate(populated_db, config)

        tle_files = list(pathlib.Path(tmpdir).glob("*.tle"))
        assert len(tle_files) > 0


def test_generate_omm_date_files(populated_db: Session):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OutputConfig(
            files=[OutputFile(type="date", format="omm", dir=tmpdir)]
        )
        generate(populated_db, config)

        omm_files = list(pathlib.Path(tmpdir).glob("*.omm"))
        assert len(omm_files) > 0
        # Check CSV content
        content = omm_files[0].read_text()
        assert "OBJECT_NAME" in content  # CSV header


def test_generate_creates_output_dirs(populated_db: Session):
    with tempfile.TemporaryDirectory() as tmpdir:
        date_dir = pathlib.Path(tmpdir) / "nested" / "dates"
        object_dir = pathlib.Path(tmpdir) / "nested" / "objects"
        config = OutputConfig(
            files=[
                OutputFile(type="date", format="tle", dir=str(date_dir)),
                OutputFile(type="object", format="tle", dir=str(object_dir)),
            ]
        )
        generate(populated_db, config)
        assert date_dir.is_dir()
        assert object_dir.is_dir()


def test_generate_no_outputs_configured_raises(populated_db: Session):
    with pytest.raises(ValueError, match="no outputs configured"):
        generate(populated_db, OutputConfig())


def test_generate_separate_dirs_per_output(populated_db: Session):
    """Each output lands in its own configured directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tle_dir = pathlib.Path(tmpdir) / "tle"
        omm_dir = pathlib.Path(tmpdir) / "omm"
        config = OutputConfig(
            files=[
                OutputFile(type="object", format="tle", dir=str(tle_dir)),
                OutputFile(type="object", format="omm", dir=str(omm_dir)),
            ]
        )
        generate(populated_db, config)

        assert list(tle_dir.glob("*.tle"))
        assert not list(tle_dir.glob("*.omm"))
        assert list(omm_dir.glob("*.omm"))
        assert not list(omm_dir.glob("*.tle"))
