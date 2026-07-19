import pathlib
import tempfile

import pytest
from sqlalchemy.orm import Session

from tests.conftest import TLES
from thistle_db.config import OutputConfig, OutputFormats, OutputTypes
from thistle_db.generator import generate
from thistle_db.ingest import ingest_tles


@pytest.fixture
def populated_db(db_session: Session) -> Session:
    ingest_tles(db_session, TLES)
    return db_session


def test_generate_tle_date_files(populated_db: Session):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OutputConfig(
            dir=tmpdir,
            formats=OutputFormats(tle=True, omm=False),
            types=OutputTypes(date_files=True, object_files=False),
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
            dir=tmpdir,
            formats=OutputFormats(tle=True, omm=False),
            types=OutputTypes(date_files=False, object_files=True),
        )
        generate(populated_db, config)

        tle_files = list(pathlib.Path(tmpdir).glob("*.tle"))
        assert len(tle_files) > 0


def test_generate_omm_date_files(populated_db: Session):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OutputConfig(
            dir=tmpdir,
            formats=OutputFormats(tle=False, omm=True),
            types=OutputTypes(date_files=True, object_files=False),
        )
        generate(populated_db, config)

        omm_files = list(pathlib.Path(tmpdir).glob("*.omm"))
        assert len(omm_files) > 0
        # Check CSV content
        content = omm_files[0].read_text()
        assert "OBJECT_NAME" in content  # CSV header


def test_generate_creates_output_dir(populated_db: Session):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir) / "nested" / "output"
        config = OutputConfig(
            dir=str(output_dir),
            formats=OutputFormats(tle=True, omm=False),
            types=OutputTypes(date_files=True, object_files=False),
        )
        generate(populated_db, config)
        assert output_dir.is_dir()
