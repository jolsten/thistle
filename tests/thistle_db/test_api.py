import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from thistle_db.api import get_tles, open_session, tles_for_object
from thistle_db.config import Database, Settings
from thistle_db.model import TLE, Base

from .conftest import TLES


@pytest.fixture()
def db_path(tmp_path):
    """An on-disk SQLite database populated with the shared TLE fixtures."""
    path = tmp_path / "api.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    for line1, line2 in TLES:
        session.add(TLE.from_twoline(line1, line2))
    session.commit()
    session.close()
    engine.dispose()
    return path


def _settings(path) -> Settings:
    return Settings(database=Database(drivername="sqlite", name=path.as_posix()))


def test_open_session_creates_schema(tmp_path):
    path = tmp_path / "fresh.db"
    session, engine = open_session(_settings(path))
    try:
        assert tles_for_object(session, 22) == []
    finally:
        session.close()
        engine.dispose()


def test_get_tles_returns_pairs_in_epoch_order(db_path):
    assert get_tles(22, _settings(db_path)) == [TLES[0], TLES[1]]


def test_get_tles_unknown_satnum(db_path):
    assert get_tles(99999, _settings(db_path)) == []


def test_get_tles_config_from_env(db_path, monkeypatch):
    """With config=None, the database comes from THISTLE_DB_* env vars."""
    monkeypatch.setenv("THISTLE_DB_DATABASE__DRIVERNAME", "sqlite")
    monkeypatch.setenv("THISTLE_DB_DATABASE__NAME", db_path.as_posix())
    assert get_tles(22) == [TLES[0], TLES[1]]
