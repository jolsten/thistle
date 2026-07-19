import os
import pathlib

import pytest

pytest.importorskip("thistle_db", reason="thistle-db not installed (requires Python >= 3.11)")

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from thistle_db.model import Base

TLE_FILES = list((pathlib.Path(__file__).parent / "data").glob("*.txt"))

TLES = [
    (
        "1 00022U 59009A   25107.90268499  .00013443  00000-0  59558-3 0  9999",
        "2 00022  50.2761  24.4877 0093242 224.4518 134.8929 15.18021433735521",
    ),
    (
        "1    22U 59009A   25108.23181848  .00011674  00000-0  51926-3 0  9993",
        "2    22  50.2762  22.8876 0093251 225.7509 133.5766 15.18029209490530",
    ),
    (
        "1 81069U          25108.62144655 +.00005722 +00000+0 +78011-2 0  9997",
        "2 81069 100.3732 355.8061 0063114 282.4083  76.9996 13.58632624483575",
    ),
]

BACKEND_URL_ENVS = {
    "mariadb": "THISTLE_DB_TEST_MARIADB_URL",
    "postgres": "THISTLE_DB_TEST_POSTGRES_URL",
}


def _database_url(backend: str) -> str:
    if backend == "sqlite":
        return "sqlite:///:memory:"
    env = BACKEND_URL_ENVS.get(backend)
    if env is None:
        raise ValueError(f"Unknown backend: {backend}")
    url = os.environ.get(env)
    if not url:
        pytest.skip(
            f"{env} not set; start {backend} via docker and export the URL "
            f"to run {backend} tests."
        )
        raise RuntimeError("unreachable")
    return url


@pytest.fixture(scope="function", params=["sqlite", "mariadb", "postgres"])
def db_session(request):
    url = _database_url(request.param)
    engine = create_engine(url)

    try:
        # Clean slate between tests for shared backends (MariaDB).
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
    except OperationalError as err:
        engine.dispose()
        pytest.skip(f"Cannot connect to {request.param} at {url}: {err}")

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
