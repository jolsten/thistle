import os
import pathlib
import uuid

import pytest

pytest.importorskip(
    "thistle_db", reason="thistle-db not installed (requires Python >= 3.11)"
)

from sqlalchemy import create_engine, text
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

# ---------------------------------------------------------------------------
# MariaDB / PostgreSQL containers (opt-in, requires Docker)
# ---------------------------------------------------------------------------
#
# Gated by ``THISTLE_DB_TEST_MARIADB=1`` / ``THISTLE_DB_TEST_POSTGRES=1`` so
# devs without Docker aren't forced to pull images — without the flags,
# ``pytest`` runs the SQLite parametrizations only. One container per test
# session; each test gets its own CREATE/DROP DATABASE inside it, so tests
# stay isolated without paying the container boot cost per test.

_MARIADB_IMAGE = os.environ.get("THISTLE_DB_MARIADB_IMAGE", "mariadb:11")
_POSTGRES_IMAGE = os.environ.get("THISTLE_DB_POSTGRES_IMAGE", "postgres:16")


@pytest.fixture(scope="session")
def mariadb_container():
    """Start a MariaDB container for the session.

    Skipped unless ``THISTLE_DB_TEST_MARIADB=1`` is set in the environment.
    """
    if not os.environ.get("THISTLE_DB_TEST_MARIADB"):
        pytest.skip("THISTLE_DB_TEST_MARIADB=1 not set")

    try:
        from testcontainers.mysql import MySqlContainer
    except ImportError as e:
        pytest.skip(f"testcontainers not installed: {e}")

    # testcontainers 4.x has no MariaDbContainer; MySqlContainer with a
    # mariadb image + pymysql dialect is the supported path.
    container = MySqlContainer(_MARIADB_IMAGE, dialect="pymysql")
    # Allow root connections from any host (needed for CREATE/DROP DATABASE).
    container.with_env("MARIADB_ROOT_HOST", "%")
    container.start()
    try:
        yield container
    finally:
        container.stop()


def _mariadb_admin_url(container) -> str:
    """SQLAlchemy URL connecting as root, for CREATE/DROP DATABASE."""
    return (
        f"mysql+pymysql://root:{container.root_password}"
        f"@{container.get_container_host_ip()}:{container.get_exposed_port(3306)}"
    )


@pytest.fixture
def mariadb_url(mariadb_container) -> str:
    """URL of a freshly-created database, dropped again on teardown."""
    db_name = f"t_{uuid.uuid4().hex[:12]}"
    admin_url = _mariadb_admin_url(mariadb_container)

    admin_engine = create_engine(admin_url)
    with admin_engine.begin() as conn:
        conn.execute(text(f"CREATE DATABASE `{db_name}`"))
    admin_engine.dispose()

    try:
        yield f"{admin_url}/{db_name}"
    finally:
        admin_engine = create_engine(admin_url)
        with admin_engine.begin() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
        admin_engine.dispose()


@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL container for the session.

    Skipped unless ``THISTLE_DB_TEST_POSTGRES=1`` is set in the environment.
    """
    if not os.environ.get("THISTLE_DB_TEST_POSTGRES"):
        pytest.skip("THISTLE_DB_TEST_POSTGRES=1 not set")

    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError as e:
        pytest.skip(f"testcontainers[postgres] not installed: {e}")

    container = PostgresContainer(_POSTGRES_IMAGE, driver="psycopg")
    container.start()
    try:
        yield container
    finally:
        container.stop()


def _postgres_admin_url(container) -> str:
    """SQLAlchemy URL connecting as superuser to the maintenance database.

    CREATE/DROP DATABASE cannot run inside a transaction on the database
    being dropped, so this targets the default ``postgres`` database.
    """
    return (
        f"postgresql+psycopg://{container.username}:{container.password}"
        f"@{container.get_container_host_ip()}:{container.get_exposed_port(5432)}"
        f"/postgres"
    )


@pytest.fixture
def postgres_url(postgres_container) -> str:
    """URL of a freshly-created database, dropped again on teardown."""
    db_name = f"t_{uuid.uuid4().hex[:12]}"
    admin_url = _postgres_admin_url(postgres_container)

    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    admin_engine.dispose()

    base = admin_url.rsplit("/", 1)[0]
    try:
        yield f"{base}/{db_name}"
    finally:
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        admin_engine.dispose()


# ---------------------------------------------------------------------------
# Parametrized session — every decorated test runs once per backend
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function", params=["sqlite", "mariadb", "postgres"])
def db_session(request):
    if request.param == "sqlite":
        url = "sqlite:///:memory:"
    elif request.param == "mariadb":
        url = request.getfixturevalue("mariadb_url")
    else:
        url = request.getfixturevalue("postgres_url")

    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()
