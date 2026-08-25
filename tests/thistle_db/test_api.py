import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from thistle_db.api import (
    DatabaseNotInitializedError,
    get_tles,
    init_db,
    open_session,
    tles_for_object,
)
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


def test_open_session_requires_init(tmp_path):
    """No DDL, no side effects: a missing SQLite file is an error, not a
    freshly created empty database."""
    path = tmp_path / "fresh.db"
    with pytest.raises(DatabaseNotInitializedError, match="init-db"):
        open_session(_settings(path))
    assert not path.exists()


def test_open_session_rejects_schemaless_db(tmp_path):
    """An existing database file without the schema is also an error."""
    import sqlite3

    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()  # file exists, no tables
    with pytest.raises(DatabaseNotInitializedError, match="init-db"):
        open_session(_settings(path))


def test_open_session_after_init(tmp_path):
    settings = _settings(tmp_path / "fresh.db")
    init_db(settings)
    session, engine = open_session(settings)
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


MODEL_TABLES = {"tle", "omm_metadata", "ingest_files", "epoch_date_state"}


def _settings_from_url(url_str: str) -> Settings:
    url = make_url(url_str)
    return Settings(
        database=Database(
            drivername=url.drivername,
            username=url.username,
            password=url.password,
            host=url.host,
            port=url.port,
            name=url.database,
        )
    )


class TestReadOnlySession:
    @pytest.fixture(params=["sqlite", "mariadb", "postgres"])
    def ro_settings(self, request, tmp_path):
        """An initialized database seeded with one TLE, per backend."""
        if request.param == "sqlite":
            settings = _settings(tmp_path / "ro.db")
        else:
            settings = _settings_from_url(
                request.getfixturevalue(f"{request.param}_url")
            )
        init_db(settings)
        session, engine = open_session(settings)
        try:
            session.add(TLE.from_twoline(*TLES[0]))
            session.commit()
        finally:
            session.close()
            engine.dispose()
        return settings

    def test_reads_work(self, ro_settings):
        session, engine = open_session(ro_settings, readonly=True)
        try:
            assert len(tles_for_object(session, 22)) == 1
        finally:
            session.close()
            engine.dispose()

    def test_writes_rejected(self, ro_settings):
        session, engine = open_session(ro_settings, readonly=True)
        try:
            session.add(TLE.from_twoline(*TLES[1]))
            with pytest.raises(DBAPIError):
                session.commit()
            session.rollback()
        finally:
            session.close()
            engine.dispose()
        # Nothing was written despite the attempt.
        assert get_tles(22, ro_settings) == [TLES[0]]


def test_init_db_creates_all_tables(tmp_path):
    settings = _settings(tmp_path / "fresh.db")
    result = init_db(settings)
    assert set(result["created"]) == MODEL_TABLES
    assert result["existing"] == []
    assert result["dropped"] == []
    engine = create_engine(settings.database.url)
    assert MODEL_TABLES <= set(inspect(engine).get_table_names())
    engine.dispose()


def test_init_db_is_idempotent(tmp_path):
    settings = _settings(tmp_path / "fresh.db")
    init_db(settings)
    result = init_db(settings)
    assert result["created"] == []
    assert set(result["existing"]) == MODEL_TABLES


def test_init_db_drop_recreates_and_wipes(db_path):
    settings = _settings(db_path)
    assert get_tles(22, settings) != []
    result = init_db(settings, drop=True)
    assert set(result["dropped"]) == MODEL_TABLES
    assert set(result["created"]) == MODEL_TABLES
    assert get_tles(22, settings) == []


DEFERRABLE = {"ix_tle_norad_cat_id_epoch", "ix_tle_created"}


def _tle_index_names(settings) -> set[str]:
    engine = create_engine(settings.database.url)
    try:
        return {ix["name"] for ix in inspect(engine).get_indexes("tle")}
    finally:
        engine.dispose()


def test_init_db_defer_indexes(tmp_path):
    """A bulk-restore schema: tables present, read-path indexes absent."""
    settings = _settings(tmp_path / "fresh.db")
    result = init_db(settings, drop=True, defer_indexes=True)
    assert set(result["indexes_deferred"]) == {
        f"tle.{name}" for name in DEFERRABLE
    }
    assert not (DEFERRABLE & _tle_index_names(settings))


def test_deferred_schema_still_deduplicates(tmp_path):
    """Deferring must never touch the dedup machinery: ingesting the same
    records twice still stores them once."""
    from thistle_db.ingest import ingest_tles

    settings = _settings(tmp_path / "fresh.db")
    init_db(settings, drop=True, defer_indexes=True)
    session, engine = open_session(settings)
    try:
        assert ingest_tles(session, TLES) == len(TLES)
        assert ingest_tles(session, TLES) == 0
    finally:
        session.close()
        engine.dispose()


def test_plain_init_db_builds_deferred_indexes(tmp_path):
    """The finalize step: a later flagless init-db reconciles the indexes."""
    settings = _settings(tmp_path / "fresh.db")
    init_db(settings, drop=True, defer_indexes=True)

    result = init_db(settings)
    assert set(result["indexes_created"]) == {
        f"tle.{name}" for name in DEFERRABLE
    }
    assert DEFERRABLE <= _tle_index_names(settings)

    # And it stays idempotent: nothing more to build.
    assert init_db(settings)["indexes_created"] == []


def test_defer_indexes_requires_drop(tmp_path):
    settings = _settings(tmp_path / "fresh.db")
    with pytest.raises(ValueError, match="drop"):
        init_db(settings, defer_indexes=True)


def test_deferred_restore_round_trip(tmp_path):
    """The full documented flow: defer -> ingest -> finalize -> query."""
    from thistle_db.ingest import ingest_tles

    settings = _settings(tmp_path / "fresh.db")
    init_db(settings, drop=True, defer_indexes=True)
    session, engine = open_session(settings)
    try:
        ingest_tles(session, TLES)
    finally:
        session.close()
        engine.dispose()

    init_db(settings)  # finalize

    assert get_tles(22, settings) != []
    assert DEFERRABLE <= _tle_index_names(settings)
