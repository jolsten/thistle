"""init-db lifecycle, session guards, and the query API."""

import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from .conftest import TLES, engine_ingest, make_settings
from thistle_db.api import (
    DatabaseNotInitializedError,
    get_tles,
    init_db,
    nearest_tles_for_date,
    open_session,
    tles_for_object,
)
from thistle_db.config import Database, Settings
from thistle_db.model import Base


@pytest.fixture
def file_settings(tmp_path) -> Settings:
    return Settings(
        database=Database(drivername="sqlite", name=str(tmp_path / "api.db"))
    )


def _table_names(settings) -> set[str]:
    engine = create_engine(settings.database.url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


class TestInitDb:
    def test_creates_all_tables(self, file_settings):
        result = init_db(file_settings)
        assert set(result["created"]) == set(Base.metadata.tables)
        assert _table_names(file_settings) >= set(Base.metadata.tables)

    def test_idempotent(self, file_settings):
        init_db(file_settings)
        result = init_db(file_settings)
        assert result["created"] == []
        assert set(result["existing"]) == set(Base.metadata.tables)

    def test_defer_requires_drop(self, file_settings):
        with pytest.raises(ValueError):
            init_db(file_settings, defer_indexes=True)

    def test_defer_and_finalize(self, file_settings):
        result = init_db(file_settings, drop=True, defer_indexes=True)
        assert result["indexes_deferred"] == ["tle.ix_tle_norad_cat_id_epoch"]
        finalize = init_db(file_settings)
        assert finalize["indexes_created"] == ["tle.ix_tle_norad_cat_id_epoch"]
        # And now nothing is missing.
        assert init_db(file_settings)["indexes_created"] == []


class TestOpenSession:
    def test_missing_sqlite_file_fails_loudly(self, tmp_path):
        settings = Settings(
            database=Database(drivername="sqlite", name=str(tmp_path / "no.db"))
        )
        with pytest.raises(DatabaseNotInitializedError):
            open_session(settings)
        assert not (tmp_path / "no.db").exists()  # not even created

    def test_missing_tables_fail_loudly(self, tmp_path):
        db_file = tmp_path / "empty.db"
        engine = create_engine(f"sqlite:///{db_file}")
        with engine.connect():
            pass  # create an empty file with no schema
        engine.dispose()
        settings = Settings(
            database=Database(drivername="sqlite", name=str(db_file))
        )
        with pytest.raises(DatabaseNotInitializedError):
            open_session(settings)

    def test_readonly_rejects_writes(self, file_settings):
        init_db(file_settings)
        session, engine = open_session(file_settings, readonly=True)
        try:
            from sqlalchemy import text

            with pytest.raises(Exception):
                session.execute(text("INSERT INTO ingest_runs (command, started, status) VALUES ('x', '2026-01-01', 'running')"))
                session.commit()
        finally:
            session.rollback()
            session.close()
            engine.dispose()


class TestQueries:
    @pytest.fixture
    def populated(self, db_session: Session, tmp_path):
        settings = make_settings(tmp_path)
        path = tmp_path / "in.tle"
        path.write_text("".join(f"{a}\n{b}\n" for a, b in TLES))
        engine_ingest(db_session, settings, files=[path])
        return db_session

    def test_tles_for_object(self, populated):
        rows = tles_for_object(populated, 22)
        assert [r.line1 for r in rows] == [TLES[0][0], TLES[1][0]]
        assert rows[0].epoch < rows[1].epoch

    def test_nearest_tles_for_date(self, populated):
        rows = nearest_tles_for_date(populated, datetime.date(2025, 4, 18), 7)
        by_norad = {r.norad_cat_id: r for r in rows}
        assert set(by_norad) == {22, 81069}
        # Nearest to 2025-04-18T12:00 for object 22 is the 25108.23 elset.
        assert by_norad[22].line1 == TLES[1][0]

    def test_nearest_window_bounds(self, populated):
        rows = nearest_tles_for_date(populated, datetime.date(2026, 1, 1), 1)
        assert rows == []


def test_get_tles_reads_configured_database(tmp_path):
    settings = Settings(
        database=Database(drivername="sqlite", name=str(tmp_path / "g.db"))
    )
    init_db(settings)
    session, engine = open_session(settings)
    try:
        ingest_settings = make_settings(tmp_path)
        path = tmp_path / "in.tle"
        path.write_text("".join(f"{a}\n{b}\n" for a, b in TLES))
        engine_ingest(session, ingest_settings, files=[path])
    finally:
        session.close()
        engine.dispose()

    pairs = get_tles(22, settings)
    assert pairs == [TLES[0], TLES[1]]
    assert get_tles(99999, settings) == []
