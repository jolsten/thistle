import datetime

import pytest
import typer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from thistle_db.cli import _parse_target, app, nearest_tles_for_date, tles_for_object
from thistle_db.model import TLE, Base

from .conftest import TLES

runner = CliRunner()


def _insert_tles(session):
    for line1, line2 in TLES:
        session.add(TLE.from_twoline(line1, line2))
    session.commit()


class TestParseTarget:
    def test_digits_are_satnum(self):
        assert _parse_target("25544") == 25544

    def test_zero_padded_satnum(self):
        assert _parse_target("00022") == 22

    def test_alpha5_satnum(self):
        assert _parse_target("E5693") == 145693

    def test_alpha5_satnum_lowercase(self):
        assert _parse_target("e5693") == 145693

    def test_compact_date(self):
        assert _parse_target("20250418") == datetime.date(2025, 4, 18)

    def test_invalid_date_raises(self):
        with pytest.raises(typer.BadParameter):
            _parse_target("20251345")

    def test_invalid_raises(self):
        with pytest.raises(typer.BadParameter):
            _parse_target("not-a-date")


class TestQueries:
    def test_tles_for_object_ordered_by_epoch(self, db_session):
        _insert_tles(db_session)
        tles = tles_for_object(db_session, 22)
        assert [(t.line1, t.line2) for t in tles] == [TLES[0], TLES[1]]

    def test_tles_for_object_unknown_satnum(self, db_session):
        _insert_tles(db_session)
        assert tles_for_object(db_session, 99999) == []

    def test_nearest_tles_for_date(self, db_session):
        _insert_tles(db_session)
        # Sat 22 has epochs on Apr 17 ~21:40 and Apr 18 ~05:34; the latter is
        # nearer to Apr 18 12:00. Sat 81069 has one epoch on Apr 18 ~14:55.
        tles = nearest_tles_for_date(db_session, datetime.date(2025, 4, 18), 7.0)
        assert [(t.line1, t.line2) for t in tles] == [TLES[1], TLES[2]]

    def test_nearest_tles_respects_window(self, db_session):
        _insert_tles(db_session)
        # All epochs are ~12 days before Apr 30: outside +/-7, inside +/-15.
        assert nearest_tles_for_date(db_session, datetime.date(2025, 4, 30), 7.0) == []
        tles = nearest_tles_for_date(db_session, datetime.date(2025, 4, 30), 15.0)
        assert len(tles) == 2


@pytest.fixture()
def cli_env(tmp_path):
    """A config.toml backed by a populated on-disk SQLite database."""
    db_path = tmp_path / "thistle-db.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    _insert_tles(session)
    session.close()
    engine.dispose()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[database]\ndrivername = "sqlite"\nname = "{db_path.as_posix()}"\n'
    )
    return config_path


class TestGetTleCommand:
    def test_satnum_prints_all_tles(self, cli_env):
        result = runner.invoke(app, ["-c", str(cli_env), "get-tle", "00022"])
        assert result.exit_code == 0
        expected = "".join(f"{l1}\n{l2}\n" for l1, l2 in TLES[:2])
        assert result.stdout == expected

    def test_date_prints_nearest_per_object(self, cli_env):
        result = runner.invoke(app, ["-c", str(cli_env), "get-tle", "20250418"])
        assert result.exit_code == 0
        expected = "".join(f"{l1}\n{l2}\n" for l1, l2 in (TLES[1], TLES[2]))
        assert result.stdout == expected

    def test_date_window_option(self, cli_env):
        result = runner.invoke(
            app, ["-c", str(cli_env), "get-tle", "20250430", "--days", "15"]
        )
        assert result.exit_code == 0
        assert len(result.stdout.splitlines()) == 4

    def test_no_results_exits_nonzero(self, cli_env):
        result = runner.invoke(app, ["-c", str(cli_env), "get-tle", "20250430"])
        assert result.exit_code == 1
        assert result.stdout == ""

    def test_invalid_target_errors(self, cli_env):
        result = runner.invoke(app, ["-c", str(cli_env), "get-tle", "not-a-date"])
        assert result.exit_code != 0
