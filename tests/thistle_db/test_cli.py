import datetime
import pathlib

import pytest
import typer
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from thistle_db.cli import _parse_target, app, nearest_tles_for_date, tles_for_object
from thistle_db.model import TLE, Base, OmmMetadata

from .conftest import TLES

DATA = pathlib.Path(__file__).parent / "data"

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


class TestInitDbCommand:
    def test_creates_schema(self, tmp_path):
        db_path = tmp_path / "new.db"
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[database]\ndrivername = "sqlite"\nname = "{db_path.as_posix()}"\n'
        )
        result = runner.invoke(app, ["-c", str(config_path), "init-db"])
        assert result.exit_code == 0
        assert "Created:" in result.stdout
        assert "tle" in result.stdout

    def test_rerun_reports_existing(self, cli_env):
        result = runner.invoke(app, ["-c", str(cli_env), "init-db"])
        assert result.exit_code == 0
        assert "Existing:" in result.stdout
        assert "Created:" not in result.stdout

    def test_drop_without_yes_refuses_non_interactive(self, cli_env):
        # CliRunner's stdin is not a TTY, so --drop must fail closed.
        result = runner.invoke(app, ["-c", str(cli_env), "init-db", "--drop"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        # And the data must be untouched.
        result = runner.invoke(app, ["-c", str(cli_env), "get-tle", "00022"])
        assert result.exit_code == 0

    def test_drop_with_yes_wipes_data(self, cli_env):
        result = runner.invoke(
            app, ["-c", str(cli_env), "init-db", "--drop", "--yes"]
        )
        assert result.exit_code == 0
        assert "Dropped:" in result.stdout
        result = runner.invoke(app, ["-c", str(cli_env), "get-tle", "00022"])
        assert result.exit_code == 1  # no TLEs left


class TestUninitializedDatabase:
    @pytest.fixture()
    def uninit_config(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[database]\ndrivername = "sqlite"\n'
            f'name = "{(tmp_path / "nope.db").as_posix()}"\n'
        )
        return config_path

    @pytest.mark.parametrize("command", [["get-tle", "25544"], ["ingest"], ["generate"]])
    def test_commands_fail_closed(self, uninit_config, tmp_path, command):
        result = runner.invoke(app, ["-c", str(uninit_config), *command])
        assert result.exit_code == 3
        assert "init-db" in result.stderr
        assert not (tmp_path / "nope.db").exists()

    def test_dump_fails_closed(self, uninit_config, tmp_path):
        result = runner.invoke(
            app, ["-c", str(uninit_config), "dump", str(tmp_path / "backup")]
        )
        assert result.exit_code == 3
        assert "init-db" in result.stderr
        assert not (tmp_path / "backup.tle").exists()


def _write_sqlite_config(tmp_path, name):
    db_path = tmp_path / f"{name}.db"
    config_path = tmp_path / f"{name}.toml"
    config_path.write_text(
        f'[database]\ndrivername = "sqlite"\nname = "{db_path.as_posix()}"\n'
    )
    return config_path, db_path


def _table_counts(db_path):
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        with engine.connect() as conn:
            tles = conn.execute(select(func.count()).select_from(TLE.__table__)).scalar()
            omms = conn.execute(
                select(func.count()).select_from(OmmMetadata.__table__)
            ).scalar()
        return tles, omms
    finally:
        engine.dispose()


class TestDumpCommand:
    @pytest.fixture()
    def populated(self, tmp_path):
        """A SQLite db loaded with plain TLEs and one OMM record."""
        config_path, db_path = _write_sqlite_config(tmp_path, "src")
        assert runner.invoke(app, ["-c", str(config_path), "init-db"]).exit_code == 0
        result = runner.invoke(
            app,
            [
                "-c",
                str(config_path),
                "ingest",
                str(DATA / "25544.txt"),
                str(DATA / "one.json"),
            ],
        )
        assert result.exit_code == 0
        return config_path, db_path

    def test_round_trip(self, populated, tmp_path):
        config_path, db_path = populated
        base = tmp_path / "backup"

        result = runner.invoke(app, ["-c", str(config_path), "dump", str(base)])
        assert result.exit_code == 0
        assert base.with_suffix(".tle").exists()
        assert base.with_suffix(".json").exists()

        # Restore into a fresh database and compare row counts.
        config2, db2 = _write_sqlite_config(tmp_path, "restored")
        assert runner.invoke(app, ["-c", str(config2), "init-db"]).exit_code == 0
        result = runner.invoke(
            app,
            [
                "-c",
                str(config2),
                "ingest",
                str(base.with_suffix(".tle")),
                str(base.with_suffix(".json")),
            ],
        )
        assert result.exit_code == 0
        assert _table_counts(db2) == _table_counts(db_path)

    def test_dump_is_idempotent_backup(self, populated, tmp_path):
        """Re-ingesting a dump into its own source database changes nothing."""
        config_path, db_path = populated
        before = _table_counts(db_path)
        base = tmp_path / "self"
        assert runner.invoke(app, ["-c", str(config_path), "dump", str(base)]).exit_code == 0
        result = runner.invoke(
            app, ["-c", str(config_path), "ingest", str(base.with_suffix(".tle"))]
        )
        assert result.exit_code == 0
        assert _table_counts(db_path) == before

    def test_refuses_overwrite_without_force(self, populated, tmp_path):
        config_path, _ = populated
        base = tmp_path / "backup"
        base.with_suffix(".tle").write_text("precious\n")
        result = runner.invoke(app, ["-c", str(config_path), "dump", str(base)])
        assert result.exit_code == 1
        assert "--force" in result.stderr
        assert base.with_suffix(".tle").read_text() == "precious\n"

        result = runner.invoke(
            app, ["-c", str(config_path), "dump", str(base), "--force"]
        )
        assert result.exit_code == 0

    def test_no_omm_metadata_skips_json(self, cli_env, tmp_path):
        # cli_env holds TLE rows only (no OMM metadata).
        base = tmp_path / "backup"
        result = runner.invoke(app, ["-c", str(cli_env), "dump", str(base)])
        assert result.exit_code == 0
        assert base.with_suffix(".tle").exists()
        assert not base.with_suffix(".json").exists()
        assert "not written" in result.stdout


class TestConfigEnvVar:
    def test_env_var_replaces_flag(self, cli_env, monkeypatch):
        monkeypatch.setenv("THISTLE_DB_CONFIG", str(cli_env))
        result = runner.invoke(app, ["get-tle", "00022"])
        assert result.exit_code == 0
        expected = "".join(f"{l1}\n{l2}\n" for l1, l2 in TLES[:2])
        assert result.stdout == expected

    def test_flag_wins_over_env_var(self, cli_env, tmp_path, monkeypatch):
        # Env var points at an empty database; the explicit -c must win.
        empty = tmp_path / "empty.toml"
        empty.write_text(
            f'[database]\ndrivername = "sqlite"\n'
            f'name = "{(tmp_path / "empty.db").as_posix()}"\n'
        )
        monkeypatch.setenv("THISTLE_DB_CONFIG", str(empty))
        result = runner.invoke(app, ["-c", str(cli_env), "get-tle", "00022"])
        assert result.exit_code == 0
        expected = "".join(f"{l1}\n{l2}\n" for l1, l2 in TLES[:2])
        assert result.stdout == expected
