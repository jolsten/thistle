"""Cross-package tests: the thistle CLI's thistle-db database fallback."""

from __future__ import annotations

import json
import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip("thistle CLI requires Python 3.10+", allow_module_level=True)

pytest.importorskip("typer")
pytest.importorskip(
    "thistle_db", reason="thistle-db not installed (requires Python >= 3.11)"
)

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from thistle.cli import _db  # noqa: E402
from thistle.cli._app import app  # noqa: E402
from thistle_db.model import TLE, Base  # noqa: E402

ISS_TLE = """\
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005
2 25544  51.6400 208.9163 0006703  30.5502 329.5947 15.49560532  1001
1 25544U 98067A   24001.75000000  .00016717  00000-0  10270-3 0  9005
2 25544  51.6400 208.9163 0006703  30.5502 329.5947 15.49560532  1002
"""

VANGUARD_TLE = """\
1 00011U 59001A   25031.53522517  .00001638  00000-0  87319-3 0  9997
2 00011  32.8626 182.3246 1451254  84.1345 292.2261 11.89131272475694
"""

runner = CliRunner()


def _tle_pairs(text: str) -> list[tuple[str, str]]:
    lines = text.strip().splitlines()
    return list(zip(lines[::2], lines[1::2]))


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    """A populated on-disk SQLite thistle-db, configured via THISTLE_DB_* env."""
    monkeypatch.delenv("THISTLE_TLE_DIR", raising=False)
    monkeypatch.delenv("THISTLE_TLE_EXT", raising=False)
    monkeypatch.delenv(_db.DB_CONFIG_ENV, raising=False)

    db_path = tmp_path / "fallback.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    for line1, line2 in _tle_pairs(ISS_TLE):
        session.add(TLE.from_twoline(line1, line2))
    session.commit()
    session.close()
    engine.dispose()

    monkeypatch.setenv("THISTLE_DB_DATABASE__DRIVERNAME", "sqlite")
    monkeypatch.setenv("THISTLE_DB_DATABASE__NAME", db_path.as_posix())
    return db_path


def test_id_resolves_from_db(db_env):
    result = runner.invoke(app, ["summary", "25544"])
    assert result.exit_code == 0
    assert "25544" in result.stdout


def test_db_miss_names_database(db_env):
    result = runner.invoke(app, ["summary", "99999"])
    assert result.exit_code == 2
    assert "thistle-db database" in result.stderr


def test_tle_dir_wins_over_db(db_env, tmp_path, monkeypatch):
    tle_dir = tmp_path / "tledir"
    tle_dir.mkdir()
    # A file named for the ISS ID but holding Vanguard 1 data: if the file
    # wins, the output shows Vanguard's international designator.
    (tle_dir / "25544.tle").write_text(VANGUARD_TLE)
    monkeypatch.setenv("THISTLE_TLE_DIR", str(tle_dir))
    result = runner.invoke(app, ["summary", "25544"])
    assert result.exit_code == 0
    assert "59001A" in result.stdout


def test_uninitialized_db_leaves_no_trace(tmp_path, monkeypatch):
    """A typo'd DB path warns and falls through — and creates no stray file."""
    monkeypatch.delenv("THISTLE_TLE_DIR", raising=False)
    monkeypatch.delenv(_db.DB_CONFIG_ENV, raising=False)
    ghost = tmp_path / "ghost.db"
    monkeypatch.setenv("THISTLE_DB_DATABASE__DRIVERNAME", "sqlite")
    monkeypatch.setenv("THISTLE_DB_DATABASE__NAME", ghost.as_posix())
    result = runner.invoke(app, ["summary", "25544"])
    assert result.exit_code == 2
    assert "thistle-db lookup failed" in result.stderr
    assert not ghost.exists()


def test_not_configured_is_not_attempted(monkeypatch):
    monkeypatch.delenv("THISTLE_TLE_DIR", raising=False)
    monkeypatch.delenv(_db.DB_CONFIG_ENV, raising=False)
    for key in ("THISTLE_DB_DATABASE__DRIVERNAME", "THISTLE_DB_DATABASE__NAME"):
        monkeypatch.delenv(key, raising=False)
    assert _db.lookup_tles("25544") is None
    # CLI keeps the pass-through behavior: plain file-not-found, no DB mention.
    result = runner.invoke(app, ["summary", "25544"])
    assert result.exit_code == 2
    assert "thistle-db database" not in result.stderr


def test_zero_padded_id_resolves_from_db(db_env):
    # The DB lookup normalizes the typed ID to an integer satnum.
    result = runner.invoke(app, ["summary", "0025544"])
    assert result.exit_code == 0
    assert "25544" in result.stdout


def test_load_tle_resolves_from_db(db_env):
    """The library-level load_tle uses the same fallback, without temp files."""
    from thistle import load_tle

    assert load_tle("25544") == _tle_pairs(ISS_TLE)


def test_load_tle_db_miss_raises(db_env):
    from thistle import load_tle

    with pytest.raises(FileNotFoundError, match="thistle-db database"):
        load_tle("99999")


def test_config_reports_active_db(db_env):
    result = runner.invoke(app, ["config", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["db_installed"] is True
    assert data["db_configured"] is True
    assert data["db_url"].startswith("sqlite:///")

    result = runner.invoke(app, ["config"])
    assert "db_fallback: active" in result.stdout
