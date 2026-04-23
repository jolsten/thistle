"""Tests for the thistle CLI."""

from __future__ import annotations

import json
import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip("thistle CLI requires Python 3.10+", allow_module_level=True)

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from thistle.cli._app import app  # noqa: E402


ISS_TLE = """\
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005
2 25544  51.6400 208.9163 0006703  30.5502 329.5947 15.49560532  1001
1 25544U 98067A   24001.75000000  .00016717  00000-0  10270-3 0  9005
2 25544  51.6400 208.9163 0006703  30.5502 329.5947 15.49560532  1002
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tle_file(tmp_path):
    path = tmp_path / "iss.tle"
    path.write_text(ISS_TLE)
    return path


# ---- top-level help -------------------------------------------------------


def test_help(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["inspect", "find-tle", "summary", "catalog", "filter", "propagate"]:
        assert cmd in result.stdout


# ---- inspect --------------------------------------------------------------


def test_inspect_columns(runner):
    result = runner.invoke(app, ["inspect", "--columns"])
    assert result.exit_code == 0
    assert "norad" in result.stdout
    assert "sma" in result.stdout


def test_inspect_stdin(runner):
    result = runner.invoke(app, ["inspect", "--header"], input=ISS_TLE)
    assert result.exit_code == 0
    assert "25544" in result.stdout
    assert "98067A" in result.stdout
    assert "norad" in result.stdout  # header row


def test_inspect_missing_file(runner):
    result = runner.invoke(app, ["inspect", "/nonexistent/file.tle"])
    assert result.exit_code == 2


# ---- summary --------------------------------------------------------------


def test_summary_text(runner, tle_file):
    result = runner.invoke(app, ["summary", str(tle_file)])
    assert result.exit_code == 0
    assert "25544" in result.stdout
    assert "Object:" in result.stdout
    assert "TLE count:" in result.stdout


def test_summary_json(runner, tle_file):
    result = runner.invoke(app, ["summary", str(tle_file), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["satnum"] == "25544"
    assert data["intl"] == "98067A"
    assert data["count"] == 2


def test_summary_missing_file(runner):
    result = runner.invoke(app, ["summary", "/nonexistent/file.tle"])
    assert result.exit_code == 2


# ---- catalog --------------------------------------------------------------


def test_catalog(runner, tmp_path, tle_file):
    result = runner.invoke(app, ["catalog", str(tmp_path)])
    assert result.exit_code == 0
    assert "Directory:" in result.stdout
    assert "TLE count:  2" in result.stdout


def test_catalog_json(runner, tmp_path, tle_file):
    result = runner.invoke(app, ["catalog", str(tmp_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["tle_count"] == 2
    assert data["objects"] == 1


def test_catalog_not_a_dir(runner, tle_file):
    result = runner.invoke(app, ["catalog", str(tle_file)])
    assert result.exit_code == 2


# ---- filter ---------------------------------------------------------------


def test_filter_satnum_match(runner, tle_file):
    result = runner.invoke(app, ["filter", str(tle_file), "--satnum", "25544"])
    assert result.exit_code == 0
    # Two TLEs = 4 lines of output
    assert result.stdout.count("\n") == 4


def test_filter_satnum_excludes(runner):
    result = runner.invoke(app, ["filter", "--satnum", "99999"], input=ISS_TLE)
    assert result.exit_code == 0
    assert result.stdout == ""


def test_filter_orbital_elements(runner, tle_file):
    # ISS inclination is ~51.64 deg; filter should match
    result = runner.invoke(
        app,
        ["filter", str(tle_file), "--min-inc", "51", "--max-inc", "52"],
    )
    assert result.exit_code == 0
    assert "25544" in result.stdout


def test_filter_orbital_elements_excludes(runner, tle_file):
    result = runner.invoke(
        app,
        ["filter", str(tle_file), "--min-inc", "60"],
    )
    assert result.exit_code == 0
    assert result.stdout == ""


# ---- find-tle -------------------------------------------------------------


def test_find_tle(runner, tle_file):
    result = runner.invoke(
        app, ["find-tle", str(tle_file)], input="2024-01-01T12:00:00\n"
    )
    assert result.exit_code == 0
    assert "25544" in result.stdout


def test_find_tle_unique(runner, tle_file):
    # Two different timestamps should yield the same TLE once with --unique
    inp = "2024-01-01T12:00:00\n2024-01-01T13:00:00\n"
    result = runner.invoke(app, ["find-tle", str(tle_file), "--unique"], input=inp)
    assert result.exit_code == 0
    # One unique TLE = 2 lines
    assert result.stdout.count("\n") == 2


# ---- propagate ------------------------------------------------------------


def test_propagate_eci(runner, tle_file):
    result = runner.invoke(
        app, ["propagate", str(tle_file), "--eci"], input="2024-01-01T12:00:00\n"
    )
    assert result.exit_code == 0
    assert "2024-01-01" in result.stdout


def test_propagate_requires_group(runner, tle_file):
    result = runner.invoke(
        app, ["propagate", str(tle_file)], input="2024-01-01T12:00:00\n"
    )
    assert result.exit_code == 2


# ---- graceful failure when typer is unavailable --------------------------


def test_main_without_typer(monkeypatch, capsys):
    """When typer isn't installed, `thistle` exits 1 with an install hint."""
    # Remove cached modules so that re-import re-evaluates the top of _app.py
    for name in list(sys.modules):
        if (
            name == "typer"
            or name.startswith("typer.")
            or name == "thistle.cli._app"
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    # Block typer from being imported; sys.modules[name]=None raises ImportError
    monkeypatch.setitem(sys.modules, "typer", None)

    from thistle.cli import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "pip install 'thistle[cli]'" in err
