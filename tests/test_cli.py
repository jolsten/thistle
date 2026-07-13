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
    for cmd in [
        "inspect",
        "find-tle",
        "summary",
        "catalog",
        "filter",
        "propagate",
        "revnum",
    ]:
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


# ---- revnum ---------------------------------------------------------------

# Vanguard 1, epoch 2025-01-31, revnum 47569 (from tests/data/leo.tle)
VANGUARD_TLE = """\
1 00011U 59001A   25031.53522517  .00001638  00000-0  87319-3 0  9997
2 00011  32.8626 182.3246 1451254  84.1345 292.2261 11.89131272475694
"""

# Epoch of the first ISS_TLE entry (24001.50000000) and its revnum field
ISS_EPOCH = "2024-01-01T12:00:00"
ISS_REVNUM = 100


@pytest.fixture
def multi_object_file(tmp_path):
    path = tmp_path / "multi.tle"
    path.write_text(ISS_TLE + VANGUARD_TLE)
    return path


def test_revnum_at_own_epoch(runner, tle_file):
    result = runner.invoke(app, ["revnum", str(tle_file), ISS_EPOCH])
    assert result.exit_code == 0
    value = float(result.stdout.strip())
    assert int(value) == ISS_REVNUM
    # fraction = mean argument of latitude / 2pi
    expected_frac = ((30.5502 + 329.5947) % 360.0) / 360.0
    assert value - ISS_REVNUM == pytest.approx(expected_frac, abs=1e-3)


def test_revnum_roundtrip(runner, tle_file):
    # Rev 101 starts near the first TLE's epoch, so the same TLE is selected
    # in both directions (the fixture's two TLEs share the same revnum).
    result = runner.invoke(app, ["revnum", str(tle_file), "101"])
    assert result.exit_code == 0
    start, stop = result.stdout.split()
    back = runner.invoke(app, ["revnum", str(tle_file), start])
    assert back.exit_code == 0
    assert float(back.stdout.strip()) == pytest.approx(101.0, abs=1e-3)


def test_revnum_rev_duration(runner, tle_file):
    from datetime import datetime

    result = runner.invoke(app, ["revnum", str(tle_file), "105"])
    assert result.exit_code == 0
    start, stop = (datetime.fromisoformat(s) for s in result.stdout.split())
    minutes = (stop - start).total_seconds() / 60.0
    # ISS nodal period ~ 1440 / 15.4956 rev/day, within a minute
    assert minutes == pytest.approx(1440.0 / 15.49560532, abs=1.0)


def test_revnum_consecutive_revs_abut(runner, tle_file):
    r105 = runner.invoke(app, ["revnum", str(tle_file), "105"])
    r106 = runner.invoke(app, ["revnum", str(tle_file), "106"])
    assert r105.stdout.split()[1] == r106.stdout.split()[0]


def test_revnum_ndot_units():
    """sgp4's ndot (rad/min^2) converts back to the raw line-1 field."""
    from sgp4.api import Satrec

    from thistle.cli._helpers import ndot_rev_per_day2

    line1, line2 = ISS_TLE.splitlines()[:2]
    sat = Satrec.twoline2rv(line1, line2)
    assert ndot_rev_per_day2(sat) == pytest.approx(0.00016717, rel=1e-6)


def test_revnum_bad_value(runner, tle_file):
    result = runner.invoke(app, ["revnum", str(tle_file), "not-a-value"])
    assert result.exit_code == 2


def test_revnum_two_of_same_kind(runner, tle_file):
    assert runner.invoke(app, ["revnum", str(tle_file), "100", "200"]).exit_code == 2
    assert (
        runner.invoke(
            app, ["revnum", str(tle_file), ISS_EPOCH, "2024-01-02T00:00:00"]
        ).exit_code
        == 2
    )


def test_revnum_missing_file(runner):
    result = runner.invoke(app, ["revnum", "/nonexistent/file.tle", "100"])
    assert result.exit_code == 2


def test_revnum_empty_file(runner, tmp_path):
    path = tmp_path / "empty.tle"
    path.write_text("no tles here\n")
    result = runner.invoke(app, ["revnum", str(path), "100"])
    assert result.exit_code == 2


def test_revnum_refine(runner, tle_file):
    from datetime import datetime

    plain = runner.invoke(app, ["revnum", str(tle_file), "105"])
    refined = runner.invoke(app, ["revnum", str(tle_file), "105", "--refine"])
    assert refined.exit_code == 0
    est_start = datetime.fromisoformat(plain.stdout.split()[0])
    ref_start = datetime.fromisoformat(refined.stdout.split()[0])
    # SGP4-exact crossing should be within a fraction of a period of the estimate
    assert abs((ref_start - est_start).total_seconds()) < 15 * 60


def test_revnum_match_mode(runner, multi_object_file):
    result = runner.invoke(
        app,
        ["revnum", str(multi_object_file), str(ISS_REVNUM), ISS_EPOCH, "--tol", "0.5"],
    )
    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].split()[0] == "25544"


def test_revnum_match_mode_huge_tol(runner, multi_object_file):
    result = runner.invoke(
        app,
        ["revnum", str(multi_object_file), str(ISS_REVNUM), ISS_EPOCH, "--tol", "1e6"],
    )
    assert result.exit_code == 0
    satnums = {line.split()[0] for line in result.stdout.strip().splitlines()}
    assert satnums == {"25544", "11"}


def test_revnum_match_mode_no_match(runner, multi_object_file):
    result = runner.invoke(
        app,
        ["revnum", str(multi_object_file), "99999", ISS_EPOCH, "--tol", "0.5"],
    )
    assert result.exit_code == 0
    assert result.stdout == ""


def test_revnum_match_mode_arg_order(runner, multi_object_file):
    a = runner.invoke(
        app, ["revnum", str(multi_object_file), str(ISS_REVNUM), ISS_EPOCH]
    )
    b = runner.invoke(
        app, ["revnum", str(multi_object_file), ISS_EPOCH, str(ISS_REVNUM)]
    )
    assert a.exit_code == b.exit_code == 0
    assert a.stdout == b.stdout


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
