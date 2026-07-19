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


@pytest.fixture(autouse=True)
def _clean_thistle_env(monkeypatch):
    """Keep the developer's real THISTLE_* environment out of the suite."""
    monkeypatch.delenv("THISTLE_TLE_DIR", raising=False)
    monkeypatch.delenv("THISTLE_TLE_EXT", raising=False)


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
        "filter",
        "propagate",
        "revnum",
        "plot",
        "maneuvers",
        "groundtrack",
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


# ---- plot -----------------------------------------------------------------


@pytest.fixture
def agg_backend(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")


def test_plot_save_png(runner, tle_file, tmp_path, agg_backend):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.png"
    result = runner.invoke(app, ["plot", str(tle_file), "-o", str(out)])
    assert result.exit_code == 0
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_plot_save_svg(runner, tle_file, tmp_path, agg_backend):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.svg"
    result = runner.invoke(app, ["plot", str(tle_file), "-o", str(out)])
    assert result.exit_code == 0
    assert b"<svg" in out.read_bytes()


def test_plot_stdin(runner, tmp_path, agg_backend):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.png"
    result = runner.invoke(app, ["plot", "-o", str(out)], input=ISS_TLE)
    assert result.exit_code == 0
    assert out.exists()


def test_plot_fields_subset(runner, tle_file, tmp_path, agg_backend):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.png"
    result = runner.invoke(
        app, ["plot", str(tle_file), "--fields", "sma,ecc", "-o", str(out)]
    )
    assert result.exit_code == 0


def test_plot_maneuvers_flag(runner, tle_file, tmp_path, agg_backend):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.png"
    result = runner.invoke(
        app, ["plot", str(tle_file), "--maneuvers", "-o", str(out)]
    )
    assert result.exit_code == 0


def test_plot_unknown_field(runner, tle_file):
    result = runner.invoke(app, ["plot", str(tle_file), "--fields", "sma,bogus"])
    assert result.exit_code == 2


def test_plot_multi_object(runner, multi_object_file, tmp_path):
    out = tmp_path / "out.png"
    result = runner.invoke(app, ["plot", str(multi_object_file), "-o", str(out)])
    assert result.exit_code == 2
    assert "filter --satnum" in result.stderr


def test_plot_missing_file(runner):
    result = runner.invoke(app, ["plot", "/nonexistent/file.tle", "-o", "x.png"])
    assert result.exit_code == 2


def test_plot_without_matplotlib(runner, tle_file, tmp_path, monkeypatch):
    """Missing matplotlib exits 1 with an install hint."""
    for name in list(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    result = runner.invoke(
        app, ["plot", str(tle_file), "-o", str(tmp_path / "x.png")]
    )
    assert result.exit_code == 1
    assert "thistle[plot]" in result.stderr


def test_plot_empty_file(runner, tmp_path):
    path = tmp_path / "empty.tle"
    path.write_text("no tles here\n")
    result = runner.invoke(app, ["plot", str(path), "-o", "x.png"])
    assert result.exit_code == 2


# ---- plot presets and derived fields --------------------------------------

# Two consecutive stationkept TLEs of a GEO object at ~145 deg E
# (from tests/data/obj/50001.txt)
GEO_TLE = """\
1 50001U 21123A   25059.54220164 -.00000245  00000-0  00000-0 0  9992
2 50001   0.0208 235.3877 0000111  15.3569 248.0225  1.00271703 11797
1 50001U 21123A   25059.83517034 -.00000246  00000-0  00000-0 0  9990
2 50001   0.0174 227.4075 0000104  26.3443 350.7728  1.00271584 11792
"""


@pytest.fixture
def geo_file(tmp_path):
    path = tmp_path / "geo.tle"
    path.write_text(GEO_TLE)
    return path


def test_plot_preset(runner, geo_file, tmp_path, agg_backend):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.png"
    result = runner.invoke(app, ["plot", str(geo_file), "--preset", "geo", "-o", str(out)])
    assert result.exit_code == 0
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_plot_preset_unknown(runner, tle_file):
    result = runner.invoke(app, ["plot", str(tle_file), "--preset", "bogus"])
    assert result.exit_code == 2


def test_plot_preset_fields_exclusive(runner, tle_file):
    result = runner.invoke(
        app, ["plot", str(tle_file), "--preset", "leo", "--fields", "sma"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


def test_plot_fields_lon_ltan(runner, geo_file, tmp_path, agg_backend):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.png"
    result = runner.invoke(
        app, ["plot", str(geo_file), "--fields", "lon,ltan", "-o", str(out)]
    )
    assert result.exit_code == 0


def test_lon_series_stationkept_geo():
    """Both TLEs of a stationkept GEO bird resolve to the same slot."""
    from sgp4.api import Satrec

    from thistle.cli._plot import _lon_series

    lines = GEO_TLE.splitlines()
    sats = [
        Satrec.twoline2rv(lines[0], lines[1]),
        Satrec.twoline2rv(lines[2], lines[3]),
    ]
    lon = _lon_series(sats)
    assert all(-180.0 <= v < 180.0 for v in lon)
    assert lon[0] == pytest.approx(145.0, abs=1.0)
    assert lon[0] == pytest.approx(lon[1], abs=0.5)


def test_ltan_series_at_equinox():
    """Near the equinox the sun's RA is ~0, so LTAN ~= RAAN/15 + 12."""
    from sgp4.api import Satrec

    from thistle.cli._plot import _ltan_series

    # ISS_TLE line 1 with the epoch moved to 2024 day 80.5 (Mar 20, equinox)
    line1 = "1 25544U 98067A   24080.50000000  .00016717  00000-0  10270-3 0  9005"
    line2 = ISS_TLE.splitlines()[1]
    sat = Satrec.twoline2rv(line1, line2)
    ltan = _ltan_series([sat])
    expected = (208.9163 / 15.0 + 12.0) % 24.0
    assert ltan[0] == pytest.approx(expected, abs=0.2)


# ---- maneuvers ------------------------------------------------------------

GEO_HISTORY = "tests/data/obj/50001.txt"


def test_maneuvers_real_data(runner):
    from datetime import datetime

    result = runner.invoke(app, ["maneuvers", GEO_HISTORY])
    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert len(lines) >= 1
    epochs = [datetime.fromisoformat(line) for line in lines]
    assert epochs == sorted(epochs)


def test_maneuvers_matches_api(runner):
    from thistle.cli._helpers import (
        epoch_to_datetime,
        maneuver_epochs,
        parse_tle_epochs,
    )

    with open(GEO_HISTORY) as f:
        parsed = parse_tle_epochs(f)
    sats = sorted(
        (s for _, _, s in parsed),
        key=lambda s: epoch_to_datetime(s.epochyr, s.epochdays),
    )
    expected = [e.isoformat() for e in maneuver_epochs(sats, 10.0)]

    result = runner.invoke(app, ["maneuvers", GEO_HISTORY])
    assert result.stdout.strip().splitlines() == expected


def test_maneuvers_huge_threshold(runner):
    result = runner.invoke(app, ["maneuvers", GEO_HISTORY, "--threshold", "1e9"])
    assert result.exit_code == 0
    assert result.stdout == ""


def test_maneuvers_short_series(runner, tle_file):
    result = runner.invoke(app, ["maneuvers", str(tle_file)])
    assert result.exit_code == 0
    assert result.stdout == ""


def test_maneuvers_stdin(runner):
    result = runner.invoke(app, ["maneuvers"], input=ISS_TLE)
    assert result.exit_code == 0


def test_maneuvers_multi_object(runner, multi_object_file):
    result = runner.invoke(app, ["maneuvers", str(multi_object_file)])
    assert result.exit_code == 2
    assert "filter --satnum" in result.stderr


def test_maneuvers_missing_file(runner):
    result = runner.invoke(app, ["maneuvers", "/nonexistent/file.tle"])
    assert result.exit_code == 2


def test_maneuvers_empty_file(runner, tmp_path):
    path = tmp_path / "empty.tle"
    path.write_text("no tles here\n")
    result = runner.invoke(app, ["maneuvers", str(path)])
    assert result.exit_code == 2


# ---- detect_maneuvers (no matplotlib required) ----------------------------


def _daily_times(n):
    from datetime import datetime, timedelta

    return [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def _noisy_series(n):
    import math

    # varied (non-degenerate) small noise so the MAD of deltas is nonzero
    return [1.0 + 0.001 * math.sin(1.7 * i) for i in range(n)]


def test_detect_maneuvers_finds_jump():
    from thistle.cli._helpers import detect_maneuvers

    times = _daily_times(20)
    values = _noisy_series(20)
    for i in range(10, 20):
        values[i] += 1.0
    events = detect_maneuvers(times, values, threshold=10.0)
    assert events == [times[10]]


def test_detect_maneuvers_pure_noise():
    from thistle.cli._helpers import detect_maneuvers

    times = _daily_times(20)
    assert detect_maneuvers(times, _noisy_series(20), threshold=10.0) == []


def test_detect_maneuvers_constant_series():
    from thistle.cli._helpers import detect_maneuvers

    times = _daily_times(20)
    assert detect_maneuvers(times, [1.0] * 20, threshold=10.0) == []


def test_detect_maneuvers_short_series():
    from thistle.cli._helpers import detect_maneuvers

    times = _daily_times(5)
    assert detect_maneuvers(times, [1.0, 1.0, 5.0, 5.0, 5.0], threshold=10.0) == []


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


# ---- groundtrack -----------------------------------------------------------

SPEC_LINE = "2024-01-01T12:00:00 2024-01-01T13:30:00"


def test_parse_spec_line_full():
    from thistle.cli._map import parse_spec_line

    spec = parse_spec_line(
        f"{SPEC_LINE} color=red style=-- width=2 alpha=0.5 marker=. "
        "label=ISS sat=25544"
    )
    assert spec is not None
    assert spec.sat == "25544"
    assert spec.plot_kwargs == {
        "color": "red",
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.5,
        "marker": ".",
        "label": "ISS",
    }


def test_parse_spec_line_unknown_key_survives(capsys):
    from thistle.cli._map import parse_spec_line

    spec = parse_spec_line(f"{SPEC_LINE} bogus=1 color=red")
    assert spec is not None
    assert spec.plot_kwargs == {"color": "red"}
    assert "unknown option" in capsys.readouterr().err


def test_parse_spec_line_rejects():
    from thistle.cli._map import parse_spec_line

    assert parse_spec_line("") is None
    assert parse_spec_line("# comment") is None
    assert parse_spec_line("not-a-time also-not") is None
    # stop before start
    assert (
        parse_spec_line("2024-01-02T00:00:00 2024-01-01T00:00:00") is None
    )


def test_groundtrack_stdin(runner, tle_file, tmp_path, agg_backend):
    pytest.importorskip("cartopy")
    out = tmp_path / "map.png"
    result = runner.invoke(
        app,
        ["groundtrack", str(tle_file), "-o", str(out)],
        input=f"{SPEC_LINE} color=red label=ISS\n",
    )
    assert result.exit_code == 0
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_groundtrack_per_line_sat(runner, tmp_path, monkeypatch, agg_backend):
    pytest.importorskip("cartopy")
    d = tmp_path / "tledir"
    d.mkdir()
    (d / "25544.tle").write_text(ISS_TLE)
    monkeypatch.setenv("THISTLE_TLE_DIR", str(d))
    out = tmp_path / "map.png"
    result = runner.invoke(
        app,
        ["groundtrack", "-o", str(out)],
        input=f"{SPEC_LINE} sat=25544\n",
    )
    assert result.exit_code == 0
    assert out.exists()


def test_groundtrack_site_ring(runner, tle_file, tmp_path, agg_backend):
    pytest.importorskip("cartopy")
    out = tmp_path / "map.png"
    result = runner.invoke(
        app,
        [
            "groundtrack", str(tle_file),
            "--site", "HAWAII:19.8:-155.5",
            "--min-el", "5",
            "-o", str(out),
        ],
        input=f"{SPEC_LINE}\n",
    )
    assert result.exit_code == 0
    assert out.exists()


def test_groundtrack_no_object(runner, tmp_path):
    result = runner.invoke(
        app,
        ["groundtrack", "-o", str(tmp_path / "map.png")],
        input=f"{SPEC_LINE}\n",
    )
    assert result.exit_code == 2
    assert "no valid traces" in result.stderr


def test_groundtrack_spec_file(runner, tle_file, tmp_path, agg_backend):
    pytest.importorskip("cartopy")
    spec = tmp_path / "spec.txt"
    spec.write_text(f"# demo\n{SPEC_LINE} color=green\n")
    out = tmp_path / "map.png"
    result = runner.invoke(
        app,
        ["groundtrack", str(tle_file), "--spec", str(spec), "-o", str(out)],
    )
    assert result.exit_code == 0


def test_groundtrack_missing_spec_file(runner, tle_file):
    result = runner.invoke(
        app, ["groundtrack", str(tle_file), "--spec", "/nonexistent/spec.txt"]
    )
    assert result.exit_code == 2


def test_groundtrack_bad_site(runner, tle_file):
    result = runner.invoke(
        app, ["groundtrack", str(tle_file), "--site", "badsite"],
        input=f"{SPEC_LINE}\n",
    )
    assert result.exit_code == 2


# ---- config / catalog-ID resolution ---------------------------------------


@pytest.fixture
def tle_dir(tmp_path, monkeypatch):
    """A THISTLE_TLE_DIR containing 25544.tle (ISS_TLE)."""
    d = tmp_path / "tledir"
    d.mkdir()
    (d / "25544.tle").write_text(ISS_TLE)
    monkeypatch.setenv("THISTLE_TLE_DIR", str(d))
    return d


def test_resolve_id_summary(runner, tle_dir):
    result = runner.invoke(app, ["summary", "25544"])
    assert result.exit_code == 0
    assert "25544" in result.stdout


def test_resolve_id_revnum(runner, tle_dir):
    result = runner.invoke(app, ["revnum", "25544", "101"])
    assert result.exit_code == 0
    assert len(result.stdout.split()) == 2


def test_resolve_id_custom_ext(runner, tmp_path, monkeypatch):
    d = tmp_path / "tledir"
    d.mkdir()
    (d / "25544.txt").write_text(ISS_TLE)
    monkeypatch.setenv("THISTLE_TLE_DIR", str(d))
    for ext in (".txt", "txt"):  # with and without leading dot
        monkeypatch.setenv("THISTLE_TLE_EXT", ext)
        result = runner.invoke(app, ["summary", "25544"])
        assert result.exit_code == 0


def test_resolve_alpha5(runner, tmp_path, monkeypatch):
    d = tmp_path / "tledir"
    d.mkdir()
    (d / "A0001.tle").write_text(ISS_TLE)
    monkeypatch.setenv("THISTLE_TLE_DIR", str(d))
    assert runner.invoke(app, ["summary", "A0001"]).exit_code == 0
    # lowercase input resolves via the uppercase retry
    assert runner.invoke(app, ["summary", "a0001"]).exit_code == 0


def test_resolve_alpha5_excluded_letters(runner, tmp_path, monkeypatch):
    d = tmp_path / "tledir"
    d.mkdir()
    (d / "I0001.tle").write_text(ISS_TLE)
    monkeypatch.setenv("THISTLE_TLE_DIR", str(d))
    # I and O are not valid Alpha-5 letters; not treated as a catalog ID
    result = runner.invoke(app, ["summary", "I0001"])
    assert result.exit_code == 2
    assert "also tried" not in result.stderr


def test_resolve_literal_path_wins(runner, tmp_path, tle_dir):
    # A real file named like an ID is used directly, not looked up in tle_dir
    literal = tmp_path / "25544"
    literal.write_text(VANGUARD_TLE)
    result = runner.invoke(app, ["summary", str(literal)])
    assert result.exit_code == 0
    assert "59001A" in result.stdout  # Vanguard, not the ISS TLE from tle_dir


def test_resolve_id_miss_names_both_paths(runner, tle_dir):
    result = runner.invoke(app, ["summary", "99999"])
    assert result.exit_code == 2
    assert "99999" in result.stderr
    assert "also tried" in result.stderr
    assert str(tle_dir) in result.stderr


def test_resolve_id_without_env_unchanged(runner):
    result = runner.invoke(app, ["summary", "25544"])
    assert result.exit_code == 2
    assert "also tried" not in result.stderr


def test_config_not_set(runner):
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "(not set)" in result.stdout
    assert "tle_ext: .tle" in result.stdout


def test_config_from_env(runner, tle_dir, monkeypatch):
    monkeypatch.setenv("THISTLE_TLE_EXT", "txt")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert str(tle_dir) in result.stdout
    assert "THISTLE_TLE_DIR" in result.stdout
    assert "tle_ext: .txt" in result.stdout


def test_config_json(runner, tle_dir):
    result = runner.invoke(app, ["config", "--json"])
    data = json.loads(result.stdout)
    assert data["tle_dir"] == str(tle_dir)
    assert data["tle_dir_source"] == "THISTLE_TLE_DIR"
    assert data["tle_ext"] == ".tle"
    assert data["tle_ext_source"] == "default"


def test_config_warns_missing_dir(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("THISTLE_TLE_DIR", str(tmp_path / "nope"))
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "non-existent" in result.stderr


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
