"""Offline map rendering: vendored Natural Earth data and fail-soft features.

`thistle groundtrack` must render on a machine that has never had internet:
the two default features (coastlines, borders) resolve from shapefiles
vendored in the package via cartopy's ``pre_existing_data_dir``, and any
feature that cannot be resolved is skipped with a warning instead of
crashing the render.
"""

import pathlib

import numpy as np
import pytest

cartopy = pytest.importorskip("cartopy")
pytest.importorskip("matplotlib")

from cartopy.io import shapereader  # noqa: E402

from thistle.cli import _map  # noqa: E402


@pytest.fixture
def isolated_cartopy(monkeypatch, tmp_path):
    """Cartopy with an empty download cache and no pre-configured data dir.

    Simulates a fresh machine: nothing in the cache, so any resolution that
    succeeds must come from the vendored directory (and anything that tries
    to download would populate ``tmp_path``, which the tests assert against).
    """
    monkeypatch.setitem(cartopy.config, "data_dir", str(tmp_path))
    monkeypatch.setitem(cartopy.config, "pre_existing_data_dir", "")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    return tmp_path


def test_vendored_shapefiles_ship_with_the_package():
    base = _map._VENDORED_CARTOPY_DATA / "shapefiles" / "natural_earth"
    for rel in (
        "physical/ne_110m_coastline.shp",
        "cultural/ne_110m_admin_0_boundary_lines_land.shp",
    ):
        assert (base / rel).is_file(), rel


def test_default_features_resolve_from_vendored_data(isolated_cartopy):
    _map._configure_cartopy()
    for category, name in (
        ("physical", "coastline"),
        ("cultural", "admin_0_boundary_lines_land"),
    ):
        path = pathlib.Path(
            shapereader.natural_earth(
                resolution="110m", category=category, name=name
            )
        )
        assert _map._VENDORED_CARTOPY_DATA in path.parents, path
    # nothing was downloaded or copied into the cache
    assert not list(isolated_cartopy.rglob("*.shp"))


def _tiny_map():
    traces = [(np.array([0.0, 10.0]), np.array([0.0, 5.0]), {})]
    return _map.make_map(traces, sites={}, ring_alt_m=None, min_el=0.0)


def test_make_map_renders_without_download_cache(isolated_cartopy, capsys):
    import matplotlib.pyplot as plt

    fig = _tiny_map()
    try:
        fig.canvas.draw()  # features resolve lazily; force the full render
    finally:
        plt.close(fig)
    assert "unavailable" not in capsys.readouterr().err
    assert not list(isolated_cartopy.rglob("*.shp"))


def test_make_map_degrades_when_features_unavailable(
    isolated_cartopy, monkeypatch, capsys
):
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    def unavailable():
        raise OSError("no internet and not cached")

    monkeypatch.setattr(cfeature.COASTLINE, "geometries", unavailable)
    monkeypatch.setattr(cfeature.BORDERS, "geometries", unavailable)

    fig = _tiny_map()
    try:
        fig.canvas.draw()
    finally:
        plt.close(fig)
    err = capsys.readouterr().err
    assert err.count("unavailable") == 2  # one warning per skipped feature
