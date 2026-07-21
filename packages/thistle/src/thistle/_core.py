"""Shared infrastructure: singletons, constants, propagation, and site utilities.

This module provides the foundational objects used across the package.
It is internal (not part of the public API) but its non-underscore names
are the stable inter-module contract.
"""

import pathlib
from typing import Dict, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from skyfield.api import EarthSatellite, load, wgs84

from thistle.utils import dt64_to_time

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).parent / "data"

ts = load.timescale()
eph = load(str(_DATA_DIR / "de421.bsp"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AU_TO_M = 149_597_870_700.0
AU_PER_DAY_TO_M_PER_S = AU_TO_M / 86_400.0

R_EARTH_KM = 6_371.0
R_SUN_KM = 696_340.0

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

GenerateResult = Dict[str, npt.NDArray]
Site = Union[Tuple[float, float], Tuple[float, float, float]]
Sites = Union[Sequence[Site], Dict[str, Site]]

# ---------------------------------------------------------------------------
# Site utilities
# ---------------------------------------------------------------------------


def normalize_site(site: tuple) -> tuple[float, float, float]:
    """Normalize a site tuple to (lat, lon, alt), defaulting alt to 0.0."""
    if len(site) == 2:
        return (float(site[0]), float(site[1]), 0.0)
    elif len(site) == 3:
        return (float(site[0]), float(site[1]), float(site[2]))
    else:
        raise ValueError(f"Site tuple must have 2 or 3 elements, got {len(site)}")


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def propagate_sat(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
):
    """Convert datetime64 array to Skyfield Time and propagate.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A (t, geocentric) tuple where t is a Skyfield Time and geocentric
        is the result of satellite.at(t).
    """
    t = dt64_to_time(times, ts)
    return t, satellite.at(t)


# ---------------------------------------------------------------------------
# Range extraction
# ---------------------------------------------------------------------------


def extract_range(t, geocentric, sites) -> GenerateResult:
    """Compute slant range and range rate from pre-computed geocentric state.

    Reuses the satellite geocentric result to avoid redundant SGP4 propagation.
    Ground site positions are computed via Earth rotation only.

    Args:
        t: Skyfield Time array.
        geocentric: Skyfield Geocentric from satellite.at(t).
        sites: List of (suffix, lat, lon, alt) tuples.

    Returns:
        Dict with range_{suffix} (m) and range_rate_{suffix} (m/s) per site.
    """
    result: GenerateResult = {}
    for suffix, lat, lon, alt in sites:
        ground = wgs84.latlon(lat, lon, elevation_m=alt).at(t)
        r = (cast(npt.NDArray, geocentric.xyz.au) - cast(npt.NDArray, ground.xyz.au)) * AU_TO_M
        v = (
            cast(npt.NDArray, geocentric.velocity.au_per_d)
            - cast(npt.NDArray, ground.velocity.au_per_d)
        ) * AU_PER_DAY_TO_M_PER_S
        slant_range = np.sqrt(np.sum(r**2, axis=0))
        range_rate = np.sum(r * v, axis=0) / slant_range
        result[f"range_{suffix}"] = slant_range
        result[f"range_rate_{suffix}"] = range_rate
    return result
