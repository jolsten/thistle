"""Ground-site-to-satellite range and range rate."""

from typing import cast

import numpy as np
import numpy.typing as npt
from skyfield.api import EarthSatellite, wgs84

from thistle.satellite import AU_PER_DAY_TO_M_PER_S, AU_TO_M, GenerateResult, ts
from thistle.utils import jday_datetime64


def generate_range(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
    lat: float,
    lon: float,
    alt: float = 0.0,
) -> GenerateResult:
    """Generate slant range and range rate from a ground site to a satellite.

    Computes the topocentric range (distance) and range rate (time derivative
    of range) from a WGS84 ground site to the satellite at each time step.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.
        lat: Ground site geodetic latitude (deg).
        lon: Ground site geodetic longitude (deg).
        alt: Ground site altitude above the WGS84 ellipsoid (m).

    Returns:
        A dict with keys: range (m), range_rate (m/s).
    """
    topos = wgs84.latlon(lat, lon, elevation_m=alt)

    jd, fr = jday_datetime64(times)
    t = ts.tt_jd(jd, fr)

    topo_pos = (satellite - topos).at(t)

    r = cast(npt.NDArray, topo_pos.xyz.au) * AU_TO_M
    v = cast(npt.NDArray, topo_pos.velocity.au_per_d) * AU_PER_DAY_TO_M_PER_S

    slant_range = np.sqrt(np.sum(r**2, axis=0))
    range_rate = np.sum(r * v, axis=0) / slant_range

    return {"range": slant_range, "range_rate": range_rate}
