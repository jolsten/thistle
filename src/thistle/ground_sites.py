"""Ground site visibility geometry on the WGS84 ellipsoid."""

from typing import cast

import numpy as np
import numpy.typing as npt
from skyfield.api import EarthSatellite, wgs84

from thistle.orbit_data import AU_PER_DAY_TO_M_PER_S, AU_TO_M, GenerateResult, ts
from thistle.utils import jday_datetime64

# WGS84 semi-major axis (m) used to convert Earth-central angle to arc distance.
_WGS84_A = 6378137.0


def visibility_circle(
    lat: float,
    lon: float,
    alt: float,
    sat_alt: float,
    min_el: float = 0.0,
    n_points: int = 100,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the ground visibility circle for a satellite altitude.

    Returns the closed polygon of lat/lon points on the WGS84 ellipsoid
    where a satellite at the given altitude is visible above the minimum
    elevation angle from the ground site.

    Args:
        lat: Ground site geodetic latitude (deg).
        lon: Ground site geodetic longitude (deg).
        alt: Ground site altitude above the ellipsoid (m).
        sat_alt: Target satellite altitude above the ellipsoid (m).
        min_el: Minimum elevation angle (deg).
        n_points: Number of polygon vertices.

    Returns:
        A tuple of (lat_array, lon_array) in degrees, each with
        shape (n_points,).
    """
    from geographiclib.geodesic import Geodesic

    R_g = _WGS84_A + alt
    R_s = _WGS84_A + sat_alt
    eps = np.radians(min_el)

    # Earth-central angle at the visibility edge
    theta = np.arccos(R_g * np.cos(eps) / R_s) - eps

    # Surface arc distance (m) along the ellipsoid
    arc_m = theta * _WGS84_A

    geod = Geodesic.WGS84
    azimuths = np.linspace(0.0, 360.0, n_points, endpoint=False)

    lats = np.empty(n_points, dtype=np.float32)
    lons = np.empty(n_points, dtype=np.float32)
    for i, az in enumerate(azimuths):
        r = geod.Direct(lat, lon, float(az), float(arc_m))
        lats[i] = r["lat2"]
        lons[i] = r["lon2"]

    return lats, lons


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
