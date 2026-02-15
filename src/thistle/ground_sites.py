"""Ground site visibility geometry on the WGS84 ellipsoid."""

import numpy as np
import numpy.typing as npt

# WGS84 semi-major axis (m) used to convert Earth-central angle to arc distance.
_WGS84_A = 6378137.0


def visibility_circle(
    lat: float,
    lon: float,
    sat_alt: float,
    min_el: float,
    alt: float = 0.0,
    n_points: int = 100,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the ground visibility circle for a satellite altitude.

    Returns the closed polygon of lat/lon points on the WGS84 ellipsoid
    where a satellite at the given altitude is visible above the minimum
    elevation angle from the ground site.

    Args:
        lat: Ground site geodetic latitude (deg).
        lon: Ground site geodetic longitude (deg).
        sat_alt: Target satellite altitude above the ellipsoid (m).
        min_el: Minimum elevation angle (deg).
        alt: Ground site altitude above the ellipsoid (m).
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
