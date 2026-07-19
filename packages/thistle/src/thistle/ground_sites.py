"""Ground site visibility geometry on the WGS84 ellipsoid."""

from typing import Union, cast

import numpy as np
import numpy.typing as npt
from skyfield.api import EarthSatellite, wgs84

from thistle._core import AU_PER_DAY_TO_M_PER_S, AU_TO_M, GenerateResult, Sites, normalize_site, ts
from thistle.utils import dt64_to_time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thistle.propagator import Propagator

# WGS84 semi-major axis (m) used to convert Earth-central angle to arc distance.
_WGS84_A = 6378137.0

SPEED_OF_LIGHT = 299_792_458.0  # m/s


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


def _generate_range_single(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
    lat: float,
    lon: float,
    alt: float = 0.0,
) -> GenerateResult:
    """Generate slant range and range rate for a single site and satellite.

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

    t = dt64_to_time(times, ts)

    topo_pos = (satellite - topos).at(t)

    r = cast(npt.NDArray, topo_pos.xyz.au) * AU_TO_M
    v = cast(npt.NDArray, topo_pos.velocity.au_per_d) * AU_PER_DAY_TO_M_PER_S

    slant_range = np.sqrt(np.sum(r**2, axis=0))
    range_rate = np.sum(r * v, axis=0) / slant_range

    return {"range": slant_range, "range_rate": range_rate}


def generate_range(
    times: npt.NDArray[np.datetime64],
    satellite: Union[EarthSatellite, "Propagator"],
    sites: Sites,
) -> GenerateResult:
    """Generate slant range and range rate from ground sites to a satellite.

    Computes the topocentric range (distance) and range rate (time derivative
    of range) from one or more WGS84 ground sites to the satellite at each
    time step.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite or a Propagator object.
        sites: Ground sites, either as:
            - A sequence of (lat, lon) or (lat, lon, alt) tuples.
              Keys are indexed: range_0, range_rate_0, range_1, ...
            - A dict mapping site names to (lat, lon) or (lat, lon, alt) tuples.
              Keys use the name: range_ksc, range_rate_ksc, ...

    Returns:
        A dict with keys range_{suffix} (m) and range_rate_{suffix} (m/s)
        for each site.
    """
    from thistle.propagator import Propagator

    # Build ordered list of (suffix, lat, lon, alt)
    if isinstance(sites, dict):
        site_list = [
            (name, *normalize_site(coords))
            for name, coords in sites.items()
        ]
    else:
        site_list = [
            (str(i), *normalize_site(coords))
            for i, coords in enumerate(sites)
        ]

    is_propagator = isinstance(satellite, Propagator)
    result: GenerateResult = {}

    for suffix, lat, lon, alt in site_list:
        if is_propagator:
            site_data = _generate_range_propagator(times, satellite, lat, lon, alt)
        else:
            site_data = _generate_range_single(times, satellite, lat, lon, alt)

        result[f"range_{suffix}"] = site_data["range"]
        result[f"range_rate_{suffix}"] = site_data["range_rate"]

    return result


def _generate_range_propagator(
    times: npt.NDArray[np.datetime64],
    propagator: "Propagator",
    lat: float,
    lon: float,
    alt: float,
) -> GenerateResult:
    """Generate range/range_rate using a Propagator with TLE switching.

    Splits the time array into segments and processes each with the
    appropriate satellite, then merges results.
    """
    segments = propagator.segment_times(times)

    segment_results = []
    for t_slice, sat in segments:
        seg_data = _generate_range_single(t_slice, sat, lat, lon, alt)
        segment_results.append((len(t_slice), seg_data))

    result: GenerateResult = {}
    if segment_results:
        for key in ("range", "range_rate"):
            output = np.empty(len(times), dtype=np.float64)
            offset = 0
            for n, seg_data in segment_results:
                output[offset : offset + n] = seg_data[key]
                offset += n
            result[key] = output

    return result


def doppler_shift(
    range_rate: npt.NDArray,
    freq: float,
) -> npt.NDArray:
    """Compute Doppler frequency shift from range rate.

    Uses the classical (non-relativistic) Doppler formula:
        doppler = -freq * range_rate / c

    Args:
        range_rate: Range rate in m/s (positive = receding).
        freq: Transmit frequency in Hz.

    Returns:
        Doppler shift in Hz (positive = approaching / compression).
    """
    return -freq * range_rate / SPEED_OF_LIGHT
