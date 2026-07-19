"""Ground-site-to-satellite range, range rate, and Doppler geolocation."""

import dataclasses
from typing import Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
from skyfield.api import EarthSatellite

from thistle._core import extract_range, propagate_sat
from thistle.orbit_data import generate_lla


@dataclasses.dataclass
class DopplerGeolocationResult:
    """Result of Doppler geolocation from a single satellite pass.

    Attributes:
        lat: Solved ground site latitude (deg).
        lon: Solved ground site longitude (deg).
        scale: Solved scale factor (doppler_units -> m/s).
        residuals: Fit residuals in doppler units.
        rms: Root-mean-square of residuals (doppler units).
        converged: Whether the optimizer converged.
    """

    lat: float
    lon: float
    scale: float
    residuals: npt.NDArray
    rms: float
    converged: bool


def _find_doppler_tca(
    times: npt.NDArray[np.datetime64],
    doppler: npt.NDArray[np.float64],
) -> np.datetime64:
    """Find the time of closest approach from the Doppler zero crossing.

    Linearly interpolates between the samples straddling the sign change.
    Falls back to the time of minimum |doppler| if no sign change exists.
    """
    signs = np.sign(doppler)
    changes = np.where(np.diff(signs) != 0)[0]

    if len(changes) == 0:
        return cast(np.datetime64, times[np.argmin(np.abs(doppler))])

    i = changes[0]
    d0, d1 = float(doppler[i]), float(doppler[i + 1])
    frac = -d0 / (d1 - d0)  # linear interpolation fraction

    dt = (times[i + 1] - times[i]).astype("timedelta64[us]").astype(np.int64)
    offset = np.timedelta64(int(frac * dt), "us")
    return cast(np.datetime64, times[i] + offset)


def _subsatellite_at_time(
    time: np.datetime64,
    satellite: EarthSatellite,
) -> Tuple[float, float]:
    """Return (lat_deg, lon_deg) of the sub-satellite point at a single time."""
    lla = generate_lla(np.atleast_1d(time), satellite)
    lat_arr = cast(npt.NDArray, lla["lat"])
    lon_arr = cast(npt.NDArray, lla["lon"])
    return float(lat_arr[0]), float(lon_arr[0])


def _reflect_across_ground_track(
    lat: float,
    lon: float,
    tca_time: np.datetime64,
    satellite: EarthSatellite,
) -> Tuple[float, float]:
    """Reflect a lat/lon point across the satellite ground track at TCA."""
    from geographiclib.geodesic import Geodesic

    geod = Geodesic.WGS84

    # Sub-satellite point at TCA
    t_arr = np.array(
        [tca_time - np.timedelta64(10, "s"), tca_time + np.timedelta64(10, "s")]
    )
    lla = generate_lla(t_arr, satellite)
    ssp_lat = float(np.mean(lla["lat"]))
    ssp_lon = float(np.mean(lla["lon"]))

    # Ground track bearing at TCA
    inv_track = geod.Inverse(
        float(lla["lat"][0]),
        float(lla["lon"][0]),
        float(lla["lat"][1]),
        float(lla["lon"][1]),
    )
    track_az = inv_track["azi1"]

    # Azimuth and distance from SSP to the candidate point
    inv_point = geod.Inverse(ssp_lat, ssp_lon, lat, lon)
    point_az = inv_point["azi1"]
    point_dist = inv_point["s12"]

    # Reflect azimuth across the ground track
    reflected_az = 2.0 * track_az - point_az

    # Project along reflected azimuth
    direct = geod.Direct(ssp_lat, ssp_lon, reflected_az, point_dist)
    return direct["lat2"], direct["lon2"]


def _extract_solution(
    opt_result: object,
    t,
    geocentric,
    doppler: npt.NDArray[np.float64],
    alt: float,
) -> DopplerGeolocationResult:
    """Build a DopplerGeolocationResult from a scipy OptimizeResult."""
    lat_sol, lon_sol = float(opt_result.x[0]), float(opt_result.x[1])
    rr = extract_range(t, geocentric, [("0", lat_sol, lon_sol, alt)])
    model_rr = cast(npt.NDArray, rr["range_rate_0"])
    scale_sol = float(np.dot(doppler, model_rr) / np.dot(model_rr, model_rr))
    residuals = doppler - scale_sol * model_rr
    rms = float(np.sqrt(np.mean(residuals**2)))
    return DopplerGeolocationResult(
        lat=lat_sol,
        lon=lon_sol,
        scale=scale_sol,
        residuals=residuals,
        rms=rms,
        converged=bool(opt_result.success),
    )


def geolocate_doppler(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
    doppler: npt.NDArray[np.float64],
    *,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
    alt: float = 0.0,
) -> list:
    """Solve for a ground site location from unscaled Doppler measurements.

    Uses a single satellite pass of Doppler (range rate in arbitrary units)
    and the known satellite orbit to estimate the ground site latitude,
    longitude, and an unknown scale factor via nonlinear least squares.

    Single-pass Doppler has an inherent left/right ambiguity relative to
    the ground track. Both sides are solved and returned, sorted by RMS
    (best fit first).

    Args:
        times: Array of datetime64 values for the pass.
        satellite: A Skyfield EarthSatellite object.
        doppler: Observed Doppler values (arbitrary units, proportional
            to range rate). Must have the same length as times.
        lat0: Initial guess for latitude (deg). If None, estimated from
            the sub-satellite point at TCA.
        lon0: Initial guess for longitude (deg). If None, estimated from
            the sub-satellite point at TCA.
        alt: Ground site altitude above the WGS84 ellipsoid (m).

    Returns:
        A list of two DopplerGeolocationResult, sorted by RMS (best
        first). Each contains lat, lon, scale, residuals, rms, and
        a convergence flag.

    Raises:
        ValueError: If inputs have mismatched shapes or fewer than 3 points.
    """
    from scipy.optimize import minimize

    times = np.asarray(times)
    doppler = np.asarray(doppler, dtype=np.float64)

    if times.shape != doppler.shape:
        raise ValueError("times and doppler must have the same shape")
    if times.ndim != 1 or len(times) < 3:
        raise ValueError("Need at least 3 measurements")

    # Propagate once — reused by every optimizer iteration
    t, geocentric = propagate_sat(times, satellite)

    # Initial guess from TCA sub-satellite point
    tca_time = _find_doppler_tca(times, doppler)
    if lat0 is None or lon0 is None:
        lat0, lon0 = _subsatellite_at_time(tca_time, satellite)

    lat_guess: float = lat0
    lon_guess: float = lon0

    def _cost(x: npt.NDArray) -> float:
        rr = extract_range(t, geocentric, [("0", float(x[0]), float(x[1]), alt)])
        model_rr = cast(npt.NDArray, rr["range_rate_0"])
        denom = np.dot(model_rr, model_rr)
        if denom < 1e-30:
            return 1e30
        scale = np.dot(doppler, model_rr) / denom
        residuals = doppler - scale * model_rr
        return float(np.dot(residuals, residuals))

    opts = {"xatol": 1e-4, "fatol": 1e-10, "maxiter": 200}

    # Try side A (initial guess)
    result_a = minimize(
        _cost, x0=[lat_guess, lon_guess], method="Nelder-Mead", options=opts
    )

    # Try side B (reflect side A's converged solution across ground track)
    lat0_b, lon0_b = _reflect_across_ground_track(
        float(result_a.x[0]),
        float(result_a.x[1]),
        tca_time,
        satellite,
    )
    result_b = minimize(_cost, x0=[lat0_b, lon0_b], method="Nelder-Mead", options=opts)

    solutions = [
        _extract_solution(result_a, t, geocentric, doppler, alt),
        _extract_solution(result_b, t, geocentric, doppler, alt),
    ]
    solutions.sort(key=lambda s: s.rms)
    return solutions
