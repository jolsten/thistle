"""Functions for generating ECI, ECEF, and LLA data from satellite propagation.

All positions are in meters, angles in degrees, velocities in m/s,
magnetic field in nanoTesla, and local solar time in fractional hours.
"""

import datetime
import pathlib
from typing import Dict, Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.elementslib import osculating_elements_of
from skyfield.framelib import itrs
from skyfield.functions import angle_between

from thistle.utils import dt64_to_time

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thistle.propagator import Propagator

AU_TO_M = 149_597_870_700.0
AU_PER_DAY_TO_M_PER_S = AU_TO_M / 86_400.0

R_EARTH_KM = 6_371.0
R_SUN_KM = 696_340.0

_DATA_DIR = pathlib.Path(__file__).parent / "data"

ts = load.timescale()
eph = load(str(_DATA_DIR / "de421.bsp"))

GenerateResult = Dict[str, npt.NDArray]


def _propagate(times, satellite):
    """Convert datetime64 array to Skyfield Time and propagate."""
    t = dt64_to_time(times, ts)
    return t, satellite.at(t)


def generate_eci(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Generate ECI (GCRS) position and velocity for the given times.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with keys: eci_x, eci_y, eci_z (m),
        eci_vx, eci_vy, eci_vz (m/s).
    """
    _, geocentric = _propagate(times, satellite)
    pos = cast(npt.NDArray, geocentric.xyz.au) * AU_TO_M
    vel = cast(npt.NDArray, geocentric.velocity.au_per_d) * AU_PER_DAY_TO_M_PER_S
    return {
        "eci_x": pos[0],
        "eci_y": pos[1],
        "eci_z": pos[2],
        "eci_vx": vel[0],
        "eci_vy": vel[1],
        "eci_vz": vel[2],
    }


def generate_ecef(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Generate ECEF (ITRS) position and velocity for the given times.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with keys: ecef_x, ecef_y, ecef_z (m),
        ecef_vx, ecef_vy, ecef_vz (m/s).
    """
    _, geocentric = _propagate(times, satellite)
    pos = cast(npt.NDArray, geocentric.frame_xyz(itrs).au) * AU_TO_M
    vel = (
        cast(npt.NDArray, geocentric.frame_xyz_and_velocity(itrs)[1].au_per_d)
        * AU_PER_DAY_TO_M_PER_S
    )
    return {
        "ecef_x": pos[0],
        "ecef_y": pos[1],
        "ecef_z": pos[2],
        "ecef_vx": vel[0],
        "ecef_vy": vel[1],
        "ecef_vz": vel[2],
    }


def generate_lla(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Generate LLA (latitude, longitude, altitude) for the given times.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with keys: lat (deg), lon (deg), alt (m).
    """
    _, geocentric = _propagate(times, satellite)
    subpoint = wgs84.subpoint(geocentric)
    return {
        "lat": cast(npt.NDArray, subpoint.latitude.degrees),
        "lon": cast(npt.NDArray, subpoint.longitude.degrees),
        "alt": cast(npt.NDArray, subpoint.elevation.km) * 1000.0,
    }


def generate_keplerian(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Generate osculating Keplerian elements for the given times.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with keys: sma (m), ecc, inc (deg), raan (deg),
        aop (deg), ta (deg), ma (deg), ea (deg), arglat (deg),
        tlon (deg), mlon (deg), lonper (deg), mm (deg/day).
    """
    _, geocentric = _propagate(times, satellite)
    elems = osculating_elements_of(geocentric)
    return {
        "sma": cast(npt.NDArray, elems.semi_major_axis.km) * 1000.0,
        "ecc": cast(npt.NDArray, elems.eccentricity),
        "inc": cast(npt.NDArray, elems.inclination.degrees),
        "raan": cast(npt.NDArray, elems.longitude_of_ascending_node.degrees),
        "aop": cast(npt.NDArray, elems.argument_of_periapsis.degrees),
        "ta": cast(npt.NDArray, elems.true_anomaly.degrees),
        "ma": cast(npt.NDArray, elems.mean_anomaly.degrees),
        "ea": cast(npt.NDArray, elems.eccentric_anomaly.degrees),
        "arglat": cast(npt.NDArray, elems.argument_of_latitude.degrees),
        "tlon": cast(npt.NDArray, elems.true_longitude.degrees),
        "mlon": cast(npt.NDArray, elems.mean_longitude.degrees),
        "lonper": cast(npt.NDArray, elems.longitude_of_periapsis.degrees),
        "mm": cast(npt.NDArray, elems.mean_motion_per_day.degrees),
    }


def generate_equinoctial(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Generate modified equinoctial elements for the given times.

    Computed from osculating Keplerian elements using the standard
    transformation (Walker 1985).

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with keys: p (m), f, g, h, k, L (deg).
    """
    _, geocentric = _propagate(times, satellite)
    elems = osculating_elements_of(geocentric)

    a = cast(npt.NDArray, elems.semi_major_axis.km) * 1000.0
    e = cast(npt.NDArray, elems.eccentricity)
    i = np.radians(cast(npt.NDArray, elems.inclination.degrees))
    raan = np.radians(cast(npt.NDArray, elems.longitude_of_ascending_node.degrees))
    omega = np.radians(cast(npt.NDArray, elems.argument_of_periapsis.degrees))
    nu = np.radians(cast(npt.NDArray, elems.true_anomaly.degrees))

    p = a * (1.0 - e**2)
    f = e * np.cos(omega + raan)
    g = e * np.sin(omega + raan)
    h = np.tan(i / 2.0) * np.cos(raan)
    k = np.tan(i / 2.0) * np.sin(raan)
    L = np.degrees(raan + omega + nu)

    return {"p": p, "f": f, "g": g, "h": h, "k": k, "L": L}


def generate_sunlight(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Classify sunlight condition using a conical shadow model.

    Uses the angular radii of the Sun and Earth as seen from the
    satellite to distinguish umbra, penumbra, and full sunlight.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with key: sun (int8, 0 = umbra, 1 = penumbra, 2 = sunlit).
    """
    t, geocentric = _propagate(times, satellite)
    sat_km = cast(npt.NDArray, geocentric.xyz.km)

    sun_km = cast(npt.NDArray, (eph["sun"] - eph["earth"]).at(t).xyz.km)

    sat_to_sun = sun_km - sat_km
    sat_to_earth = -sat_km

    d_sun = np.linalg.norm(sat_to_sun, axis=0)
    d_earth = np.linalg.norm(sat_to_earth, axis=0)

    theta_sun = np.arcsin(R_SUN_KM / d_sun)
    theta_earth = np.arcsin(R_EARTH_KM / d_earth)

    cos_sep = np.sum(sat_to_sun * sat_to_earth, axis=0) / (d_sun * d_earth)
    theta_sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))

    result = np.ones(len(times), dtype=np.int8)  # default penumbra
    result[theta_sep >= theta_earth + theta_sun] = 2  # sunlit
    result[theta_sep <= theta_earth - theta_sun] = 0  # umbra
    return {"sun": result}


def generate_beta_angle(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Generate the beta angle (orbit plane vs. Sun) for the given times.

    Beta angle is the angle between the orbit plane and the Sun direction
    vector, computed as 90 degrees minus the angle between the orbit
    normal (r x v) and the Sun vector.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with key: beta (deg). Positive when the Sun is above
        the orbit plane.
    """
    t, geocentric = _propagate(times, satellite)

    r = cast(npt.NDArray, geocentric.xyz.km)
    v = cast(npt.NDArray, geocentric.velocity.km_per_s)
    orbit_normal = np.cross(r.T, v.T).T

    sun_vec = cast(npt.NDArray, (eph["sun"] - eph["earth"]).at(t).xyz.km)

    beta_rad = angle_between(orbit_normal, sun_vec)
    return {"beta": 90.0 - np.degrees(beta_rad)}


def generate_local_solar_time(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
) -> GenerateResult:
    """Generate apparent local solar time at the subsatellite point.

    Computed from the subsatellite longitude, Greenwich Mean Sidereal
    Time, and the Sun's right ascension.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.

    Returns:
        A dict with key: lst (fractional hours [0, 24)).
    """
    t, geocentric = _propagate(times, satellite)

    lon_deg = cast(npt.NDArray, wgs84.subpoint(geocentric).longitude.degrees)
    gmst = cast(npt.NDArray, t.gmst)
    sun_ra_hours = cast(
        npt.NDArray, eph["earth"].at(t).observe(eph["sun"]).apparent().radec()[0].hours
    )

    lst_hours = gmst + lon_deg / 15.0
    local_solar_time = (lst_hours - sun_ra_hours + 12.0) % 24.0
    return {"lst": local_solar_time}


def _magnetic_field_enu(times, satellite, epoch):
    """Compute IGRF field in ENU and return components with lat/lon."""
    import ppigrf

    _, geocentric = _propagate(times, satellite)
    subpoint = wgs84.subpoint(geocentric)
    lat = cast(npt.NDArray, subpoint.latitude.degrees)
    lon = cast(npt.NDArray, subpoint.longitude.degrees)
    alt_km = cast(npt.NDArray, subpoint.elevation.km)

    if epoch is None:
        mid = times[len(times) // 2]
        epoch = mid.astype("datetime64[us]").astype(datetime.datetime)

    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt_km, epoch)
    return Be.ravel(), Bn.ravel(), Bu.ravel(), lat, lon


def generate_magnetic_field_enu(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
    epoch: Optional[datetime.datetime] = None,
) -> GenerateResult:
    """Generate IGRF magnetic field in local ENU coordinates.

    Uses the IGRF model via ppigrf evaluated at the subsatellite point.
    A single IGRF epoch is used for all time steps since the model
    coefficients vary slowly (~100 nT/year).

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.
        epoch: IGRF evaluation date. Defaults to the midpoint of times.

    Returns:
        A dict with keys: Be, Bn, Bu (nT).
    """
    Be, Bn, Bu, _, _ = _magnetic_field_enu(times, satellite, epoch)
    return {"Be": Be, "Bn": Bn, "Bu": Bu}


def generate_magnetic_field_total(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
    epoch: Optional[datetime.datetime] = None,
) -> GenerateResult:
    """Generate total IGRF magnetic field magnitude.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.
        epoch: IGRF evaluation date. Defaults to the midpoint of times.

    Returns:
        A dict with key: Bt (nT).
    """
    Be, Bn, Bu, _, _ = _magnetic_field_enu(times, satellite, epoch)
    return {"Bt": np.sqrt(Be**2 + Bn**2 + Bu**2)}


def generate_magnetic_field_ecef(
    times: npt.NDArray[np.datetime64],
    satellite: EarthSatellite,
    epoch: Optional[datetime.datetime] = None,
) -> GenerateResult:
    """Generate IGRF magnetic field rotated into ECEF (ITRS) coordinates.

    Transforms the local ENU field components to Earth-Centered
    Earth-Fixed using the subsatellite geodetic latitude and longitude.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object.
        epoch: IGRF evaluation date. Defaults to the midpoint of times.

    Returns:
        A dict with keys: Bx, By, Bz (nT).
    """
    Be, Bn, Bu, lat_deg, lon_deg = _magnetic_field_enu(times, satellite, epoch)

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    Bx = -sin_lon * Be - sin_lat * cos_lon * Bn + cos_lat * cos_lon * Bu
    By = cos_lon * Be - sin_lat * sin_lon * Bn + cos_lat * sin_lon * Bu
    Bz = cos_lat * Bn + sin_lat * Bu
    return {"Bx": Bx, "By": By, "Bz": Bz}


GENERATORS = {
    "eci": generate_eci,
    "ecef": generate_ecef,
    "lla": generate_lla,
    "keplerian": generate_keplerian,
    "equinoctial": generate_equinoctial,
    "sunlight": generate_sunlight,
    "beta": generate_beta_angle,
    "lst": generate_local_solar_time,
    "mag_enu": generate_magnetic_field_enu,
    "mag_total": generate_magnetic_field_total,
    "mag_ecef": generate_magnetic_field_ecef,
}


# Keys that downcast to float32 (angles, dimensionless, magnetic field, lst).
_F32_KEYS = {
    "lat",
    "lon",
    "inc",
    "raan",
    "aop",
    "ta",
    "ma",
    "ea",
    "arglat",
    "tlon",
    "mlon",
    "lonper",
    "mm",
    "ecc",
    "f",
    "g",
    "h",
    "k",
    "L",
    "beta",
    "lst",
    "Be",
    "Bn",
    "Bu",
    "Bt",
    "Bx",
    "By",
    "Bz",
}


def _generate_with_propagator(
    times: npt.NDArray[np.datetime64],
    propagator: "Propagator",
    groups: Sequence[str],
) -> GenerateResult:
    """Generate data using a Propagator with automatic TLE switching.

    Splits the time array into segments based on transition times and
    processes each segment with the appropriate satellite.

    Args:
        times: Array of datetime64 values.
        propagator: A Propagator object.
        groups: Which data groups to compute.

    Returns:
        A dict with all requested data groups.
    """
    segments = propagator.segment_times(times)

    # Process each segment
    segment_results = []
    for t_slice, sat in segments:
        segment_data = {}  # type: GenerateResult
        for name in groups:
            segment_data.update(GENERATORS[name](t_slice, sat))
        segment_results.append((len(t_slice), segment_data))

    # Merge segments back into full arrays
    result = {}  # type: GenerateResult
    if segment_results:
        all_keys = set(segment_results[0][1].keys())
        for key in all_keys:
            first_segment_value = segment_results[0][1][key]
            output_array = np.empty(len(times), dtype=first_segment_value.dtype)

            offset = 0
            for n, segment_data in segment_results:
                output_array[offset : offset + n] = segment_data[key]
                offset += n

            result[key] = output_array

    return result


def generate(
    times: npt.NDArray[np.datetime64],
    satellite: Union[EarthSatellite, "Propagator"],
    groups: Sequence[str],
) -> GenerateResult:
    """Run one or more generate functions and merge the results.

    Arrays are downcast where full float64 precision is unnecessary:
    float32 for angles, dimensionless elements, and magnetic field;
    float16 for local solar time. Positions and velocities in meters
    remain float64.

    When a Propagator is provided, the time array is split into segments
    based on the transition times in the propagator, and each segment is
    computed using the appropriate EarthSatellite object. Results are
    merged at the end.

    Args:
        times: Array of datetime64 values.
        satellite: A Skyfield EarthSatellite object or a Propagator.
        groups: Which data groups to compute. Valid names:
            eci, ecef, lla, keplerian, equinoctial, sunlight,
            beta, lst, mag_enu, mag_total, mag_ecef.

    Returns:
        A single dict merging all requested groups.

    Raises:
        ValueError: If a group name is not recognized.
    """
    # Validate group names
    for name in groups:
        if name not in GENERATORS:
            raise ValueError(
                f"Unknown group {name!r}, expected one of {list(GENERATORS)}"
            )

    # Dispatch to appropriate implementation
    from thistle.propagator import Propagator

    if isinstance(satellite, Propagator):
        result = _generate_with_propagator(times, satellite, groups)
    else:
        # Single satellite case
        result = {}  # type: GenerateResult
        for name in groups:
            result.update(GENERATORS[name](times, satellite))

    # Downcast arrays where appropriate
    for key, arr in result.items():
        if key in _F32_KEYS:
            result[key] = arr.astype(np.float32)

    return result
