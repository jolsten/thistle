"""Functions for finding satellite events: passes, node crossings, sunlit/eclipse and ascending/descending periods."""

from typing import Optional, Union, cast

import numpy as np
import numpy.typing as npt
import skyfield.timelib
from pydantic import BaseModel, ConfigDict
from skyfield import almanac
from skyfield.api import EarthSatellite, wgs84

from thistle.orbit_data import eph, ts
from thistle.utils import jday_datetime64, time_to_dt64

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thistle.propagator import Propagator


class Event(BaseModel):
    """Base class for all satellite events.

    Attributes:
        start: Start time of the event.
        stop: End time of the event. Equal to start for instantaneous events.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: np.datetime64
    stop: np.datetime64


class SatellitePass(Event):
    """A satellite pass over a ground site.

    Attributes:
        start: Time the satellite rises above the minimum elevation.
        stop: Time the satellite sets below the minimum elevation.
        peak_time: Time of maximum elevation (culmination).
        peak_elevation: Maximum elevation angle (deg).
    """

    peak_time: np.datetime64
    peak_elevation: float


class NodeCrossing(Event):
    """An orbital node crossing (equator crossing).

    start and stop are both set to the crossing time.

    Attributes:
        start: Time of the equator crossing.
        stop: Time of the equator crossing (same as start).
        longitude: Geodetic longitude at the crossing (deg).
        ascending: True if the satellite is moving northbound.
    """

    longitude: float
    ascending: bool


class SunlitPeriod(Event):
    """A period during which the satellite is in sunlight."""


class EclipsePeriod(Event):
    """A period during which the satellite is in Earth's shadow."""


class AscendingPeriod(Event):
    """A period during which the satellite latitude is increasing (moving northward)."""


class DescendingPeriod(Event):
    """A period during which the satellite latitude is decreasing (moving southward)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_skyfield(time: np.datetime64) -> skyfield.timelib.Time:
    """Convert a single datetime64 to a scalar Skyfield Time."""
    jd, fr = jday_datetime64(np.atleast_1d(time))
    return ts.ut1_jd(float(jd[0]) + float(fr[0]))


def _split_window(
    start: np.datetime64,
    stop: np.datetime64,
    propagator: "Propagator",
) -> list[tuple[np.datetime64, np.datetime64, EarthSatellite]]:
    """Split a time window into sub-windows at TLE transition boundaries."""
    transitions = propagator.switcher.transitions
    inner = transitions[(transitions > start) & (transitions < stop)]
    boundaries = np.concatenate([[start], inner, [stop]])
    result = []
    for i in range(len(boundaries) - 1):
        sub_start, sub_stop = boundaries[i], boundaries[i + 1]
        sat = propagator.find_satellite(sub_start + (sub_stop - sub_start) // 2)
        result.append((sub_start, sub_stop, sat))
    return result


def _merge_periods(periods: list, cls: type) -> list:
    """Merge consecutive periods that touch at a boundary."""
    if not periods:
        return periods
    merged = [periods[0]]
    for p in periods[1:]:
        if p.start == merged[-1].stop:
            merged[-1] = cls(start=merged[-1].start, stop=p.stop)
        else:
            merged.append(p)
    return merged


def _peak_elevation(
    peak_time: np.datetime64,
    satellite: EarthSatellite,
    topos: object,
) -> float:
    """Compute the elevation angle at a given time."""
    t = _to_skyfield(peak_time)
    alt_deg, _, _ = (satellite - topos).at(t).altaz()
    return float(cast(float, alt_deg.degrees))


def _group_periods(
    start: np.datetime64,
    stop: np.datetime64,
    event_dt64: npt.NDArray[np.datetime64],
    event_values: npt.NDArray,
    active_at_start: bool,
    cls: type,
) -> list:
    """Group boolean transitions into start/stop period objects."""
    periods = []
    period_start: Optional[np.datetime64] = start if active_at_start else None

    for dt, val in zip(event_dt64, event_values):
        if val:  # transition to active
            period_start = dt
        else:  # transition to inactive
            if period_start is not None:
                periods.append(cls(start=period_start, stop=dt))
                period_start = None

    # Close open period at window end
    if period_start is not None:
        periods.append(cls(start=period_start, stop=stop))

    return periods


# ---------------------------------------------------------------------------
# Public event-finding functions
# ---------------------------------------------------------------------------


def find_passes(
    start: np.datetime64,
    stop: np.datetime64,
    satellite: Union[EarthSatellite, "Propagator"],
    lat: float,
    lon: float,
    alt: float = 0.0,
    min_elevation: float = 5.0,
) -> list[SatellitePass]:
    """Find satellite passes over a ground site within a time window.

    Uses Skyfield's event detection with a padded search window to
    capture true rise/set times for passes clipped at the boundaries.
    Incomplete passes at the edges use the window boundary as the
    missing rise or set time.

    Args:
        start: Start of the time window.
        stop: End of the time window.
        satellite: A Skyfield EarthSatellite or Propagator object.
        lat: Ground site geodetic latitude (deg).
        lon: Ground site geodetic longitude (deg).
        alt: Ground site altitude above the WGS84 ellipsoid (m).
        min_elevation: Minimum elevation angle (deg).

    Returns:
        A list of SatellitePass objects sorted by rise time.

    Raises:
        ValueError: If start >= stop.
    """
    if start >= stop:
        raise ValueError("start must be before stop")

    from thistle.propagator import Propagator

    if isinstance(satellite, Propagator):
        passes: list[SatellitePass] = []
        for sub_start, sub_stop, sat in _split_window(start, stop, satellite):
            passes.extend(
                find_passes(sub_start, sub_stop, sat, lat, lon, alt, min_elevation)
            )
        return passes

    topos = wgs84.latlon(lat, lon, elevation_m=alt)

    # Pad by 100 min to capture full passes clipped at boundaries
    pad = np.timedelta64(100, "m")
    t0 = _to_skyfield(start - pad)
    t1 = _to_skyfield(stop + pad)

    event_times, event_types = satellite.find_events(
        topos,
        t0,
        t1,
        altitude_degrees=min_elevation,
    )

    if len(event_times) == 0:
        return []

    event_dt64 = time_to_dt64(event_times)

    # Group events into passes
    passes = []
    rise: Optional[np.datetime64] = None
    peak: Optional[np.datetime64] = None

    for dt, etype in zip(event_dt64, event_types):
        if etype == 0:  # rise
            rise = dt
            peak = None
        elif etype == 1:  # culminate
            peak = dt
        elif etype == 2:  # set
            rise_t = rise if rise is not None else start
            peak_t = peak if peak is not None else rise_t
            set_t = dt

            # Only include passes that overlap the requested window
            if set_t > start and rise_t < stop:
                # Compute peak elevation; discard near-grazing passes
                # where find_events produced spurious events
                peak_el = _peak_elevation(peak_t, satellite, topos)
                if peak_el >= min_elevation:
                    passes.append(
                        SatellitePass(
                            start=rise_t,
                            stop=set_t,
                            peak_time=peak_t,
                            peak_elevation=peak_el,
                        )
                    )
            rise = None
            peak = None

    # Handle trailing incomplete pass (rise or culminate without set)
    if rise is not None and rise < stop:
        peak_t = peak if peak is not None else rise
        peak_el = _peak_elevation(peak_t, satellite, topos)
        if peak_el >= min_elevation:
            passes.append(
                SatellitePass(
                    start=rise,
                    stop=stop,
                    peak_time=peak_t,
                    peak_elevation=peak_el,
                )
            )

    return passes


def find_node_crossings(
    start: np.datetime64,
    stop: np.datetime64,
    satellite: Union[EarthSatellite, "Propagator"],
) -> list[NodeCrossing]:
    """Find orbital node crossings (equator crossings) within a time window.

    Args:
        start: Start of the time window.
        stop: End of the time window.
        satellite: A Skyfield EarthSatellite or Propagator object.

    Returns:
        A list of NodeCrossing objects sorted by time.

    Raises:
        ValueError: If start >= stop.
    """
    if start >= stop:
        raise ValueError("start must be before stop")

    from thistle.propagator import Propagator

    if isinstance(satellite, Propagator):
        crossings: list[NodeCrossing] = []
        for sub_start, sub_stop, sat in _split_window(start, stop, satellite):
            crossings.extend(find_node_crossings(sub_start, sub_stop, sat))
        return crossings

    t0 = _to_skyfield(start)
    t1 = _to_skyfield(stop)

    def _is_north(t: skyfield.timelib.Time) -> npt.NDArray:
        geo = satellite.at(t)
        lat = cast(npt.NDArray, wgs84.subpoint(geo).latitude.degrees)
        return np.array(lat) >= 0

    _is_north.step_days = 1 / 1440  # type: ignore[attr-defined]

    event_times, event_values = almanac.find_discrete(t0, t1, _is_north)
    event_dt64 = time_to_dt64(event_times)

    crossings = []
    for dt, t_sky, val in zip(event_dt64, event_times, event_values):
        geo = satellite.at(t_sky)
        lon = float(cast(npt.NDArray, wgs84.subpoint(geo).longitude.degrees))
        crossings.append(
            NodeCrossing(
                start=dt,
                stop=dt,
                longitude=lon,
                ascending=bool(val),
            )
        )

    return crossings


def find_sunlit_periods(
    start: np.datetime64,
    stop: np.datetime64,
    satellite: Union[EarthSatellite, "Propagator"],
) -> list[SunlitPeriod]:
    """Find periods when the satellite is in sunlight within a time window.

    Args:
        start: Start of the time window.
        stop: End of the time window.
        satellite: A Skyfield EarthSatellite or Propagator object.

    Returns:
        A list of SunlitPeriod objects sorted by start time.

    Raises:
        ValueError: If start >= stop.
    """
    if start >= stop:
        raise ValueError("start must be before stop")

    from thistle.propagator import Propagator

    if isinstance(satellite, Propagator):
        periods: list[SunlitPeriod] = []
        for sub_start, sub_stop, sat in _split_window(start, stop, satellite):
            periods.extend(find_sunlit_periods(sub_start, sub_stop, sat))
        return _merge_periods(periods, SunlitPeriod)

    t0 = _to_skyfield(start)
    t1 = _to_skyfield(stop)

    def _is_sunlit(t: skyfield.timelib.Time) -> npt.NDArray:
        return np.array(satellite.at(t).is_sunlit(eph))

    _is_sunlit.step_days = 1 / 1440  # type: ignore[attr-defined]

    event_times, event_values = almanac.find_discrete(t0, t1, _is_sunlit)
    event_dt64 = (
        time_to_dt64(event_times)
        if len(event_times)
        else np.array([], dtype="datetime64[us]")
    )

    # Determine state at start
    sunlit_at_start = bool(satellite.at(t0).is_sunlit(eph))

    return _group_periods(
        start, stop, event_dt64, event_values, sunlit_at_start, SunlitPeriod
    )


def find_eclipse_periods(
    start: np.datetime64,
    stop: np.datetime64,
    satellite: Union[EarthSatellite, "Propagator"],
) -> list[EclipsePeriod]:
    """Find periods when the satellite is in Earth's shadow within a time window.

    Args:
        start: Start of the time window.
        stop: End of the time window.
        satellite: A Skyfield EarthSatellite or Propagator object.

    Returns:
        A list of EclipsePeriod objects sorted by start time.

    Raises:
        ValueError: If start >= stop.
    """
    if start >= stop:
        raise ValueError("start must be before stop")

    from thistle.propagator import Propagator

    if isinstance(satellite, Propagator):
        periods: list[EclipsePeriod] = []
        for sub_start, sub_stop, sat in _split_window(start, stop, satellite):
            periods.extend(find_eclipse_periods(sub_start, sub_stop, sat))
        return _merge_periods(periods, EclipsePeriod)

    t0 = _to_skyfield(start)
    t1 = _to_skyfield(stop)

    def _is_sunlit(t: skyfield.timelib.Time) -> npt.NDArray:
        return np.array(satellite.at(t).is_sunlit(eph))

    _is_sunlit.step_days = 1 / 1440  # type: ignore[attr-defined]

    event_times, event_values = almanac.find_discrete(t0, t1, _is_sunlit)
    event_dt64 = (
        time_to_dt64(event_times)
        if len(event_times)
        else np.array([], dtype="datetime64[us]")
    )

    # Eclipse is the inverse of sunlit
    eclipse_at_start = not bool(satellite.at(t0).is_sunlit(eph))
    inv_values = ~np.array(event_values, dtype=bool)

    return _group_periods(
        start, stop, event_dt64, inv_values, eclipse_at_start, EclipsePeriod
    )


def find_ascending_periods(
    start: np.datetime64,
    stop: np.datetime64,
    satellite: Union[EarthSatellite, "Propagator"],
) -> list[AscendingPeriod]:
    """Find periods when the satellite latitude is increasing (moving northward).

    Ascending periods run from the minimum latitude point (southernmost)
    to the maximum latitude point (northernmost) of each orbit.

    Args:
        start: Start of the time window.
        stop: End of the time window.
        satellite: A Skyfield EarthSatellite or Propagator object.

    Returns:
        A list of AscendingPeriod objects sorted by start time.

    Raises:
        ValueError: If start >= stop.
    """
    if start >= stop:
        raise ValueError("start must be before stop")

    from thistle.propagator import Propagator

    if isinstance(satellite, Propagator):
        periods: list[AscendingPeriod] = []
        for sub_start, sub_stop, sat in _split_window(start, stop, satellite):
            periods.extend(find_ascending_periods(sub_start, sub_stop, sat))
        return _merge_periods(periods, AscendingPeriod)

    t0 = _to_skyfield(start)
    t1 = _to_skyfield(stop)

    def _is_ascending(t: skyfield.timelib.Time) -> npt.NDArray:
        vz = cast(npt.NDArray, satellite.at(t).velocity.km_per_s)
        return np.array(vz[2]) > 0

    _is_ascending.step_days = 1 / 1440  # type: ignore[attr-defined]

    event_times, event_values = almanac.find_discrete(t0, t1, _is_ascending)
    event_dt64 = (
        time_to_dt64(event_times)
        if len(event_times)
        else np.array([], dtype="datetime64[us]")
    )

    ascending_at_start = bool(_is_ascending(t0))

    return _group_periods(
        start, stop, event_dt64, event_values, ascending_at_start, AscendingPeriod
    )


def find_descending_periods(
    start: np.datetime64,
    stop: np.datetime64,
    satellite: Union[EarthSatellite, "Propagator"],
) -> list[DescendingPeriod]:
    """Find periods when the satellite latitude is decreasing (moving southward).

    Descending periods run from the maximum latitude point (northernmost)
    to the minimum latitude point (southernmost) of each orbit.

    Args:
        start: Start of the time window.
        stop: End of the time window.
        satellite: A Skyfield EarthSatellite or Propagator object.

    Returns:
        A list of DescendingPeriod objects sorted by start time.

    Raises:
        ValueError: If start >= stop.
    """
    if start >= stop:
        raise ValueError("start must be before stop")

    from thistle.propagator import Propagator

    if isinstance(satellite, Propagator):
        periods: list[DescendingPeriod] = []
        for sub_start, sub_stop, sat in _split_window(start, stop, satellite):
            periods.extend(find_descending_periods(sub_start, sub_stop, sat))
        return _merge_periods(periods, DescendingPeriod)

    t0 = _to_skyfield(start)
    t1 = _to_skyfield(stop)

    def _is_ascending(t: skyfield.timelib.Time) -> npt.NDArray:
        vz = cast(npt.NDArray, satellite.at(t).velocity.km_per_s)
        return np.array(vz[2]) > 0

    _is_ascending.step_days = 1 / 1440  # type: ignore[attr-defined]

    event_times, event_values = almanac.find_discrete(t0, t1, _is_ascending)
    event_dt64 = (
        time_to_dt64(event_times)
        if len(event_times)
        else np.array([], dtype="datetime64[us]")
    )

    descending_at_start = not bool(_is_ascending(t0))
    inv_values = ~np.array(event_values, dtype=bool)

    return _group_periods(
        start, stop, event_dt64, inv_values, descending_at_start, DescendingPeriod
    )
