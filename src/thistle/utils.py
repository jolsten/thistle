import datetime
import itertools
from typing import Any, Callable, Iterable, TypeVar

import numpy as np
import numpy.typing as npt
import skyfield.timelib

from thistle.typing import DateTime, TLETuple


def pairwise(iterable: Iterable) -> Iterable[tuple[Any, Any]]:
    """Iterate over consecutive pairs in an iterable.

    Fallback for itertools.pairwise on Python < 3.10.

    Args:
        iterable: The input iterable.

    Returns:
        An iterable of (s0, s1), (s1, s2), (s2, s3), ... pairs.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


TIME_SCALE = "us"
EPOCH_DTYPE = np.dtype(f"datetime64[{TIME_SCALE}]")
ONE_SECOND_IN_TIME_SCALE = np.timedelta64(1, "s").astype(f"timedelta64[{TIME_SCALE}]")

DATETIME_MIN = datetime.datetime(1957, 1, 1)
DATETIME_MAX = datetime.datetime(2056, 12, 31, 23, 59, 59, 999999)
DATETIME64_MIN = np.datetime64(DATETIME_MIN, TIME_SCALE)
DATETIME64_MAX = np.datetime64(DATETIME_MAX, TIME_SCALE)

JDAY_1957 = 2435839.5


def datetime_to_dt64(dt: datetime.datetime) -> np.datetime64:
    """Convert a datetime to a numpy datetime64 in microseconds.

    Args:
        dt: The datetime to convert. Timezone info is stripped.

    Returns:
        A numpy datetime64 value with microsecond resolution.
    """
    dt = dt.replace(tzinfo=None)
    return np.datetime64(dt, TIME_SCALE)


def dt64_to_datetime(dt: np.datetime64) -> datetime.datetime:
    """Convert a numpy datetime64 to a timezone-naive datetime.

    Args:
        dt: The numpy datetime64 value.

    Returns:
        A timezone-naive datetime.
    """
    return datetime.datetime.fromisoformat(str(dt))


def validate_datetime64(value: DateTime) -> np.datetime64:
    """Coerce a datetime or datetime64 to a datetime64 with microsecond resolution.

    Args:
        value: A datetime or numpy datetime64.

    Returns:
        A numpy datetime64 with microsecond resolution.
    """
    return np.datetime64(value, TIME_SCALE)


def trange(
    start: datetime.datetime, stop: datetime.datetime, step: float
) -> npt.NDArray[np.datetime64]:
    """Generate an array of datetime64 values at a fixed interval.

    Args:
        start: Start of the range (inclusive).
        stop: End of the range (exclusive).
        step: Step size in seconds.

    Returns:
        An array of datetime64 values with microsecond resolution.
    """
    times = np.arange(
        datetime_to_dt64(start),
        datetime_to_dt64(stop),
        step * ONE_SECOND_IN_TIME_SCALE,
    )
    return times


def datetime_to_tle_epoch(dt: datetime.datetime) -> tuple[int, float]:
    """Convert a datetime to a TLE-style epoch (two-digit year, day-of-year).

    Args:
        dt: The datetime to convert. Must be timezone-aware (UTC).

    Returns:
        A tuple of (two-digit year, fractional day-of-year).
    """
    midnight = datetime.datetime.combine(
        dt.date(), datetime.time(0, 0, 0), tzinfo=datetime.timezone.utc
    )
    fday = (dt - midnight).total_seconds()
    yr = int(dt.strftime("%y"))
    days = int(dt.strftime("%j")) + fday / 86_400
    return yr, days


def jday_datetime64(
    array: npt.NDArray[np.datetime64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert a datetime64 array to Julian date components.

    Args:
        array: Array of datetime64 values.

    Returns:
        A tuple of (integer Julian day, fractional day) arrays.
    """
    times = (
        (array - np.datetime64("1957-01-01", "us")).astype("i8") / 86_400 / 1_000_000
    )
    jd = np.floor(times)
    fr = times - jd
    jd += JDAY_1957
    return jd, fr


def time_to_dt64(time: skyfield.timelib.Time) -> npt.NDArray[np.datetime64]:
    """Convert a skyfield Time to a datetime64 array.

    Leap seconds are folded into the resulting datetimes.

    Args:
        time: A skyfield Time object (array).

    Returns:
        An array of datetime64 values with microsecond resolution.
    """
    dt, ls = time.utc_datetime_and_leap_second()
    dt = [
        a.replace(tzinfo=None) + datetime.timedelta(seconds=int(b))
        for a, b in zip(dt, ls)
    ]
    return np.array(dt, dtype=EPOCH_DTYPE)


def tle_epoch(tle: TLETuple) -> float:
    """Get the epoch from a TLE as a float, adjusted for Y2K.

    The returned value encodes the full year and fractional day-of-year
    (e.g. ``2025032.5`` for noon on Feb 1, 2025).

    Args:
        tle: A (line1, line2) TLE tuple.

    Returns:
        The epoch as a year-prefixed float (e.g. ``2025032.0``).
    """
    epoch = float(tle[0][18:32].replace(" ", "0"))
    epoch += 1900_000 if epoch // 1000 >= 57 else 2000_000
    return epoch


def tle_date(tle: TLETuple) -> str:
    """Get the date from a TLE as a ``YYYYMMDD`` string.

    Args:
        tle: A (line1, line2) TLE tuple.

    Returns:
        The epoch date formatted as ``YYYYMMDD``.
    """
    epoch = tle_epoch(tle)
    year, doy = divmod(epoch, 1000)
    doy = doy // 1
    dt = datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=int(doy - 1))
    return dt.strftime("%Y%m%d")


def tle_satnum(tle: TLETuple) -> str:
    """Extract the satellite catalog number from a TLE.

    Args:
        tle: A (line1, line2) TLE tuple.

    Returns:
        The 5-character Alpha-5 satellite number.
    """
    return tle[0][2:7].replace(" ", "0")


GroupByKey = TypeVar("GroupByKey")


def group_by(
    tles: list[TLETuple], key: Callable[[TLETuple], GroupByKey]
) -> dict[GroupByKey, list[TLETuple]]:
    """Group TLEs by values produced by a key function.

    Args:
        tles: List of (line1, line2) TLE tuples.
        key: A callable that maps a TLE to a grouping key.

    Returns:
        A dict mapping each key value to its list of TLEs.
    """
    results: dict[GroupByKey, list[TLETuple]] = {}
    for tle in tles:
        group = key(tle)
        if group not in results:
            results[group] = []
        results[group].append(tle)
    return results


T = TypeVar("T")


def unique(tles: Iterable[T]) -> list[T]:
    """Remove duplicate entries while preserving order.

    Args:
        tles: An iterable of hashable items.

    Returns:
        A list with duplicates removed, in first-seen order.
    """
    return list(dict.fromkeys(tles).keys())
