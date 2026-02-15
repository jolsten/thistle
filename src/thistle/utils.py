"""Utility functions for time conversion, TLE parsing, and collection helpers."""

import datetime
import itertools
from typing import Any, Callable, Iterable, Sequence, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import skyfield.timelib

from thistle.typing import DateTime, PathLike, TLETuple


def read_tle(
    file: PathLike,
) -> list[TLETuple]:
    """Read a single TLE file.

    Parses a file containing Two-Line Element sets, extracting line 1/line 2
    pairs based on the leading character of each line.

    Args:
        file: Path to the TLE file.

    Returns:
        A list of (line1, line2) tuples for each TLE in the file.
    """
    results: list[TLETuple] = []
    with open(file, "r") as f:
        line1 = None
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line[0] == "1":
                line1 = line
            elif line[0] == "2" and line1 is not None:
                results.append((line1, line))
                line1 = None
    return results


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
"""Numpy time resolution used throughout the package (microseconds)."""

EPOCH_DTYPE = np.dtype(f"datetime64[{TIME_SCALE}]")
"""Numpy dtype for datetime64 values at :data:`TIME_SCALE` resolution."""

ONE_SECOND_IN_TIME_SCALE = np.timedelta64(1, "s").astype(f"timedelta64[{TIME_SCALE}]")
"""One second expressed as a timedelta64 at :data:`TIME_SCALE` resolution."""

DATETIME_MIN = datetime.datetime(1957, 1, 1)
"""Earliest representable datetime (start of the space age)."""

DATETIME_MAX = datetime.datetime(2056, 12, 31, 23, 59, 59, 999999)
"""Latest representable datetime (end of the TLE two-digit year range)."""

DATETIME64_MIN = np.datetime64(DATETIME_MIN, TIME_SCALE)
"""Earliest representable datetime as a datetime64."""

DATETIME64_MAX = np.datetime64(DATETIME_MAX, TIME_SCALE)
"""Latest representable datetime as a datetime64."""

JDAY_1957 = 2435839.5
"""Julian date of 1957-01-01 00:00:00 UTC."""


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

    Raises:
        TypeError: If value is not a datetime or datetime64.
    """
    if isinstance(value, datetime.datetime):
        value = value.replace(tzinfo=None)
        return np.datetime64(value, TIME_SCALE)
    if isinstance(value, np.datetime64):
        return value.astype(f"datetime64[{TIME_SCALE}]")
    raise TypeError(f"Expected datetime or datetime64, got {type(value).__name__}")


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
    dt_arr, ls_arr = time.utc_datetime_and_leap_second()
    dt = [
        a.replace(tzinfo=None) + datetime.timedelta(seconds=int(b))
        for a, b in zip(
            cast(Sequence[datetime.datetime], dt_arr),
            cast(Sequence[float], ls_arr),
        )
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


# ---------------------------------------------------------------------------
# Alpha-5 satellite catalog number encoding/decoding
# ---------------------------------------------------------------------------

_A_TO_I = {
    "A": 10,
    "B": 11,
    "C": 12,
    "D": 13,
    "E": 14,
    "F": 15,
    "G": 16,
    "H": 17,
    "J": 18,
    "K": 19,
    "L": 20,
    "M": 21,
    "N": 22,
    "P": 23,
    "Q": 24,
    "R": 25,
    "S": 26,
    "T": 27,
    "U": 28,
    "V": 29,
    "W": 30,
    "X": 31,
    "Y": 32,
    "Z": 33,
}
_I_TO_A = {val: key for key, val in _A_TO_I.items()}


def to_alpha5(satnum: int) -> str:
    """Encode an integer satellite number to an Alpha-5 string.

    Numbers 0-99,999 are zero-padded to 5 digits. Numbers 100,000-339,999
    use a letter prefix (A-Z, skipping I and O) followed by 4 digits.

    Args:
        satnum: Satellite catalog number (0 to 339,999).

    Returns:
        A 5-character Alpha-5 encoded string.

    Raises:
        ValueError: If satnum is outside the valid range.
    """
    if satnum < 0 or satnum > 339_999:
        msg = "Alpha-5 satnum must be >= 0 and <= 333,999 (encoded as Z9999)"
        raise ValueError(msg)

    if satnum < 100_000:
        return f"{satnum:05}"

    a, b = divmod(satnum, 10_000)
    return f"{_I_TO_A[a]}{b:04}"


def from_alpha5(satnum: str) -> int:
    """Decode an Alpha-5 string to an integer satellite number.

    Args:
        satnum: A 5-character Alpha-5 encoded string.

    Returns:
        The decoded integer satellite catalog number.
    """
    satnum = str(satnum)
    if satnum[0].isnumeric():
        return int(satnum)
    return _A_TO_I[satnum[0]] * 10_000 + int(satnum[1:])


def ensure_alpha5(satnum: Union[int, str]) -> str:
    """Ensure a satellite number is in Alpha-5 string format.

    If the input is an integer it is encoded; if it is already a string
    it is returned as-is.

    Args:
        satnum: An integer or string satellite number.

    Returns:
        The Alpha-5 encoded string.

    Raises:
        TypeError: If satnum is neither an int nor a str.
    """
    if isinstance(satnum, int):
        return to_alpha5(satnum)
    elif isinstance(satnum, str):
        return satnum
    raise TypeError
