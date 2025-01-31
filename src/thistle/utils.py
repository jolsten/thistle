import datetime
import itertools
from typing import Any, Iterable

import numpy as np


def pairwise_recipe(iterable: Iterable) -> Iterable[tuple[Any, Any]]:
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
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
    dt = dt.replace(tzinfo=None)
    return np.datetime64(dt, TIME_SCALE)


def dt64_to_datetime(dt: np.datetime64) -> datetime.datetime:
    return datetime.datetime.fromisoformat(str(dt))


def trange(
    start: datetime.datetime, stop: datetime.datetime, step: float
) -> np.ndarray[np.datetime64]:
    times = np.arange(
        datetime_to_dt64(start),
        datetime_to_dt64(stop),
        step * ONE_SECOND_IN_TIME_SCALE,
    )
    return times


def datetime_to_yy_days(dt: datetime.datetime) -> tuple[int, float]:
    midnight = datetime.datetime.combine(
        dt.date(), datetime.time(0, 0, 0), tzinfo=datetime.timezone.utc
    )
    fday = (dt - midnight).total_seconds()
    yr = int(dt.strftime("%y"))
    days = int(dt.strftime("%j")) + fday / 86_400
    return yr, days


def jday_datetime64(
    array: np.ndarray[np.datetime64],
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    times = (
        (array - np.datetime64("1957-01-01", "us")).astype("i8") / 86_400 / 1_000_000
    )
    jd = np.floor(times)
    fr = times - jd
    jd += JDAY_1957
    return jd, fr
