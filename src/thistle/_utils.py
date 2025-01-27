import datetime
import itertools
from typing import Iterable

import numpy as np

TIME_SCALE = "us"
EPOCH_DTYPE = np.dtype(f"datetime64[{TIME_SCALE}]")
ONE_SECOND_IN_TIME_SCALE = np.timedelta64(1, "s").astype(f"timedelta64[{TIME_SCALE}]")

DATETIME_MIN = datetime.datetime(1957,1,1)
DATETIME_MAX = datetime.datetime(2056,12,31,23,59,59,999999)
DATETIME64_MIN = np.datetime64(DATETIME_MIN.isoformat(sep="T", timespec="microseconds"))
DATETIME64_MAX = np.datetime64(DATETIME_MAX.isoformat(sep="T", timespec="microseconds"))

def datetime_to_dt64(dt: datetime.datetime) -> np.datetime64:
    dt = dt.replace(tzinfo=None)
    return np.datetime64(dt.isoformat(sep="T", timespec="microseconds"), TIME_SCALE)


def dt64_to_datetime(dt: np.datetime64) -> datetime.datetime:
    return datetime.datetime.fromisoformat(str(dt))


def pairwise(iterable: Iterable) -> Iterable:
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def trange(
    start: datetime.datetime, stop: datetime.datetime, step: float
) -> np.ndarray[np.datetime64]:
    times = np.arange(
        datetime_to_dt64(start),
        datetime_to_dt64(stop),
        step * ONE_SECOND_IN_TIME_SCALE,
    )
    return times
