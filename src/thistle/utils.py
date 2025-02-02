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
COUNTS_PER_DAY = np.int64(86_400_000_000)

DATETIME_MIN = datetime.datetime(1957, 1, 1)
DATETIME_MAX = datetime.datetime(2056, 12, 31, 23, 59, 59, 999999)
DATETIME64_MIN = np.datetime64(DATETIME_MIN, TIME_SCALE)
DATETIME64_MAX = np.datetime64(DATETIME_MAX, TIME_SCALE)

JDAY_1957 = 2435839.5
JDAY_1970 = 2440587.5
JDAY_2057 = 2472364.5

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



def dt64_to_jday(
    array: np.ndarray[np.datetime64],
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    array = array.astype("datetime64[us]") - np.datetime64("1957-01-01", "us")

    jd, fr = np.divmod(array.astype("i8"), COUNTS_PER_DAY)
    jd = jd.astype("f8") + JDAY_1957
    fr = fr.astype("f8") / COUNTS_PER_DAY

    # quo1, rem1 = np.divmod(fr, np.int64(86_400))
    # quo2, rem2 = np.divmod(quo1, np.int64(1_000_000))

    # print(quo1, rem1, quo2, rem2)

    # a = quo2.astype("f8")
    # b = rem2.astype("f8") / np.float64(1_000_000)
    # c = rem1.astype("f8") / np.float64(86_400_000_000)
 
    # print(a, b, c)

    # fr = quo2.astype("f8") + rem2.astype("f8") / np.float64(1_000_000) + rem1.astype("f8") / np.float64(86_400_000_000)

    hard = fr < 0
    jd[hard] = jd[hard] - 1
    fr[hard] = fr[hard] + 1
    
    return jd, fr


def jday_to_dt64(jd: np.ndarray[np.float64], fr: np.ndarray[np.float64]) -> np.datetime64:
    jd = jd - JDAY_1970
    jd = (jd + fr) * 86_400.0
    return (np.round(jd).astype("i8") * 1_000_000).astype(EPOCH_DTYPE)
