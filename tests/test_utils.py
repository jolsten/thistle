import datetime
from typing import Callable, Iterable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sgp4.conveniences import jday_datetime

from thistle.utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    DATETIME_MAX,
    DATETIME_MIN,
    TIME_SCALE,
    datetime_to_dt64,
    datetime_to_tle_epoch,
    dt64_to_datetime,
    group_by,
    jday_datetime64,
    tle_date,
    tle_epoch,
    tle_satnum,
    unique,
)


@given(st.datetimes(min_value=DATETIME_MIN, max_value=DATETIME_MAX))
def test_convert_1(time_in: datetime.datetime):
    assert time_in == dt64_to_datetime(datetime_to_dt64(time_in))


@given(
    st.integers(
        min_value=DATETIME64_MIN.astype(int), max_value=DATETIME64_MAX.astype(int)
    )
)
def test_convert_2(integer: int):
    dt64 = np.datetime64(integer, TIME_SCALE)
    assert dt64 == datetime_to_dt64(dt64_to_datetime(dt64))


@pytest.mark.parametrize(
    "dt, yy, days",
    [
        (datetime.datetime(2000, 1, 1, 0, 0, 0), 0, 1.0),
        (datetime.datetime(2000, 1, 1, 12, 0, 0), 0, 1.5),
        (datetime.datetime(2000, 1, 2, 0, 0, 0), 0, 2.0),
        (datetime.datetime(2001, 1, 1, 0, 0, 0), 1, 1.0),
        (datetime.datetime(1957, 1, 1, 0, 0, 0), 57, 1.0),
    ],
)
def test_datetime_to_tle_epoch(dt: datetime.datetime, yy: int, days: float):
    got_yy, got_days = datetime_to_tle_epoch(dt.replace(tzinfo=datetime.timezone.utc))
    assert got_yy == yy
    assert got_days == days


@given(
    st.lists(
        st.datetimes(
            min_value=DATETIME_MIN,
            max_value=DATETIME_MAX,
            timezones=st.sampled_from([datetime.timezone.utc]),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_jday_datetime64(dt_list: list[datetime.datetime]) -> None:
    exp_jd, exp_fr = [], []
    for dt in dt_list:
        jd, fr = jday_datetime(dt)
        exp_jd.append(jd)
        exp_fr.append(fr)
    exp_jd = np.array(exp_jd, dtype="f8")
    exp_fr = np.array(exp_fr, dtype="f8")

    times = np.array([datetime_to_dt64(dt) for dt in dt_list], dtype="datetime64[us]")
    jd, fr = jday_datetime64(times)

    assert jd == pytest.approx(exp_jd.tolist())
    assert fr == pytest.approx(exp_fr.tolist())


@pytest.mark.parametrize(
    "epoch_str, date, epoch, satnum",
    [
        ("57001.00000000", "19570101", 1957001.0, "25544"),
        ("25032.00000000", "20250201", 2025032.0, "25544"),
        ("56366.00000000", "20561231", 2056366.0, "A0001"),
    ],
)
class TestTLEFuncs:
    def test_tle_epoch(self, epoch_str: str, date: str, epoch: float, satnum: str):
        line1 = f"1 25544U 98067A   {epoch_str}  .00020137  00000-0  16538-3 0  9993"
        line2 = "2 25544  51.6335 344.7760 0007976 126.2523 325.9359 15.70406856328906"
        tle = (line1, line2)
        assert epoch == tle_epoch(tle)

    def test_tle_date(self, epoch_str: str, date: str, epoch: float, satnum: str):
        line1 = f"1 25544U 98067A   {epoch_str}  .00020137  00000-0  16538-3 0  9993"
        line2 = "2 25544  51.6335 344.7760 0007976 126.2523 325.9359 15.70406856328906"
        tle = (line1, line2)
        assert tle_date(tle) == date

    def test_tle_satnum(self, epoch_str: str, date: str, epoch: float, satnum: str):
        line1 = f"1 {satnum:5}U 98067A   25077.00000000  .00020137  00000-0  16538-3 0  9993"
        line2 = f"2 {satnum:5}  51.6335 344.7760 0007976 126.2523 325.9359 15.70406856328906"
        tle = (line1, line2)
        assert tle_satnum(tle) == satnum


@pytest.mark.parametrize(
    "value, exp",
    [
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZZZZZZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ([1, 1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ],
)
def test_unique(value: Iterable, exp: Iterable):
    value, exp = list(value), list(exp)
    assert unique(value) == exp


@pytest.mark.parametrize(
    "value, exp, key",
    [
        (
            [1, 1.5, 2, 2.5, 3, 3.5],
            {1: [1, 1.5], 2: [2, 2.5], 3: [3, 3.5]},
            lambda x: x // 1,
        ),
        ("AaBb", {"A": ["A", "a"], "B": ["B", "b"]}, lambda x: x.upper()),
    ],
)
def test_group_by(value: Iterable, exp: dict, key: Callable):
    assert group_by(value, key=key) == exp
