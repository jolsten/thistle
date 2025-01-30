import datetime

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
    datetime_to_yy_days,
    dt64_to_datetime,
    jday_datetime64,
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
def test_datetime_to_yy_days(dt: datetime.datetime, yy: int, days: float):
    got_yy, got_days = datetime_to_yy_days(dt.replace(tzinfo=datetime.timezone.utc))
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
