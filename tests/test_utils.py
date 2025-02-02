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
    JDAY_1957,
    JDAY_1970,
    JDAY_2057,
    TIME_SCALE,
    datetime_to_dt64,
    datetime_to_yy_days,
    dt64_to_datetime,
    dt64_to_jday,
    jday_to_dt64,
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


# @given(
#     st.lists(
#         st.datetimes(
#             min_value=DATETIME_MIN,
#             max_value=DATETIME_MAX,
#             timezones=st.sampled_from([datetime.timezone.utc]),
#         ),
#         min_size=1,
#         max_size=100,
#     )
# )
# def test_dt64_to_jday(dt_list: list[datetime.datetime]) -> None:
#     exp_jd, exp_fr = [], []
#     for dt in dt_list:
#         jd, fr = jday_datetime(dt)
#         exp_jd.append(jd)
#         exp_fr.append(fr)
#     exp_jd = np.array(exp_jd, dtype="f8")
#     exp_fr = np.array(exp_fr, dtype="f8")

#     times = np.array([datetime_to_dt64(dt) for dt in dt_list], dtype="datetime64[us]")
#     jd, fr = dt64_to_jday(times)

#     assert jd == pytest.approx(exp_jd.tolist())
#     assert fr == pytest.approx(exp_fr.tolist())




@pytest.mark.parametrize("dt64, jd, fr", [
    # ("1957-01-01T00:00", JDAY_1957, 0.000000),
    ("1957-01-01T06:00", JDAY_1957, 0.250000),
    ("1957-01-01T12:00", JDAY_1957, 0.500000),
    ("1957-01-01T18:00", JDAY_1957, 0.750000),

    # ("1970-01-01T00:00", JDAY_1970, 0.000000),
    ("1970-01-01T06:00", JDAY_1970, 0.250000),
    ("1970-01-01T12:00", JDAY_1970, 0.500000),
    ("1970-01-01T18:00", JDAY_1970, 0.750000),


    ("2001-09-11T08:46", 2452163.5, 0.36527777777777776), # F Around
    ("2011-05-01T20:00", 2455682.5, 0.83333333333333333), # Find Out
])
class TestJDayDateTime64:
    def test_forward(self, dt64: str, jd: float, fr: float):
        dt64 = np.array([dt64], dtype="datetime64[us]")
        jd = np.array([jd], dtype="f8")
        fr = np.array([fr], dtype="f8")

        got_jd, got_fr = dt64_to_jday(dt64)
        assert jd.tolist() == pytest.approx(got_jd.tolist(), abs=1e-6)
        assert fr.tolist() == pytest.approx(got_fr.tolist(), abs=1e-6)

    def test_reverse(self, dt64: str, jd: float, fr: float):
        dt64 = np.array([dt64], dtype="datetime64[us]")
        jd = np.array([jd], dtype="f8")
        fr = np.array([fr], dtype="f8")

        got_dt64 = jday_to_dt64(jd, fr)
        print(dt64, got_dt64)
        assert dt64.tolist() == got_dt64.tolist()


@given(st.datetimes(min_value=DATETIME_MIN, max_value=DATETIME_MAX))
def test_jday_to_dt64(time: datetime.datetime):
    dt64 = np.array([time], dtype="datetime64[us]")
    jd, fr = dt64_to_jday(dt64)
    got_dt64 = jday_to_dt64(jd, fr)
    print(dt64, got_dt64)
    assert dt64.tolist() == got_dt64.tolist()


@st.composite
def julian_midnights(draw, min_value=JDAY_1957, max_value=JDAY_2057) -> float:
    jday = draw(st.integers(min_value=int(min_value), max_value=int(max_value)))
    return jday + 0.5


@given(julian_midnights(), st.floats(min_value=0, max_value=0.99999999))
def test_dt64_to_jday(jd: float, fr: float):
    jd = np.array([jd], "f8")
    fr = np.array([fr], "f8")
    dt64 = jday_to_dt64(jd, fr)
    got_jd, got_fr = dt64_to_jday(dt64)

    assert jd.tolist() == got_jd.tolist()
    assert fr.tolist() == got_fr.tolist()
