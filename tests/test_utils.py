from thistle._utils import DATETIME64_MAX, DATETIME64_MIN, DATETIME_MAX, DATETIME_MIN, TIME_SCALE, datetime_to_dt64, dt64_to_datetime
import datetime
from hypothesis import given, strategies as st
import numpy as np

@given(st.datetimes(min_value=DATETIME_MIN, max_value=DATETIME_MAX))
def test_convert_1(time_in: datetime.datetime):
    assert time_in == dt64_to_datetime(datetime_to_dt64(time_in))

@given(st.integers(min_value=DATETIME64_MIN.astype(int), max_value=DATETIME64_MAX.astype(int)))
def test_convert_2(integer: int):
    dt64 = np.datetime64(integer, TIME_SCALE)
    assert dt64 == datetime_to_dt64(dt64_to_datetime(dt64))
