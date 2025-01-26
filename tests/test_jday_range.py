import datetime

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from sgp4.conveniences import jday_datetime
from thistle.store import jday_range

TOLERANCE = 1e-6


def test_jday_range_single_example():
    start = datetime.datetime(2019, 12, 9, 20, 42, 0)
    stop = datetime.datetime(2019, 12, 9, 20, 42, 0)
    jd, fr = jday_range(start, stop)
    assert float(jd[0]) == pytest.approx(2458826.5)
    assert float(fr[0]) == pytest.approx(0.8625)


@given(
    st.datetimes(
        min_value=datetime.datetime(1970, 1, 1),
        max_value=datetime.datetime(2056, 12, 31),
    )
)
def test_jday_range_single_time(time: datetime.datetime):
    jd, fr = jday_range(time, time)
    ex_jd, ex_fr = jday_datetime(time.replace(tzinfo=datetime.UTC))
    print(time.isoformat(sep="T", timespec="milliseconds"), jd[0], fr[0], ex_jd, ex_fr)
    assert float(jd[0]) == pytest.approx(ex_jd, abs=TOLERANCE)
    assert float(fr[0]) == pytest.approx(ex_fr, abs=TOLERANCE)


@st.composite
def date_range(
    draw,
    min_value: datetime.datetime = datetime.datetime(1957, 1, 1),
    max_value: datetime.datetime = datetime.datetime(2056, 12, 31),
    max_duration: float = 86400.0,
) -> tuple[datetime.datetime, datetime.datetime]:
    start = draw(st.datetimes(min_value=min_value, max_value=max_value))
    duration = draw(st.floats(min_value=0.001, max_value=max_duration))
    stop = start + datetime.timedelta(seconds=duration)

    assume(start < stop)
    assume((stop - start).total_seconds() > 0)
    assume(stop < max_value)

    return start, stop


@given(date_range())
def test_jday_range(date_range: tuple[datetime.datetime, datetime.datetime]):
    NUM_SAMPLES = 10
    start, stop = date_range
    step = (stop - start).total_seconds() / (NUM_SAMPLES - 1)
    jd, fr = jday_range(start, stop, step)
    start_jd, start_fr = jday_datetime(start.replace(tzinfo=datetime.UTC))
    stop_jd, stop_fr = jday_datetime(stop.replace(tzinfo=datetime.UTC))
    print(
        start.isoformat(sep="T", timespec="milliseconds"),
        stop.isoformat(sep="T", timespec="milliseconds"),
        step,
    )
    # Ensure first time value is the input "start" time
    assert float(jd[0]) == pytest.approx(start_jd, abs=TOLERANCE)
    assert float(fr[0]) == pytest.approx(start_fr, abs=TOLERANCE)

    # Esnure final time value is within one step of the input "stop" time
    stop_jday_exp = stop_jd + stop_fr
    stop_jday_out = jd[-1] + fr[-1]
    assert abs(stop_jday_exp - stop_jday_out) < (step * 86400)

    # Ensure vector list is appropriate size, plus/minus 1
    assert len(jd) == pytest.approx(NUM_SAMPLES, abs=1)
    assert len(fr) == pytest.approx(NUM_SAMPLES, abs=1)
