import datetime

from thistle._utils import DATETIME_MAX, DATETIME_MIN, datetime_to_dt64, trange
from hypothesis import assume
from hypothesis import strategies as st
import numpy as np

DATETIME_1957 = datetime.datetime(1957, 1, 1)
DATETIME_2056 = datetime.datetime(2056, 12, 31)


@st.composite
def datetime_bounds(
    draw, min_value=DATETIME_1957, max_value=DATETIME_2056
) -> tuple[datetime.datetime, datetime.datetime]:
    t0 = draw(st.datetimes(min_value=min_value, max_value=max_value))
    td = draw(
        st.timedeltas(
            min_value=datetime.timedelta(seconds=10),
            max_value=datetime.timedelta(days=30),
        )
    )
    t1 = t0 + td
    assume(t1 < DATETIME_2056)
    return t0, t1


@st.composite
def times(draw, min_size: int = 1, max_size: int = 10) -> np.ndarray[np.datetime64]:
    t0, t1 = draw(datetime_bounds())
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    step = (t1 - t0).total_seconds() / size
    times = trange(t0, t1, step)
    return times


@st.composite
def transitions(draw, min_count: int = 1, max_count: int = 3):
    t0, t1 = draw(datetime_bounds())
    transitions = draw(
        st.lists(
            st.datetimes(min_value=t0, max_value=t1),
            min_size=min_count,
            max_size=max_count,
            unique_by=lambda x: x,
        )
    )
    transitions = [DATETIME_MIN] + sorted(transitions) + [DATETIME_MAX]
    transitions = np.atleast_1d(
        np.array([datetime_to_dt64(t) for t in transitions], dtype="datetime64[us]")
    )
    return transitions
