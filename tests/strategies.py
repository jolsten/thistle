import datetime

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st
from sgp4.api import Satrec
from thistle.utils import (
    DATETIME_MAX,
    DATETIME_MIN,
    datetime_to_dt64,
    datetime_to_yy_days,
    trange,
)

GENERIC_TLE = [
    "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996",
    "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05",
]


@st.composite
def satrecs(
    draw,
    min_epoch: datetime.datetime = DATETIME_MIN,
    max_epoch: datetime.datetime = DATETIME_MAX,
) -> Satrec:
    sat = Satrec.twoline2rv(*GENERIC_TLE)
    dt = draw(st.datetimes(min_value=min_epoch, max_value=max_epoch))
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    sat.epochyr, sat.epochdays = datetime_to_yy_days(dt)
    return sat


def unique_by_epoch(sat: Satrec) -> float:
    return sat.jdsatepoch + sat.jdsatepochF


@st.composite
def satrec_lists(
    draw,
    min_epoch: datetime.datetime = DATETIME_MIN,
    max_epoch: datetime.datetime = DATETIME_MAX,
    min_size: int = 1,
    max_size: int = 100,
) -> list[Satrec]:
    satrec_list = draw(
        st.lists(
            satrecs(min_epoch=min_epoch, max_epoch=max_epoch),
            min_size=min_size,
            max_size=max_size,
            unique_by=unique_by_epoch,
        )
    )
    satrec_list = sorted(satrec_list, key=unique_by_epoch)
    return satrec_list


@st.composite
def datetime_bounds(
    draw, min_value=DATETIME_MIN, max_value=DATETIME_MAX
) -> tuple[datetime.datetime, datetime.datetime]:
    t0 = draw(st.datetimes(min_value=min_value, max_value=max_value))
    td = draw(
        st.timedeltas(
            min_value=datetime.timedelta(seconds=10),
            max_value=datetime.timedelta(days=30),
        )
    )
    t1 = t0 + td
    assume(t1 < DATETIME_MAX)
    return t0, t1


@st.composite
def times(draw, min_size: int = 1, max_size: int = 10) -> np.ndarray[np.datetime64]:
    t0, t1 = draw(datetime_bounds())
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    step = (t1 - t0).total_seconds() / size
    times = trange(t0, t1, step)
    return times


@st.composite
def transitions(
    draw,
    min_value: datetime.datetime = DATETIME_MIN,
    max_value: datetime.datetime = DATETIME_MAX,
    min_count: int = 1,
    max_count: int = 3,
) -> np.ndarray[np.datetime64]:
    t0, t1 = draw(datetime_bounds(min_value=min_value, max_value=max_value))
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
