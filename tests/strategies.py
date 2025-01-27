import datetime

from hypothesis import assume
from hypothesis import strategies as st

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
