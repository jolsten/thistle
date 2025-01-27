import datetime

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from thistle._utils import datetime_to_dt64, dt64_to_datetime, trange
from thistle.reader import read_tle_file
from thistle.switcher import EpochSwitcher, MidpointSwitcher, slices_by_transitions

from .strategies import datetime_bounds


@given(st.data(), datetime_bounds())
def test_slices(data, bounds: tuple[datetime.datetime, datetime.datetime]):
    t0, t1 = bounds
    step = (t1 - t0).total_seconds() / 100
    times = trange(t0, t1, step)
    transitions = data.draw(
        st.lists(
            st.datetimes(min_value=t0, max_value=t1),
            min_size=1,
            max_size=3,
            unique_by=lambda x: x,
        )
    )
    transitions = sorted(transitions)
    transitions = np.atleast_1d(
        np.array([datetime_to_dt64(t) for t in transitions], dtype="datetime64[us]")
    )
    print("=" * 20)
    print(t0, "step =", step)
    # print(transitions)
    for t in transitions:
        print(" ", t)
    print(t1)
    print("-" * 20)
    slices = slices_by_transitions(transitions, times)
    for idx, slc_ in slices:
        print(transitions[idx])
        print(" ", times[slc_][0], times[slc_][-1], len(times[slc_]))
    # print(slices)

    # for val in times[slice_]:
    #     assert val < transitions

    assert False


def test_transitions_epoch_switcher():
    file = "tests/data/25544.tle"
    tles = read_tle_file(file)
    satrecs = [Satrec.twoline2rv(tle[0], tle[1]) for tle in tles]

    switcher = EpochSwitcher(satrecs)
    switcher.compute_transitions()
    for sat, time in zip(satrecs[1:], switcher.transitions):
        sat_epoch = (
            sat_epoch_datetime(sat)
            .replace(tzinfo=None)
            .isoformat(sep="T", timespec="milliseconds")
        )
        assert sat_epoch == str(time)[0:23]


def test_transitions_midpoint_switcher():
    file = "tests/data/25544.tle"
    tles = read_tle_file(file)
    satrecs = [Satrec.twoline2rv(tle[0], tle[1]) for tle in tles]

    switcher = MidpointSwitcher(satrecs)
    switcher.compute_transitions()
    for idx, t_time in enumerate(switcher.transitions):
        t_time = dt64_to_datetime(t_time)
        sat_a, sat_b = switcher.satrecs[idx : idx + 2]
        time_a = sat_epoch_datetime(sat_a).replace(tzinfo=None)
        time_b = sat_epoch_datetime(sat_b).replace(tzinfo=None)

        print(time_a, t_time, time_b)

        # transition time must be between t0 and t1
        assert time_a <= t_time
        assert t_time <= time_b

        # transition should be halfway between t0 and t1... duh
        dur1 = (t_time - time_a).total_seconds()
        dur2 = (time_b - t_time).total_seconds()
        assert dur1 == pytest.approx(dur2, abs=1e-5)
