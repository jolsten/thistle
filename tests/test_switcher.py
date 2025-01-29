import datetime
from itertools import pairwise
from typing import Type

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from thistle._utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    DATETIME_MAX,
    DATETIME_MIN,
    datetime_to_dt64,
    dt64_to_datetime,
    trange,
)
from thistle.reader import parse_tle_file, read_tle_file
from thistle.switcher import (
    EpochSwitcher,
    MidpointSwitcher,
    TLESwitcher,
    slices_by_transitions,
)

from . import strategies as cst
from .strategies import datetime_bounds

np.set_printoptions(linewidth=300)

BASIC_TIMES = trange(
    datetime.datetime(2000, 1, 1, 0), datetime.datetime(2000, 1, 2, 0), step=360
)
BASIC_SATRECS = parse_tle_file("tests/data/25544.tle")


class SwitcherBasic:
    class_: Type[TLESwitcher]

    def setup_class(self):
        self.switcher = self.class_(BASIC_SATRECS)
        self.switcher.compute_transitions()

    def test_switcher_transition_count(self):
        # One transition per satrec, plus one  after
        assert len(self.switcher.transitions) == len(BASIC_SATRECS) + 1

    def test_switcher_first_epoch(self):
        assert self.switcher.transitions[0] == DATETIME64_MIN

    def test_switcher_last_epoch(self):
        assert self.switcher.transitions[-1] == DATETIME64_MAX


class TestEpochSwitcherBasic(SwitcherBasic):
    class_ = EpochSwitcher

    def test_transitions(self):
        for idx, t in enumerate(self.switcher.transitions[1:-1]):
            # First Satrec period of validity starts at -inf
            # (ergo its epoch should not be a transition time)
            epoch = sat_epoch_datetime(self.switcher.satrecs[idx + 1]).replace(
                tzinfo=None
            )
            assert epoch == dt64_to_datetime(t)


class TestMidpointSwitcherBasic(SwitcherBasic):
    class_ = MidpointSwitcher

    def test_transitions(self):
        print(type(self.switcher))
        print([sat_epoch_datetime(sat) for sat in self.switcher.satrecs[1:3]])
        print(self.switcher.transitions[1])
        for idx, bounds in enumerate(pairwise(self.switcher.transitions)):
            time_a, time_b = [dt64_to_datetime(t) for t in bounds]
            # Midpoints should be between Satrecs on either side
            # idx1 is between a and b
            epoch = sat_epoch_datetime(self.switcher.satrecs[idx]).replace(tzinfo=None)
            # print(time_a, epoch, time_b)
            assert time_a < epoch
            assert epoch < time_b
        assert False


@given(cst.transitions(), cst.times())
def test_slices(
    transitions: np.ndarray[np.datetime64], times: np.ndarray[np.datetime64]
):
    slices = slices_by_transitions(transitions, times)
    print("=" * 20)
    for idx, slc_ in slices:
        print("a", transitions[idx])
        print("b", times[slc_])
        print("c", transitions[idx + 1])
        print("-" * 20)
        assert (transitions[idx] <= times[slc_]).all()
        assert (times[slc_] < transitions[idx + 1]).all()

    # assert False


def test_transitions_epoch_switcher():
    file = "tests/data/25544.tle"
    tles = read_tle_file(file)
    satrecs = [Satrec.twoline2rv(tle[0], tle[1]) for tle in tles]

    switcher = EpochSwitcher(satrecs)
    switcher.compute_transitions()

    for sat, time in zip(satrecs, switcher.transitions[1:-1]):
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
