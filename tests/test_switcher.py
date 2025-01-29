import datetime
from itertools import pairwise
from typing import Type

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from thistle._utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    datetime_to_dt64,
    dt64_to_datetime,
    trange,
)
from thistle.reader import parse_tle_file
from thistle.switcher import (
    EpochSwitcher,
    MidpointSwitcher,
    TLESwitcher,
    slices_by_transitions,
)

from . import strategies as cst

np.set_printoptions(linewidth=300)

BASIC_TIMES = trange(
    datetime.datetime(2000, 1, 1, 0), datetime.datetime(2000, 1, 2, 0), step=360
)
BASIC_SATRECS = parse_tle_file("tests/data/25544.tle")


@given(cst.satrec_lists())
def test_midpoint_switcher(satrec_list: list[Satrec]) -> None:
    switcher = MidpointSwitcher(satrec_list)
    switcher.compute_transitions()

    for idx, bounds in enumerate(pairwise(switcher.transitions)):
        time_a, time_b = [dt64_to_datetime(t) for t in bounds]
        # Midpoints should be between Satrecs on either side
        # idx1 is between a and b
        epoch = sat_epoch_datetime(switcher.satrecs[idx]).replace(tzinfo=None)
        assert time_a <= epoch
        assert epoch <= time_b

    # assert False


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

    def test_example_1(self):
        line1 = "2 25544 051.5927 164.4358 0123823 089.5260 271.9768 16.05621877000127"
        line2 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"

        sat = Satrec.twoline2rv(line1, line2)
        dt64 = datetime_to_dt64(sat_epoch_datetime(sat))

        times = np.array([dt64])
        tle = self.switcher.select_satrec()


class TestMidpointSwitcherBasic(SwitcherBasic):
    class_ = MidpointSwitcher

    def test_transitions(self):
        for idx, bounds in enumerate(pairwise(self.switcher.transitions)):
            time_a, time_b = [dt64_to_datetime(t) for t in bounds]
            # Midpoints should be between Satrecs on either side idx1 is between a and b
            # less than or equal to is required in the case of two consecutive, identical epochs
            epoch = sat_epoch_datetime(self.switcher.satrecs[idx]).replace(tzinfo=None)
            assert time_a <= epoch
            assert epoch <= time_b


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


def test_specific_epoch_switcher():
    line1 = "2 25544 051.5927 164.4358 0123823 089.5260 271.9768 16.05621877000127"
    line2 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"

    sat = Satrec.twoline2rv(line1, line2)
    epoch = sat_epoch_datetime(sat)

    switcher = EpochSwitcher(BASIC_SATRECS)
    switcher.compute_transitions()
