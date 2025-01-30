from itertools import pairwise
from typing import Type

import numpy as np
from hypothesis import given
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from thistle.switcher import (
    EpochSwitcher,
    MidpointSwitcher,
    TLESwitcher,
)
from thistle.utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    dt64_to_datetime,
)

from . import strategies as cst
from .conftest import ISS_SATRECS

np.set_printoptions(linewidth=300)


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


class SwitcherBasic:
    class_: Type[TLESwitcher]

    def setup_class(self):
        self.switcher = self.class_(ISS_SATRECS)
        self.switcher.compute_transitions()

    def test_switcher_transition_count(self):
        # One transition per satrec, plus one  after
        assert len(self.switcher.transitions) == len(ISS_SATRECS) + 1

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
        for idx, bounds in enumerate(pairwise(self.switcher.transitions)):
            time_a, time_b = [dt64_to_datetime(t) for t in bounds]
            # Midpoints should be between Satrecs on either side idx1 is between a and b
            # less than or equal to is required in the case of two consecutive, identical epochs
            epoch = sat_epoch_datetime(self.switcher.satrecs[idx]).replace(tzinfo=None)
            assert time_a <= epoch
            assert epoch <= time_b
