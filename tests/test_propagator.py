import datetime
from typing import Type

import numpy as np
import pytest
from hypothesis import given
from sgp4.api import Satrec
from sgp4.exporter import export_tle
from skyfield.api import EarthSatellite, load

from thistle.propagator import (
    EpochSwitchStrategy,
    MidpointSwitchStrategy,
    Propagator,
    SwitchingStrategy,
    _slices_by_transitions,
)
from thistle.utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    datetime_to_dt64,
    dt64_to_datetime,
    pairwise,
    trange,
)

from . import strategies as cst
from .conftest import ISS_SATRECS, ISS_TLES

UTC = datetime.timezone.utc

np.set_printoptions(linewidth=300)


@given(cst.transitions(), cst.times())
def test_slices(
    transitions: np.ndarray[np.datetime64], times: np.ndarray[np.datetime64]
):
    slices = _slices_by_transitions(transitions, times)
    for idx, slc_ in slices:
        assert (transitions[idx] <= times[slc_]).all()
        assert (times[slc_] < transitions[idx + 1]).all()


@given(cst.satrec_lists())
def test_midpoint_switcher(satrec_list: list[Satrec]) -> None:
    ts = load.timescale()
    satellite_list = [EarthSatellite.from_satrec(satrec, ts) for satrec in satrec_list]
    switcher = MidpointSwitchStrategy(satellite_list)
    switcher.compute_transitions()

    for idx, bounds in enumerate(pairwise(switcher.transitions)):
        time_a, time_b = [dt64_to_datetime(t) for t in bounds]
        # Midpoints should be between Satrecs on either side
        # idx1 is between a and b
        epoch = switcher.satellites[idx].epoch.utc_datetime().replace(tzinfo=None)
        assert time_a <= epoch
        assert epoch <= time_b


class SwitchStrategyBasic:
    class_: Type[SwitchingStrategy]

    def setup_class(self):
        self.ts = load.timescale()
        self.switcher = self.class_(
            [EarthSatellite.from_satrec(satrec, self.ts) for satrec in ISS_SATRECS]
        )
        self.switcher.compute_transitions()

    def test_switcher_transition_count(self):
        # One transition per satrec, plus one  after
        assert len(self.switcher.transitions) == len(ISS_SATRECS) + 1

    def test_switcher_first_epoch(self):
        assert self.switcher.transitions[0] == DATETIME64_MIN

    def test_switcher_last_epoch(self):
        assert self.switcher.transitions[-1] == DATETIME64_MAX


class TestEpochSwitchStrategy(SwitchStrategyBasic):
    class_ = EpochSwitchStrategy

    def test_transitions(self):
        for idx, t in enumerate(self.switcher.transitions[1:-1]):
            # First Satrec period of validity starts at -inf
            # (ergo its epoch should not be a transition time)
            epoch = (
                self.switcher.satellites[idx + 1]
                .epoch.utc_datetime()
                .replace(tzinfo=None)
            )
            assert epoch == dt64_to_datetime(t)


class TestMidpointSwitchStrategy(SwitchStrategyBasic):
    class_ = MidpointSwitchStrategy

    def test_transitions(self):
        for idx, bounds in enumerate(pairwise(self.switcher.transitions)):
            time_a, time_b = [dt64_to_datetime(t) for t in bounds]
            # Midpoints should be between Satrecs on either side idx1 is between a and b
            # less than or equal to is required in the case of two consecutive, identical epochs
            epoch = (
                self.switcher.satellites[idx].epoch.utc_datetime().replace(tzinfo=None)
            )
            assert time_a <= epoch
            assert epoch <= time_b


class PropagatorBaseClass:
    method: str

    def setup_class(self):
        self.ts = load.timescale()
        self.tles = ISS_TLES
        self.propagator = Propagator(ISS_TLES, method=self.method)


class TestPropagatorEpoch(PropagatorBaseClass):
    method: str = "epoch"

    def test_find_satrec_by_epoch(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        sat = self.propagator.find_satellite(datetime_to_dt64(dt))
        assert export_tle(sat.model) == export_tle(exp_sat.model)

    def test_at(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        sat = self.propagator.find_satellite(datetime_to_dt64(dt))
        times = trange(dt, dt + datetime.timedelta(seconds=60), 10)
        times = [dt64_to_datetime(t).replace(tzinfo=UTC) for t in times]
        times = self.ts.from_datetimes(times)

        exp_geo = exp_sat.at(times)
        geo = sat.at(times)

        assert geo.position.au.flatten().tolist() == pytest.approx(
            exp_geo.position.au.flatten().tolist()
        )
        assert geo.velocity.au_per_d.flatten().tolist() == pytest.approx(
            exp_geo.velocity.au_per_d.flatten().tolist()
        )
        assert geo.t.tt.flatten().tolist() == exp_geo.t.tt.flatten().tolist()


class TestPropagatorMidpoint(PropagatorBaseClass):
    method: str = "midpoint"

    def test_at(self):
        a1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        a2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"
        b1 = "1 25544U 98067A   98325.51671211  .01832406  18178-2  41610-2 0  9996"
        b2 = "2 25544 051.5928 161.7497 0074408 097.6565 263.2450 15.92278419   200"

        sat_a = EarthSatellite(a1, a2)
        sat_b = EarthSatellite(b1, b2)
        epoch_a = sat_a.epoch.utc_datetime().replace(tzinfo=None)
        epoch_b = sat_b.epoch.utc_datetime().replace(tzinfo=None)
        delta = epoch_b - epoch_a
        midpoint = epoch_a + delta / 2
        step = delta.total_seconds() / 100

        # Check first half of range
        times = trange(epoch_a, midpoint, step)
        dt = [dt64_to_datetime(t).replace(tzinfo=UTC) for t in times]
        tt = self.ts.from_datetimes(dt)

        geo = self.propagator.at(tt)
        exp_geo = sat_a.at(tt)

        satrec = self.propagator.find_satellite(times[-1]).model
        assert export_tle(satrec) == export_tle(sat_a.model)

        assert geo.position.au.flatten().tolist() == pytest.approx(
            exp_geo.position.au.flatten().tolist()
        )
        assert geo.velocity.au_per_d.flatten().tolist() == pytest.approx(
            exp_geo.velocity.au_per_d.flatten().tolist()
        )
        assert geo.t.tt.flatten().tolist() == exp_geo.t.tt.flatten().tolist()

        # Check second half of range
        times = trange(midpoint, epoch_b, step)
        dt = [dt64_to_datetime(t).replace(tzinfo=UTC) for t in times]
        tt = self.ts.from_datetimes(dt)

        geo = self.propagator.at(tt)
        exp_geo = sat_b.at(tt)

        satrec = self.propagator.find_satellite(times[-1]).model
        assert export_tle(satrec) == export_tle(sat_b.model)

        assert geo.position.au.flatten().tolist() == pytest.approx(
            exp_geo.position.au.flatten().tolist()
        )
        assert geo.velocity.au_per_d.flatten().tolist() == pytest.approx(
            exp_geo.velocity.au_per_d.flatten().tolist()
        )
        assert geo.t.tt.flatten().tolist() == exp_geo.t.tt.flatten().tolist()
