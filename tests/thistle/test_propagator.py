import datetime
from typing import Type

import numpy as np
import numpy.typing as npt
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
    TCASwitchStrategy,
    _find_tca,
    _slices_by_transitions,
)
from thistle.utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    datetime_to_dt64,
    dt64_to_datetime,
    pairwise,
    read_tle,
    trange,
)

from . import strategies as cst
from .conftest import ISS_SATRECS, ISS_TLES

UTC = datetime.timezone.utc

np.set_printoptions(linewidth=300)


@given(cst.transitions(), cst.times())
def test_slices(
    transitions: npt.NDArray[np.datetime64], times: npt.NDArray[np.datetime64]
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

    def test_at_single_time(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        t = self.ts.from_datetimes([dt.replace(tzinfo=UTC)])

        geo = self.propagator.at(t)
        exp_geo = exp_sat.at(t)

        assert geo.position.au.flatten().tolist() == pytest.approx(
            exp_geo.position.au.flatten().tolist()
        )
        assert geo.velocity.au_per_d.flatten().tolist() == pytest.approx(
            exp_geo.velocity.au_per_d.flatten().tolist()
        )

    def test_find_satrec_by_epoch(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        sat = self.propagator.find_satellite(datetime_to_dt64(dt))
        assert export_tle(sat.model) == export_tle(exp_sat.model)

    def test_find_satrec_returns_satrec(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        satrec = self.propagator.find_satrec(datetime_to_dt64(dt))
        assert isinstance(satrec, Satrec)
        assert export_tle(satrec) == export_tle(exp_sat.model)

    def test_find_tle_returns_tuple(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        tle = self.propagator.find_tle(datetime_to_dt64(dt))
        assert isinstance(tle, tuple)
        assert len(tle) == 2
        assert tle == export_tle(exp_sat.model)

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

    def test_find_satrec_returns_satrec(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        satrec = self.propagator.find_satrec(datetime_to_dt64(dt))
        assert isinstance(satrec, Satrec)
        assert export_tle(satrec) == export_tle(exp_sat.model)

    def test_find_tle_returns_tuple(self):
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        tle = self.propagator.find_tle(datetime_to_dt64(dt))
        assert isinstance(tle, tuple)
        assert len(tle) == 2
        assert tle == export_tle(exp_sat.model)


class TestFindTCA:
    def setup_class(self):
        self.ts = load.timescale()

    def test_tca_between_epochs(self):
        """TCA should fall between the two satellite epochs."""
        a1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        a2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"
        b1 = "1 25544U 98067A   98325.51671211  .01832406  18178-2  41610-2 0  9996"
        b2 = "2 25544 051.5928 161.7497 0074408 097.6565 263.2450 15.92278419   200"

        sat_a = EarthSatellite(a1, a2, ts=self.ts)
        sat_b = EarthSatellite(b1, b2, ts=self.ts)
        epoch_a = sat_a.epoch.utc_datetime().replace(tzinfo=None)
        epoch_b = sat_b.epoch.utc_datetime().replace(tzinfo=None)

        tca = _find_tca(sat_a, sat_b, self.ts)

        assert epoch_a <= tca
        assert tca <= epoch_b

    def test_tca_identical_epochs(self):
        """When epochs are identical, TCA should return the epoch itself."""
        a1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        a2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        sat_a = EarthSatellite(a1, a2, ts=self.ts)
        sat_b = EarthSatellite(a1, a2, ts=self.ts)
        epoch_a = sat_a.epoch.utc_datetime().replace(tzinfo=None)

        tca = _find_tca(sat_a, sat_b, self.ts)

        # Allow for floating-point precision (within 1 millisecond)
        assert abs((tca - epoch_a).total_seconds()) < 0.001

    def test_tca_is_near_minimum_distance(self):
        """Distance at TCA should be less than or equal to distance at endpoints."""
        a1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        a2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"
        b1 = "1 25544U 98067A   98325.51671211  .01832406  18178-2  41610-2 0  9996"
        b2 = "2 25544 051.5928 161.7497 0074408 097.6565 263.2450 15.92278419   200"

        sat_a = EarthSatellite(a1, a2, ts=self.ts)
        sat_b = EarthSatellite(b1, b2, ts=self.ts)
        epoch_a = sat_a.epoch.utc_datetime().replace(tzinfo=None)
        epoch_b = sat_b.epoch.utc_datetime().replace(tzinfo=None)

        tca = _find_tca(sat_a, sat_b, self.ts)

        def distance_at(dt):
            t = self.ts.from_datetimes([dt.replace(tzinfo=UTC)])
            ga = sat_a.at(t)
            gb = sat_b.at(t)
            diff = ga.xyz.au - gb.xyz.au
            return float(np.sqrt(np.sum(diff**2)))

        d_tca = distance_at(tca)
        d_a = distance_at(epoch_a)
        d_b = distance_at(epoch_b)

        assert d_tca <= d_a
        assert d_tca <= d_b


class TestTCASwitchStrategy(SwitchStrategyBasic):
    class_ = TCASwitchStrategy

    def setup_class(self):
        self.ts = load.timescale()
        self.switcher = self.class_(
            [EarthSatellite.from_satrec(satrec, self.ts) for satrec in ISS_SATRECS[:20]],
            ts=self.ts,
        )
        self.switcher.compute_transitions()

    def test_switcher_transition_count(self):
        # One transition per satrec, plus one after
        assert len(self.switcher.transitions) == 20 + 1

    def test_transitions_between_epochs(self):
        """Each TCA transition should fall between the neighboring epochs."""
        tolerance = datetime.timedelta(microseconds=10)
        for idx, bounds in enumerate(pairwise(self.switcher.transitions)):
            time_a, time_b = [dt64_to_datetime(t) for t in bounds]
            epoch = (
                self.switcher.satellites[idx]
                .epoch.utc_datetime()
                .replace(tzinfo=None)
            )
            # Allow small tolerance for floating-point precision
            assert time_a <= epoch + tolerance
            assert epoch <= time_b + tolerance

    def test_transitions_differ_from_midpoint(self):
        """TCA transitions should not exactly equal midpoint transitions."""
        midpoint_switcher = MidpointSwitchStrategy(
            [EarthSatellite.from_satrec(satrec, self.ts) for satrec in ISS_SATRECS[:20]]
        )
        midpoint_switcher.compute_transitions()

        inner_tca = self.switcher.transitions[1:-1]
        inner_mid = midpoint_switcher.transitions[1:-1]
        assert not np.array_equal(inner_tca, inner_mid)


class TestPropagatorTCA(PropagatorBaseClass):
    method: str = "tca"

    def setup_class(self):
        self.ts = load.timescale()
        self.tles = ISS_TLES[:20]
        self.propagator = Propagator(self.tles, method="tca")

    def test_at_single_time(self):
        """Position at a known TLE epoch should match that TLE's prediction."""
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        t = self.ts.from_datetimes([dt.replace(tzinfo=UTC)])

        geo = self.propagator.at(t)
        exp_geo = exp_sat.at(t)

        assert geo.position.au.flatten().tolist() == pytest.approx(
            exp_geo.position.au.flatten().tolist()
        )
        assert geo.velocity.au_per_d.flatten().tolist() == pytest.approx(
            exp_geo.velocity.au_per_d.flatten().tolist()
        )

    def test_find_satellite_at_epoch(self):
        """find_satellite at a TLE epoch should return that TLE's satellite."""
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        sat = self.propagator.find_satellite(datetime_to_dt64(dt))
        assert export_tle(sat.model) == export_tle(exp_sat.model)

    def test_find_satrec_at_epoch(self):
        """find_satrec at a TLE epoch should return that TLE's Satrec."""
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        satrec = self.propagator.find_satrec(datetime_to_dt64(dt))
        assert isinstance(satrec, Satrec)
        assert export_tle(satrec) == export_tle(exp_sat.model)

    def test_find_tle_at_epoch(self):
        """find_tle at a TLE epoch should return that TLE's lines."""
        line1 = "1 25544U 98067A   98325.45376114  .01829530  18113-2  41610-2 0  9996"
        line2 = "2 25544 051.5938 162.0926 0074012 097.3081 262.5015 15.92299093   191"

        exp_sat = EarthSatellite(line1, line2)
        dt = exp_sat.epoch.utc_datetime().replace(tzinfo=None)
        tle = self.propagator.find_tle(datetime_to_dt64(dt))
        assert isinstance(tle, tuple)
        assert len(tle) == 2
        assert tle == export_tle(exp_sat.model)


class TestSegmentTimes:
    def setup_class(self):
        tles = read_tle("tests/data/25544.tle")
        self.propagator = Propagator(tles, method="epoch")
        # Time range spanning multiple TLEs
        self.times = trange(
            datetime.datetime(1998, 11, 20),
            datetime.datetime(1998, 12, 20),
            step=60,
        )

    def test_returns_list_of_tuples(self):
        segments = self.propagator.segment_times(self.times)
        assert isinstance(segments, list)
        for t_slice, sat in segments:
            assert isinstance(t_slice, np.ndarray)
            assert isinstance(sat, EarthSatellite)

    def test_segments_cover_all_times(self):
        segments = self.propagator.segment_times(self.times)
        reconstructed = np.concatenate([t for t, _ in segments])
        np.testing.assert_array_equal(reconstructed, self.times)

    def test_single_satellite(self):
        """A time range within one TLE's window yields one segment."""
        first_sat = self.propagator.satellites[0]
        epoch = first_sat.epoch.utc_datetime().replace(tzinfo=None)
        short_times = trange(epoch, epoch + datetime.timedelta(minutes=10), step=10)
        segments = self.propagator.segment_times(short_times)
        assert len(segments) == 1
        assert segments[0][1] is first_sat

    def test_empty_gaps_omitted(self):
        """Segments with no matching times are not returned."""
        segments = self.propagator.segment_times(self.times)
        for t_slice, _ in segments:
            assert len(t_slice) > 0

    def test_correct_satellite_per_segment(self):
        """Each segment's satellite matches what find_satellite returns."""
        segments = self.propagator.segment_times(self.times)
        for t_slice, sat in segments:
            mid = t_slice[len(t_slice) // 2]
            expected_sat = self.propagator.find_satellite(mid)
            assert export_tle(sat.model) == export_tle(expected_sat.model)


class TestStrategyInstance:
    """Test passing a SwitchingStrategy instance to Propagator."""

    def test_epoch_instance(self):
        """Passing an EpochSwitchStrategy instance works like method='epoch'."""
        prop_str = Propagator(ISS_TLES, method="epoch")
        prop_inst = Propagator(ISS_TLES, method=EpochSwitchStrategy([]))
        np.testing.assert_array_equal(
            prop_str.switcher.transitions, prop_inst.switcher.transitions
        )

    def test_midpoint_instance(self):
        """Passing a MidpointSwitchStrategy instance works like method='midpoint'."""
        prop_str = Propagator(ISS_TLES, method="midpoint")
        prop_inst = Propagator(ISS_TLES, method=MidpointSwitchStrategy([]))
        np.testing.assert_array_equal(
            prop_str.switcher.transitions, prop_inst.switcher.transitions
        )

    def test_tca_instance(self):
        """Passing a TCASwitchStrategy instance works like method='tca'."""
        ts = load.timescale()
        tles = ISS_TLES[:20]
        prop_str = Propagator(tles, method="tca")
        prop_inst = Propagator(tles, method=TCASwitchStrategy([], ts=ts))
        np.testing.assert_array_equal(
            prop_str.switcher.transitions, prop_inst.switcher.transitions
        )

    def test_instance_satellites_replaced(self):
        """Strategy instance's satellites are replaced with those built from TLEs."""
        strategy = EpochSwitchStrategy([])
        assert len(strategy.satellites) == 0
        prop = Propagator(ISS_TLES, method=strategy)
        assert len(prop.switcher.satellites) == len(ISS_TLES)

    def test_instance_with_start_stop(self):
        """Strategy instance respects start/stop filtering."""
        strategy = MidpointSwitchStrategy([])
        prop = Propagator(
            ISS_TLES,
            method=strategy,
            start=datetime.datetime(1998, 11, 25),
            stop=datetime.datetime(1998, 12, 5),
        )
        assert len(prop.satellites) < len(ISS_TLES)

    def test_instance_is_same_object(self):
        """The Propagator uses the same strategy instance passed in."""
        strategy = EpochSwitchStrategy([])
        prop = Propagator(ISS_TLES, method=strategy)
        assert prop.switcher is strategy

    def test_invalid_method_type_raises(self):
        """Passing an invalid type raises TypeError."""
        with pytest.raises(TypeError, match="strategy name or SwitchingStrategy"):
            Propagator(ISS_TLES, method=42)
