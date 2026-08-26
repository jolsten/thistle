import datetime
import warnings
from typing import Type

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from sgp4.api import Satrec
from sgp4.exporter import export_tle
from skyfield.api import EarthSatellite, load

from thistle.events import find_node_crossings
from thistle.propagator import (
    EpochSwitchStrategy,
    MidpointSwitchStrategy,
    Propagator,
    SwitchingStrategy,
    TCASwitchStrategy,
    TLEExtrapolationWarning,
    _find_tca,
    _slices_by_transitions,
)
from thistle.utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    datetime_to_dt64,
    dt64_to_datetime,
    dt64_to_time,
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
        tles = read_tle("tests/thistle/data/25544.tle")
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


# ---------------------------------------------------------------------------
# TLEExtrapolationWarning
# ---------------------------------------------------------------------------


def _tle_checksum(line: str) -> str:
    total = sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in line)
    return str(total % 10)


def _tle_at_epoch(epoch: str) -> tuple[str, str]:
    """An ISS-like TLE with the given line-1 epoch field (cols 19-32)."""
    base_l1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
    base_l2 = "2 25544  51.6400 208.9163 0006703  30.5502 329.5947 15.49560532  1001"
    line1 = base_l1[:18] + epoch + base_l1[32:68]
    return line1 + _tle_checksum(line1), base_l2


WARN_TLE_A = _tle_at_epoch("24001.50000000")  # epoch 2024-01-01T12:00
WARN_TLE_B = _tle_at_epoch("24041.50000000")  # epoch 2024-02-10T12:00


@pytest.fixture
def gap_propagator() -> Propagator:
    """Two TLEs 40 days apart, default 7-day warn threshold."""
    return Propagator([WARN_TLE_A, WARN_TLE_B])


class TestExtrapolationWarning:
    def test_array_beyond_last_epoch(self, gap_propagator: Propagator) -> None:
        times = trange(
            datetime.datetime(2024, 3, 14), datetime.datetime(2024, 3, 15), 3600
        )
        with pytest.warns(
            TLEExtrapolationWarning,
            match=r"satellite 25544: all 24 propagation times are more than 7 days",
        ) as rec:
            gap_propagator.segment_times(times)
        assert len(rec) == 1
        w = rec[0].message
        assert w.n_bad == w.n_total == 24
        assert w.threshold == datetime.timedelta(days=7)
        assert w.span == (
            datetime.datetime(2024, 1, 1, 12),
            datetime.datetime(2024, 2, 10, 12),
        )
        assert datetime.timedelta(days=32) < w.max_offset < datetime.timedelta(days=34)
        assert str(w).endswith("warn_threshold=None disables.")

    def test_at_beyond_last_epoch(self, gap_propagator: Propagator) -> None:
        times = trange(
            datetime.datetime(2024, 3, 14), datetime.datetime(2024, 3, 15), 3600
        )
        t = dt64_to_time(times, gap_propagator.ts)
        with pytest.warns(
            TLEExtrapolationWarning, match="TLEs cover 2024-01-01 to 2024-02-10"
        ):
            gap_propagator.at(t)

    def test_array_before_first_epoch(self, gap_propagator: Propagator) -> None:
        times = trange(
            datetime.datetime(2023, 12, 1), datetime.datetime(2023, 12, 2), 3600
        )
        with pytest.warns(TLEExtrapolationWarning, match="more than 7 days"):
            gap_propagator.segment_times(times)

    def test_partial_count_phrasing(self, gap_propagator: Propagator) -> None:
        # Daily times Feb 10-19 at 00:00; only Feb 18 and 19 are > 7 days
        # from the Feb 10 12:00 epoch.
        times = trange(
            datetime.datetime(2024, 2, 10), datetime.datetime(2024, 2, 20), 86400
        )
        with pytest.warns(
            TLEExtrapolationWarning, match=r"2 of 10 propagation times"
        ):
            gap_propagator.segment_times(times)

    def test_scalar_interior_gap(self, gap_propagator: Propagator) -> None:
        with pytest.warns(
            TLEExtrapolationWarning,
            match=r"requested time 2024-01-21T12:00Z is 20 days",
        ):
            gap_propagator.find_satellite(datetime.datetime(2024, 1, 21, 12))

    def test_scalar_find_tle(self, gap_propagator: Propagator) -> None:
        with pytest.warns(
            TLEExtrapolationWarning, match="requested time 2024-03-15T00:00Z"
        ):
            gap_propagator.find_tle(datetime.datetime(2024, 3, 15))

    def test_no_warning_within_coverage(self, gap_propagator: Propagator) -> None:
        times = trange(
            datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 3), 3600
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", TLEExtrapolationWarning)
            gap_propagator.segment_times(times)
            gap_propagator.find_satellite(datetime.datetime(2024, 2, 12))

    @pytest.mark.parametrize("threshold", [None, 0, -1.0])
    def test_disabled(self, threshold) -> None:
        prop = Propagator([WARN_TLE_A, WARN_TLE_B], warn_threshold=threshold)
        assert prop.warn_threshold is None
        with warnings.catch_warnings():
            warnings.simplefilter("error", TLEExtrapolationWarning)
            prop.find_satellite(datetime.datetime(2025, 6, 1))

    def test_threshold_as_days_number(self) -> None:
        prop = Propagator([WARN_TLE_A, WARN_TLE_B], warn_threshold=45)
        with warnings.catch_warnings():
            warnings.simplefilter("error", TLEExtrapolationWarning)
            prop.find_satellite(datetime.datetime(2024, 3, 15))

        prop = Propagator(
            [WARN_TLE_A, WARN_TLE_B], warn_threshold=datetime.timedelta(days=1)
        )
        with pytest.warns(TLEExtrapolationWarning, match=r"warn threshold: 1 day\)"):
            prop.find_satellite(datetime.datetime(2024, 1, 4))

    def test_check_coverage_window(self, gap_propagator: Propagator) -> None:
        with pytest.warns(
            TLEExtrapolationWarning,
            match=r"requested window 2024-04-01T00:00Z to 2024-05-01T00:00Z extends",
        ):
            gap_propagator.check_coverage(
                datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", TLEExtrapolationWarning)
            gap_propagator.check_coverage(
                datetime.datetime(2024, 2, 8), datetime.datetime(2024, 2, 12)
            )

    def test_events_emit_single_window_warning(
        self, gap_propagator: Propagator
    ) -> None:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            find_node_crossings(
                np.datetime64("2024-04-01T00:00:00", "us"),
                np.datetime64("2024-04-01T03:00:00", "us"),
                gap_propagator,
            )
        found = [w for w in rec if issubclass(w.category, TLEExtrapolationWarning)]
        assert len(found) == 1
        assert "requested window" in str(found[0].message)
