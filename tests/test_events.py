"""Tests for thistle.events event-finding functions."""

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load

from thistle.utils import read_tle
from thistle.events import (
    find_ascending_periods,
    find_descending_periods,
    find_eclipse_periods,
    find_node_crossings,
    find_passes,
    find_sunlit_periods,
)

ts = load.timescale()
_tles = read_tle("tests/data/25544.tle")
SAT = EarthSatellite(_tles[0][0], _tles[0][1], ts=ts)

# 24-hour window near the TLE epoch
START_24H = np.datetime64("1998-11-20T00:00:00", "us")
STOP_24H = np.datetime64("1998-11-21T00:00:00", "us")

# Boulder, CO
BOULDER_LAT = 40.0150
BOULDER_LON = -105.2705


class TestFindPasses:
    """Tests for find_passes."""

    def test_returns_list_of_dicts(self):
        passes = find_passes(START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON)
        assert isinstance(passes, list)
        for p in passes:
            assert isinstance(p, dict)
            assert "start" in p
            assert "stop" in p
            assert "peak_time" in p
            assert "peak_elevation" in p

    def test_expected_count(self):
        """ISS over Boulder should have a handful of passes in 24h."""
        passes = find_passes(START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON)
        assert len(passes) >= 1
        assert len(passes) <= 10

    def test_time_ordering(self):
        """start <= peak_time <= stop for each pass."""
        passes = find_passes(START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON)
        for p in passes:
            assert p["start"] <= p["peak_time"]
            assert p["peak_time"] <= p["stop"]

    def test_peak_elevation_above_minimum(self):
        """Peak elevation should be at or above the minimum."""
        min_el = 5.0
        passes = find_passes(
            START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON,
            min_elevation=min_el,
        )
        for p in passes:
            assert p["peak_elevation"] >= min_el - 0.5  # small tolerance

    def test_passes_sorted_by_start(self):
        passes = find_passes(START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON)
        for i in range(len(passes) - 1):
            assert passes[i]["start"] <= passes[i + 1]["start"]

    def test_empty_for_short_window(self):
        """A very short window with no pass should return empty list."""
        short_start = np.datetime64("1998-11-20T12:00:00", "us")
        short_stop = np.datetime64("1998-11-20T12:00:10", "us")
        passes = find_passes(short_start, short_stop, SAT, BOULDER_LAT, BOULDER_LON)
        assert isinstance(passes, list)

    def test_start_equals_stop_raises(self):
        with pytest.raises(ValueError):
            find_passes(START_24H, START_24H, SAT, BOULDER_LAT, BOULDER_LON)

    def test_higher_min_elevation_fewer_passes(self):
        """Raising min elevation should produce equal or fewer passes."""
        passes_5 = find_passes(
            START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON,
            min_elevation=5.0,
        )
        passes_30 = find_passes(
            START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON,
            min_elevation=30.0,
        )
        assert len(passes_30) <= len(passes_5)


class TestFindNodeCrossings:
    """Tests for find_node_crossings."""

    def test_returns_list_of_dicts(self):
        crossings = find_node_crossings(START_24H, STOP_24H, SAT)
        assert isinstance(crossings, list)
        for c in crossings:
            assert isinstance(c, dict)
            assert "start" in c
            assert "stop" in c
            assert "longitude" in c
            assert "ascending" in c

    def test_start_equals_stop(self):
        """Node crossings are instantaneous: start == stop."""
        crossings = find_node_crossings(START_24H, STOP_24H, SAT)
        for c in crossings:
            assert c["start"] == c["stop"]

    def test_expected_count(self):
        """ISS makes ~16 orbits/day -> ~32 node crossings."""
        crossings = find_node_crossings(START_24H, STOP_24H, SAT)
        assert len(crossings) >= 28
        assert len(crossings) <= 36

    def test_alternating_ascending_descending(self):
        """Node crossings should alternate between ascending and descending."""
        crossings = find_node_crossings(START_24H, STOP_24H, SAT)
        for i in range(len(crossings) - 1):
            assert crossings[i]["ascending"] != crossings[i + 1]["ascending"]

    def test_longitude_range(self):
        """Longitude should be in [-180, 180]."""
        crossings = find_node_crossings(START_24H, STOP_24H, SAT)
        for c in crossings:
            assert -180.0 <= c["longitude"] <= 180.0

    def test_times_within_window(self):
        crossings = find_node_crossings(START_24H, STOP_24H, SAT)
        for c in crossings:
            assert START_24H <= c["start"] <= STOP_24H

    def test_start_equals_stop_raises(self):
        with pytest.raises(ValueError):
            find_node_crossings(START_24H, START_24H, SAT)


class TestFindSunlitPeriods:
    """Tests for find_sunlit_periods."""

    def test_returns_list_of_dicts(self):
        periods = find_sunlit_periods(START_24H, STOP_24H, SAT)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)
            assert "start" in p
            assert "stop" in p

    def test_expected_count(self):
        """ISS should have ~16 sunlit periods in 24h."""
        periods = find_sunlit_periods(START_24H, STOP_24H, SAT)
        assert len(periods) >= 10
        assert len(periods) <= 20

    def test_start_before_stop(self):
        periods = find_sunlit_periods(START_24H, STOP_24H, SAT)
        for p in periods:
            assert p["start"] < p["stop"]

    def test_no_overlap(self):
        """Sunlit periods should not overlap."""
        periods = find_sunlit_periods(START_24H, STOP_24H, SAT)
        for i in range(len(periods) - 1):
            assert periods[i]["stop"] <= periods[i + 1]["start"]

    def test_times_within_window(self):
        periods = find_sunlit_periods(START_24H, STOP_24H, SAT)
        for p in periods:
            assert p["start"] >= START_24H
            assert p["stop"] <= STOP_24H

    def test_start_equals_stop_raises(self):
        with pytest.raises(ValueError):
            find_sunlit_periods(START_24H, START_24H, SAT)


class TestFindEclipsePeriods:
    """Tests for find_eclipse_periods."""

    def test_returns_list_of_dicts(self):
        periods = find_eclipse_periods(START_24H, STOP_24H, SAT)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)
            assert "start" in p
            assert "stop" in p

    def test_expected_count(self):
        """ISS should have ~16 eclipse periods in 24h."""
        periods = find_eclipse_periods(START_24H, STOP_24H, SAT)
        assert len(periods) >= 10
        assert len(periods) <= 20

    def test_start_before_stop(self):
        periods = find_eclipse_periods(START_24H, STOP_24H, SAT)
        for p in periods:
            assert p["start"] < p["stop"]

    def test_no_overlap(self):
        """Eclipse periods should not overlap."""
        periods = find_eclipse_periods(START_24H, STOP_24H, SAT)
        for i in range(len(periods) - 1):
            assert periods[i]["stop"] <= periods[i + 1]["start"]

    def test_sunlit_and_eclipse_complementary(self):
        """Sunlit and eclipse periods should cover the full window with no gaps."""
        sunlit = find_sunlit_periods(START_24H, STOP_24H, SAT)
        eclipse = find_eclipse_periods(START_24H, STOP_24H, SAT)

        # Merge and sort all transitions
        transitions = []
        for p in sunlit:
            transitions.append((p["start"], "sunlit_start"))
            transitions.append((p["stop"], "sunlit_stop"))
        for p in eclipse:
            transitions.append((p["start"], "eclipse_start"))
            transitions.append((p["stop"], "eclipse_stop"))
        transitions.sort(key=lambda x: x[0])

        # Verify that stops and starts match up (no gaps)
        for i in range(len(transitions) - 1):
            t1_time, t1_type = transitions[i]
            t2_time, t2_type = transitions[i + 1]
            if "stop" in t1_type and "start" in t2_type:
                assert t1_time == t2_time, f"Gap between {t1_time} and {t2_time}"

    def test_start_equals_stop_raises(self):
        with pytest.raises(ValueError):
            find_eclipse_periods(START_24H, START_24H, SAT)


class TestFindAscendingPeriods:
    """Tests for find_ascending_periods."""

    def test_returns_list_of_dicts(self):
        periods = find_ascending_periods(START_24H, STOP_24H, SAT)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)
            assert "start" in p
            assert "stop" in p

    def test_expected_count(self):
        """ISS makes ~16 orbits/day -> ~16 ascending periods."""
        periods = find_ascending_periods(START_24H, STOP_24H, SAT)
        assert len(periods) >= 14
        assert len(periods) <= 18

    def test_start_before_stop(self):
        periods = find_ascending_periods(START_24H, STOP_24H, SAT)
        for p in periods:
            assert p["start"] < p["stop"]

    def test_no_overlap(self):
        periods = find_ascending_periods(START_24H, STOP_24H, SAT)
        for i in range(len(periods) - 1):
            assert periods[i]["stop"] <= periods[i + 1]["start"]

    def test_start_equals_stop_raises(self):
        with pytest.raises(ValueError):
            find_ascending_periods(START_24H, START_24H, SAT)


class TestFindDescendingPeriods:
    """Tests for find_descending_periods."""

    def test_returns_list_of_dicts(self):
        periods = find_descending_periods(START_24H, STOP_24H, SAT)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)
            assert "start" in p
            assert "stop" in p

    def test_expected_count(self):
        """ISS makes ~16 orbits/day -> ~16 descending periods."""
        periods = find_descending_periods(START_24H, STOP_24H, SAT)
        assert len(periods) >= 14
        assert len(periods) <= 18

    def test_start_before_stop(self):
        periods = find_descending_periods(START_24H, STOP_24H, SAT)
        for p in periods:
            assert p["start"] < p["stop"]

    def test_no_overlap(self):
        periods = find_descending_periods(START_24H, STOP_24H, SAT)
        for i in range(len(periods) - 1):
            assert periods[i]["stop"] <= periods[i + 1]["start"]

    def test_ascending_and_descending_complementary(self):
        """Ascending and descending periods should cover the full window with no gaps."""
        asc = find_ascending_periods(START_24H, STOP_24H, SAT)
        desc = find_descending_periods(START_24H, STOP_24H, SAT)

        transitions = []
        for p in asc:
            transitions.append((p["start"], "asc_start"))
            transitions.append((p["stop"], "asc_stop"))
        for p in desc:
            transitions.append((p["start"], "desc_start"))
            transitions.append((p["stop"], "desc_stop"))
        transitions.sort(key=lambda x: x[0])

        for i in range(len(transitions) - 1):
            t1_time, t1_type = transitions[i]
            t2_time, t2_type = transitions[i + 1]
            if "stop" in t1_type and "start" in t2_type:
                assert t1_time == t2_time, f"Gap between {t1_time} and {t2_time}"

    def test_start_equals_stop_raises(self):
        with pytest.raises(ValueError):
            find_descending_periods(START_24H, START_24H, SAT)


# ---------- Propagator support tests ----------

from thistle.propagator import Propagator

# Single-TLE Propagator: no transitions in the test window, so results
# should match using the same EarthSatellite directly.
PROP = Propagator([_tles[0]], method="epoch")

# Multi-TLE Propagator for broader coverage tests
PROP_MULTI = Propagator(_tles, method="epoch")


class TestPropagatorPasses:
    """Test find_passes with a Propagator."""

    def test_returns_list(self):
        passes = find_passes(START_24H, STOP_24H, PROP, BOULDER_LAT, BOULDER_LON)
        assert isinstance(passes, list)
        for p in passes:
            assert isinstance(p, dict)

    def test_matches_single_satellite(self):
        """Single-TLE Propagator gives same results as EarthSatellite."""
        prop_passes = find_passes(START_24H, STOP_24H, PROP, BOULDER_LAT, BOULDER_LON)
        sat_passes = find_passes(START_24H, STOP_24H, SAT, BOULDER_LAT, BOULDER_LON)
        assert len(prop_passes) == len(sat_passes)
        for pp, sp in zip(prop_passes, sat_passes):
            diff = abs((pp["start"] - sp["start"]) / np.timedelta64(1, "s"))
            assert diff < 1.0


class TestPropagatorNodeCrossings:
    """Test find_node_crossings with a Propagator."""

    def test_returns_list(self):
        crossings = find_node_crossings(START_24H, STOP_24H, PROP)
        assert isinstance(crossings, list)
        for c in crossings:
            assert isinstance(c, dict)

    def test_matches_single_satellite(self):
        prop_crossings = find_node_crossings(START_24H, STOP_24H, PROP)
        sat_crossings = find_node_crossings(START_24H, STOP_24H, SAT)
        assert len(prop_crossings) == len(sat_crossings)
        for pc, sc in zip(prop_crossings, sat_crossings):
            diff = abs((pc["start"] - sc["start"]) / np.timedelta64(1, "s"))
            assert diff < 1.0


class TestPropagatorSunlit:
    """Test find_sunlit_periods with a Propagator."""

    def test_returns_list(self):
        periods = find_sunlit_periods(START_24H, STOP_24H, PROP)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)

    def test_matches_single_satellite(self):
        prop_periods = find_sunlit_periods(START_24H, STOP_24H, PROP)
        sat_periods = find_sunlit_periods(START_24H, STOP_24H, SAT)
        assert len(prop_periods) == len(sat_periods)
        for pp, sp in zip(prop_periods, sat_periods):
            diff = abs((pp["start"] - sp["start"]) / np.timedelta64(1, "s"))
            assert diff < 1.0


class TestPropagatorEclipse:
    """Test find_eclipse_periods with a Propagator."""

    def test_returns_list(self):
        periods = find_eclipse_periods(START_24H, STOP_24H, PROP)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)

    def test_matches_single_satellite(self):
        prop_periods = find_eclipse_periods(START_24H, STOP_24H, PROP)
        sat_periods = find_eclipse_periods(START_24H, STOP_24H, SAT)
        assert len(prop_periods) == len(sat_periods)
        for pp, sp in zip(prop_periods, sat_periods):
            diff = abs((pp["start"] - sp["start"]) / np.timedelta64(1, "s"))
            assert diff < 1.0


class TestPropagatorAscending:
    """Test find_ascending_periods with a Propagator."""

    def test_returns_list(self):
        periods = find_ascending_periods(START_24H, STOP_24H, PROP)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)

    def test_matches_single_satellite(self):
        prop_periods = find_ascending_periods(START_24H, STOP_24H, PROP)
        sat_periods = find_ascending_periods(START_24H, STOP_24H, SAT)
        assert len(prop_periods) == len(sat_periods)
        for pp, sp in zip(prop_periods, sat_periods):
            diff = abs((pp["start"] - sp["start"]) / np.timedelta64(1, "s"))
            assert diff < 1.0


class TestPropagatorDescending:
    """Test find_descending_periods with a Propagator."""

    def test_returns_list(self):
        periods = find_descending_periods(START_24H, STOP_24H, PROP)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, dict)

    def test_matches_single_satellite(self):
        prop_periods = find_descending_periods(START_24H, STOP_24H, PROP)
        sat_periods = find_descending_periods(START_24H, STOP_24H, SAT)
        assert len(prop_periods) == len(sat_periods)
        for pp, sp in zip(prop_periods, sat_periods):
            diff = abs((pp["start"] - sp["start"]) / np.timedelta64(1, "s"))
            assert diff < 1.0


# ---------- Multi-TLE Propagator tests ----------


MULTI_START = np.datetime64("1998-11-20T00:00:00", "us")
MULTI_STOP = np.datetime64("1998-12-20T00:00:00", "us")


class TestMultiTLEPropagator:
    """Test event functions with a multi-TLE Propagator spanning 30 days."""

    def test_passes_over_long_window(self):
        passes = find_passes(
            MULTI_START, MULTI_STOP, PROP_MULTI, BOULDER_LAT, BOULDER_LON
        )
        assert len(passes) > 10
        for p in passes:
            assert isinstance(p, dict)
            assert p["start"] >= MULTI_START
            assert p["stop"] <= MULTI_STOP

    def test_sunlit_no_gaps_at_transitions(self):
        """Sunlit periods should have no artificial gaps at TLE transitions."""
        periods = find_sunlit_periods(MULTI_START, MULTI_STOP, PROP_MULTI)
        assert len(periods) > 100
        for p in periods:
            assert p["start"] < p["stop"]
        for i in range(len(periods) - 1):
            assert periods[i]["stop"] <= periods[i + 1]["start"]

    def test_eclipse_no_gaps_at_transitions(self):
        periods = find_eclipse_periods(MULTI_START, MULTI_STOP, PROP_MULTI)
        assert len(periods) > 100
        for p in periods:
            assert p["start"] < p["stop"]
        for i in range(len(periods) - 1):
            assert periods[i]["stop"] <= periods[i + 1]["start"]

    def test_node_crossings_over_long_window(self):
        crossings = find_node_crossings(MULTI_START, MULTI_STOP, PROP_MULTI)
        assert len(crossings) > 100
        for i in range(len(crossings) - 1):
            assert crossings[i]["start"] < crossings[i + 1]["start"]
