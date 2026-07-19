"""Ground truth tests for sunlight and eclipse detection."""

import numpy as np
import pytest

from thistle.orbit_data import generate_sunlight

from .conftest import get_iss_satellite, load_eclipse_times, load_truth_csv


@pytest.fixture(scope="module")
def iss_satellite():
    """Get ISS satellite for testing."""
    return get_iss_satellite()


@pytest.fixture(scope="module")
def sun_periods():
    """Load ground truth sunlight periods."""
    return load_eclipse_times("25544_Sun.csv")


@pytest.fixture(scope="module")
def umbra_periods():
    """Load ground truth umbra periods."""
    return load_eclipse_times("25544_Umbra.csv")


class TestSunlightGroundTruth:
    """Test sunlight/eclipse detection against ground truth data."""

    def test_sunlight_detection(self, iss_satellite, sun_periods, umbra_periods):
        """Test sunlight condition detection against ground truth periods.

        During sunlight periods (from 25544_Sun.csv), satellite should report
        sunlit (2). During umbra periods (from 25544_Umbra.csv), satellite
        should report umbra (0).
        """
        # Sample times during known sunlight periods
        for start_time, stop_time in sun_periods[:5]:  # Test first 5 periods
            # Sample middle of the sunlight period
            mid_time = start_time + (stop_time - start_time) / 2
            times = np.array([mid_time])

            result = generate_sunlight(times, iss_satellite)

            # Should be sunlit (2)
            assert result["sun"][0] == 2, f"Expected sunlit at {mid_time}, got {result['sun'][0]}"

    def test_umbra_detection(self, iss_satellite, umbra_periods):
        """Test umbra detection against ground truth periods."""
        # Sample times during known umbra periods
        for start_time, stop_time in umbra_periods[:5]:  # Test first 5 periods
            # Sample middle of the umbra period
            mid_time = start_time + (stop_time - start_time) / 2
            times = np.array([mid_time])

            result = generate_sunlight(times, iss_satellite)

            # Should be in umbra (0)
            assert result["sun"][0] == 0, f"Expected umbra at {mid_time}, got {result['sun'][0]}"

    def test_eclipse_transition_near_boundaries(self, iss_satellite, umbra_periods):
        """Test that eclipse transitions happen near ground truth boundaries.

        Sample points 30 seconds before umbra entry and 30 seconds after umbra
        exit to verify transitions happen near the expected times.
        """
        for start_time, stop_time in umbra_periods[:3]:  # Test first 3 periods
            # Sample before umbra entry (should be sunlit or penumbra)
            before_entry = start_time - np.timedelta64(30, 's')
            times_before = np.array([before_entry])
            result_before = generate_sunlight(times_before, iss_satellite)
            assert result_before["sun"][0] >= 1, "Should be in sunlight or penumbra before umbra"

            # Sample after umbra exit (should be sunlit or penumbra)
            after_exit = stop_time + np.timedelta64(30, 's')
            times_after = np.array([after_exit])
            result_after = generate_sunlight(times_after, iss_satellite)
            assert result_after["sun"][0] >= 1, "Should be in sunlight or penumbra after umbra"

    def test_eclipse_count_over_day(self, iss_satellite):
        """Test that eclipse count is reasonable over one day.

        ISS typically experiences 15-16 orbits per day, with approximately
        one eclipse per orbit, so we expect ~15 eclipses per day.
        """
        # Generate sunlight data for 24 hours
        start_time = np.datetime64('2020-04-01T00:00:00', 'us')
        times = start_time + np.arange(0, 24 * 60 * 60, 10, dtype='timedelta64[s]')

        result = generate_sunlight(times, iss_satellite)

        # Count transitions into umbra
        sun_condition = result["sun"]
        umbra_entries = np.sum((sun_condition[:-1] != 0) & (sun_condition[1:] == 0))

        # Should have approximately 15 umbra entries over 24 hours
        assert 12 <= umbra_entries <= 18, f"Expected ~15 umbra entries, got {umbra_entries}"

    def test_sunlight_values_are_valid(self, iss_satellite):
        """Test that sunlight values are always 0, 1, or 2."""
        start_time = np.datetime64('2020-04-01T00:00:00', 'us')
        times = start_time + np.arange(0, 60 * 60, 10, dtype='timedelta64[s]')

        result = generate_sunlight(times, iss_satellite)
        sun = result["sun"]

        # All values must be 0 (umbra), 1 (penumbra), or 2 (sunlit)
        assert np.all(np.isin(sun, [0, 1, 2])), "Invalid sunlight values"

    def test_continuous_sunlight_duration(self, iss_satellite, sun_periods):
        """Test that continuous sunlight durations match ground truth.

        Compare the duration of sunlight periods computed from our
        implementation against ground truth values.
        """
        for start_time, stop_time in sun_periods[:3]:  # Test first 3 periods
            # Sample every second during this period
            duration_sec = int((stop_time - start_time) / np.timedelta64(1, 's'))
            times = start_time + np.arange(0, duration_sec, 1, dtype='timedelta64[s]')

            result = generate_sunlight(times, iss_satellite)
            sun = result["sun"]

            # Count how many samples are in sunlight (value 2)
            sunlit_count = np.sum(sun == 2)
            sunlit_fraction = sunlit_count / len(sun)

            # Most of this period should be in sunlight
            # (allow some samples at boundaries to be penumbra)
            assert sunlit_fraction > 0.8, "Too few sunlit samples in sunlight period"
