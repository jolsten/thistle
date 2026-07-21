"""Ground truth tests for local solar time with different propagator strategies."""

import numpy as np
import pytest

from thistle.orbit_data import generate_local_solar_time
from thistle.propagator import Propagator

from .conftest import load_truth_csv


def parse_time_to_hours(time_str: str) -> float:
    """Parse time string 'HH:MM:SS.mmm' to fractional hours.

    Args:
        time_str: Time string in format "HH:MM:SS.mmm"

    Returns:
        Fractional hours [0, 24)
    """
    parts = time_str.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours + minutes / 60.0 + seconds / 3600.0


# Parametrize tests across all three switching strategies
@pytest.fixture(scope="module", params=["Epoch", "Midpoint", "TCA"])
def strategy_data(request, iss_tles, april_2020_dates):
    """Load ground truth data and create propagator for each strategy.

    Returns:
        tuple: (strategy_name, propagator, times, ground_truth_data)
    """
    strategy = request.param

    # Load ground truth data for this strategy
    filename = f"25544_SAT_{strategy}.csv"
    times, data = load_truth_csv(filename)

    # Create propagator with appropriate strategy
    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, times, data


class TestLocalSolarTimeGroundTruthByStrategy:
    """Test local solar time against ground truth data for each switching strategy.

    Ground truth data is generated from STK using different TLE switching strategies:
    - Epoch: Switch at TLE epoch times
    - Midpoint: Switch at midpoint between TLE epochs
    - TCA: Switch at time of closest approach between consecutive TLEs

    Local solar time is the apparent solar time at the subsatellite point, computed
    from subsatellite longitude, GMST, and the Sun's right ascension.
    """

    def test_local_solar_time(self, strategy_data):
        """Test local solar time matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_local_solar_time(times, propagator)

        computed_lst = result["lst"]

        # Parse expected LST from "Apparent Time" column
        # Note: CSV has string data that wasn't converted to float, so we need to
        # parse it manually. The load_truth_csv function skips non-numeric columns,
        # so we need to load the file again to get the time strings.
        import csv
        import pathlib

        truth_dir = pathlib.Path(__file__).parent.parent / "data" / "truth"
        filepath = truth_dir / f"25544_SAT_{strategy}.csv"

        expected_lst = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row["Apparent Time"]
                hours = parse_time_to_hours(time_str)
                expected_lst.append(hours)

        expected_lst = np.array(expected_lst)

        # Handle wrapping at 24 hours (e.g., 23.9 vs 0.1 hour difference)
        lst_diff = np.abs(computed_lst - expected_lst)
        lst_diff = np.minimum(lst_diff, 24.0 - lst_diff)

        # Allow 0.1 hour (6 minute) tolerance
        # This accounts for differences in Sun position calculation and GMST computation
        assert np.all(lst_diff < 0.1), \
            f"Local solar time mismatch for {strategy} strategy. " \
            f"Max error: {np.max(lst_diff):.3f} hours"

    def test_local_solar_time_range(self, strategy_data):
        """Test that local solar time is in valid range [0, 24) for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_local_solar_time(times, propagator)

        lst = result["lst"]

        assert np.all(lst >= 0.0), f"LST should be >= 0 for {strategy} strategy"
        assert np.all(lst < 24.0), f"LST should be < 24 for {strategy} strategy"

    def test_local_solar_time_continuity(self, strategy_data):
        """Test that local solar time changes continuously for each strategy.

        LST should generally increase over time (with occasional wraps from 23.x to 0.x).
        """
        strategy, propagator, times, data = strategy_data
        result = generate_local_solar_time(times, propagator)

        lst = result["lst"]
        lst_diff = np.diff(lst)

        # Account for wrapping at 24 hours
        # If difference is large negative (e.g., -23), it's actually a forward wrap
        wrapped_diffs = np.where(lst_diff < -12, lst_diff + 24.0, lst_diff)

        # LST should generally increase (positive diffs) over 10-second intervals
        # For LEO, ~90-95 minute orbital period means ~0.25 hours (~15 minutes) of LST change
        # per orbit. In 10 seconds, expect ~0.003 hours (~10 seconds) of LST change.
        # Most steps should show increasing LST
        positive_diffs = np.sum(wrapped_diffs > 0)
        assert positive_diffs > 0.9 * len(wrapped_diffs), \
            f"LST should generally increase for {strategy} strategy"

    def test_local_solar_time_variation_over_orbit(self, strategy_data):
        """Test that local solar time varies over one orbit for each strategy.

        Over the ~600 samples (~100 minutes), LST should vary significantly as the
        satellite moves in longitude.
        """
        strategy, propagator, times, data = strategy_data
        result = generate_local_solar_time(times, propagator)

        lst = result["lst"]

        # LST should vary by at least a few hours over the full dataset
        lst_range = np.max(lst) - np.min(lst)
        # If range is small, might be due to wrapping - check for wrap
        if lst_range < 5.0:
            # Check if values span the wrap point
            has_small = np.any(lst < 5.0)
            has_large = np.any(lst > 19.0)
            if has_small and has_large:
                # Wrapped data - adjust range calculation
                lst_range = 24.0 - lst_range

        assert lst_range > 1.0, \
            f"LST should vary by at least 1 hour for {strategy} strategy. Got {lst_range:.2f}"
