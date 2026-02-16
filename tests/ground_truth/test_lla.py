"""Ground truth tests for LLA position with different propagator strategies."""

import numpy as np
import pytest

from thistle.orbit_data import generate_lla
from thistle.propagator import Propagator

from .conftest import load_truth_csv


# Parametrize tests across all three switching strategies
@pytest.fixture(scope="module", params=["Epoch", "Midpoint", "TCA"])
def strategy_data(request, iss_tles, april_2020_dates):
    """Load ground truth data and create propagator for each strategy.

    Returns:
        tuple: (strategy_name, propagator, times, ground_truth_data)
    """
    strategy = request.param

    # Load ground truth data for this strategy
    filename = f"25544_LLA_{strategy}.csv"
    times, data = load_truth_csv(filename)

    # Create propagator with appropriate strategy
    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, times, data


class TestLLAGroundTruthByStrategy:
    """Test LLA position against ground truth data for each switching strategy.

    Ground truth data is generated from STK using different TLE switching strategies:
    - Epoch: Switch at TLE epoch times
    - Midpoint: Switch at midpoint between TLE epochs
    - TCA: Switch at time of closest approach between consecutive TLEs
    """

    def test_latitude(self, strategy_data):
        """Test latitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_lla(times, propagator)

        computed_lat = result["lat"]
        expected_lat = data["Lat (deg)"]

        # Allow 0.1 degree tolerance
        np.testing.assert_allclose(
            computed_lat,
            expected_lat,
            atol=0.1,
            err_msg=f"Latitude mismatch for {strategy} strategy"
        )

    def test_longitude(self, strategy_data):
        """Test longitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_lla(times, propagator)

        computed_lon = result["lon"]
        expected_lon = data["Lon (deg)"]

        # Handle longitude wrapping at ±180 degrees
        lon_diff = np.abs(computed_lon - expected_lon)
        lon_diff = np.minimum(lon_diff, 360.0 - lon_diff)

        # Allow 0.1 degree tolerance
        assert np.all(lon_diff < 0.1), f"Longitude mismatch for {strategy} strategy"

    def test_altitude(self, strategy_data):
        """Test altitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_lla(times, propagator)

        # Convert from meters to km
        computed_alt = result["alt"] / 1000.0
        expected_alt = data["Alt (km)"]

        # Allow 10 km tolerance
        np.testing.assert_allclose(
            computed_alt,
            expected_alt,
            atol=10.0,
            rtol=0.01,
            err_msg=f"Altitude mismatch for {strategy} strategy"
        )

    def test_subsatellite_point_consistency(self, strategy_data):
        """Test that subsatellite point moves consistently over time for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_lla(times, propagator)

        # Check that latitude oscillates (for inclined orbit)
        lat = result["lat"]
        lat_range = np.max(lat) - np.min(lat)
        assert lat_range > 10.0, f"Latitude should vary significantly for {strategy} strategy"

        # Check that longitude increases monotonically (or wraps)
        lon = result["lon"]
        lon_diff = np.diff(lon)

        # Account for wrapping
        lon_diff[lon_diff < -180] += 360
        lon_diff[lon_diff > 180] -= 360

        # Most steps should show increasing longitude (eastward motion)
        assert np.sum(lon_diff > 0) > len(lon_diff) * 0.9, f"Longitude should generally increase for {strategy} strategy"

    def test_altitude_stability(self, strategy_data):
        """Test that altitude doesn't change drastically between samples for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_lla(times, propagator)

        alt_km = result["alt"] / 1000.0
        alt_diff = np.abs(np.diff(alt_km))

        # Maximum altitude change between 10-second samples should be < 1 km
        assert np.all(alt_diff < 1.0), f"Altitude changing too rapidly for {strategy} strategy"
