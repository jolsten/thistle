"""Ground truth tests for ECEF position and velocity with different propagator strategies."""

import numpy as np
import pytest

from thistle.orbit_data import generate_ecef
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
    filename = f"25544_ECEF_{strategy}.csv"
    times, data = load_truth_csv(filename)

    # Create propagator with appropriate strategy
    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, times, data


class TestECEFGroundTruthByStrategy:
    """Test ECEF position and velocity against ground truth data for each switching strategy.

    Ground truth data is generated from STK using different TLE switching strategies:
    - Epoch: Switch at TLE epoch times
    - Midpoint: Switch at midpoint between TLE epochs
    - TCA: Switch at time of closest approach between consecutive TLEs
    """

    def test_ecef_position_x(self, strategy_data):
        """Test ECEF x position matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        # Convert from meters to km for comparison
        computed_x = result["ecef_x"] / 1000.0
        expected_x = data["x (km)"]

        # Allow 10 km tolerance (TLE propagation accuracy)
        np.testing.assert_allclose(
            computed_x,
            expected_x,
            atol=10.0,
            rtol=0.001,
            err_msg=f"ECEF x position mismatch for {strategy} strategy",
        )

    def test_ecef_position_y(self, strategy_data):
        """Test ECEF y position matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        computed_y = result["ecef_y"] / 1000.0
        expected_y = data["y (km)"]

        np.testing.assert_allclose(
            computed_y,
            expected_y,
            atol=10.0,
            rtol=0.001,
            err_msg=f"ECEF y position mismatch for {strategy} strategy",
        )

    def test_ecef_position_z(self, strategy_data):
        """Test ECEF z position matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        computed_z = result["ecef_z"] / 1000.0
        expected_z = data["z (km)"]

        np.testing.assert_allclose(
            computed_z,
            expected_z,
            atol=10.0,
            rtol=0.001,
            err_msg=f"ECEF z position mismatch for {strategy} strategy",
        )

    def test_ecef_position_magnitude(self, strategy_data):
        """Test ECEF position magnitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        # Compute magnitudes
        computed_r = (
            np.sqrt(result["ecef_x"] ** 2 + result["ecef_y"] ** 2 + result["ecef_z"] ** 2)
            / 1000.0
        )

        expected_r = np.sqrt(
            data["x (km)"] ** 2 + data["y (km)"] ** 2 + data["z (km)"] ** 2
        )

        # Magnitude should match closely
        np.testing.assert_allclose(
            computed_r,
            expected_r,
            atol=10.0,
            rtol=0.001,
            err_msg=f"ECEF position magnitude mismatch for {strategy} strategy",
        )

    def test_ecef_velocity_x(self, strategy_data):
        """Test ECEF x velocity matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        # Convert from m/s to km/s
        computed_vx = result["ecef_vx"] / 1000.0
        expected_vx = data["vx (km/sec)"]

        # Allow 0.01 km/s (10 m/s) tolerance
        np.testing.assert_allclose(
            computed_vx,
            expected_vx,
            atol=0.01,
            rtol=0.001,
            err_msg=f"ECEF vx mismatch for {strategy} strategy",
        )

    def test_ecef_velocity_y(self, strategy_data):
        """Test ECEF y velocity matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        computed_vy = result["ecef_vy"] / 1000.0
        expected_vy = data["vy (km/sec)"]

        np.testing.assert_allclose(
            computed_vy,
            expected_vy,
            atol=0.01,
            rtol=0.001,
            err_msg=f"ECEF vy mismatch for {strategy} strategy",
        )

    def test_ecef_velocity_z(self, strategy_data):
        """Test ECEF z velocity matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        computed_vz = result["ecef_vz"] / 1000.0
        expected_vz = data["vz (km/sec)"]

        np.testing.assert_allclose(
            computed_vz,
            expected_vz,
            atol=0.01,
            rtol=0.001,
            err_msg=f"ECEF vz mismatch for {strategy} strategy",
        )

    def test_ecef_velocity_magnitude(self, strategy_data):
        """Test ECEF velocity magnitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_ecef(times, propagator)

        computed_v = (
            np.sqrt(
                result["ecef_vx"] ** 2 + result["ecef_vy"] ** 2 + result["ecef_vz"] ** 2
            )
            / 1000.0
        )

        expected_v = np.sqrt(
            data["vx (km/sec)"] ** 2
            + data["vy (km/sec)"] ** 2
            + data["vz (km/sec)"] ** 2
        )

        np.testing.assert_allclose(
            computed_v,
            expected_v,
            atol=0.01,
            rtol=0.001,
            err_msg=f"ECEF velocity magnitude mismatch for {strategy} strategy",
        )
