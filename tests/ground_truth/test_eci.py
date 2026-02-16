"""Ground truth tests for ECI position and velocity with different propagator strategies."""

import numpy as np
import pytest

from thistle.orbit_data import generate_eci
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
    filename = f"25544_ECI_{strategy}.csv"
    times, data = load_truth_csv(filename)

    # Create propagator with appropriate strategy
    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, times, data


class TestECIGroundTruthByStrategy:
    """Test ECI position and velocity against ground truth data for each switching strategy.

    Ground truth data is generated from STK using different TLE switching strategies:
    - Epoch: Switch at TLE epoch times
    - Midpoint: Switch at midpoint between TLE epochs
    - TCA: Switch at time of closest approach between consecutive TLEs
    """

    def test_eci_position_x(self, strategy_data):
        """Test ECI x position matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        # Convert from meters to km for comparison
        computed_x = result["eci_x"] / 1000.0
        expected_x = data["x (km)"]

        error_x = np.abs(computed_x - expected_x)
        print(error_x)
        print(np.std(error_x))

        # Allow 10 km tolerance (TLE propagation accuracy)
        np.testing.assert_allclose(
            computed_x,
            expected_x,
            atol=10.0,
            rtol=0.001,
            err_msg=f"ECI x position mismatch for {strategy} strategy",
        )

    def test_eci_position_y(self, strategy_data):
        """Test ECI y position matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        computed_y = result["eci_y"] / 1000.0
        expected_y = data["y (km)"]

        np.testing.assert_allclose(
            computed_y,
            expected_y,
            atol=10.0,
            rtol=0.001,
            err_msg=f"ECI y position mismatch for {strategy} strategy",
        )

    def test_eci_position_z(self, strategy_data):
        """Test ECI z position matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        computed_z = result["eci_z"] / 1000.0
        expected_z = data["z (km)"]

        np.testing.assert_allclose(
            computed_z,
            expected_z,
            atol=10.0,
            rtol=0.001,
            err_msg=f"ECI z position mismatch for {strategy} strategy",
        )

    def test_eci_position_magnitude(self, strategy_data):
        """Test ECI position magnitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        # Compute magnitudes
        computed_r = (
            np.sqrt(result["eci_x"] ** 2 + result["eci_y"] ** 2 + result["eci_z"] ** 2)
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
            err_msg=f"ECI position magnitude mismatch for {strategy} strategy",
        )

    def test_eci_velocity_x(self, strategy_data):
        """Test ECI x velocity matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        # Convert from m/s to km/s
        computed_vx = result["eci_vx"] / 1000.0
        expected_vx = data["vx (km/sec)"]

        # Allow 0.01 km/s (10 m/s) tolerance
        np.testing.assert_allclose(
            computed_vx,
            expected_vx,
            atol=0.01,
            rtol=0.001,
            err_msg=f"ECI vx mismatch for {strategy} strategy",
        )

    def test_eci_velocity_y(self, strategy_data):
        """Test ECI y velocity matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        computed_vy = result["eci_vy"] / 1000.0
        expected_vy = data["vy (km/sec)"]

        np.testing.assert_allclose(
            computed_vy,
            expected_vy,
            atol=0.01,
            rtol=0.001,
            err_msg=f"ECI vy mismatch for {strategy} strategy",
        )

    def test_eci_velocity_z(self, strategy_data):
        """Test ECI z velocity matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        computed_vz = result["eci_vz"] / 1000.0
        expected_vz = data["vz (km/sec)"]

        np.testing.assert_allclose(
            computed_vz,
            expected_vz,
            atol=0.01,
            rtol=0.001,
            err_msg=f"ECI vz mismatch for {strategy} strategy",
        )

    def test_eci_velocity_magnitude(self, strategy_data):
        """Test ECI velocity magnitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_eci(times, propagator)

        computed_v = (
            np.sqrt(
                result["eci_vx"] ** 2 + result["eci_vy"] ** 2 + result["eci_vz"] ** 2
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
            err_msg=f"ECI velocity magnitude mismatch for {strategy} strategy",
        )
