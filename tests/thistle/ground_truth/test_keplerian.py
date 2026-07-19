"""Ground truth tests for Keplerian orbital elements with different propagator strategies."""

import numpy as np
import pytest

from thistle.orbit_data import generate_keplerian
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
    filename = f"25544_COE_{strategy}.csv"
    times, data = load_truth_csv(filename)

    # Create propagator with appropriate strategy
    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, times, data


class TestKeplerianGroundTruthByStrategy:
    """Test Keplerian orbital elements against ground truth data for each switching strategy.

    Ground truth data is generated from STK using different TLE switching strategies:
    - Epoch: Switch at TLE epoch times
    - Midpoint: Switch at midpoint between TLE epochs
    - TCA: Switch at time of closest approach between consecutive TLEs
    """

    def test_semi_major_axis(self, strategy_data):
        """Test semi-major axis matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        # Convert from meters to km
        computed_sma = result["sma"] / 1000.0
        expected_sma = data["Semi-major Axis (km)"]

        # Allow 10 km tolerance
        np.testing.assert_allclose(
            computed_sma,
            expected_sma,
            atol=10.0,
            rtol=0.001,
            err_msg=f"Semi-major axis mismatch for {strategy} strategy"
        )

    def test_eccentricity(self, strategy_data):
        """Test eccentricity matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        computed_ecc = result["ecc"]
        expected_ecc = data["Eccentricity"]

        # Allow 0.001 tolerance (ISS is nearly circular)
        np.testing.assert_allclose(
            computed_ecc,
            expected_ecc,
            atol=0.001,
            err_msg=f"Eccentricity mismatch for {strategy} strategy"
        )

    def test_inclination(self, strategy_data):
        """Test inclination matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        computed_inc = result["inc"]
        expected_inc = data["Inclination (deg)"]

        # Allow 0.1 degree tolerance
        np.testing.assert_allclose(
            computed_inc,
            expected_inc,
            atol=0.1,
            err_msg=f"Inclination mismatch for {strategy} strategy"
        )

    def test_raan(self, strategy_data):
        """Test RAAN matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        computed_raan = result["raan"]
        expected_raan = data["RAAN (deg)"]

        # Handle angle wrapping
        angle_diff = np.abs(computed_raan - expected_raan)
        angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

        # Allow 1 degree tolerance
        assert np.all(angle_diff < 1.0), f"RAAN mismatch for {strategy} strategy"

    def test_argument_of_perigee(self, strategy_data):
        """Test argument of perigee matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        computed_aop = result["aop"]
        expected_aop = data["Arg of Perigee (deg)"]

        # Handle angle wrapping
        angle_diff = np.abs(computed_aop - expected_aop)
        angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

        # Allow 2 degrees tolerance (AOP less well-defined for circular orbits)
        assert np.all(angle_diff < 2.0), f"Argument of perigee mismatch for {strategy} strategy"

    def test_true_anomaly(self, strategy_data):
        """Test true anomaly matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        computed_ta = result["ta"]
        expected_ta = data["True Anomaly (deg)"]

        # Handle angle wrapping
        angle_diff = np.abs(computed_ta - expected_ta)
        angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

        # Allow 2 degree tolerance (anomaly timing affected by propagation)
        assert np.all(angle_diff < 2.0), f"True anomaly mismatch for {strategy} strategy"

    def test_mean_anomaly(self, strategy_data):
        """Test mean anomaly matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        computed_ma = result["ma"]
        expected_ma = data["Mean Anomaly (deg)"]

        # Handle angle wrapping
        angle_diff = np.abs(computed_ma - expected_ma)
        angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

        # Allow 2 degree tolerance (anomaly timing affected by propagation)
        assert np.all(angle_diff < 2.0), f"Mean anomaly mismatch for {strategy} strategy"

    def test_orbital_period_from_sma(self, strategy_data):
        """Test that orbital period computed from SMA is consistent for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        # Compute period from Kepler's 3rd law: T = 2π√(a³/μ)
        sma_m = result["sma"]
        mu = 3.986004418e14  # Earth's gravitational parameter (m³/s²)
        period_sec = 2.0 * np.pi * np.sqrt(sma_m**3 / mu)
        period_min = period_sec / 60.0

        # ISS orbital period should be ~90-95 minutes
        assert np.all(period_min > 90), f"Period too short for {strategy} strategy"
        assert np.all(period_min < 95), f"Period too long for {strategy} strategy"

    def test_mean_motion_consistency(self, strategy_data):
        """Test that mean motion is consistent with semi-major axis for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_keplerian(times, propagator)

        # Mean motion from SMA using Kepler's 3rd law
        sma_m = result["sma"]
        mu = 3.986004418e14
        mm_rad_per_sec = np.sqrt(mu / sma_m**3)
        mm_deg_per_day = np.degrees(mm_rad_per_sec) * 86400

        # Compare with reported mean motion
        mm_reported = result["mm"]

        # Allow 1% tolerance
        np.testing.assert_allclose(
            mm_deg_per_day,
            mm_reported,
            rtol=0.01,
            err_msg=f"Mean motion inconsistent with SMA for {strategy} strategy"
        )
