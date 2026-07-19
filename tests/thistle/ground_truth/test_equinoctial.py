"""Ground truth tests for equinoctial orbital elements with different propagator strategies."""

import numpy as np
import pytest

from thistle.orbit_data import generate_equinoctial
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
    filename = f"25544_Equinoctial_{strategy}.csv"
    times, data = load_truth_csv(filename)

    # Create propagator with appropriate strategy
    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, times, data


class TestEquinoctialGroundTruthByStrategy:
    """Test equinoctial elements against ground truth data for each switching strategy.

    Ground truth data is generated from STK using different TLE switching strategies:
    - Epoch: Switch at TLE epoch times
    - Midpoint: Switch at midpoint between TLE epochs
    - TCA: Switch at time of closest approach between consecutive TLEs

    Note: STK exports semi-major axis (a) while we compute semi-parameter (p = a(1-e²)).
    The conversion is done using p = a * (1 - f² - g²) where f and g are eccentricity components.
    """

    def test_semi_parameter(self, strategy_data):
        """Test semi-parameter p matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        # Convert semi-major axis from CSV to semi-parameter
        # p = a * (1 - e²) = a * (1 - (f² + g²))
        a_km = data["Semi-Major Axis (km)"]
        f_csv = data["e * cos(omegaBar)"]
        g_csv = data["e * sin(omegaBar)"]
        e_squared = f_csv**2 + g_csv**2
        expected_p_m = a_km * 1000.0 * (1.0 - e_squared)

        # Convert from meters to km for comparison
        computed_p_km = result["p"] / 1000.0
        expected_p_km = expected_p_m / 1000.0

        # Allow 10 km tolerance
        np.testing.assert_allclose(
            computed_p_km,
            expected_p_km,
            atol=10.0,
            rtol=0.001,
            err_msg=f"Semi-parameter mismatch for {strategy} strategy",
        )

    def test_f_component(self, strategy_data):
        """Test f component (e * cos(ω̄)) matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        computed_f = result["f"]
        expected_f = data["e * cos(omegaBar)"]

        # Allow 0.001 tolerance (ISS is nearly circular)
        np.testing.assert_allclose(
            computed_f,
            expected_f,
            atol=0.001,
            err_msg=f"f component mismatch for {strategy} strategy",
        )

    def test_g_component(self, strategy_data):
        """Test g component (e * sin(ω̄)) matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        computed_g = result["g"]
        expected_g = data["e * sin(omegaBar)"]

        # Allow 0.001 tolerance (ISS is nearly circular)
        np.testing.assert_allclose(
            computed_g,
            expected_g,
            atol=0.001,
            err_msg=f"g component mismatch for {strategy} strategy",
        )

    def test_h_component(self, strategy_data):
        """Test h component (tan(i/2) * cos(Ω)) matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        computed_h = result["h"]
        expected_h = data["tan(i/2) * cos(raan)"]

        # Allow 0.001 tolerance
        np.testing.assert_allclose(
            computed_h,
            expected_h,
            atol=0.001,
            err_msg=f"h component mismatch for {strategy} strategy",
        )

    def test_k_component(self, strategy_data):
        """Test k component (tan(i/2) * sin(Ω)) matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        computed_k = result["k"]
        expected_k = data["tan(i/2) * sin(raan)"]

        # Allow 0.001 tolerance
        np.testing.assert_allclose(
            computed_k,
            expected_k,
            atol=0.001,
            err_msg=f"k component mismatch for {strategy} strategy",
        )

    def test_mean_longitude(self, strategy_data):
        """Test mean longitude L matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        computed_L = result["L"]
        expected_L = data["Mean Lon (deg)"]

        # Handle angle wrapping at 360 degrees
        angle_diff = np.abs(computed_L - expected_L)
        angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

        # Allow 2 degree tolerance (timing sensitive like true anomaly)
        assert np.all(angle_diff < 2.0), f"Mean longitude mismatch for {strategy} strategy"

    def test_eccentricity_magnitude(self, strategy_data):
        """Test that eccentricity magnitude from f,g components is consistent for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        # Compute eccentricity from f and g components
        computed_e = np.sqrt(result["f"] ** 2 + result["g"] ** 2)

        # ISS eccentricity should be small (nearly circular orbit)
        assert np.all(computed_e < 0.01), f"Eccentricity too large for {strategy} strategy"
        assert np.all(computed_e > 0.0), f"Eccentricity should be positive for {strategy} strategy"

    def test_inclination_from_h_k(self, strategy_data):
        """Test that inclination computed from h,k components is reasonable for each strategy."""
        strategy, propagator, times, data = strategy_data
        result = generate_equinoctial(times, propagator)

        # Compute inclination from h and k components
        # tan(i/2) = sqrt(h² + k²)
        tan_half_i = np.sqrt(result["h"] ** 2 + result["k"] ** 2)
        computed_i_deg = 2.0 * np.degrees(np.arctan(tan_half_i))

        # ISS inclination should be around 51-52 degrees
        assert np.all(computed_i_deg > 50), f"Inclination too low for {strategy} strategy"
        assert np.all(computed_i_deg < 53), f"Inclination too high for {strategy} strategy"
