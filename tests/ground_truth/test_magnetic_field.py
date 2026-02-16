"""Ground truth tests for magnetic field in ECEF coordinates with different propagator strategies."""

import datetime

import numpy as np
import pytest

from thistle.orbit_data import generate_magnetic_field_ecef, generate_magnetic_field_total
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
    filename = f"25544_MAG_{strategy}.csv"
    times, data = load_truth_csv(filename)

    # Create propagator with appropriate strategy
    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, times, data


class TestMagneticFieldECEFGroundTruthByStrategy:
    """Test magnetic field ECEF components against ground truth data for each switching strategy.

    Ground truth data is generated from STK using different TLE switching strategies:
    - Epoch: Switch at TLE epoch times
    - Midpoint: Switch at midpoint between TLE epochs
    - TCA: Switch at time of closest approach between consecutive TLEs

    The magnetic field is computed using the IGRF model in ECEF (Earth-Centered Earth-Fixed) coordinates.
    """

    def test_magnetic_field_x(self, strategy_data):
        """Test magnetic field x component matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        epoch = datetime.datetime(2020, 4, 1)
        result = generate_magnetic_field_ecef(times, propagator, epoch=epoch)

        computed_bx = result["Bx"]
        expected_bx = data["x (nT)"]

        # Allow 100 nT tolerance (IGRF model precision + coordinate transformation)
        np.testing.assert_allclose(
            computed_bx,
            expected_bx,
            atol=100.0,
            rtol=0.01,
            err_msg=f"Magnetic field Bx mismatch for {strategy} strategy",
        )

    def test_magnetic_field_y(self, strategy_data):
        """Test magnetic field y component matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        epoch = datetime.datetime(2020, 4, 1)
        result = generate_magnetic_field_ecef(times, propagator, epoch=epoch)

        computed_by = result["By"]
        expected_by = data["y (nT)"]

        np.testing.assert_allclose(
            computed_by,
            expected_by,
            atol=100.0,
            rtol=0.01,
            err_msg=f"Magnetic field By mismatch for {strategy} strategy",
        )

    def test_magnetic_field_z(self, strategy_data):
        """Test magnetic field z component matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        epoch = datetime.datetime(2020, 4, 1)
        result = generate_magnetic_field_ecef(times, propagator, epoch=epoch)

        computed_bz = result["Bz"]
        expected_bz = data["z (nT)"]

        np.testing.assert_allclose(
            computed_bz,
            expected_bz,
            atol=100.0,
            rtol=0.01,
            err_msg=f"Magnetic field Bz mismatch for {strategy} strategy",
        )

    def test_magnetic_field_magnitude(self, strategy_data):
        """Test total magnetic field magnitude matches ground truth for each strategy."""
        strategy, propagator, times, data = strategy_data
        epoch = datetime.datetime(2020, 4, 1)
        result = generate_magnetic_field_total(times, propagator, epoch=epoch)

        computed_bt = result["Bt"]
        expected_bt = data["Magnitude (nT)"]

        # Allow 100 nT tolerance
        np.testing.assert_allclose(
            computed_bt,
            expected_bt,
            atol=100.0,
            rtol=0.005,
            err_msg=f"Total magnetic field mismatch for {strategy} strategy",
        )

    def test_magnetic_field_components_match_magnitude(self, strategy_data):
        """Test that field components match reported magnitude for each strategy."""
        strategy, propagator, times, data = strategy_data
        epoch = datetime.datetime(2020, 4, 1)
        result = generate_magnetic_field_ecef(times, propagator, epoch=epoch)

        # Compute magnitude from components
        computed_bt = np.sqrt(result["Bx"]**2 + result["By"]**2 + result["Bz"]**2)
        expected_bt = data["Magnitude (nT)"]

        # Should match very closely (within numerical precision)
        np.testing.assert_allclose(
            computed_bt,
            expected_bt,
            atol=100.0,
            rtol=0.005,
            err_msg=f"Magnitude doesn't match components for {strategy} strategy",
        )

    def test_magnetic_field_strength_range(self, strategy_data):
        """Test that magnetic field strength is in expected range for LEO for each strategy."""
        strategy, propagator, times, data = strategy_data
        epoch = datetime.datetime(2020, 4, 1)
        result = generate_magnetic_field_total(times, propagator, epoch=epoch)

        bt = result["Bt"]

        # At LEO altitude (400 km), field should be 20,000 - 50,000 nT
        assert np.all(bt > 15000), f"Magnetic field too weak for {strategy} strategy"
        assert np.all(bt < 55000), f"Magnetic field too strong for {strategy} strategy"

    def test_magnetic_field_variation_over_orbit(self, strategy_data):
        """Test that magnetic field varies significantly over one orbit for each strategy."""
        strategy, propagator, times, data = strategy_data
        epoch = datetime.datetime(2020, 4, 1)
        result = generate_magnetic_field_total(times, propagator, epoch=epoch)

        bt = result["Bt"]

        # Field should vary by at least 5000 nT over the orbit
        field_range = np.max(bt) - np.min(bt)
        assert field_range > 5000, \
            f"Magnetic field not varying enough over orbit for {strategy} strategy"
