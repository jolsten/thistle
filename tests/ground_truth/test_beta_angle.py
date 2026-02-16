"""Ground truth tests for beta angle."""

import numpy as np
import pytest

from thistle.orbit_data import generate_beta_angle

from .conftest import get_iss_satellite, load_truth_csv


@pytest.fixture(scope="module")
def iss_satellite():
    """Get ISS satellite for testing."""
    return get_iss_satellite()


@pytest.fixture(scope="module")
def truth_data():
    """Load ground truth beta angle data."""
    return load_truth_csv("25544_Beta_Angle.csv")


class TestBetaAngleGroundTruth:
    """Test beta angle against ground truth data."""

    def test_beta_angle_values(self, iss_satellite, truth_data):
        """Test beta angle matches ground truth."""
        times, data = truth_data
        result = generate_beta_angle(times, iss_satellite)

        computed_beta = result["beta"]
        expected_beta = data["Beta Angle (deg)"]

        # Allow 0.5 degree tolerance
        np.testing.assert_allclose(
            computed_beta, expected_beta, atol=0.5, err_msg="Beta angle mismatch"
        )

    def test_beta_angle_stability(self, iss_satellite, truth_data):
        """Test that beta angle changes slowly over time."""
        times, data = truth_data
        result = generate_beta_angle(times, iss_satellite)

        beta = result["beta"]
        beta_diff = np.abs(np.diff(beta))

        # Beta angle should change by less than 0.1 degrees over 10 seconds
        assert np.all(beta_diff < 0.1), "Beta angle changing too rapidly"

    def test_beta_angle_range(self, iss_satellite, truth_data):
        """Test that beta angle is within valid range."""
        times, data = truth_data
        result = generate_beta_angle(times, iss_satellite)

        beta = result["beta"]

        # Beta angle must be in [-90, 90] degrees
        assert np.all(beta >= -90), "Beta angle too negative"
        assert np.all(beta <= 90), "Beta angle too positive"

    def test_beta_angle_nearly_constant(self, iss_satellite, truth_data):
        """Test that beta angle is nearly constant over short periods.

        Beta angle varies slowly (few degrees per day), so over a few
        hours it should be essentially constant. We test this by looking
        at only the first hour of data.
        """
        times, data = truth_data

        # Take only first hour of data by checking timestamps
        start_time = times[0]
        end_time = start_time + np.timedelta64(1, 'h')
        mask = times < end_time
        times_short = times[mask]

        result = generate_beta_angle(times_short, iss_satellite)
        beta = result["beta"]
        beta_std = np.std(beta)

        # Standard deviation should be very small over one hour (< 0.1 degrees)
        assert beta_std < 0.1, f"Beta angle varying too much over 1 hour: std={beta_std:.4f}"

    def test_beta_angle_matches_ground_truth_average(self, iss_satellite, truth_data):
        """Test that average beta angle matches ground truth."""
        times, data = truth_data
        result = generate_beta_angle(times, iss_satellite)

        computed_avg = np.mean(result["beta"])
        expected_avg = np.mean(data["Beta Angle (deg)"])

        # Averages should match very closely
        np.testing.assert_allclose(
            computed_avg, expected_avg, atol=0.1, err_msg="Average beta angle mismatch"
        )
