"""Ground truth tests for slant range against STK AER data."""

import csv
import pathlib

import numpy as np
import pytest

from thistle.ground_sites import generate_range
from thistle.propagator import Propagator

from .conftest import TRUTH_DATA_DIR, parse_time

# Kennedy Space Center ground site (matches STK facility)
KSC_LAT = 28.57
KSC_LON = -80.65


def _load_aer_csv(filename: str):
    """Load AER CSV with multiple pass sections separated by statistics blocks.

    Returns a list of passes, each a dict with 'times' (datetime64 array)
    and 'range_km' (float64 array).
    """
    filepath = TRUTH_DATA_DIR / filename

    passes = []
    current_times = []
    current_range = []
    in_data = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header row starts a new data section
            if line.startswith('"Time (UTCG)"'):
                # Save previous pass if any
                if current_times:
                    passes.append({
                        "times": np.array(current_times, dtype="datetime64[us]"),
                        "range_km": np.array(current_range),
                    })
                current_times = []
                current_range = []
                in_data = True
                continue

            # Statistics or global blocks end data sections
            if "Statistics" in line:
                if current_times:
                    passes.append({
                        "times": np.array(current_times, dtype="datetime64[us]"),
                        "range_km": np.array(current_range),
                    })
                    current_times = []
                    current_range = []
                in_data = False
                continue

            # Skip statistic rows (start with a quoted label)
            if line.startswith('"'):
                continue

            if in_data:
                parts = line.split(",")
                if len(parts) >= 4:
                    current_times.append(parse_time(parts[0]))
                    current_range.append(float(parts[3]))

    # Save last pass if file didn't end with statistics
    if current_times:
        passes.append({
            "times": np.array(current_times, dtype="datetime64[us]"),
            "range_km": np.array(current_range),
        })

    return passes


@pytest.fixture(scope="module", params=["Epoch", "Midpoint", "TCA"])
def strategy_range_data(request, iss_tles, april_2020_dates):
    """Load AER ground truth and create propagator for each strategy."""
    strategy = request.param
    passes = _load_aer_csv(f"25544_KSC_AER_{strategy}.csv")

    start_date, end_date = april_2020_dates
    propagator = Propagator(
        iss_tles, method=strategy.lower(), start=start_date, stop=end_date
    )

    return strategy, propagator, passes


class TestRangeGroundTruth:
    """Test slant range against STK AER ground truth for KSC."""

    def test_range_values(self, strategy_range_data):
        """Computed range matches STK range for all passes."""
        strategy, propagator, passes = strategy_range_data

        for i, p in enumerate(passes):
            result = generate_range(
                p["times"], propagator, sites=[(KSC_LAT, KSC_LON)]
            )
            computed_km = result["range_0"] / 1000.0
            expected_km = p["range_km"]

            np.testing.assert_allclose(
                computed_km,
                expected_km,
                atol=5.0,
                rtol=0.002,
                err_msg=f"Range mismatch for {strategy} pass {i + 1}",
            )

    def test_range_minimum_at_peak(self, strategy_range_data):
        """Minimum range in each pass is within 5 km of STK minimum."""
        strategy, propagator, passes = strategy_range_data

        for i, p in enumerate(passes):
            result = generate_range(
                p["times"], propagator, sites=[(KSC_LAT, KSC_LON)]
            )
            computed_min = np.min(result["range_0"]) / 1000.0
            expected_min = np.min(p["range_km"])

            assert abs(computed_min - expected_min) < 5.0, (
                f"{strategy} pass {i + 1}: min range {computed_min:.1f} km "
                f"vs expected {expected_min:.1f} km"
            )

    def test_pass_count(self, strategy_range_data):
        """CSV contains the expected number of passes (6 for April 1, 2020)."""
        _, _, passes = strategy_range_data
        assert len(passes) == 6
