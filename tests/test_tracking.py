"""Tests for thistle.tracking.generate_range."""

import numpy as np
from skyfield.api import EarthSatellite, load

from thistle.io import read_tle
from thistle.tracking import generate_range

ts = load.timescale()
_tles = read_tle("tests/data/25544.tle")
SAT = EarthSatellite(_tles[0][0], _tles[0][1], ts=ts)

T0 = np.datetime64("1998-11-20T06:50:00")
TIMES = T0 + np.arange(0, 60 * 60, 60, dtype="timedelta64[s]")
N = len(TIMES)

# Ground site roughly below the ISS orbit
SITE_LAT, SITE_LON = 40.0, -105.0


class TestGenerateRange:
    """Tests for generate_range."""

    def test_keys(self):
        result = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        assert set(result) == {"range", "range_rate"}

    def test_shapes(self):
        result = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        assert result["range"].shape == (N,)
        assert result["range_rate"].shape == (N,)

    def test_dtype_float64(self):
        result = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        assert result["range"].dtype == np.float64
        assert result["range_rate"].dtype == np.float64

    def test_range_positive(self):
        result = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        assert np.all(result["range"] > 0)

    def test_range_magnitude_leo(self):
        """Range to LEO should be between ~150 km and ~13000 km."""
        result = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        assert np.all(result["range"] > 150_000)
        assert np.all(result["range"] < 13_000_000)

    def test_range_rate_bounded_by_velocity(self):
        """Range rate magnitude cannot exceed satellite velocity (~7.5 km/s)."""
        result = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        assert np.all(np.abs(result["range_rate"]) < 8_500)

    def test_alt_default_zero(self):
        """Calling with and without alt=0.0 gives the same result."""
        r1 = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        r2 = generate_range(TIMES, SAT, SITE_LAT, SITE_LON, alt=0.0)
        np.testing.assert_array_equal(r1["range"], r2["range"])

    def test_different_sites_different_range(self):
        """Two different ground sites produce different range profiles."""
        r1 = generate_range(TIMES, SAT, 0.0, 0.0)
        r2 = generate_range(TIMES, SAT, 45.0, 90.0)
        assert not np.allclose(r1["range"], r2["range"])

    def test_range_varies_over_orbit(self):
        """Range should not be constant over a full hour."""
        result = generate_range(TIMES, SAT, SITE_LAT, SITE_LON)
        assert result["range"].std() > 1000

    def test_single_time(self):
        """Works with a single-element time array."""
        single = TIMES[:1]
        result = generate_range(single, SAT, SITE_LAT, SITE_LON)
        assert result["range"].shape == (1,)
        assert result["range_rate"].shape == (1,)
