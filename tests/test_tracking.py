"""Tests for thistle.tracking."""

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load

from thistle.tracking import (
    DopplerGeolocationResult,
    generate_range,
    geolocate_doppler,
)
from thistle.utils import read_tle

ts = load.timescale()
_tles = read_tle("tests/data/25544.tle")
SAT = EarthSatellite(_tles[0][0], _tles[0][1], ts=ts)

T0 = np.datetime64("1998-11-20T06:50:00")
TIMES = T0 + np.arange(0, 60 * 60, 60, dtype="timedelta64[s]")
N = len(TIMES)

# Ground site roughly below the ISS orbit
SITE_LAT, SITE_LON = 40.0, -105.0

# A pass window where the ISS flies nearly overhead a ground site,
# giving a clean range-rate zero crossing for geolocation tests.
_PASS_SITE_LAT, _PASS_SITE_LON = 42.0, 116.57
_PASS_CENTER = np.datetime64("1998-11-20T08:29:20")
_PASS_TIMES = _PASS_CENTER + np.arange(-300, 300, 10, dtype="timedelta64[s]")
_PASS_RR = generate_range(_PASS_TIMES, SAT, _PASS_SITE_LAT, _PASS_SITE_LON)[
    "range_rate"
]


class TestGeolocateDoppler:
    """Tests for geolocate_doppler."""

    def test_returns_two_solutions(self):
        """Should return a list of exactly two results."""
        doppler = 2.5 * _PASS_RR
        solutions = geolocate_doppler(_PASS_TIMES, SAT, doppler)
        assert isinstance(solutions, list)
        assert len(solutions) == 2
        assert all(isinstance(s, DopplerGeolocationResult) for s in solutions)

    def test_sorted_by_rms(self):
        """Solutions should be sorted by RMS, best first."""
        doppler = 2.5 * _PASS_RR
        solutions = geolocate_doppler(_PASS_TIMES, SAT, doppler)
        assert solutions[0].rms <= solutions[1].rms

    def test_best_recovers_known_site_noiseless(self):
        """Best solution from noiseless Doppler recovers the ground site."""
        doppler = 2.5 * _PASS_RR
        best = geolocate_doppler(_PASS_TIMES, SAT, doppler)[0]
        assert abs(best.lat - _PASS_SITE_LAT) < 0.1
        assert abs(best.lon - _PASS_SITE_LON) < 0.1
        assert abs(best.scale - 2.5) < 0.01
        assert best.rms < 1.0

    def test_both_solutions_converge(self):
        """Both ambiguity solutions should converge."""
        doppler = 2.5 * _PASS_RR
        solutions = geolocate_doppler(_PASS_TIMES, SAT, doppler)
        assert solutions[0].converged is True
        assert solutions[1].converged is True

    def test_recovers_with_noise(self):
        """Noisy synthetic Doppler recovers the ground site approximately."""
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 50.0, len(_PASS_RR))
        doppler = 2.5 * _PASS_RR + noise
        best = geolocate_doppler(_PASS_TIMES, SAT, doppler)[0]
        assert abs(best.lat - _PASS_SITE_LAT) < 0.5
        assert abs(best.lon - _PASS_SITE_LON) < 0.5

    def test_negative_scale(self):
        """Negative scale (inverted Doppler convention) still works."""
        doppler = -3.0 * _PASS_RR
        best = geolocate_doppler(_PASS_TIMES, SAT, doppler)[0]
        assert abs(best.lat - _PASS_SITE_LAT) < 0.1
        assert abs(best.lon - _PASS_SITE_LON) < 0.1
        assert abs(best.scale - (-3.0)) < 0.01

    def test_custom_initial_guess(self):
        """Providing lat0/lon0 near the true site converges."""
        doppler = 2.5 * _PASS_RR
        best = geolocate_doppler(
            _PASS_TIMES,
            SAT,
            doppler,
            lat0=_PASS_SITE_LAT + 1.0,
            lon0=_PASS_SITE_LON + 1.0,
        )[0]
        assert abs(best.lat - _PASS_SITE_LAT) < 0.2
        assert abs(best.lon - _PASS_SITE_LON) < 0.2

    def test_return_types(self):
        """Result fields have the correct types."""
        doppler = 2.5 * _PASS_RR
        best = geolocate_doppler(_PASS_TIMES, SAT, doppler)[0]
        assert isinstance(best, DopplerGeolocationResult)
        assert isinstance(best.lat, float)
        assert isinstance(best.lon, float)
        assert isinstance(best.scale, float)
        assert isinstance(best.rms, float)
        assert isinstance(best.converged, bool)
        assert best.residuals.shape == _PASS_TIMES.shape

    def test_converged_flag(self):
        """Noiseless case should converge for both solutions."""
        doppler = 2.5 * _PASS_RR
        solutions = geolocate_doppler(_PASS_TIMES, SAT, doppler)
        assert solutions[0].converged is True

    def test_shape_mismatch_raises(self):
        """Mismatched times/doppler shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            geolocate_doppler(_PASS_TIMES, SAT, _PASS_RR[:5])

    def test_too_few_points_raises(self):
        """Fewer than 3 measurements raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            geolocate_doppler(_PASS_TIMES[:2], SAT, _PASS_RR[:2])
