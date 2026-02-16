import numpy as np
from skyfield.api import EarthSatellite, load

from thistle.ground_sites import (
    _generate_range_single,
    doppler_shift,
    generate_range,
    visibility_circle,
    SPEED_OF_LIGHT,
)
from thistle.propagator import Propagator
from thistle.utils import read_tle


class TestVisibilityCircle:
    """Tests for visibility_circle."""

    def test_output_shape(self):
        """Output arrays have length n_points."""
        lats, lons = visibility_circle(
            0.0, 0.0, 0.0, sat_alt=400_000, min_el=10.0, n_points=72
        )
        assert lats.shape == (72,)
        assert lons.shape == (72,)

    def test_output_dtype(self):
        """Output arrays are float32."""
        lats, lons = visibility_circle(0.0, 0.0, 0.0, sat_alt=400_000, min_el=10.0)
        assert lats.dtype == np.float32
        assert lons.dtype == np.float32

    def test_symmetry_equator(self):
        """Circle at the equator/prime meridian is symmetric in latitude."""
        lats, lons = visibility_circle(
            0.0, 0.0, 0.0, sat_alt=400_000, min_el=5.0, n_points=360
        )
        # Max and min latitude should be roughly equal in magnitude
        assert abs(lats.max() + lats.min()) < 0.5

    def test_symmetry_longitude(self):
        """Circle at the equator/prime meridian is symmetric in longitude."""
        lats, lons = visibility_circle(
            0.0, 0.0, 0.0, sat_alt=400_000, min_el=5.0, n_points=360
        )
        assert abs(lons.max() + lons.min()) < 0.5

    def test_circle_centered_on_site(self):
        """Mean of the circle points is close to the ground site."""
        lat_site, lon_site = 45.0, -90.0
        lats, lons = visibility_circle(lat_site, lon_site, 0.0, sat_alt=400_000, min_el=10.0)
        assert abs(np.mean(lats) - lat_site) < 1.0
        assert abs(np.mean(lons) - lon_site) < 1.0

    def test_higher_elevation_smaller_circle(self):
        """Higher minimum elevation yields a smaller visibility circle."""
        lats_5, _ = visibility_circle(0.0, 0.0, 0.0, sat_alt=400_000, min_el=5.0)
        lats_30, _ = visibility_circle(0.0, 0.0, 0.0, sat_alt=400_000, min_el=30.0)
        radius_5 = lats_5.max() - lats_5.min()
        radius_30 = lats_30.max() - lats_30.min()
        assert radius_30 < radius_5

    def test_higher_altitude_larger_circle(self):
        """Higher satellite altitude yields a larger visibility circle."""
        lats_leo, _ = visibility_circle(0.0, 0.0, 0.0, sat_alt=400_000, min_el=10.0)
        lats_meo, _ = visibility_circle(0.0, 0.0, 0.0, sat_alt=20_200_000, min_el=10.0)
        radius_leo = lats_leo.max() - lats_leo.min()
        radius_meo = lats_meo.max() - lats_meo.min()
        assert radius_meo > radius_leo

    def test_zero_elevation_leo_radius(self):
        """At 0 deg elevation, LEO ~400 km gives a radius of roughly 20 deg."""
        lats, _ = visibility_circle(0.0, 0.0, 0.0, sat_alt=400_000, min_el=0.0)
        half_span = (lats.max() - lats.min()) / 2.0
        # Should be roughly 18-24 degrees for 400 km altitude
        assert 15.0 < half_span < 28.0

    def test_geo_altitude_large_radius(self):
        """GEO altitude at 0 deg elevation gives a radius > 60 deg."""
        lats, _ = visibility_circle(0.0, 0.0, 0.0, sat_alt=35_786_000, min_el=0.0)
        half_span = (lats.max() - lats.min()) / 2.0
        assert half_span > 60.0

    def test_ground_site_altitude(self):
        """Non-zero ground site altitude changes the circle size.

        With sat_alt measured from the ellipsoid, a higher ground site
        is closer to the satellite, so the circle shrinks slightly.
        """
        lats_sea, _ = visibility_circle(0.0, 0.0, 0.0, sat_alt=400_000, min_el=10.0)
        lats_mtn, _ = visibility_circle(0.0, 0.0, 5000.0, sat_alt=400_000, min_el=10.0)
        radius_sea = lats_sea.max() - lats_sea.min()
        radius_mtn = lats_mtn.max() - lats_mtn.min()
        assert radius_mtn < radius_sea

    def test_polar_site(self):
        """Circle near the pole has valid latitudes (clamped to +-90)."""
        lats, lons = visibility_circle(89.0, 0.0, 0.0, sat_alt=400_000, min_el=5.0)
        assert lats.max() <= 90.1
        assert lats.min() >= -90.1

    def test_n_points_one(self):
        """Single point still works."""
        lats, lons = visibility_circle(
            0.0, 0.0, 0.0, sat_alt=400_000, min_el=10.0, n_points=1
        )
        assert lats.shape == (1,)
        assert lons.shape == (1,)


# ---------- shared test fixtures ----------

ts = load.timescale()
_tles = read_tle("tests/data/25544.tle")
SAT = EarthSatellite(_tles[0][0], _tles[0][1], ts=ts)

T0 = np.datetime64("1998-11-20T06:50:00")
TIMES = T0 + np.arange(0, 60 * 60, 60, dtype="timedelta64[s]")
N = len(TIMES)

# Ground sites
SITE_LAT, SITE_LON = 40.0, -105.0
SITE2_LAT, SITE2_LON = 0.0, 0.0


# ---------- generate_range (multi-site) tests ----------


class TestGenerateRange:
    """Tests for the multi-site generate_range."""

    def test_single_site_list_keys(self):
        """Sequence of one site produces range_0, range_rate_0."""
        result = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert set(result) == {"range_0", "range_rate_0"}

    def test_single_site_dict_keys(self):
        """Dict site produces range_{name}, range_rate_{name}."""
        result = generate_range(TIMES, SAT, sites={"ksc": (SITE_LAT, SITE_LON)})
        assert set(result) == {"range_ksc", "range_rate_ksc"}

    def test_multi_site_keys(self):
        """Two positional sites produce indexed keys for both."""
        result = generate_range(
            TIMES, SAT, sites=[(SITE_LAT, SITE_LON), (SITE2_LAT, SITE2_LON)]
        )
        assert set(result) == {
            "range_0", "range_rate_0",
            "range_1", "range_rate_1",
        }

    def test_multi_site_named_keys(self):
        """Two named sites produce named keys."""
        result = generate_range(
            TIMES, SAT,
            sites={"denver": (SITE_LAT, SITE_LON), "origin": (SITE2_LAT, SITE2_LON)},
        )
        assert set(result) == {
            "range_denver", "range_rate_denver",
            "range_origin", "range_rate_origin",
        }

    def test_shapes(self):
        result = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert result["range_0"].shape == (N,)
        assert result["range_rate_0"].shape == (N,)

    def test_dtype_float64(self):
        result = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert result["range_0"].dtype == np.float64
        assert result["range_rate_0"].dtype == np.float64

    def test_range_positive(self):
        result = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert np.all(result["range_0"] > 0)

    def test_range_magnitude_leo(self):
        """Range to LEO should be between ~150 km and ~13000 km."""
        result = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert np.all(result["range_0"] > 150_000)
        assert np.all(result["range_0"] < 13_000_000)

    def test_range_rate_bounded_by_velocity(self):
        """Range rate magnitude cannot exceed satellite velocity (~7.5 km/s)."""
        result = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert np.all(np.abs(result["range_rate_0"]) < 8_500)

    def test_range_varies_over_orbit(self):
        """Range should not be constant over a full hour."""
        result = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert result["range_0"].std() > 1000

    def test_single_time(self):
        """Works with a single-element time array."""
        single = TIMES[:1]
        result = generate_range(single, SAT, sites=[(SITE_LAT, SITE_LON)])
        assert result["range_0"].shape == (1,)
        assert result["range_rate_0"].shape == (1,)

    def test_multi_site_values_match_single(self):
        """Each site's arrays match calling _generate_range_single individually."""
        multi = generate_range(
            TIMES, SAT, sites=[(SITE_LAT, SITE_LON), (SITE2_LAT, SITE2_LON)]
        )
        single_0 = _generate_range_single(TIMES, SAT, SITE_LAT, SITE_LON)
        single_1 = _generate_range_single(TIMES, SAT, SITE2_LAT, SITE2_LON)

        np.testing.assert_array_equal(multi["range_0"], single_0["range"])
        np.testing.assert_array_equal(multi["range_rate_0"], single_0["range_rate"])
        np.testing.assert_array_equal(multi["range_1"], single_1["range"])
        np.testing.assert_array_equal(multi["range_rate_1"], single_1["range_rate"])

    def test_lat_lon_tuple_defaults_alt_zero(self):
        """(lat, lon) gives same result as (lat, lon, 0.0)."""
        r1 = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        r2 = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON, 0.0)])
        np.testing.assert_array_equal(r1["range_0"], r2["range_0"])

    def test_different_sites_different_range(self):
        """Two different ground sites produce different range profiles."""
        result = generate_range(
            TIMES, SAT, sites=[(SITE_LAT, SITE_LON), (SITE2_LAT, SITE2_LON)]
        )
        assert not np.allclose(result["range_0"], result["range_1"])


# ---------- Propagator support ----------


class TestGenerateRangePropagator:
    """Tests for generate_range with a Propagator."""

    def test_propagator_support(self):
        """Propagator input produces correct shape."""
        prop = Propagator(_tles, method="epoch")
        result = generate_range(TIMES, prop, sites=[(SITE_LAT, SITE_LON)])
        assert result["range_0"].shape == (N,)
        assert result["range_rate_0"].shape == (N,)

    def test_propagator_matches_single_satellite(self):
        """For a single-TLE window, Propagator gives same result as EarthSatellite."""
        prop = Propagator(_tles, method="epoch")
        r_prop = generate_range(TIMES, prop, sites=[(SITE_LAT, SITE_LON)])
        r_sat = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        np.testing.assert_allclose(
            r_prop["range_0"], r_sat["range_0"], rtol=1e-10
        )
        np.testing.assert_allclose(
            r_prop["range_rate_0"], r_sat["range_rate_0"], rtol=1e-10
        )


# ---------- doppler_shift tests ----------


class TestDopplerShift:
    """Tests for the doppler_shift helper."""

    def test_values(self):
        """Known range_rate + freq gives expected Doppler."""
        rr = np.array([1000.0])  # 1 km/s receding
        freq = 437e6
        expected = -freq * 1000.0 / SPEED_OF_LIGHT
        result = doppler_shift(rr, freq)
        np.testing.assert_allclose(result, expected)

    def test_sign_receding(self):
        """Positive range_rate (receding) gives negative Doppler."""
        rr = np.array([100.0, 200.0, 300.0])
        result = doppler_shift(rr, freq=437e6)
        assert np.all(result < 0)

    def test_sign_approaching(self):
        """Negative range_rate (approaching) gives positive Doppler."""
        rr = np.array([-100.0, -200.0, -300.0])
        result = doppler_shift(rr, freq=437e6)
        assert np.all(result > 0)

    def test_zero(self):
        """Zero range_rate gives zero Doppler."""
        rr = np.array([0.0])
        result = doppler_shift(rr, freq=437e6)
        assert result[0] == 0.0

    def test_array_shape_preserved(self):
        """Output shape matches input shape."""
        rr = np.linspace(-1000, 1000, 50)
        result = doppler_shift(rr, freq=437e6)
        assert result.shape == rr.shape

    def test_typical_leo_magnitude(self):
        """For LEO range rate ~7 km/s at 437 MHz, Doppler ~ 10 kHz."""
        rr = np.array([7000.0])
        result = doppler_shift(rr, freq=437e6)
        assert 5_000 < abs(result[0]) < 15_000
