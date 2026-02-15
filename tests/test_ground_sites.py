import numpy as np
import pytest

from thistle.ground_sites import visibility_circle


class TestVisibilityCircle:
    """Tests for visibility_circle."""

    def test_output_shape(self):
        """Output arrays have length n_points."""
        lats, lons = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=10.0, n_points=72)
        assert lats.shape == (72,)
        assert lons.shape == (72,)

    def test_output_dtype(self):
        """Output arrays are float32."""
        lats, lons = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=10.0)
        assert lats.dtype == np.float32
        assert lons.dtype == np.float32

    def test_symmetry_equator(self):
        """Circle at the equator/prime meridian is symmetric in latitude."""
        lats, lons = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=5.0, n_points=360)
        # Max and min latitude should be roughly equal in magnitude
        assert abs(lats.max() + lats.min()) < 0.5

    def test_symmetry_longitude(self):
        """Circle at the equator/prime meridian is symmetric in longitude."""
        lats, lons = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=5.0, n_points=360)
        assert abs(lons.max() + lons.min()) < 0.5

    def test_circle_centered_on_site(self):
        """Mean of the circle points is close to the ground site."""
        lat_site, lon_site = 45.0, -90.0
        lats, lons = visibility_circle(lat_site, lon_site, sat_alt=400_000, min_el=10.0)
        assert abs(np.mean(lats) - lat_site) < 1.0
        assert abs(np.mean(lons) - lon_site) < 1.0

    def test_higher_elevation_smaller_circle(self):
        """Higher minimum elevation yields a smaller visibility circle."""
        lats_5, _ = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=5.0)
        lats_30, _ = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=30.0)
        radius_5 = lats_5.max() - lats_5.min()
        radius_30 = lats_30.max() - lats_30.min()
        assert radius_30 < radius_5

    def test_higher_altitude_larger_circle(self):
        """Higher satellite altitude yields a larger visibility circle."""
        lats_leo, _ = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=10.0)
        lats_meo, _ = visibility_circle(0.0, 0.0, sat_alt=20_200_000, min_el=10.0)
        radius_leo = lats_leo.max() - lats_leo.min()
        radius_meo = lats_meo.max() - lats_meo.min()
        assert radius_meo > radius_leo

    def test_zero_elevation_leo_radius(self):
        """At 0 deg elevation, LEO ~400 km gives a radius of roughly 20 deg."""
        lats, _ = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=0.0)
        half_span = (lats.max() - lats.min()) / 2.0
        # Should be roughly 18-24 degrees for 400 km altitude
        assert 15.0 < half_span < 28.0

    def test_geo_altitude_large_radius(self):
        """GEO altitude at 0 deg elevation gives a radius > 60 deg."""
        lats, _ = visibility_circle(0.0, 0.0, sat_alt=35_786_000, min_el=0.0)
        half_span = (lats.max() - lats.min()) / 2.0
        assert half_span > 60.0

    def test_ground_site_altitude(self):
        """Non-zero ground site altitude changes the circle size.

        With sat_alt measured from the ellipsoid, a higher ground site
        is closer to the satellite, so the circle shrinks slightly.
        """
        lats_sea, _ = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=10.0, alt=0.0)
        lats_mtn, _ = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=10.0, alt=5000.0)
        radius_sea = lats_sea.max() - lats_sea.min()
        radius_mtn = lats_mtn.max() - lats_mtn.min()
        assert radius_mtn < radius_sea

    def test_polar_site(self):
        """Circle near the pole has valid latitudes (clamped to +-90)."""
        lats, lons = visibility_circle(89.0, 0.0, sat_alt=400_000, min_el=5.0)
        assert lats.max() <= 90.1
        assert lats.min() >= -90.1

    def test_n_points_one(self):
        """Single point still works."""
        lats, lons = visibility_circle(0.0, 0.0, sat_alt=400_000, min_el=10.0, n_points=1)
        assert lats.shape == (1,)
        assert lons.shape == (1,)
