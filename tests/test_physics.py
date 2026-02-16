"""Physics validation tests based on fundamental orbital mechanics principles.

These tests validate that fundamental physical relationships hold true,
serving as cross-checks for the implementation. Unlike ground truth tests
that compare against external reference data, these tests verify internal
consistency and conservation laws.
"""

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load

from thistle.orbit_data import (
    generate_ecef,
    generate_eci,
    generate_equinoctial,
    generate_keplerian,
    generate_lla,
)
from thistle.utils import read_tle

ts = load.timescale()


@pytest.fixture(scope="module")
def iss_satellite():
    """Get ISS satellite for testing."""
    _tles = read_tle("tests/data/25544.tle")
    return EarthSatellite(_tles[0][0], _tles[0][1], ts=ts)


class TestPhysicsValidation:
    """Test that outputs obey fundamental physics and orbital mechanics relationships."""

    def test_energy_conservation_over_orbit(self, iss_satellite):
        """Specific mechanical energy should be nearly constant over one orbit.

        For a Keplerian orbit, specific energy E = -μ/(2a) where μ is the
        gravitational parameter and a is the semi-major axis. This should
        remain constant (within numerical precision) over one orbit period.
        """
        # Generate data over one orbital period (~90 minutes for ISS)
        t0 = np.datetime64("1998-11-20T06:50:00")
        times = t0 + np.arange(0, 100 * 60, 30, dtype="timedelta64[s]")

        eci = generate_eci(times, iss_satellite)
        kep = generate_keplerian(times, iss_satellite)

        # Compute specific energy from position and velocity
        r = np.sqrt(eci["eci_x"] ** 2 + eci["eci_y"] ** 2 + eci["eci_z"] ** 2)
        v2 = eci["eci_vx"] ** 2 + eci["eci_vy"] ** 2 + eci["eci_vz"] ** 2
        mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        energy_from_state = v2 / 2.0 - mu / r

        # Compute specific energy from semi-major axis
        energy_from_sma = -mu / (2.0 * kep["sma"])

        # Energy should be conserved (within ~1% due to perturbations and SGP4)
        np.testing.assert_allclose(
            energy_from_state,
            energy_from_sma,
            rtol=0.01,
            err_msg="Specific energy not conserved over orbit",
        )

    def test_angular_momentum_conservation(self, iss_satellite):
        """Angular momentum magnitude should be nearly constant over one orbit.

        For a Keplerian orbit, h = sqrt(μ * p) where p is the semi-latus rectum.
        """
        t0 = np.datetime64("1998-11-20T06:50:00")
        times = t0 + np.arange(0, 100 * 60, 30, dtype="timedelta64[s]")

        eci = generate_eci(times, iss_satellite)
        equi = generate_equinoctial(times, iss_satellite)

        # Compute angular momentum from state vectors
        r = np.array([eci["eci_x"], eci["eci_y"], eci["eci_z"]])
        v = np.array([eci["eci_vx"], eci["eci_vy"], eci["eci_vz"]])
        h_vec = np.cross(r.T, v.T).T
        h_mag = np.linalg.norm(h_vec, axis=0)

        # Compute angular momentum from semi-latus rectum
        mu = 3.986004418e14
        h_from_p = np.sqrt(mu * equi["p"])

        # Should match within ~1% due to perturbations
        np.testing.assert_allclose(
            h_mag, h_from_p, rtol=0.01, err_msg="Angular momentum not conserved"
        )

    def test_eccentricity_from_state_vectors(self, iss_satellite):
        """Eccentricity computed from state vectors should match Keplerian elements."""
        t0 = np.datetime64("1998-11-20T06:50:00")
        times = t0 + np.arange(0, 60 * 60, 60, dtype="timedelta64[s]")

        eci = generate_eci(times, iss_satellite)
        kep = generate_keplerian(times, iss_satellite)

        # Compute eccentricity vector from state vectors
        mu = 3.986004418e14
        r = np.array([eci["eci_x"], eci["eci_y"], eci["eci_z"]])
        v = np.array([eci["eci_vx"], eci["eci_vy"], eci["eci_vz"]])
        r_mag = np.linalg.norm(r, axis=0)

        # e = (v × h) / μ - r / |r|
        h = np.cross(r.T, v.T).T
        e_vec = np.cross(v.T, h.T).T / mu - r / r_mag
        e_mag = np.linalg.norm(e_vec, axis=0)

        # Should match Keplerian eccentricity within numerical precision
        # Allow slightly larger tolerance (1e-5) for cross product roundoff
        np.testing.assert_allclose(
            e_mag, kep["ecc"], rtol=1e-5, err_msg="Eccentricity mismatch"
        )

    def test_position_velocity_orthogonality_at_apoapsis_periapsis(self, iss_satellite):
        """At apoapsis and periapsis, position and velocity are orthogonal."""
        t0 = np.datetime64("1998-11-20T06:50:00")
        times = t0 + np.arange(0, 100 * 60, 10, dtype="timedelta64[s]")

        eci = generate_eci(times, iss_satellite)
        kep = generate_keplerian(times, iss_satellite)

        # Find points near apoapsis and periapsis (TA ≈ 0° or 180°)
        ta = kep["ta"]
        near_periapsis = np.abs(ta) < 5.0
        near_apoapsis = np.abs(ta - 180.0) < 5.0
        apsides = near_periapsis | near_apoapsis

        if np.any(apsides):
            r = np.array(
                [eci["eci_x"][apsides], eci["eci_y"][apsides], eci["eci_z"][apsides]]
            )
            v = np.array(
                [eci["eci_vx"][apsides], eci["eci_vy"][apsides], eci["eci_vz"][apsides]]
            )

            # Dot product should be near zero at apsides
            dot_product = np.sum(r * v, axis=0)
            r_mag = np.linalg.norm(r, axis=0)
            v_mag = np.linalg.norm(v, axis=0)

            # Normalized dot product (cosine of angle)
            cos_angle = dot_product / (r_mag * v_mag)

            # Should be nearly zero (within 0.1 due to discrete sampling)
            assert np.all(
                np.abs(cos_angle) < 0.1
            ), "Position and velocity not orthogonal at apsides"

    def test_eci_ecef_rotation_preserves_magnitude(self, iss_satellite):
        """Rotation from ECI to ECEF should preserve position magnitude."""
        t0 = np.datetime64("1998-11-20T06:50:00")
        times = t0 + np.arange(0, 60 * 60, 60, dtype="timedelta64[s]")

        eci = generate_eci(times, iss_satellite)
        ecef = generate_ecef(times, iss_satellite)

        r_eci = np.sqrt(eci["eci_x"] ** 2 + eci["eci_y"] ** 2 + eci["eci_z"] ** 2)
        r_ecef = np.sqrt(
            ecef["ecef_x"] ** 2 + ecef["ecef_y"] ** 2 + ecef["ecef_z"] ** 2
        )

        # Magnitudes should be identical (rotation preserves norm)
        np.testing.assert_allclose(
            r_eci, r_ecef, rtol=1e-10, err_msg="Rotation changed position magnitude"
        )

    def test_lla_altitude_matches_eci_radius(self, iss_satellite):
        """LLA altitude should match ECI radius minus Earth's radius.

        Using WGS84 ellipsoid for Earth's radius at the latitude.
        """
        t0 = np.datetime64("1998-11-20T06:50:00")
        times = t0 + np.arange(0, 60 * 60, 60, dtype="timedelta64[s]")

        eci = generate_eci(times, iss_satellite)
        lla = generate_lla(times, iss_satellite)

        r_eci = np.sqrt(eci["eci_x"] ** 2 + eci["eci_y"] ** 2 + eci["eci_z"] ** 2)

        # WGS84 parameters
        a = 6378137.0  # semi-major axis (m)
        f = 1.0 / 298.257223563  # flattening
        lat_rad = np.radians(lla["lat"])

        # Radius of Earth at latitude (using simplified formula)
        r_earth = a * (1.0 - f * np.sin(lat_rad) ** 2)

        # Altitude should approximately equal r_eci - r_earth
        # Allow 1 km tolerance due to ellipsoid approximation
        altitude_from_eci = r_eci - r_earth
        np.testing.assert_allclose(
            lla["alt"],
            altitude_from_eci,
            atol=1000.0,
            err_msg="LLA altitude doesn't match ECI-derived altitude",
        )
