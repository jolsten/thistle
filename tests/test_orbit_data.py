"""Tests for thistle.orbit_data generate_* functions."""

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load

from thistle.utils import read_tle
from thistle.ground_sites import generate_range
from thistle.orbit_data import (
    GENERATORS,
    generate,
    generate_beta_angle,
    generate_ecef,
    generate_eci,
    generate_equinoctial,
    generate_keplerian,
    generate_lla,
    generate_local_solar_time,
    generate_magnetic_field_ecef,
    generate_magnetic_field_enu,
    generate_magnetic_field_total,
    generate_sunlight,
)
from thistle.propagator import Propagator

ts = load.timescale()
_tles = read_tle("tests/data/25544.tle")
SAT = EarthSatellite(_tles[0][0], _tles[0][1], ts=ts)

# Near the TLE epoch to keep propagation error small.
T0 = np.datetime64("1998-11-20T06:50:00")
TIMES = T0 + np.arange(0, 60 * 60, 60, dtype="timedelta64[s]")  # 1 hour, 1-min steps
N = len(TIMES)


# ---------------------------------------------------------------------------
# ECI
# ---------------------------------------------------------------------------
class TestGenerateECI:
    """Tests for generate_eci."""

    def test_keys(self):
        result = generate_eci(TIMES, SAT)
        assert set(result) == {"eci_x", "eci_y", "eci_z", "eci_vx", "eci_vy", "eci_vz"}

    def test_shapes(self):
        result = generate_eci(TIMES, SAT)
        for arr in result.values():
            assert arr.shape == (N,)

    def test_position_magnitude_leo(self):
        """ISS position magnitude should be ~6700-6800 km from Earth center."""
        result = generate_eci(TIMES, SAT)
        r = np.sqrt(result["eci_x"] ** 2 + result["eci_y"] ** 2 + result["eci_z"] ** 2)
        assert np.all(r > 6_300_000)
        assert np.all(r < 7_200_000)

    def test_velocity_magnitude_leo(self):
        """ISS velocity should be ~7.5 km/s."""
        result = generate_eci(TIMES, SAT)
        v = np.sqrt(
            result["eci_vx"] ** 2 + result["eci_vy"] ** 2 + result["eci_vz"] ** 2
        )
        assert np.all(v > 6_500)
        assert np.all(v < 8_500)


# ---------------------------------------------------------------------------
# ECEF
# ---------------------------------------------------------------------------
class TestGenerateECEF:
    """Tests for generate_ecef."""

    def test_keys(self):
        result = generate_ecef(TIMES, SAT)
        assert set(result) == {
            "ecef_x",
            "ecef_y",
            "ecef_z",
            "ecef_vx",
            "ecef_vy",
            "ecef_vz",
        }

    def test_shapes(self):
        result = generate_ecef(TIMES, SAT)
        for arr in result.values():
            assert arr.shape == (N,)

    def test_position_magnitude_matches_eci(self):
        """ECEF position magnitude should match ECI (rotation preserves norm)."""
        eci = generate_eci(TIMES, SAT)
        ecef = generate_ecef(TIMES, SAT)
        r_eci = np.sqrt(eci["eci_x"] ** 2 + eci["eci_y"] ** 2 + eci["eci_z"] ** 2)
        r_ecef = np.sqrt(
            ecef["ecef_x"] ** 2 + ecef["ecef_y"] ** 2 + ecef["ecef_z"] ** 2
        )
        np.testing.assert_allclose(r_ecef, r_eci, rtol=1e-6)


# ---------------------------------------------------------------------------
# LLA
# ---------------------------------------------------------------------------
class TestGenerateLLA:
    """Tests for generate_lla."""

    def test_keys(self):
        result = generate_lla(TIMES, SAT)
        assert set(result) == {"lat", "lon", "alt"}

    def test_shapes(self):
        result = generate_lla(TIMES, SAT)
        for arr in result.values():
            assert arr.shape == (N,)

    def test_latitude_bounds(self):
        """Latitude must stay within +-90 deg."""
        result = generate_lla(TIMES, SAT)
        assert np.all(result["lat"] >= -90.0)
        assert np.all(result["lat"] <= 90.0)

    def test_longitude_bounds(self):
        """Longitude must stay within +-180 deg."""
        result = generate_lla(TIMES, SAT)
        assert np.all(result["lon"] >= -180.0)
        assert np.all(result["lon"] <= 180.0)

    def test_iss_latitude_bounded_by_inclination(self):
        """ISS latitude should not exceed its ~52 deg inclination."""
        result = generate_lla(TIMES, SAT)
        assert np.all(np.abs(result["lat"]) < 55.0)

    def test_altitude_leo(self):
        """ISS altitude should be roughly 150-600 km."""
        result = generate_lla(TIMES, SAT)
        alt_km = result["alt"] / 1000.0
        assert np.all(alt_km > 150)
        assert np.all(alt_km < 600)


# ---------------------------------------------------------------------------
# Keplerian
# ---------------------------------------------------------------------------
class TestGenerateKeplerian:
    """Tests for generate_keplerian."""

    EXPECTED_KEYS = {
        "sma",
        "ecc",
        "inc",
        "raan",
        "aop",
        "ta",
        "ma",
        "ea",
        "arglat",
        "tlon",
        "mlon",
        "lonper",
        "mm",
    }

    def test_keys(self):
        result = generate_keplerian(TIMES, SAT)
        assert set(result) == self.EXPECTED_KEYS

    def test_shapes(self):
        result = generate_keplerian(TIMES, SAT)
        for arr in result.values():
            assert arr.shape == (N,)

    def test_sma_leo(self):
        """ISS semi-major axis should be ~6700 km."""
        result = generate_keplerian(TIMES, SAT)
        sma_km = result["sma"] / 1000.0
        assert np.all(sma_km > 6_400)
        assert np.all(sma_km < 7_000)

    def test_eccentricity_near_circular(self):
        """ISS orbit is near-circular (e < 0.02)."""
        result = generate_keplerian(TIMES, SAT)
        assert np.all(result["ecc"] < 0.02)

    def test_inclination_iss(self):
        """ISS inclination should be ~51.6 deg."""
        result = generate_keplerian(TIMES, SAT)
        assert np.all(np.abs(result["inc"] - 51.6) < 1.0)

    def test_angles_bounded(self):
        """Angular elements should be within +-360 deg."""
        result = generate_keplerian(TIMES, SAT)
        for key in ("ta", "ma", "ea", "arglat", "tlon", "mlon"):
            assert np.all(np.abs(result[key]) <= 360.0), f"{key} out of range"

    def test_mean_motion_iss(self):
        """ISS mean motion should be ~5800-6000 deg/day (~16 rev/day)."""
        result = generate_keplerian(TIMES, SAT)
        assert np.all(result["mm"] > 5_500)
        assert np.all(result["mm"] < 6_200)


# ---------------------------------------------------------------------------
# Equinoctial
# ---------------------------------------------------------------------------
class TestGenerateEquinoctial:
    """Tests for generate_equinoctial."""

    def test_keys(self):
        result = generate_equinoctial(TIMES, SAT)
        assert set(result) == {"p", "f", "g", "h", "k", "L"}

    def test_shapes(self):
        result = generate_equinoctial(TIMES, SAT)
        for arr in result.values():
            assert arr.shape == (N,)

    def test_semi_latus_rectum_leo(self):
        """p = a(1 - e^2), should be close to sma for near-circular orbit."""
        kep = generate_keplerian(TIMES, SAT)
        equi = generate_equinoctial(TIMES, SAT)
        expected_p = kep["sma"] * (1.0 - kep["ecc"] ** 2)
        np.testing.assert_allclose(equi["p"], expected_p, rtol=1e-6)

    def test_fg_magnitude_equals_eccentricity(self):
        """sqrt(f^2 + g^2) should equal eccentricity."""
        kep = generate_keplerian(TIMES, SAT)
        equi = generate_equinoctial(TIMES, SAT)
        e_from_equi = np.sqrt(equi["f"] ** 2 + equi["g"] ** 2)
        np.testing.assert_allclose(e_from_equi, kep["ecc"], rtol=1e-6)

    def test_hk_from_inclination(self):
        """sqrt(h^2 + k^2) should equal tan(i/2)."""
        kep = generate_keplerian(TIMES, SAT)
        equi = generate_equinoctial(TIMES, SAT)
        tan_half_i = np.tan(np.radians(kep["inc"]) / 2.0)
        hk = np.sqrt(equi["h"] ** 2 + equi["k"] ** 2)
        np.testing.assert_allclose(hk, tan_half_i, rtol=1e-6)


# ---------------------------------------------------------------------------
# Sunlight
# ---------------------------------------------------------------------------
class TestGenerateSunlight:
    """Tests for generate_sunlight."""

    def test_keys(self):
        result = generate_sunlight(TIMES, SAT)
        assert set(result) == {"sun"}

    def test_shape(self):
        result = generate_sunlight(TIMES, SAT)
        assert result["sun"].shape == (N,)

    def test_dtype(self):
        result = generate_sunlight(TIMES, SAT)
        assert result["sun"].dtype == np.int8

    def test_values_in_range(self):
        """Sunlight flag must be 0 (umbra), 1 (penumbra), or 2 (sunlit)."""
        result = generate_sunlight(TIMES, SAT)
        assert np.all(np.isin(result["sun"], [0, 1, 2]))

    def test_has_transitions(self):
        """Over a full orbit the ISS should see at least two distinct states."""
        full_orbit = T0 + np.arange(0, 100 * 60, 30, dtype="timedelta64[s]")
        result = generate_sunlight(full_orbit, SAT)
        assert len(np.unique(result["sun"])) >= 2


# ---------------------------------------------------------------------------
# Beta angle
# ---------------------------------------------------------------------------
class TestGenerateBetaAngle:
    """Tests for generate_beta_angle."""

    def test_keys(self):
        result = generate_beta_angle(TIMES, SAT)
        assert set(result) == {"beta"}

    def test_shape(self):
        result = generate_beta_angle(TIMES, SAT)
        assert result["beta"].shape == (N,)

    def test_range(self):
        """Beta angle should be within +-90 deg."""
        result = generate_beta_angle(TIMES, SAT)
        assert np.all(result["beta"] >= -90.0)
        assert np.all(result["beta"] <= 90.0)

    def test_nearly_constant_over_one_orbit(self):
        """Beta angle varies slowly; stdev over 1 hour should be small."""
        result = generate_beta_angle(TIMES, SAT)
        assert np.std(result["beta"]) < 1.0


# ---------------------------------------------------------------------------
# Local solar time
# ---------------------------------------------------------------------------
class TestGenerateLocalSolarTime:
    """Tests for generate_local_solar_time."""

    def test_keys(self):
        result = generate_local_solar_time(TIMES, SAT)
        assert set(result) == {"lst"}

    def test_shape(self):
        result = generate_local_solar_time(TIMES, SAT)
        assert result["lst"].shape == (N,)

    def test_range(self):
        """LST should be in [0, 24)."""
        result = generate_local_solar_time(TIMES, SAT)
        assert np.all(result["lst"] >= 0.0)
        assert np.all(result["lst"] < 24.0)


# ---------------------------------------------------------------------------
# Magnetic field ENU
# ---------------------------------------------------------------------------
class TestGenerateMagneticFieldENU:
    """Tests for generate_magnetic_field_enu."""

    def test_keys(self):
        result = generate_magnetic_field_enu(TIMES, SAT)
        assert set(result) == {"Be", "Bn", "Bu"}

    def test_shapes(self):
        result = generate_magnetic_field_enu(TIMES, SAT)
        for arr in result.values():
            assert arr.shape == (N,)

    def test_magnitude_reasonable(self):
        """Total field at LEO should be roughly 20000-60000 nT."""
        result = generate_magnetic_field_enu(TIMES, SAT)
        bt = np.sqrt(result["Be"] ** 2 + result["Bn"] ** 2 + result["Bu"] ** 2)
        assert np.all(bt > 15_000)
        assert np.all(bt < 65_000)


# ---------------------------------------------------------------------------
# Magnetic field total
# ---------------------------------------------------------------------------
class TestGenerateMagneticFieldTotal:
    """Tests for generate_magnetic_field_total."""

    def test_keys(self):
        result = generate_magnetic_field_total(TIMES, SAT)
        assert set(result) == {"Bt"}

    def test_shape(self):
        result = generate_magnetic_field_total(TIMES, SAT)
        assert result["Bt"].shape == (N,)

    def test_positive(self):
        result = generate_magnetic_field_total(TIMES, SAT)
        assert np.all(result["Bt"] > 0)

    def test_matches_enu_norm(self):
        """Total should equal sqrt(Be^2 + Bn^2 + Bu^2)."""
        enu = generate_magnetic_field_enu(TIMES, SAT)
        total = generate_magnetic_field_total(TIMES, SAT)
        expected = np.sqrt(enu["Be"] ** 2 + enu["Bn"] ** 2 + enu["Bu"] ** 2)
        np.testing.assert_allclose(total["Bt"], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Magnetic field ECEF
# ---------------------------------------------------------------------------
class TestGenerateMagneticFieldECEF:
    """Tests for generate_magnetic_field_ecef."""

    def test_keys(self):
        result = generate_magnetic_field_ecef(TIMES, SAT)
        assert set(result) == {"Bx", "By", "Bz"}

    def test_shapes(self):
        result = generate_magnetic_field_ecef(TIMES, SAT)
        for arr in result.values():
            assert arr.shape == (N,)

    def test_magnitude_matches_enu(self):
        """ECEF rotation preserves the field magnitude."""
        enu = generate_magnetic_field_enu(TIMES, SAT)
        ecef = generate_magnetic_field_ecef(TIMES, SAT)
        bt_enu = np.sqrt(enu["Be"] ** 2 + enu["Bn"] ** 2 + enu["Bu"] ** 2)
        bt_ecef = np.sqrt(ecef["Bx"] ** 2 + ecef["By"] ** 2 + ecef["Bz"] ** 2)
        np.testing.assert_allclose(bt_ecef, bt_enu, rtol=1e-6)


# ---------------------------------------------------------------------------
# generate() wrapper
# ---------------------------------------------------------------------------
class TestGenerate:
    """Tests for the generate() wrapper."""

    def test_single_group(self):
        result = generate(TIMES, SAT, ["eci"])
        assert "eci_x" in result

    def test_multiple_groups(self):
        result = generate(TIMES, SAT, ["eci", "lla"])
        assert "eci_x" in result
        assert "lat" in result

    def test_all_groups(self):
        result = generate(TIMES, SAT, list(GENERATORS))
        for gen_name, gen_fn in GENERATORS.items():
            individual = gen_fn(TIMES, SAT)
            for key in individual:
                assert key in result

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown group"):
            generate(TIMES, SAT, ["nonexistent"])

    def test_f32_downcast(self):
        """Keys in _F32_KEYS should be float32 after generate()."""
        result = generate(TIMES, SAT, ["lla", "keplerian"])
        for key in ("lat", "lon", "inc", "ecc"):
            assert result[key].dtype == np.float32, f"{key} not float32"

    def test_position_stays_f64(self):
        """Position keys should remain float64."""
        result = generate(TIMES, SAT, ["eci"])
        for key in ("eci_x", "eci_y", "eci_z"):
            assert result[key].dtype == np.float64, f"{key} not float64"

    def test_generate_with_propagator(self):
        """Test generate() with a Propagator object."""
        # Load multiple TLEs
        tles = read_tle("tests/data/25544.tle")[:3]  # Use first 3 TLEs

        # Create a propagator
        propagator = Propagator(tles, method="epoch")

        # Generate data across time range that spans multiple TLEs
        times = T0 + np.arange(0, 10 * 24 * 60 * 60, 60 * 60, dtype="timedelta64[s]")
        result = generate(times, propagator, ["eci", "lla"])

        # Verify all expected keys are present
        assert "eci_x" in result
        assert "eci_y" in result
        assert "eci_z" in result
        assert "lat" in result
        assert "lon" in result
        assert "alt" in result

        # Verify output shapes match input times
        for key in result:
            assert result[key].shape == (len(times),), f"{key} shape mismatch"

        # Verify position magnitude is reasonable for LEO
        r = np.sqrt(result["eci_x"]**2 + result["eci_y"]**2 + result["eci_z"]**2)
        assert np.all(r > 6_300_000)
        assert np.all(r < 7_200_000)

    def test_generate_with_propagator_matches_single_satellite(self):
        """Test that Propagator with single TLE matches EarthSatellite result."""
        # Create a propagator with a single TLE
        tles = read_tle("tests/data/25544.tle")[:1]
        propagator = Propagator(tles, method="epoch")

        # Generate with both methods
        result_prop = generate(TIMES, propagator, ["eci", "lla"])
        result_sat = generate(TIMES, SAT, ["eci", "lla"])

        # Results should be identical
        for key in result_prop:
            np.testing.assert_allclose(
                result_prop[key],
                result_sat[key],
                rtol=1e-10,
                err_msg=f"{key} mismatch"
            )


# ---------------------------------------------------------------------------
# generate() with sites
# ---------------------------------------------------------------------------
SITE_LAT, SITE_LON = 40.0, -105.0


class TestGenerateWithSites:
    """Tests for generate() with the sites parameter."""

    def test_sites_list_keys(self):
        """Indexed site produces range_0, range_rate_0."""
        result = generate(TIMES, SAT, ["eci"], sites=[(SITE_LAT, SITE_LON)])
        assert "range_0" in result
        assert "range_rate_0" in result

    def test_sites_dict_keys(self):
        """Named site produces range_ksc, range_rate_ksc."""
        result = generate(TIMES, SAT, ["eci"], sites={"ksc": (SITE_LAT, SITE_LON)})
        assert "range_ksc" in result
        assert "range_rate_ksc" in result

    def test_sites_and_groups_coexist(self):
        """Orbit groups and range keys coexist in the result."""
        result = generate(TIMES, SAT, ["eci", "lla"], sites=[(SITE_LAT, SITE_LON)])
        assert "eci_x" in result
        assert "lat" in result
        assert "range_0" in result
        assert "range_rate_0" in result

    def test_sites_shape_and_dtype(self):
        """Range arrays have correct shape and stay float64."""
        result = generate(TIMES, SAT, ["eci"], sites=[(SITE_LAT, SITE_LON)])
        assert result["range_0"].shape == (N,)
        assert result["range_rate_0"].shape == (N,)
        assert result["range_0"].dtype == np.float64
        assert result["range_rate_0"].dtype == np.float64

    def test_sites_matches_standalone(self):
        """Integrated range matches standalone generate_range (no light-time)."""
        integrated = generate(TIMES, SAT, ["eci"], sites=[(SITE_LAT, SITE_LON)])
        standalone = generate_range(TIMES, SAT, sites=[(SITE_LAT, SITE_LON)])
        # Should be very close — only difference is light-time correction
        np.testing.assert_allclose(
            integrated["range_0"], standalone["range_0"], rtol=1e-6
        )
        np.testing.assert_allclose(
            integrated["range_rate_0"], standalone["range_rate_0"], rtol=1e-4
        )

    def test_sites_with_propagator(self):
        """Propagator + sites produces correct shapes."""
        prop = Propagator(_tles[:1], method="epoch")
        result = generate(TIMES, prop, ["eci"], sites=[(SITE_LAT, SITE_LON)])
        assert result["range_0"].shape == (N,)
        assert result["range_rate_0"].shape == (N,)

    def test_no_sites_no_range_keys(self):
        """Omitting sites produces no range keys."""
        result = generate(TIMES, SAT, ["eci"])
        assert not any(k.startswith("range") for k in result)
