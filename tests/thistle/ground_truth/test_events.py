"""Ground truth tests for satellite events against STK data for April 1, 2020."""

import csv
import pathlib

import numpy as np
import pytest

from skyfield.api import wgs84

from thistle.events import (
    find_eclipse_periods,
    find_node_crossings,
    find_passes,
    find_sunlit_periods,
)
from thistle.ground_sites import visibility_circle
from thistle.orbit_data import ts
from thistle.propagator import Propagator
from thistle.utils import dt64_to_time

from .conftest import parse_time

# Kennedy Space Center ground site
KSC_LAT = 28.57
KSC_LON = -80.65

# April 1 time window
T0 = np.datetime64("2020-04-01T00:00:00", "us")
T1 = np.datetime64("2020-04-02T00:00:00", "us")

_TRUTH_DIR = pathlib.Path(__file__).parent.parent / "data" / "truth"


def _load_intervals_csv(filename: str) -> list[dict]:
    """Load start/stop/duration intervals from an STK CSV."""
    filepath = _TRUTH_DIR / filename
    intervals = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intervals.append(
                {
                    "start": parse_time(row["Start Time (UTCG)"]),
                    "stop": parse_time(row["Stop Time (UTCG)"]),
                    "duration_sec": float(row["Duration (sec)"]),
                }
            )
    return intervals


def _load_ascending_node_csv(filename: str) -> list[np.datetime64]:
    """Load ascending node crossing times from an STK CSV."""
    filepath = _TRUTH_DIR / filename
    times = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(parse_time(row["Time of Ascen Node (UTCG)"]))
    return times


# ---------- shared satellite fixture ----------


@pytest.fixture(scope="module")
def iss_satellite(iss_tles, april_2020_dates):
    """Get the ISS EarthSatellite valid for April 1 midday."""
    start_date, end_date = april_2020_dates
    propagator = Propagator(iss_tles, method="epoch", start=start_date, stop=end_date)
    midday = np.datetime64("2020-04-01T12:00:00", "us")
    return propagator.find_satellite(midday)


# ---------- access (pass) tests ----------


@pytest.fixture(scope="module")
def truth_accesses():
    return _load_intervals_csv("25544_Access_KSC.csv")


@pytest.fixture(scope="module")
def computed_passes(iss_satellite):
    return find_passes(T0, T1, iss_satellite, KSC_LAT, KSC_LON, min_elevation=0.0)


class TestAccessGroundTruth:
    """Test satellite access times against STK ground truth for KSC."""

    def test_pass_count(self, computed_passes, truth_accesses):
        assert len(computed_passes) == len(truth_accesses)

    def test_start_times(self, computed_passes, truth_accesses):
        for computed, truth in zip(computed_passes, truth_accesses):
            diff = abs((computed["start"] - truth["start"]) / np.timedelta64(1, "s"))
            assert diff < 30.0, (
                f"Start time mismatch: {computed['start']} vs {truth['start']}, "
                f"diff {diff:.1f}s"
            )

    def test_stop_times(self, computed_passes, truth_accesses):
        for computed, truth in zip(computed_passes, truth_accesses):
            diff = abs((computed["stop"] - truth["stop"]) / np.timedelta64(1, "s"))
            assert diff < 30.0, (
                f"Stop time mismatch: {computed['stop']} vs {truth['stop']}, "
                f"diff {diff:.1f}s"
            )

    def test_durations(self, computed_passes, truth_accesses):
        for computed, truth in zip(computed_passes, truth_accesses):
            computed_dur = (computed["stop"] - computed["start"]) / np.timedelta64(1, "s")
            assert abs(computed_dur - truth["duration_sec"]) < 60.0

    def test_passes_ordered(self, computed_passes):
        for i in range(len(computed_passes) - 1):
            assert computed_passes[i]["start"] < computed_passes[i + 1]["start"]

    def test_peak_elevation_positive(self, computed_passes):
        for p in computed_passes:
            assert p["peak_elevation"] > 0.0


# ---------- sunlit period tests ----------


@pytest.fixture(scope="module")
def truth_sunlit():
    return _load_intervals_csv("25544_Sun.csv")


@pytest.fixture(scope="module")
def computed_sunlit(iss_satellite):
    return find_sunlit_periods(T0, T1, iss_satellite)


class TestSunlitGroundTruth:
    """Test sunlit periods against STK ground truth."""

    def test_period_count(self, computed_sunlit, truth_sunlit):
        assert len(computed_sunlit) == len(truth_sunlit)

    def test_start_times(self, computed_sunlit, truth_sunlit):
        for computed, truth in zip(computed_sunlit, truth_sunlit):
            diff = abs((computed["start"] - truth["start"]) / np.timedelta64(1, "s"))
            assert diff < 30.0, (
                f"Sunlit start mismatch: {computed['start']} vs {truth['start']}, "
                f"diff {diff:.1f}s"
            )

    def test_stop_times(self, computed_sunlit, truth_sunlit):
        for computed, truth in zip(computed_sunlit, truth_sunlit):
            diff = abs((computed["stop"] - truth["stop"]) / np.timedelta64(1, "s"))
            assert diff < 30.0, (
                f"Sunlit stop mismatch: {computed['stop']} vs {truth['stop']}, "
                f"diff {diff:.1f}s"
            )

    def test_durations(self, computed_sunlit, truth_sunlit):
        for computed, truth in zip(computed_sunlit, truth_sunlit):
            computed_dur = (computed["stop"] - computed["start"]) / np.timedelta64(1, "s")
            assert abs(computed_dur - truth["duration_sec"]) < 60.0

    def test_periods_ordered(self, computed_sunlit):
        for i in range(len(computed_sunlit) - 1):
            assert computed_sunlit[i]["start"] < computed_sunlit[i + 1]["start"]

    def test_no_overlap(self, computed_sunlit):
        for i in range(len(computed_sunlit) - 1):
            assert computed_sunlit[i]["stop"] <= computed_sunlit[i + 1]["start"]


# ---------- eclipse (umbra) period tests ----------


@pytest.fixture(scope="module")
def truth_umbra():
    return _load_intervals_csv("25544_Umbra.csv")


@pytest.fixture(scope="module")
def computed_eclipse(iss_satellite):
    return find_eclipse_periods(T0, T1, iss_satellite)


class TestEclipseGroundTruth:
    """Test eclipse periods against STK umbra ground truth.

    Our find_eclipse_periods uses Skyfield's is_sunlit(), which transitions
    at the penumbra boundary (~8s wider than umbra on each side). Tolerances
    account for this difference.
    """

    def test_period_count(self, computed_eclipse, truth_umbra):
        assert len(computed_eclipse) == len(truth_umbra)

    def test_start_times(self, computed_eclipse, truth_umbra):
        """Eclipse start should be near umbra start (within penumbra + tolerance)."""
        for computed, truth in zip(computed_eclipse, truth_umbra):
            diff = abs((computed["start"] - truth["start"]) / np.timedelta64(1, "s"))
            assert diff < 30.0, (
                f"Eclipse start mismatch: {computed['start']} vs {truth['start']}, "
                f"diff {diff:.1f}s"
            )

    def test_stop_times(self, computed_eclipse, truth_umbra):
        """Eclipse stop should be near umbra stop (within penumbra + tolerance)."""
        for computed, truth in zip(computed_eclipse, truth_umbra):
            diff = abs((computed["stop"] - truth["stop"]) / np.timedelta64(1, "s"))
            assert diff < 30.0, (
                f"Eclipse stop mismatch: {computed['stop']} vs {truth['stop']}, "
                f"diff {diff:.1f}s"
            )

    def test_eclipse_longer_than_umbra(self, computed_eclipse, truth_umbra):
        """Eclipse (penumbra + umbra) should be at least as long as umbra alone."""
        for computed, truth in zip(computed_eclipse, truth_umbra):
            computed_dur = (computed["stop"] - computed["start"]) / np.timedelta64(1, "s")
            assert computed_dur >= truth["duration_sec"] - 30.0

    def test_periods_ordered(self, computed_eclipse):
        for i in range(len(computed_eclipse) - 1):
            assert computed_eclipse[i]["start"] < computed_eclipse[i + 1]["start"]

    def test_no_overlap(self, computed_eclipse):
        for i in range(len(computed_eclipse) - 1):
            assert computed_eclipse[i]["stop"] <= computed_eclipse[i + 1]["start"]


# ---------- ascending node crossing tests ----------


@pytest.fixture(scope="module")
def truth_ascending_nodes():
    return _load_ascending_node_csv("25544_Ascending_Node.csv")


@pytest.fixture(scope="module")
def computed_node_crossings(iss_satellite):
    return find_node_crossings(T0, T1, iss_satellite)


class TestAscendingNodeGroundTruth:
    """Test ascending node crossings against STK ground truth."""

    def test_ascending_count(self, computed_node_crossings, truth_ascending_nodes):
        """Number of ascending crossings matches ground truth."""
        ascending = [c for c in computed_node_crossings if c["ascending"] and c["start"] >= T0]
        assert len(ascending) == len(truth_ascending_nodes)

    def test_ascending_times(self, computed_node_crossings, truth_ascending_nodes):
        """Ascending node times match ground truth within 30 seconds."""
        ascending = [c for c in computed_node_crossings if c["ascending"] and c["start"] >= T0]
        for computed, truth_time in zip(ascending, truth_ascending_nodes):
            diff = abs((computed["start"] - truth_time) / np.timedelta64(1, "s"))
            assert diff < 30.0, (
                f"Node crossing mismatch: {computed['start']} vs {truth_time}, "
                f"diff {diff:.1f}s"
            )

    def test_ascending_longitude_range(self, computed_node_crossings):
        """Ascending node longitudes should be in [-180, 180]."""
        ascending = [c for c in computed_node_crossings if c["ascending"]]
        for c in ascending:
            assert -180.0 <= c["longitude"] <= 180.0

    def test_crossings_alternate(self, computed_node_crossings):
        """Node crossings should alternate between ascending and descending."""
        for i in range(len(computed_node_crossings) - 1):
            assert computed_node_crossings[i]["ascending"] != computed_node_crossings[i + 1]["ascending"]

    def test_crossings_ordered(self, computed_node_crossings):
        for i in range(len(computed_node_crossings) - 1):
            assert computed_node_crossings[i]["start"] < computed_node_crossings[i + 1]["start"]


# ---------- visibility circle cross-validation ----------


class TestVisibilityCircleCrossValidation:
    """Cross-validate visibility_circle against access boundary times.

    At each access start/stop (0 deg elevation), the great-circle distance
    from KSC to the ISS subsatellite point should equal the visibility
    circle radius for the satellite's altitude at that moment.
    """

    def test_subsatellite_distance_matches_visibility_radius(
        self, iss_satellite, computed_passes
    ):
        """At computed pass boundaries, subsatellite distance ~ visibility radius."""
        from geographiclib.geodesic import Geodesic

        geod = Geodesic.WGS84

        for p in computed_passes:
            for boundary_time in [p["start"], p["stop"]]:
                # ISS position at boundary time
                t = dt64_to_time(np.atleast_1d(boundary_time), ts)
                geo = iss_satellite.at(t)
                subpoint = wgs84.subpoint(geo)
                sat_lat = subpoint.latitude.degrees.item()
                sat_lon = subpoint.longitude.degrees.item()
                sat_alt = subpoint.elevation.m.item()

                # Great-circle distance from KSC to subsatellite point
                actual_dist = geod.Inverse(KSC_LAT, KSC_LON, sat_lat, sat_lon)["s12"]

                # Visibility circle radius for this altitude (any single point
                # on the polygon is at distance arc_m from center by construction)
                vis_lats, vis_lons = visibility_circle(
                    KSC_LAT, KSC_LON, 0.0, sat_alt, min_el=0.0, n_points=4
                )
                vis_radius = geod.Inverse(
                    KSC_LAT, KSC_LON, float(vis_lats[0]), float(vis_lons[0])
                )["s12"]

                assert abs(actual_dist - vis_radius) < 50_000, (
                    f"At {boundary_time}: subsatellite distance {actual_dist/1e3:.0f} km "
                    f"vs visibility radius {vis_radius/1e3:.0f} km, "
                    f"diff {abs(actual_dist - vis_radius)/1e3:.0f} km"
                )
