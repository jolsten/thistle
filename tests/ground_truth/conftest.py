"""Shared fixtures and utilities for ground truth tests."""

import csv
import datetime
import pathlib
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pytest
from skyfield.api import EarthSatellite, load

from thistle.utils import read_tle

# Test data directory
TRUTH_DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "truth"

# Load ISS TLE for April 1, 2020
ts = load.timescale()


@pytest.fixture(scope="session")
def iss_tles():
    """Load ISS TLEs once for all ground truth tests.

    Returns:
        List of TLE tuples (line1, line2) for ISS from tests/data/25544.tle
    """
    return read_tle("tests/data/25544.tle")


@pytest.fixture(scope="session")
def april_2020_dates():
    """Date range for April 2020 ground truth data.

    Returns:
        Tuple of (start_date, end_date) for filtering TLEs around April 2020
    """
    start = datetime.datetime(2020, 3, 25, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2020, 4, 8, tzinfo=datetime.timezone.utc)
    return start, end


def get_iss_satellite() -> EarthSatellite:
    """Get ISS EarthSatellite for the ground truth data period (April 2020).

    The ground truth data was generated for April 1, 2020 00:00 UTC.
    We use TLE index 40395 which has epoch 2020-04-01 02:54:32 UTC
    (closest to the ground truth data period).
    """
    tles = read_tle("tests/data/25544.tle")

    # Use TLE index 40395 (epoch 2020-04-01 02:54:32 UTC)
    # This is the closest TLE to April 1, 2020 00:00 UTC
    tle_index = 40395
    line1, line2 = tles[tle_index]

    return EarthSatellite(line1, line2, ts=ts)


def load_truth_csv(filename: str) -> tuple[npt.NDArray[np.datetime64], Dict[str, npt.NDArray]]:
    """Load ground truth CSV file and return times and data arrays.

    Args:
        filename: Name of CSV file in truth data directory (e.g., "25544_eci.csv")

    Returns:
        Tuple of (times, data_dict) where:
        - times: Array of datetime64 values
        - data_dict: Dictionary mapping column names to numpy arrays
    """
    filepath = TRUTH_DATA_DIR / filename

    times = []
    data = {}
    skip_columns = set()  # Track columns that can't be converted to float

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)

        # Initialize data arrays from first row
        first_row = next(reader)
        time_key = [k for k in first_row.keys() if "Time" in k][0]

        # Try to convert each column to float, skip if it fails
        for key in first_row.keys():
            if "Time" in key:
                continue
            try:
                float(first_row[key])
                data[key] = []
            except (ValueError, TypeError):
                # Skip non-numeric columns
                skip_columns.add(key)

        # Parse time from first row
        times.append(parse_time(first_row[time_key]))
        for key in data.keys():
            data[key].append(float(first_row[key]))

        # Read remaining rows
        for row in reader:
            times.append(parse_time(row[time_key]))
            for key in data.keys():
                try:
                    data[key].append(float(row[key]))
                except (ValueError, TypeError):
                    # If a value becomes non-numeric later, use NaN
                    data[key].append(np.nan)

    # Convert to numpy arrays
    times_array = np.array(times, dtype='datetime64[us]')
    for key in data.keys():
        data[key] = np.array(data[key])

    return times_array, data


def parse_time(time_str: str) -> np.datetime64:
    """Parse time string from CSV format to datetime64.

    Format: "1 Apr 2020 00:00:00.000"
    """
    # Parse the datetime string
    dt = datetime.datetime.strptime(time_str, "%d %b %Y %H:%M:%S.%f")
    # Convert to datetime64 with microsecond precision
    return np.datetime64(dt, 'us')


def load_eclipse_times(filename: str) -> List[tuple[np.datetime64, np.datetime64]]:
    """Load eclipse time intervals from CSV.

    Args:
        filename: Name of CSV file (e.g., "25544_Umbra.csv")

    Returns:
        List of (start_time, stop_time) tuples
    """
    filepath = TRUTH_DATA_DIR / filename

    intervals = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = parse_time(row["Start Time (UTCG)"])
            stop = parse_time(row["Stop Time (UTCG)"])
            intervals.append((start, stop))

    return intervals
