import pathlib

import pytest

from .conftest import TLE_FILES
from thistle_db.reader import group_by, read_tle, read_tles, tle_satnum


@pytest.mark.parametrize("file", TLE_FILES)
def test_read_tle(file: pathlib.Path):
    tles = read_tle(file)
    for line1, line2 in tles:
        assert line1 and line2
        assert line1[0] == "1"
        assert line2[0] == "2"


def test_read_tles():
    tles = read_tles(TLE_FILES)
    for line1, line2 in tles:
        assert line1 and line2
        assert line1[0] == "1"
        assert line2[0] == "2"


@pytest.mark.parametrize(
    "file",
    [
        "tests/thistle_db/data/25544.txt",
    ],
)
def test_group_satnum(file: pathlib.Path):
    tles = read_tle(file)
    grouped = group_by(tles, key=tle_satnum)
    assert len(grouped) == 1
