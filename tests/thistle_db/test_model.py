import pathlib

import pytest

from .conftest import TLE_FILES
from thistle_db.model import TLE
from thistle_db.reader import read_tle


@pytest.mark.parametrize("file", TLE_FILES)
def test_from_twoline(file: pathlib.Path):
    tles = read_tle(file)
    for line1, line2 in tles:
        item = TLE.from_twoline(line1, line2)
        assert isinstance(item, TLE)
