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


_L1 = "1 00022U 59009A   25107.90268499  .00013443  00000-0  59558-3 0  9999"
_L2 = "2 00022  50.2761  24.4877 0093242 224.4518 134.8929 15.18021433735521"


def test_from_twoline_numeric_intldesg_gets_century():
    item = TLE.from_twoline(_L1, _L2)
    assert item.object_id == "1959009A"


def test_from_twoline_non_numeric_intldesg_kept_verbatim():
    # Placeholder designators (e.g. "TBA") must not reject an otherwise
    # valid TLE as malformed — the value is stored as-is.
    line1 = _L1[:9] + "TBA".ljust(8) + _L1[17:]
    item = TLE.from_twoline(line1, _L2)
    assert item.object_id == "TBA"


def test_from_twoline_blank_intldesg():
    # Analyst objects are often distributed with a blank designator.
    line1 = _L1[:9] + " " * 8 + _L1[17:]
    item = TLE.from_twoline(line1, _L2)
    assert item.object_id == ""
