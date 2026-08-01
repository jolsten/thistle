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


def test_read_tle_ignores_digit_leading_name_lines(tmp_path):
    # 3LE name lines are conventionally "0 NAME", but bare names occur; a
    # satellite name starting with a digit ("1KUNS-PF", "2019-XYZ") must
    # not be mistaken for line1/line2 (which always have "1 "/"2 ").
    line1 = "1 00022U 59009A   25107.90268499  .00013443  00000-0  59558-3 0  9999"
    line2 = "2 00022  50.2761  24.4877 0093242 224.4518 134.8929 15.18021433735521"
    file = tmp_path / "named.tle"
    file.write_text(f"1KUNS-PF\n{line1}\n{line2}\n2019-XYZ FRAGMENT\n")
    assert list(read_tle(file)) == [(line1, line2)]
