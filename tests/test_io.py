import pytest

from thistle.io import read_tle, read_tles

from .conftest import DAILY_FILES, OBJECT_FILES


@pytest.mark.parametrize("file", DAILY_FILES + OBJECT_FILES)
def test_read_one(file):
    tles = read_tle(file)
    assert len(tles)


def test_read_many(files):
    tles = read_tles(DAILY_FILES + OBJECT_FILES)
    assert len(tles)
