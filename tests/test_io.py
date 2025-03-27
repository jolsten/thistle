import pytest

from thistle.config import Settings
from thistle.io import Loader, read_tle, read_tles

from .conftest import DAILY_FILES, OBJECT_FILES


@pytest.mark.parametrize("file", DAILY_FILES + OBJECT_FILES)
def test_read_one(file):
    tles = read_tle(file)
    assert len(tles)


def test_read_many():
    tles = read_tles(DAILY_FILES + OBJECT_FILES)
    assert len(tles)


class TestLoader:
    def setup_class(self):
        test_settings = Settings(
            archive=".", daily="tests/data/day", object="tests/data/obj", suffix=".txt"
        )
        self.loader = Loader(config=test_settings)

    def test_load_obj_str(self):
        tles = self.loader.load_object("25544")
        assert len(tles)

    def test_load_obj_int(self):
        tles = self.loader.load_object(25544)
        assert len(tles)

    def test_load_daily(self):
        tles = self.loader.load_day("20250301")
        assert len(tles)

    def test_load_object_no_exist(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load_object(99999)
