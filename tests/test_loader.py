import pytest

from thistle.config import Settings
from thistle.loader import Loader


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
