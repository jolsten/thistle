import datetime
import pathlib

import pytest
from sgp4.api import Satrec

from thistle.io import read_tle
from thistle.utils import trange

BASIC_TIMES = trange(
    datetime.datetime(2000, 1, 1, 0), datetime.datetime(2000, 1, 2, 0), step=360
)
ISS_TLES = read_tle("tests/data/25544.tle")
ISS_SATRECS = [Satrec.twoline2rv(a, b) for a, b in ISS_TLES]

DAILY_FILES = list(pathlib.Path("tests/data/day").glob("*.txt"))
OBJECT_FILES = list(pathlib.Path("tests/data/obj").glob("*.txt"))


@pytest.fixture
def iss_satrecs() -> list[Satrec]:
    return ISS_SATRECS
