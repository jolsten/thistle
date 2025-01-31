import datetime

import pytest
from sgp4.api import Satrec

from thistle.reader import _parse_tle_file
from thistle.utils import trange

BASIC_TIMES = trange(
    datetime.datetime(2000, 1, 1, 0), datetime.datetime(2000, 1, 2, 0), step=360
)
ISS_SATRECS = [Satrec.twoline2rv(a, b) for a, b in _parse_tle_file("tests/data/25544.tle")]

@pytest.fixture
def iss_satrecs() -> list[Satrec]:
    return ISS_SATRECS
