import datetime

from thistle.reader import parse_tle_file
from thistle.utils import trange

BASIC_TIMES = trange(
    datetime.datetime(2000, 1, 1, 0), datetime.datetime(2000, 1, 2, 0), step=360
)
ISS_SATRECS = parse_tle_file("tests/data/25544.tle")
