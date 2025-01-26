from numpy import dtype
import pytest

from thistle.reader import read_tle_file
from thistle._utils import dt64_to_datetime
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

from thistle.switcher import EpochSwitcher, MidpointSwitcher

def test_epoch_switcher():
    file = "tests/data/25544.tle"
    tles = read_tle_file(file)
    satrecs = [Satrec.twoline2rv(tle[0], tle[1]) for tle in tles]

    switcher = EpochSwitcher(satrecs)
    switcher.compute_transitions()
    for sat, time in zip(satrecs[1:], switcher.transitions):
        sat_epoch = sat_epoch_datetime(sat).replace(tzinfo=None).isoformat(sep="T", timespec="milliseconds")
        assert sat_epoch == str(time)[0:23]

def test_midpoint_switcher():
    file = "tests/data/25544.tle"
    tles = read_tle_file(file)
    satrecs = [Satrec.twoline2rv(tle[0], tle[1]) for tle in tles]
    
    switcher = MidpointSwitcher(satrecs)
    switcher.compute_transitions()
    for idx, t_time in enumerate(switcher.transitions):
        t_time = dt64_to_datetime(t_time)
        sat_a, sat_b = switcher.satrecs[idx:idx+2]
        time_a = sat_epoch_datetime(sat_a).replace(tzinfo=None)
        time_b = sat_epoch_datetime(sat_b).replace(tzinfo=None)

        print(time_a, t_time, time_b)

        assert time_a <= t_time
        assert t_time <= time_b
        assert t_time - time_a == time_b - t_time
