import abc
import datetime
from typing import Literal, get_args

import numpy as np
import skyfield.timelib
from skyfield.api import EarthSatellite, Time, Timescale, load
from skyfield.positionlib import Distance, Geocentric, Velocity

from thistle.typing import DateTime, TLETuple
from thistle.utils import (
    DATETIME_MAX,
    DATETIME_MIN,
    EPOCH_DTYPE,
    datetime_to_dt64,
    dt64_to_datetime,
    time_to_dt64,
    validate_datetime64,
)

try:
    from itertools import pairwise
except ImportError:
    from thistle.utils import pairwise


SwitchingStrategies = Literal["epoch", "midpoint", "tca"]


# Transition Examples
# Epoch Switching
# -     A     B     C     D     E     +
# |-----~-----|-----|-----|-----|-----|
# Transitions: n + 1
# Segments: n
#
# MidpointSWitching
# -     A     B     C     D     E     +
# |-----~--|--~--|--~--|--~--|--~-----|
# Transitions: n + 1
# Segments: n
#
# TCA Switching
# -     A     B     C     D     E     +
# |-----~--|--~--|--~--|--~--|--~-----|
# Transitions: n + 1
# Segments: n


class SwitchingStrategy(abc.ABC):
    satellites: list[EarthSatellite]
    transitions: np.ndarray

    def __init__(
        self,
        satellites: list[EarthSatellite],
    ) -> None:
        self.satellites = sorted(satellites, key=lambda sat: sat.epoch.utc_datetime())
        self.transitions = None

    @abc.abstractmethod
    def compute_transitions(self) -> None: ...


class EpochSwitchStrategy(SwitchingStrategy):
    """Switching based on the TLE epoch.

    This TLE switching strategy selects the TLE generated with an epoch
    closest to the time without being "in the future".
    """

    def compute_transitions(self) -> None:
        transitions = [
            sat.epoch.utc_datetime().replace(tzinfo=None) for sat in self.satellites
        ]
        transitions = [DATETIME_MIN] + transitions[1:] + [DATETIME_MAX]
        self.transitions = np.array(
            [datetime_to_dt64(dt) for dt in transitions],
            dtype=EPOCH_DTYPE,
        )


class MidpointSwitchStrategy(SwitchingStrategy):
    """Switching based on the midpoint between neighboring TLE epoch times.

    This TLE switching strategy selects the TLE nearest to the desired time,
    regardless of whether that TLE is precedes it or is "in the future".
    """

    def compute_transitions(self) -> None:
        transitions = []
        for sat_a, sat_b in pairwise(self.satellites):
            time_a = sat_a.epoch.utc_datetime().replace(tzinfo=None)
            time_b = sat_b.epoch.utc_datetime().replace(tzinfo=None)

            delta = time_b - time_a
            midpoint = time_a + delta / 2
            midpoint = datetime_to_dt64(midpoint)
            transitions.append(midpoint)

        transitions = [DATETIME_MIN] + transitions + [DATETIME_MAX]
        self.transitions = np.array(transitions, dtype=EPOCH_DTYPE)


class TCASwitchStrategy(SwitchingStrategy):
    """Switching based on the time of closest approach for neighboring TLEs.

    This TLE switching method attempts to determine the time of closest approach
    for each pair of neighboring TLEs and use those times as the transitions.
    """


def _slices_by_transitions(
    transitions: np.ndarray[np.datetime64], times: np.ndarray[np.datetime64]
) -> list[tuple[int, np.ndarray[np.int64]]]:
    """Split a time vector into slices based on a sequence of transition times."""
    indices = []
    t0 = times[0]
    t1 = times[-1]

    # Avoid traversing the ENTIRE Satrec list by checking
    # the first & last progataion times

    # Find the first transition index to search
    start_idx = np.nonzero(transitions <= t0)[0][-1]

    # Find the last transition index to search
    stop_idx = np.nonzero(t1 < transitions)[0][0]

    search_space = transitions[start_idx : stop_idx + 1]
    for idx, bounds in enumerate(pairwise(search_space), start=start_idx):
        time_a, time_b = bounds
        cond1 = time_a <= times
        cond2 = times < time_b
        comb = np.logical_and(cond1, cond2)
        slice_ = np.nonzero(comb)[0]
        indices.append((idx, slice_))
    return indices


def merge_geos(geos: list[Geocentric], ts: Timescale) -> Geocentric:
    center = geos[0].center
    target = geos[0].target
    pos = Distance(au=np.concatenate([g.xyz.au for g in geos], axis=1))
    vel = Velocity(au_per_d=np.concatenate([g.velocity.au_per_d for g in geos], axis=1))
    times = Time(ts=ts, tt=np.concatenate([g.t.tt for g in geos]))
    return Geocentric(pos.au, vel.au_per_d, times, center, target)


class Propagator:
    satellites: list[EarthSatellite]
    switcher: SwitchingStrategy
    ts: skyfield.timelib.Timescale

    def __init__(
        self, tles: list[TLETuple], *, method: SwitchingStrategies = "epoch"
    ) -> None:
        self.ts = load.timescale()
        self.satellites = [EarthSatellite(a, b, ts=self.ts) for a, b in tles]
        # dt, leap_second = earth_satellites[0].utc_datetime_and_leap_second()
        # dt += datetime.timedelta(seconds=leap_second)

        method = method.lower()
        if method == "epoch":
            switcher = EpochSwitchStrategy(self.satellites)
        elif method == "midpoint":
            switcher = MidpointSwitchStrategy(self.satellites)
        elif method == "tca":
            switcher = TCASwitchStrategy(self.satellites)
        else:
            msg = f"Switching method {method!r} must be in {get_args(SwitchingStrategies)!r}"
            raise ValueError(msg)

        self.switcher = switcher
        self.switcher.compute_transitions()

    def find_satellite(self, time: DateTime) -> EarthSatellite:
        time = validate_datetime64(time)
        indices = _slices_by_transitions(self.switcher.transitions, np.atleast_1d(time))
        idx, _ = indices[0]
        return self.satellites[idx]

    def at(self, tt: skyfield.timelib.Time) -> Geocentric:
        dt64 = time_to_dt64(tt)
        indices = _slices_by_transitions(self.switcher.transitions, dt64)
        geos = []
        for idx, slice_ in indices:
            t = self.ts.from_datetimes(
                [dt64_to_datetime(t).replace(tzinfo=datetime.UTC) for t in dt64[slice_]]
            )
            g = self.satellites[idx].at(t)
            geos.append(g)
        return merge_geos(geos, self.ts)
