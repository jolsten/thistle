import abc
import datetime
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
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

UTC = datetime.timezone.utc

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
    """Base class for TLE switching strategies.

    Subclasses define how transition boundaries between consecutive TLEs
    are computed.

    Attributes:
        satellites: EarthSatellite objects sorted by epoch.
        transitions: Boundary times separating each satellite's validity
            window. Set by ``compute_transitions``.
    """

    satellites: list[EarthSatellite]
    transitions: np.ndarray

    def __init__(
        self,
        satellites: list[EarthSatellite],
    ) -> None:
        self.satellites = sorted(satellites, key=lambda sat: sat.epoch.utc_datetime())

    @abc.abstractmethod
    def compute_transitions(self) -> None:
        """Compute the transition boundary array and store it in ``self.transitions``."""
        ...


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

    def compute_transitions(self) -> None:
        raise NotImplementedError("TCA switching is not yet implemented")


def _slices_by_transitions(
    transitions: npt.NDArray[np.datetime64], times: npt.NDArray[np.datetime64]
) -> list[tuple[int, npt.NDArray[np.intp]]]:
    """Split a time array into segments based on transition boundaries.

    Each segment maps to the satellite index whose validity window
    contains those times.

    Args:
        transitions: Sorted boundary times (length n + 1 for n satellites).
        times: Sorted array of propagation times.

    Returns:
        A list of (satellite_index, index_array) tuples, where
        index_array contains the positions in ``times`` that fall
        within that satellite's window.
    """
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
    """Concatenate multiple Geocentric results into a single object.

    Args:
        geos: Geocentric position results to merge.
        ts: The skyfield Timescale used to reconstruct the time array.

    Returns:
        A single Geocentric with concatenated position, velocity, and
        time arrays.
    """
    center = geos[0].center
    target = geos[0].target
    pos = Distance(au=np.concatenate([g.xyz.au for g in geos], axis=1))
    vel = Velocity(au_per_d=np.concatenate([g.velocity.au_per_d for g in geos], axis=1))
    times = Time(ts=ts, tt=np.concatenate([g.t.tt for g in geos]))
    return Geocentric(pos.au, vel.au_per_d, times, center, target)


class Propagator:
    """Satellite propagator with automatic TLE switching.

    Manages multiple TLEs for a satellite and automatically selects the
    most appropriate TLE for a given propagation time based on the
    configured switching strategy.

    Attributes:
        satellites: EarthSatellite objects created from the input TLEs.
        switcher: The active TLE switching strategy.
        ts: Skyfield timescale used for time conversions.
    """

    satellites: list[EarthSatellite]
    switcher: SwitchingStrategy
    ts: Timescale

    def __init__(
        self, tles: list[TLETuple], *, method: SwitchingStrategies = "epoch"
    ) -> None:
        """Initialize the propagator.

        Args:
            tles: List of (line1, line2) TLE tuples for a single satellite.
            method: TLE switching strategy. One of ``"epoch"``,
                ``"midpoint"``, or ``"tca"``.

        Raises:
            ValueError: If method is not a valid switching strategy.
        """
        self.ts = load.timescale()
        self.satellites = [EarthSatellite(a, b, ts=self.ts) for a, b in tles]

        strategy = method.lower()
        switcher: SwitchingStrategy
        if strategy == "epoch":
            switcher = EpochSwitchStrategy(self.satellites)
        elif strategy == "midpoint":
            switcher = MidpointSwitchStrategy(self.satellites)
        elif strategy == "tca":
            switcher = TCASwitchStrategy(self.satellites)
        else:
            msg = f"Switching method {strategy!r} must be in {get_args(SwitchingStrategies)!r}"
            raise ValueError(msg)

        self.switcher = switcher
        self.switcher.compute_transitions()

    def find_satellite(self, time: DateTime) -> EarthSatellite:
        """Find the appropriate satellite for a given time.

        Args:
            time: The target time as a datetime or numpy.datetime64.

        Returns:
            The EarthSatellite whose TLE is most appropriate for the
            given time according to the switching strategy.
        """
        time = validate_datetime64(time)
        indices = _slices_by_transitions(self.switcher.transitions, np.atleast_1d(time))
        idx, _ = indices[0]
        return self.satellites[idx]

    def at(self, tt: Time) -> Geocentric:
        """Propagate satellite position at the given time(s).

        Automatically switches between TLEs based on the configured
        strategy to produce the most accurate position across the time
        range.

        Args:
            tt: A skyfield Time object, scalar or array.

        Returns:
            A Geocentric position combining results from all TLE
            segments.
        """
        dt64 = time_to_dt64(tt)
        indices = _slices_by_transitions(self.switcher.transitions, dt64)
        geos = []
        for idx, slice_ in indices:
            t = self.ts.from_datetimes(
                [dt64_to_datetime(t).replace(tzinfo=UTC) for t in dt64[slice_]]
            )
            g = self.satellites[idx].at(t)
            geos.append(g)
        return merge_geos(geos, self.ts)
