"""Satellite orbit propagation with automatic TLE switching.

Provides a :class:`Propagator` that manages multiple TLEs for a satellite and
selects the most appropriate one at each propagation time using a configurable
:class:`SwitchingStrategy`.
"""

import abc
import datetime
from typing import Literal, Union, get_args

import numpy as np
import numpy.typing as npt
from sgp4.api import Satrec
from sgp4.exporter import export_tle
from skyfield.api import EarthSatellite, Time, Timescale, load
from skyfield.positionlib import Distance, Geocentric, Velocity

from thistle.typing import DateTime, TLETuple
from thistle.utils import (
    DATETIME_MAX,
    DATETIME_MIN,
    EPOCH_DTYPE,
    datetime_to_dt64,
    time_to_dt64,
    validate_datetime64,
)

try:
    from itertools import pairwise
except ImportError:
    from thistle.utils import pairwise

UTC = datetime.timezone.utc

SwitchingStrategies = Literal["epoch", "midpoint", "tca"]
"""Valid switching strategy names for :class:`Propagator`."""


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
    transitions: npt.NDArray[np.datetime64]

    def __init__(
        self,
        satellites: list[EarthSatellite],
    ) -> None:
        """Initialize the strategy with a list of satellites.

        Args:
            satellites: EarthSatellite objects to manage. They will be
                sorted by epoch internally.
        """
        self.satellites = sorted(satellites, key=lambda sat: sat.epoch.utc_datetime())

    @abc.abstractmethod
    def compute_transitions(self) -> None:
        """Compute the transition boundary array and store it in ``self.transitions``."""
        ...


class EpochSwitchStrategy(SwitchingStrategy):
    """Switching based on the TLE epoch.

    Selects the TLE whose epoch is closest to the target time without
    being in the future. Transition boundaries are placed at each TLE's
    epoch time.
    """

    def compute_transitions(self) -> None:
        """Compute transitions at each satellite's epoch time."""
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

    Selects the TLE nearest to the desired time, regardless of whether
    it precedes the target or is in the future. Transition boundaries
    are placed at the temporal midpoint between consecutive epochs.
    """

    def compute_transitions(self) -> None:
        """Compute transitions at the midpoint between consecutive epochs."""
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


def _find_tca(
    sat_a: EarthSatellite,
    sat_b: EarthSatellite,
    ts: Timescale,
    coarse_step: float = 60.0,
    fine_step: float = 5.0,
) -> datetime.datetime:
    """Find the Time of Closest Approach between two satellite TLEs.

    Uses a two-pass grid search over Julian dates: a coarse sweep across
    the full epoch window followed by a finer sweep around the coarse
    minimum. All time arithmetic is done in numpy with TT Julian dates
    to avoid creating Python datetime objects.

    Args:
        sat_a: The earlier satellite (by epoch).
        sat_b: The later satellite (by epoch).
        ts: Skyfield Timescale for building Time objects.
        coarse_step: Time step in seconds for the coarse search grid.
        fine_step: Time step in seconds for the refinement search grid.

    Returns:
        The datetime (timezone-naive, UTC) of closest approach.
    """
    tt_a = sat_a.epoch.tt
    tt_b = sat_b.epoch.tt
    delta_days = tt_b - tt_a
    delta_seconds = delta_days * 86400.0

    if delta_seconds < 1.0:  # less than 1 second apart
        mid_tt = (tt_a + tt_b) / 2.0
        return ts.tt_jd(mid_tt).utc_datetime().replace(tzinfo=None)

    # Coarse grid search — use fixed time step
    n_coarse = max(2, int(delta_seconds / coarse_step) + 1)
    coarse_tt = np.linspace(tt_a, tt_b, n_coarse)
    coarse_times = ts.tt_jd(coarse_tt)
    diff = sat_a.at(coarse_times).xyz.au - sat_b.at(coarse_times).xyz.au
    dist_sq = np.sum(diff**2, axis=0)
    ci = int(np.argmin(dist_sq))

    # Fine grid refinement around the coarse minimum
    # Refine over +/- one coarse step centered on the minimum
    coarse_step_jd = coarse_step / 86400.0
    fine_start = max(tt_a, coarse_tt[ci] - coarse_step_jd)
    fine_end = min(tt_b, coarse_tt[ci] + coarse_step_jd)
    fine_window_seconds = (fine_end - fine_start) * 86400.0
    n_fine = max(2, int(fine_window_seconds / fine_step) + 1)

    fine_tt = np.linspace(fine_start, fine_end, n_fine)
    fine_times = ts.tt_jd(fine_tt)
    diff = sat_a.at(fine_times).xyz.au - sat_b.at(fine_times).xyz.au
    dist_sq = np.sum(diff**2, axis=0)
    fi = int(np.argmin(dist_sq))

    return ts.tt_jd(fine_tt[fi]).utc_datetime().replace(tzinfo=None)


class TCASwitchStrategy(SwitchingStrategy):
    """Switching based on the time of closest approach for neighboring TLEs.

    This TLE switching method attempts to determine the time of closest
    approach for each pair of neighboring TLEs and use those times as the
    transitions.

    Attributes:
        ts: Skyfield Timescale used for time conversions during TCA search.
        coarse_step: Time step in seconds for the coarse TCA search.
        fine_step: Time step in seconds for the fine TCA refinement.
    """

    def __init__(
        self,
        satellites: list[EarthSatellite],
        ts: Timescale,
        coarse_step: float = 60.0,
        fine_step: float = 5.0,
    ) -> None:
        """Initialize the TCA switching strategy.

        Args:
            satellites: EarthSatellite objects to manage.
            ts: Skyfield Timescale for time conversions during TCA search.
            coarse_step: Time step in seconds for the coarse search grid.
            fine_step: Time step in seconds for the fine refinement grid.
        """
        super().__init__(satellites)
        self.ts = ts
        self.coarse_step = coarse_step
        self.fine_step = fine_step

    def compute_transitions(self) -> None:
        """Compute transitions at the time of closest approach between consecutive TLEs."""
        transitions = []
        for sat_a, sat_b in pairwise(self.satellites):
            tca = _find_tca(
                sat_a,
                sat_b,
                self.ts,
                coarse_step=self.coarse_step,
                fine_step=self.fine_step,
            )
            transitions.append(datetime_to_dt64(tca))

        transitions = [DATETIME_MIN] + transitions + [DATETIME_MAX]
        self.transitions = np.array(transitions, dtype=EPOCH_DTYPE)


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
    bins = np.searchsorted(transitions, times, side="right") - 1
    indices = []
    for idx in range(int(bins[0]), int(bins[-1]) + 1):
        slice_ = np.nonzero(bins == idx)[0]
        if len(slice_) > 0:
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
    pos = Distance(au=np.concatenate([g.xyz.au for g in geos], axis=1))  # type: ignore[call-overload]
    vel = Velocity(au_per_d=np.concatenate([g.velocity.au_per_d for g in geos], axis=1))  # type: ignore[call-overload]
    times = Time(ts=ts, tt=np.concatenate([g.t.tt for g in geos]))  # type: ignore[call-overload]
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
        self,
        tles: list[TLETuple],
        *,
        method: SwitchingStrategies = "epoch",
        start: Union[datetime.datetime, None] = None,
        stop: Union[datetime.datetime, None] = None,
    ) -> None:
        """Initialize the propagator.

        Args:
            tles: List of (line1, line2) TLE tuples for a single satellite.
            method: TLE switching strategy. One of ``"epoch"``,
                ``"midpoint"``, or ``"tca"``.
            start: Optional start time to filter TLEs. Only TLEs relevant
                to this time range will be kept.
            stop: Optional stop time to filter TLEs. Only TLEs relevant
                to this time range will be kept.

        Raises:
            ValueError: If method is not a valid switching strategy.
        """
        self.ts = load.timescale()
        self.satellites = [EarthSatellite(a, b, ts=self.ts) for a, b in tles]
        self.satellites.sort(key=lambda s: s.epoch.tt)

        # Filter satellites to time range if specified
        if start is not None or stop is not None:
            epochs = np.array([s.epoch.tt for s in self.satellites])
            lo = 0
            hi = len(self.satellites)

            if start is not None:
                start_tt = self.ts.from_datetime(start.replace(tzinfo=UTC)).tt
                # Keep one satellite before start for boundary coverage
                lo = max(0, int(np.searchsorted(epochs, start_tt)) - 1)

            if stop is not None:
                stop_tt = self.ts.from_datetime(stop.replace(tzinfo=UTC)).tt
                # Keep one satellite after stop for boundary coverage
                hi = min(hi, int(np.searchsorted(epochs, stop_tt)) + 1)

            self.satellites = self.satellites[lo:hi]

        strategy = method.lower()
        switcher: SwitchingStrategy
        if strategy == "epoch":
            switcher = EpochSwitchStrategy(self.satellites)
        elif strategy == "midpoint":
            switcher = MidpointSwitchStrategy(self.satellites)
        elif strategy == "tca":
            switcher = TCASwitchStrategy(self.satellites, ts=self.ts)
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

    def find_satrec(self, time: DateTime) -> Satrec:
        """Find the appropriate Satrec for a given time.

        Args:
            time: The target time as a datetime or numpy.datetime64.

        Returns:
            The sgp4 Satrec whose TLE is most appropriate for the
            given time according to the switching strategy.
        """
        time = validate_datetime64(time)
        return self.find_satellite(time).model

    def find_tle(self, time: DateTime) -> TLETuple:
        """Find the appropriate TLE lines for a given time.

        Args:
            time: The target time as a datetime or numpy.datetime64.

        Returns:
            A (line1, line2) tuple of TLE strings for the most appropriate
            TLE according to the switching strategy.
        """
        satrec = self.find_satrec(time)
        return export_tle(satrec)

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
            g = self.satellites[idx].at(tt[slice_])
            geos.append(g)
        return merge_geos(geos, self.ts)
