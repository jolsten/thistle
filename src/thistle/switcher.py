import abc

import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

from thistle._utils import (
    DATETIME_MAX,
    DATETIME_MIN,
    EPOCH_DTYPE,
    datetime_to_dt64,
)

try:
    from itertools import pairwise
except ImportError:
    from thistle._utils import pairwise_recipe as pairwise


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


def slices_by_transitions(
    transitions: np.ndarray[np.datetime64], times: np.ndarray[np.datetime64]
) -> list[tuple[int, np.ndarray[np.int64]]]:
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


class TLESwitcher(abc.ABC):
    satrecs: list[Satrec]
    transitions: np.ndarray

    def __init__(
        self,
        satrecs: list[Satrec],
    ) -> None:
        self.satrecs = sorted(satrecs, key=lambda sat: sat_epoch_datetime(sat))
        self.transitions = None

    @abc.abstractmethod
    def compute_transitions(self) -> None:
        pass

    def propagate(
        self, times: np.ndarray[np.datetime64]
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = slices_by_transitions(self.transitions, times)

        E, R, V = [], [], []

        for idx, slice_ in indices:
            jd, fr = datetime64_to_jd_fr(times[slice_])
            e, r, v = self.satrecs[idx].sgp4_array(jd, fr)
            E.append(e)
            R.append(r)
            V.append(v)
            # print(idx, self.transitions[idx], slice_)
        return np.asarray(R).flatten(), np.asarray(V).flatten()


class EpochSwitcher(TLESwitcher):
    def compute_transitions(self) -> None:
        transitions = [
            sat_epoch_datetime(sat).replace(tzinfo=None) for sat in self.satrecs
        ]
        transitions = [DATETIME_MIN] + transitions[1:] + [DATETIME_MAX]
        self.transitions = np.array(
            [datetime_to_dt64(dt) for dt in transitions],
            dtype=EPOCH_DTYPE,
        )


class MidpointSwitcher(TLESwitcher):
    def compute_transitions(self) -> None:
        transitions = []
        # print(self.satrecs)
        # print("="*20)
        for sat_a, sat_b in pairwise(self.satrecs):
            time_a = sat_epoch_datetime(sat_a).replace(tzinfo=None)
            time_b = sat_epoch_datetime(sat_b).replace(tzinfo=None)

            # if time_a < time_b:
            delta = time_b - time_a
            midpoint = time_a + delta / 2
            midpoint = datetime_to_dt64(midpoint)
            transitions.append(midpoint)
            # else:
            #     if midpoint == time_b:
            #         print(time_a, midpoint, time_b)
            #         raise ValueError
            #     transitions.append(time_a)
        transitions = [DATETIME_MIN] + transitions + [DATETIME_MAX]
        self.transitions = np.array(transitions, dtype=EPOCH_DTYPE)


class TCASwitcher(TLESwitcher):
    pass
