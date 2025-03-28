from typing import Literal, get_args

import numpy as np
from sgp4.api import Satrec

from thistle.switcher import (
    EpochSwitcher,
    MidpointSwitcher,
    SwitchingStrategy,
    TCASwitcher,
)
from thistle.typing import DateTime, TLETuple
from thistle.utils import jday_datetime64, validate_datetime64

try:
    from itertools import pairwise
except ImportError:
    from thistle.utils import pairwise


SwitchingStrategies = Literal["epoch", "midpoint", "tca"]


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


class Propagator:
    satrecs: list[Satrec]
    switcher: SwitchingStrategy

    def __init__(
        self, tles: list[TLETuple], *, method: SwitchingStrategies = "epoch"
    ) -> None:
        self.satrecs = [Satrec.twoline2rv(a, b) for a, b in tles]

        method = method.lower()
        if method == "epoch":
            switcher = EpochSwitcher(self.satrecs)
        elif method == "midpoint":
            switcher = MidpointSwitcher(self.satrecs)
        elif method == "tca":
            switcher = TCASwitcher(self.satrecs)
        else:
            msg = f"Switching method {method!r} must be in {get_args(SwitchingStrategies)!r}"
            raise ValueError(msg)

        self.switcher = switcher
        self.switcher.compute_transitions()

    def find_satrec(self, time: DateTime) -> Satrec:
        time = validate_datetime64(time)
        indices = _slices_by_transitions(self.switcher.transitions, np.atleast_1d(time))
        idx, _ = indices[0]
        return self.satrecs[idx]

    def propagate(
        self, times: np.ndarray[np.datetime64]
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
        indices = _slices_by_transitions(self.switcher.transitions, times)

        e, r, v = [], [], []
        for idx, slice_ in indices:
            jd, fr = jday_datetime64(times[slice_])
            a, b, c = self.satrecs[idx].sgp4_array(jd, fr)
            e.append(a)
            r.append(b)
            v.append(c)
        e = np.asarray(e).flatten()
        r = np.asarray(r).flatten()
        v = np.asarray(v).flatten()
        return e, r, v
