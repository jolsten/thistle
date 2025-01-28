import abc

import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

from thistle._utils import (
    DATETIME_MAX,
    DATETIME_MIN,
    EPOCH_DTYPE,
    TIME_SCALE,
    datetime_to_dt64,
    dt64_to_datetime,
    pairwise,
)


def slices_by_transitions(
    transitions: np.ndarray[np.datetime64], times: np.ndarray[np.datetime64]
) -> list[tuple[int, np.ndarray[np.int64]]]:
    indices = []
    t0 = times[0]
    t1 = times[-1]

    print(f"Start: {t0}")
    print(f"Stop:  {t1}")

    # Avoid traversing the ENTIRE Satrec list by checking
    # the first & last progataion times

    # Find the first transition index to search
    print(
        "start idx",
        np.nonzero(transitions <= t0)[0],
        "->",
        np.nonzero(transitions <= t0)[0][-1],
        "->",
        transitions[np.nonzero(transitions <= t0)[0][-1]],
    )
    start_idx = np.nonzero(transitions <= t0)[0][-1]
    # print(
    #     "start idx",
    #     np.nonzero(t0 < transitions)[0],
    #     "->",
    #     np.nonzero(t0 < transitions)[0][0],
    #     "->",
    #     transitions[np.nonzero(t0 < transitions)[0][0]],
    # )
    # start_idx = np.nonzero(t0 < transitions)[0][0]

    # Find the last transition index to search
    print(
        "stop  idx",
        np.nonzero(t1 < transitions)[0],
        "->",
        np.nonzero(t1 < transitions)[0][0],
        "->",
        transitions[np.nonzero(t1 < transitions)[0][0]],
    )
    stop_idx = np.nonzero(t1 < transitions)[0][0]
    # print(
    #     "stop  idx",
    #     np.nonzero(transitions < t1)[0],
    #     "->",
    #     np.nonzero(transitions < t1)[0][-1] + 1,
    #     "->",
    #     transitions[np.nonzero(transitions < t1)[0][-1] + 1],
    # )
    # stop_idx = np.nonzero(transitions < t1)[0][-1] + 1

    # print("bb", transitions < t1)
    # print(f"Start Idx: {start_idx}")
    # print(f"Stop  Idx: {stop_idx}")

    search_space = transitions[start_idx : stop_idx + 1]
    print("time range", times[0], times[-1])
    print("transitions", transitions)
    print("search space", search_space)

    for idx, bounds in enumerate(pairwise(search_space), start=start_idx):
        # print("d", idx, bounds)
        time_a, time_b = bounds
        cond1 = time_a <= times
        cond2 = times < time_b
        comb = np.logical_and(cond1, cond2)
        slice_ = np.nonzero(comb)[0]
        # print("e", times[slice_])
        indices.append((idx, slice_))

    # print("y", times[slice_])
    # print("z", indices)

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

        for idx, slice_ in indices:
            print(idx, self.transitions[idx], slice_)
        return


class EpochSwitcher(TLESwitcher):
    def compute_transitions(self) -> None:
        transitions = [
            sat_epoch_datetime(sat).replace(tzinfo=None) for sat in self.satrecs
        ]
        transitions = [DATETIME_MIN] + transitions + [DATETIME_MAX]
        self.transitions = np.array(
            [datetime_to_dt64(dt) for dt in transitions],
            dtype=EPOCH_DTYPE,
        )


class MidpointSwitcher(TLESwitcher):
    def compute_transitions(self) -> None:
        transitions = []
        for sat_a, sat_b in pairwise(self.satrecs):
            time_a = sat_epoch_datetime(sat_a).replace(tzinfo=None)
            time_b = sat_epoch_datetime(sat_b).replace(tzinfo=None)
            delta = time_b - time_a
            midpoint = time_a + delta / 2
            midpoint = np.datetime64(midpoint, TIME_SCALE)
            transitions.append(midpoint)
        transitions = [DATETIME_MIN] + transitions + [DATETIME_MAX]
        self.transitions = np.array(transitions, dtype=EPOCH_DTYPE)


class TCASwitcher(TLESwitcher):
    pass
