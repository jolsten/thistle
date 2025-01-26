import abc
import datetime
from typing import Optional, Union, Iterable
import itertools

import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

TIME_SCALE = "us"
EPOCH_DTYPE = np.dtype(f"datetime64[{TIME_SCALE}]")


class TLESwitcher(abc.ABC):
    satrecs: list[Satrec]
    transitions: list[datetime.datetime]

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
        pass


class EpochSwitcher(TLESwitcher):
    def compute_transitions(self) -> None:
        transitions = [sat_epoch_datetime(sat).replace(tzinfo=None) for sat in self.satrecs[1:]]
        self.transitions = np.array(
            [
                np.datetime64(
                    dt.isoformat(sep="T", timespec="milliseconds"), TIME_SCALE
                )
                for dt in transitions
            ],
            dtype=EPOCH_DTYPE,
        )


def _pairwise(iterable: Iterable) -> Iterable:
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class MidpointSwitcher(TLESwitcher):
    def compute_transitions(self) -> None:
        transitions = []
        for sat_a, sat_b in _pairwise(self.satrecs):
            time_a = sat_epoch_datetime(sat_a).replace(tzinfo=None)
            time_b = sat_epoch_datetime(sat_b).replace(tzinfo=None)
            delta = time_b - time_a
            midpoint = time_a + delta / 2
            midpoint = np.datetime64(midpoint, TIME_SCALE)
            transitions.append(midpoint)
        self.transitions = np.array(transitions, dtype=EPOCH_DTYPE)


class TCASwitcher(TLESwitcher):
    pass
