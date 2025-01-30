from typing import Literal, get_args

import numpy as np
from sgp4.api import Satrec

from thistle.switcher import EpochSwitcher, MidpointSwitcher, TCASwitcher

SwitchingStrategies = Literal["epoch", "midpoint", "tca"]


class Propagator:
    def __init__(
        self, satrecs: list[Satrec], *, method: SwitchingStrategies = "epoch"
    ) -> None:
        match method.lower():
            case "epoch":
                switcher = EpochSwitcher(satrecs)
            case "midpoint":
                switcher = MidpointSwitcher(satrecs)
            case "tca":
                switcher = TCASwitcher(satrecs)
            case _:
                msg = f"Switching method {method!r} must be in {get_args(SwitchingStrategies)!r}"
                raise ValueError(msg)

        self.switcher = switcher
        self.switcher.compute_transitions()

    def __call__(
        self, times: np.ndarray[np.datetime64]
    ) -> tuple[np.ndarray, np.ndarray[np.float64], np.ndarray[np.float64]]:
        return self.switcher.propagate(times)
