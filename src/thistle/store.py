import datetime
import itertools
import os
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, Literal, Union

import numpy as np
from sgp4.api import Satrec, SatrecArray
from sgp4.conveniences import sat_epoch_datetime

PathLike = Union[str, bytes, pathlib.Path, os.PathLike]


def _init() -> list:
    return [None] * 2


def _read_tle_file(path: PathLike) -> list[tuple[str]]:
    tles = []
    with open(path, "r") as reader:
        lines = _init()
        for line in reader:
            line = line[:-1]
            line_no = int(line[0])
            if line_no == 1:
                lines[0] = line
            elif line_no == 2:
                lines[1] = line
                tles.append(tuple(lines))
                lines = _init()
    return tles


def _datetime_to_dt64ms(dt: datetime.datetime) -> np.datetime64:
    return np.datetime64(dt.isoformat(sep="T", timespec="milliseconds"), "ms")


def _pairwise(iterable: Iterable) -> Iterable:
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


JDAY_1970 = 2440587.5

_time_scales = {
    "D": 1 / 86_400,
    "s": 1,
    "ms": 1_000,
    "us": 1_000_000,
    "ns": 1_000_000_000,
}


def jday_range(
    start: datetime.datetime,
    stop: datetime.datetime,
    step: float = 60,
    dtype: Literal["D", "s", "ms", "us", "ns"] = "us",
) -> tuple[np.ndarray, np.ndarray]:
    if dtype in _time_scales:
        scale = _time_scales[dtype]
    else:
        msg = f"dtype {dtype!r} is invalid"
        raise ValueError(msg)

    t0 = np.datetime64(start.isoformat(sep="T", timespec="microseconds"), dtype)
    t1 = np.datetime64(stop.isoformat(sep="T", timespec="microseconds"), dtype)
    dt = np.timedelta64(int(step * scale), dtype)
    # print(t0, t1, dt)
    times = np.arange(t0, t1 + dt, dt)
    times = times.astype("i8") / (86_400 * scale)
    jd = np.floor(times)
    fr = times - jd
    jd = jd + JDAY_1970
    return jd, fr


@dataclass
class Satellite:
    satnum: str
    _epochs: np.ndarray[np.datetime64] = field(repr=False, init=False, default=None)
    _satrec: list[Satrec] = field(repr=False, default_factory=list)

    def add_satrec(self, satrec: Union[Satrec, list[Satrec]]) -> None:
        satrec = np.atleast_1d(satrec).tolist()
        self._satrec.extend(satrec)

    def _build_epochs(self) -> None:
        if self._epochs is None or len(self._satrec) > len(self._epochs):
            epochs = [_datetime_to_dt64ms(sat_epoch_datetime(s)) for s in self._satrec]
            self._epochs = np.asarray(epochs, dtype="datetime64[ms]")

    def compute_nearest(self) -> None:
        self._build_epochs()

    def find_nearest(self, epoch: datetime.datetime) -> Satrec:
        self._build_epochs()
        epoch = _datetime_to_dt64ms(epoch)
        deltas = np.abs(self._epochs - epoch)
        min_idx = np.argmin(deltas)
        return self._satrec[min_idx]

    def compute_tcas(self) -> None:
        delta_t = 60.0 / 86_400
        self._tcas = []
        for sat_a, sat_b in _pairwise(self._satrec):
            jd_a, fr_a = sat_a.jdsatepoch, sat_a.jdsatepochF
            jd_b, fr_b = sat_b.jdsatepoch, sat_b.jdsatepochF

            jd, fr = [], []
            times = []
            jd_x, fr_x = jd_a, fr_a
            while jd_x <= jd_b and fr_x <= fr_b:
                fr_x += delta_t
                if fr_x >= 1.000:
                    jd_x += 1
                    fr_x += -1
                jd.append(jd_x)
                fr.append(fr_x)
                times.append((jd_x, fr_x))

            jd = np.array(jd, dtype="f8")
            fr = np.array(fr, dtype="f8")
            sat_array = SatrecArray([sat_a, sat_b])
            e, r, _ = sat_array.sgp4(jd, fr)

            # TODO: Error checking?
            # if e != 0:
            #     msg = f"spg4 propogator encountered error {e}"
            #     raise Exception(msg)

            try:
                distance = np.linalg.norm(r[0] - r[1], axis=1)
                pca_idx = np.argmin(distance)
                jd_tca, fr_tca = times[pca_idx]
            except ValueError:
                self._tcas.append(None)
            else:
                self._tcas.append((jd_tca, fr_tca))

    def propagate_nearest(
        self, times: np.ndarray[np.datetime64]
    ) -> tuple[int, float, float]:
        first_satrec = self.find_nearest(times[0])


@dataclass
class TLEStore:
    _single_stores: dict[str, Satellite] = field(default_factory=dict, repr=False)

    def load(self, path: PathLike) -> None:
        tles = _read_tle_file(path)
        for tle in tles:
            sat = Satrec.twoline2rv(tle[0], tle[1])

            if sat.satnum_str not in self._single_stores:
                self._single_stores[sat.satnum_str] = Satellite(satnum=sat.satnum_str)
            self._single_stores[sat.satnum_str].add_satrec(sat)

    def find_nearest(self, satnum: str, epoch: datetime.datetime) -> Satrec:
        return self._single_stores[satnum].find_nearest(epoch)
