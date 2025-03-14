import datetime
import os
import pathlib
from typing import Optional, Union

from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

from thistle.alpha5 import Satnum, from_alpha5

PathLike = Union[str, bytes, os.PathLike, pathlib.Path]
TLETuple = tuple[str, str]


def tle_datetime(tle: TLETuple) -> datetime.datetime:
    sat = Satrec.twoline2rv(*tle)
    dt = sat_epoch_datetime(sat)
    return dt


def tle_satnum(tle: TLETuple) -> str:
    return tle[0][2:7].replace(" ", "0")


def read_tle(
    file: PathLike,
) -> list[TLETuple]:
    results = []
    with open(file, "r") as f:
        current = [None, None]
        for line in f:
            if line[0] == "1":
                current[0] = line
                current[1] = None
            elif line[0] == "2":
                current[1] = line
                results.append(tuple(current))
    return results


def _tles_to_satrecs(tles: list[TLETuple]) -> list[Satrec]:
    return [Satrec.twoline2rv(a, b) for a, b in tles]


def _satrec_epoch(satrec: Satrec) -> float:
    return satrec.jdsatepoch + satrec.jdsatepochF


class SatrecDict:
    satrecs: dict[int, list[Satrec]]

    def __init__(self) -> None:
        self.satrecs = {}

    def append(self, satrec: Satrec) -> None:
        if satrec.satnum not in self.satrecs:
            self.satrecs[satrec.satnum] = []
        self.satrecs[satrec.satnum].append(satrec)

    def extend(self, satrecs: list[Satrec]) -> None:
        for satrec in satrecs:
            self.append(satrec)

    def get(self, satnum: Satnum) -> list[Satrec]:
        satnum = from_alpha5(satnum)
        if satnum not in self.satrecs:
            return []
        self.satrecs[satnum] = sorted(self.satrecs[satnum], key=_satrec_epoch)
        return self.satrecs[satnum]


class TLEReader:
    satnums: Optional[list[int]]
    _satrecs: SatrecDict

    def __init__(self, satnums: Optional[list[Satnum]] = None) -> None:
        self._satrecs = SatrecDict()
        self.satnums = None
        if satnums is not None:
            self.satnums = [from_alpha5(satnum) for satnum in satnums]

    def read(
        self,
        path: PathLike,
    ) -> None:
        tles = read_tle(path)
        satrecs = _tles_to_satrecs(tles)

        if self.satnums is not None:
            satrecs = [sat for sat in satrecs if sat.satnum in self.satnums]

        self._satrecs.extend(satrecs)

    def select(self, satnum: Satnum) -> list[Satrec]:
        return self._satrecs.get(satnum)
