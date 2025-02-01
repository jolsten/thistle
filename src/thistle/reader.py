import os
import pathlib
from typing import Optional, Union

from sgp4.api import Satrec

from thistle.alpha5 import Satnum, from_alpha5

PathLike = Union[str, bytes, pathlib.Path, os.PathLike]


def _init() -> list:
    return [None] * 2


def _parse_tle_file(path: PathLike) -> list[tuple[str, str]]:
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


def _tles_to_satrecs(tles: list[tuple[str, str]]) -> list[Satrec]:
    return [Satrec.twoline2rv(a, b) for a, b in tles]


def _satrec_epoch(satrec: Satrec) -> float:
    return satrec.epochyr + satrec.epochdays


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
        tles = _parse_tle_file(path)
        satrecs = _tles_to_satrecs(tles)

        if self.satnums is not None:
            satrecs = [sat for sat in satrecs if sat.satnum in self.satnums]

        self._satrecs.extend(satrecs)

    def select(self, satnum: Satnum) -> list[Satrec]:
        return self._satrecs.get(satnum)
