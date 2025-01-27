import os
import pathlib
from typing import Union

from sgp4.api import Satrec

PathLike = Union[str, bytes, pathlib.Path, os.PathLike]


def _init() -> list:
    return [None] * 2


def read_tle_file(path: PathLike) -> list[tuple[str]]:
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


def parse_tle_file(path: PathLike) -> list[Satrec]:
    return [Satrec.twoline2rv(a, b) for a, b in read_tle_file(path)]
