import concurrent.futures
import datetime
import os
import pathlib
from typing import Iterable, Union

import tqdm
from sgp4.api import Satrec

from thistle import utils
from thistle.alpha5 import to_alpha5
from thistle.config import Settings
from thistle.utils import group_by, tle_epoch, tle_satnum

PathLike = Union[str, bytes, os.PathLike, pathlib.Path]
TLETuple = tuple[str, str]


def read_tle(
    file: PathLike,
) -> list[TLETuple]:
    """Read a single TLE file."""
    results = []
    with open(file, "r") as f:
        current = [None, None]
        for line in f:
            line = line.rstrip()
            if line[0] == "1":
                current[0] = line
                current[1] = None
            elif line[0] == "2":
                current[1] = line
                results.append(tuple(current))
    return results


def read_tles(files: Iterable[PathLike]) -> list[TLETuple]:
    """Read multiple TLE files."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_tle, file) for file in files]
        tles = []
        for future in futures:
            results = future.result()
            tles.extend(results)
    return tles


def write_tle(
    file_path: PathLike,
    tles: Iterable[TLETuple],
    *,
    sort: bool = False,
    unique: bool = False,
) -> None:
    if unique:
        tles = utils.unique(tles)

    if sort:
        tles = sorted(tles, key=tle_epoch)
        tles = sorted(tles, key=tle_satnum)

    with open(file_path, "w") as f:
        for line1, line2 in tles:
            print(line1, file=f)
            print(line2, file=f)


def write_tles(
    files: dict[pathlib.Path, Iterable[TLETuple]],
    *,
    unique: bool = True,
    sort: bool = False,
    progress_bar: bool = False,
) -> None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with tqdm.tqdm(
            total=len(files), desc="writing tle files", disable=not progress_bar
        ) as pbar:
            futures: dict[concurrent.futures.Future, pathlib.Path] = {
                executor.submit(write_tle, file, tles, unique=unique, sort=sort): file
                for file, tles in files.items()
            }

            for file, future in futures.items():
                _ = future.result()
                pbar.update(1)


class Loader:
    satrecs: dict[int, list[Satrec]]
    settings: Settings

    def __init__(self, config: Settings) -> None:
        self.settings = config
        self.satrecs = {}

    def load_object(self, satnum: Union[str, int]) -> None:
        satnum = to_alpha5(satnum)
        file = self.settings.daily / f"{satnum}{self.settings.suffix}"
        tles = read_tle(file)
        self.satrecs[satnum] = tles

    def load_daily(self, date: datetime.date) -> None:
        file = self.settings.daily / f"{date.strftime('%Y%m%d')}{self.settings.suffix}"
        all_tles = read_tle(file)
        grouped = group_by(all_tles, key=tle_satnum)
        for satnum, tles in grouped.items():
            if satnum not in self.satrecs:
                self.satrecs[satnum] = []
            self.satrecs[satnum].extend(tles)
