import concurrent.futures
import pathlib
from typing import Iterable

import tqdm

from thistle import utils
from thistle.typing import PathLike, TLETuple
from thistle.utils import tle_epoch, tle_satnum


def read_tle(
    file: PathLike,
) -> list[TLETuple]:
    """Read a single TLE file.

    Parses a file containing Two-Line Element sets, extracting line 1/line 2
    pairs based on the leading character of each line.

    Args:
        file: Path to the TLE file.

    Returns:
        A list of (line1, line2) tuples for each TLE in the file.
    """
    results: list[TLETuple] = []
    with open(file, "r") as f:
        line1 = None
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line[0] == "1":
                line1 = line
            elif line[0] == "2" and line1 is not None:
                results.append((line1, line))
                line1 = None
    return results


def read_tles(files: Iterable[PathLike]) -> list[TLETuple]:
    """Read multiple TLE files concurrently.

    Uses a thread pool to read files in parallel and combines the results
    into a single list.

    Args:
        files: Paths to the TLE files.

    Returns:
        A combined list of (line1, line2) tuples from all files.
    """
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
    """Write TLEs to a single file.

    Args:
        file_path: Destination file path.
        tles: TLE (line1, line2) tuples to write.
        sort: If True, sort TLEs by satellite number then epoch.
        unique: If True, remove duplicate TLEs before writing.
    """
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
    """Write TLEs to multiple files concurrently.

    Uses a thread pool to write files in parallel.

    Args:
        files: Mapping of file paths to their TLE tuples.
        unique: If True, remove duplicate TLEs before writing.
        sort: If True, sort TLEs by satellite number then epoch.
        progress_bar: If True, display a tqdm progress bar.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with tqdm.tqdm(
            total=len(files), desc="writing tle files", disable=not progress_bar
        ) as pbar:
            futures: dict[concurrent.futures.Future, pathlib.Path] = {
                executor.submit(write_tle, file, tles, unique=unique, sort=sort): file
                for file, tles in files.items()
            }

            for future, file in futures.items():
                _ = future.result()
                pbar.update(1)
