import argparse
import pathlib
import shutil
import tempfile
from typing import Iterable

from benchmark import TLETuple
from thistle.reader import PathLike, read_tle, tle_datetime, tle_satnum


def write_tle(filename: PathLike, tle_list: Iterable[TLETuple]) -> None:
    with open(filename, "w") as f:
        for line1, line2 in sorted(tle_list, key=tle_datetime):
            print(line1, file=f)
            print(line2, file=f)


def tle_jday(tle: TLETuple) -> float:
    jday = float(tle[0][18:32]) + 1900
    if jday < 1957:
        jday += 100
    return jday


def main():
    parser = argparse.ArgumentParser("fix-tle")
    parser.add_argument("file", type=pathlib.Path, help="path to tle file")
    parser.add_argument(
        "output", type=pathlib.Path, default=None, help="output file", nargs="?"
    )
    args = parser.parse_args()

    infile = pathlib.Path(args.file).absolute()

    tles = read_tle(infile)

    # Use ordered dictionary to de-dupe entries
    results = {}
    for tle in tles:
        results[tle] = None
    results = results.keys()

    results = sorted(results, key=tle_satnum)  # Sort by second attribute first
    results = sorted(results, key=tle_jday)  # Sort by first attribute last

    with tempfile.TemporaryFile("w", dir=infile.parent) as outfile:
        write_tle(outfile, results)
        shutil.move(outfile, infile)


if __name__ == "__main__":
    main()
