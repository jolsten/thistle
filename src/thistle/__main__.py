import argparse
import pathlib
import shutil
import tempfile

from thistle.io import read_tle, write_tle
from thistle.utils import tle_epoch, tle_satnum


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
    results = sorted(results, key=tle_epoch)  # Sort by first attribute last

    with tempfile.TemporaryFile("w", dir=infile.parent, delete=False) as outfile:
        write_tle(outfile, results)
    shutil.move(outfile, infile)


if __name__ == "__main__":
    main()
