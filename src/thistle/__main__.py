import argparse
import pathlib
import shutil
import tempfile
from argparse import ArgumentTypeError

from thistle.alpha5 import ensure_alpha5
from thistle.config import Settings
from thistle.io import read_tle
from thistle.utils import tle_epoch, tle_satnum, unique


def file_exists(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.exists() and path.is_file():
        return path
    msg = f"Path must be a file that already exists, but got {value!r}"
    raise ArgumentTypeError(msg)


def satnum(value: str) -> str:
    value = ensure_alpha5(value)
    if len(value) == 5:
        return value
    msg = f"{value!r} is not a valid satnum"
    raise ValueError(msg)


def fix(args: argparse.Namespace) -> None:
    file: pathlib.Path = args.file
    tles = read_tle(file)

    tles = unique(tles)
    tles = sorted(tles, key=tle_satnum)
    tles = sorted(tles, key=tle_epoch)

    with tempfile.TemporaryFile(delete=False) as tmp:
        for tle in tles:
            tmp.writelines(tle)

    shutil.move(tmp, file)


def find(args: argparse.Namespace) -> None:
    if args.object and args.day:
        msg = "Provide either an object number (satnum) or a date, not both"
        raise ValueError(msg)

    settings = Settings()

    if args.object:
        file = settings.object / f"{args.object}{settings.suffix}"
    elif args.day:
        file = settings.daily / f"{args.day}{settings.suffix}"
    else:
        raise ValueError

    file = file.expanduser().absolute()
    print(str(file))


def main():
    parser = argparse.ArgumentParser("thistle")
    subparsers = parser.add_subparsers()

    sp_fix = subparsers.add_parser("fix")
    sp_fix.add_argument("file", type=file_exists, help="path to tle file")
    sp_fix.set_defaults(func=fix)

    sp_find = subparsers.add_parser("find")
    sp_find.add_argument("--object", "-o", type=satnum, help="find an object file")
    sp_find.add_argument("--day", "-d", type=str, help="find a day file")
    sp_find.set_defaults(func=find)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
