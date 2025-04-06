import datetime
import pathlib
import shutil
import tempfile
from argparse import ArgumentTypeError
from typing import Optional

import click

from thistle.alpha5 import ensure_alpha5
from thistle.config import Settings, _config_template, user_config_file
from thistle.io import read_tle, write_tle
from thistle.utils import tle_epoch, tle_satnum, unique


def file_exists(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.exists() and path.is_file():
        return path
    msg = f"Path must be a file that already exists, but got {value!r}"
    raise ArgumentTypeError(msg)


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        writable=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
def fix(file: pathlib.Path) -> None:
    click.echo(f"Fixing {file}", color="green")
    tles = read_tle(file)

    click.echo(f"Read {len(tles)} TLEs", color="green")
    tles = unique(tles)
    tles = sorted(tles, key=tle_satnum)
    tles = sorted(tles, key=tle_epoch)

    with tempfile.TemporaryFile(delete=False) as tmp:
        write_tle(tmp.name, tles)

    shutil.move(tmp.name, file)
    click.echo(f"Wrote {len(tles)} TLEs", color="green")


def satnum(value: str) -> str:
    value = ensure_alpha5(value)
    if len(value) == 5:
        return value
    msg = f"{value!r} is not a valid satnum"
    raise ValueError(msg)


def date(value: str) -> str:
    try:
        _ = datetime.datetime.strptime(value, "%Y%m%d")
    except ValueError:
        click.echo(f"Invalid date: {value!r}, must be YYYYMMDD")
    else:
        return value


@cli.command()
@click.option("-s", "--satnum", type=satnum)
@click.option("-d", "--date", type=date)
def find(satnum: Optional[int], date: Optional[str]) -> None:
    settings = Settings()
    if satnum and date:
        msg = "Provide either an object number (satnum) or a date, not both"
        raise ValueError(msg)
    elif satnum:
        file = settings.object / f"{satnum}{settings.suffix}"
    elif date:
        file = settings.daily / f"{date}{settings.suffix}"
    file = file.expanduser().absolute()
    click.echo(str(file), color="blue")


def _ask_overwrite() -> bool:
    while True:
        overwrite = click.prompt(
            "Overwrite existing config?",
            type=click.Choice("yn"),
            default="n",
        )
        if overwrite.strip().lower() == "y":
            return True
        elif overwrite.strip().lower() == "n":
            return False


@cli.command()
def init():
    if user_config_file.is_file() and not _ask_overwrite():
        return

    click.echo(f"Creating {user_config_file}")
    user_config_file.touch()

    archive: pathlib.Path = click.prompt(
        "Archive root path",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    )
    archive = archive.absolute().as_posix()
    daily = click.prompt("Daily path", default="day")
    object_ = click.prompt("Object path", default="obj")
    suffix = click.prompt("TLE file suffix", default=".txt")

    text = _config_template.substitute(
        archive=archive, daily=daily, object=object_, suffix=suffix
    )

    with open(user_config_file, "w") as f:
        print(text.strip(), file=f)


if __name__ == "__main__":
    cli()
