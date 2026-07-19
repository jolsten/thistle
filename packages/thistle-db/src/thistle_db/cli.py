import datetime
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from sgp4.alpha5 import from_alpha5
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from thistle_db.config import load_config
from thistle_db.generator import generate as generate_outputs
from thistle_db.ingest import FileStatus, ingest_source_file, ingest_sources
from thistle_db.model import TLE, Base

app = typer.Typer(
    name="thistle-db",
    help="TLE database management tool",
    no_args_is_help=True,
)


def _setup_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def _get_session(config):
    engine = config.database.engine
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


CONFIG_TEMPLATE = """\
# thistle-db configuration
# See: https://github.com/jolsten/thistle

# ----------------------------------------------
# Database connection
# ----------------------------------------------
[database]
# SQLAlchemy driver string. Examples:
#   "sqlite"          - local SQLite file (default)
#   "mysql+pymysql"   - MariaDB/MySQL via PyMySQL (install with: pip install thistle-db[mysql])
drivername = "sqlite"

# For SQLite: path to the database file (relative or absolute)
# For MariaDB/MySQL: the database name
name = "thistle-db.db"

# Uncomment for MariaDB/MySQL:
# host = "localhost"
# port = 3306
#
# Credentials are loaded separately (never put passwords here).
# Resolution order (highest priority first):
#   1. Environment variables:  THISTLE_DB_DATABASE__USERNAME / THISTLE_DB_DATABASE__PASSWORD
#   2. User secrets file:      ~/.config/thistle-db.toml
#   3. System secrets file:    set secrets_file below
#
# secrets_file = "/etc/thistle-db/secrets.toml"

# ----------------------------------------------
# Ingest sources
# ----------------------------------------------
# Each [[ingest.sources]] entry defines a directory to scan for TLE/OMM files.
# You can have multiple entries. File format is auto-detected by extension:
#   .tle / .txt / .3le  ->  Two-Line Element format
#   .json               ->  Space-Track OMM JSON
#   .csv                ->  OMM CSV
#   .xml                ->  OMM XML

[[ingest.sources]]
path = "./incoming"
pattern = "*.tle"    # glob pattern for matching files

# Add more sources as needed:
# [[ingest.sources]]
# path = "/data/spacetrack/daily"
# pattern = "*.json"

# ----------------------------------------------
# Output generation
# ----------------------------------------------
[output]
dir = "./output"     # directory where generated files are written

# Which output formats to produce
[output.formats]
tle = true           # two-line element format (.tle files)
omm = true           # OMM CSV format (.omm files)

# Which output file types to generate
[output.types]
date_files = true    # one file per date: YYYYMMDD.{tle,omm}
                     # contains the latest TLE per satellite for that date
object_files = true  # one file per satellite: NORAD_ID.{tle,omm}
                     # contains all TLEs for that satellite, ordered by epoch

# ----------------------------------------------
# Logging
# ----------------------------------------------
[logging]
level = "INFO"       # DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

SECRETS_TEMPLATE = """\
# thistle-db user secrets
# This file stores database credentials for your user account.
# Keep this file private (chmod 600 on Linux/macOS).
#
# These values are overridden by environment variables:
#   THISTLE_DB_DATABASE__USERNAME
#   THISTLE_DB_DATABASE__PASSWORD

username = ""
password = ""
"""


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            help="Path to config.toml (default: ./config.toml)",
        ),
    ] = Path("config.toml"),
) -> None:
    ctx.obj = config


@app.command()
def init(ctx: typer.Context) -> None:
    """Scaffold config.toml and ~/.config/thistle-db.toml."""
    config_path: Path = ctx.obj
    secrets_path = Path.home() / ".config" / "thistle-db.toml"

    created = []

    if config_path.exists():
        print(f"Config already exists: {config_path}")
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(CONFIG_TEMPLATE)
        created.append(str(config_path))

    if secrets_path.exists():
        print(f"Secrets file already exists: {secrets_path}")
    else:
        secrets_path.parent.mkdir(parents=True, exist_ok=True)
        secrets_path.write_text(SECRETS_TEMPLATE)
        created.append(str(secrets_path))

    if created:
        print("Created:")
        for path in created:
            print(f"  {path}")
        print("\nEdit these files to configure your database and credentials.")
    else:
        print("Nothing to do - all files already exist.")


@app.command()
def ingest(
    ctx: typer.Context,
    files: Annotated[
        Optional[list[Path]],
        typer.Argument(
            help="Specific files to ingest (if omitted, scans configured source dirs)"
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-ingest files even if their recorded state is unchanged",
        ),
    ] = False,
) -> None:
    """Ingest TLE/OMM files into the database."""
    config = load_config(ctx.obj)
    _setup_logging(config.logging.level)

    session, engine = _get_session(config)
    try:
        if files:
            total = 0
            failed = 0
            for file in files:
                # Explicitly named files always parse; state is still recorded.
                status, count = ingest_source_file(session, file, force=True)
                if status == FileStatus.FAILED:
                    failed += 1
                total += count
            logger.info(
                f"Ingested {total} new records from {len(files)} files"
                + (f" ({failed} failed)" if failed else "")
            )
        else:
            total = ingest_sources(session, config.ingest.sources, force=force)
            logger.info(f"Ingested {total} new records from configured sources")
    finally:
        session.close()
        engine.dispose()


@app.command()
def generate(ctx: typer.Context) -> None:
    """Generate output TLE/OMM files from the database."""
    config = load_config(ctx.obj)
    _setup_logging(config.logging.level)

    session, engine = _get_session(config)
    try:
        generate_outputs(session, config.output)
    finally:
        session.close()
        engine.dispose()


def _parse_target(value: str) -> int | datetime.date:
    """Parse a get-tle target: 8-digit YYYYMMDD -> date, else alpha-5 NORAD ID."""
    if len(value) == 8 and value.isdigit():
        try:
            return datetime.datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            raise typer.BadParameter(
                f"{value!r} is not a valid YYYYMMDD date"
            ) from None
    try:
        return from_alpha5(value.upper())
    except ValueError:
        raise typer.BadParameter(
            f"{value!r} is neither an alpha-5 NORAD ID nor a YYYYMMDD date"
        ) from None


def tles_for_object(session, satnum: int) -> list[TLE]:
    """All TLEs for one satellite, ordered by epoch."""
    stmt = select(TLE).where(TLE.norad_cat_id == satnum).order_by(TLE.epoch)
    return list(session.execute(stmt).scalars().all())


def nearest_tles_for_date(
    session, date: datetime.date, days: float
) -> list[TLE]:
    """Nearest TLE per satellite to 12:00 UTC on `date`, within +/- `days`."""
    center = datetime.datetime.combine(date, datetime.time(12))
    window = datetime.timedelta(days=days)
    stmt = select(TLE).where(
        TLE.epoch >= center - window,
        TLE.epoch <= center + window,
    )

    nearest: dict[int, TLE] = {}
    for tle in session.execute(stmt).scalars():
        if tle.norad_cat_id is None:
            continue
        best = nearest.get(tle.norad_cat_id)
        if best is None or abs(tle.epoch - center) < abs(best.epoch - center):
            nearest[tle.norad_cat_id] = tle
    return [nearest[satnum] for satnum in sorted(nearest)]


@app.command(name="get-tle")
def get_tle(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(
            help="NORAD ID (alpha-5, e.g. 25544 or E5693) or date (YYYYMMDD)",
            show_default=False,
        ),
    ],
    days: Annotated[
        float,
        typer.Option(
            "--days",
            "-d",
            help="Date mode: search window around the date, in days",
        ),
    ] = 7.0,
) -> None:
    """Print TLEs to stdout.

    With a NORAD ID (alpha-5): all TLEs for that object, ordered by epoch.
    With a YYYYMMDD date: the nearest TLE per object to 12:00 UTC on that
    date, within +/- DAYS days.
    """
    parsed = _parse_target(target)

    config = load_config(ctx.obj)
    _setup_logging(config.logging.level)

    session, engine = _get_session(config)
    try:
        if isinstance(parsed, int):
            tles = tles_for_object(session, parsed)
        else:
            tles = nearest_tles_for_date(session, parsed, days)

        for tle in tles:
            print(tle.line1)
            print(tle.line2)
    finally:
        session.close()
        engine.dispose()

    if not tles:
        logger.warning(f"No TLEs found for {target}")
        raise typer.Exit(code=1)


def main() -> None:
    app()
