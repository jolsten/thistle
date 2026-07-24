import datetime
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from sgp4.alpha5 import from_alpha5

from thistle_db.api import (
    DatabaseNotInitializedError,
    dump_db,
    init_db,
    nearest_tles_for_date,
    open_session,
    tles_for_object,
)
from thistle_db.config import load_config
from thistle_db.generator import generate as generate_outputs
from thistle_db.ingest import FileStatus, ingest_source_file, ingest_sources

app = typer.Typer(
    name="thistle-db",
    help="TLE database management tool",
    no_args_is_help=True,
)


def _setup_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def _open_session_or_exit(config):
    try:
        return open_session(config)
    except DatabaseNotInitializedError as err:
        print(f"Error: {err}", file=sys.stderr)
        raise typer.Exit(code=3) from None


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
            envvar="THISTLE_DB_CONFIG",
            help="Path to config.toml (default: $THISTLE_DB_CONFIG or ./config.toml)",
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


@app.command(name="init-db")
def init_db_cmd(
    ctx: typer.Context,
    drop: Annotated[
        bool,
        typer.Option(
            "--drop",
            help="Drop all tables first, destroying all data (DESTRUCTIVE)",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the --drop confirmation prompt (for scripts)",
        ),
    ] = False,
) -> None:
    """Create the database schema in the configured database.

    Idempotent: existing tables are left untouched. With --drop, all tables
    are dropped and recreated (asks for confirmation unless --yes).
    """
    config = load_config(ctx.obj)
    _setup_logging(config.logging.level)
    url = config.database.url.render_as_string(hide_password=True)

    if drop and not yes:
        if not sys.stdin.isatty():
            print(
                "Error: --drop requires confirmation; pass --yes to skip the "
                "prompt in non-interactive use.",
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
        if not typer.confirm(f"Drop ALL tables in {url}?"):
            raise typer.Exit(code=1)

    result = init_db(config, drop=drop)

    print(f"Database: {url}")
    if result["dropped"]:
        print(f"Dropped:  {', '.join(result['dropped'])}")
    if result["created"]:
        print(f"Created:  {', '.join(result['created'])}")
    if result["existing"]:
        print(f"Existing: {', '.join(result['existing'])} (left untouched)")


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

    session, engine = _open_session_or_exit(config)
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

    session, engine = _open_session_or_exit(config)
    try:
        generate_outputs(session, config.output)
    finally:
        session.close()
        engine.dispose()


@app.command()
def dump(
    ctx: typer.Context,
    output: Annotated[
        Path,
        typer.Argument(
            help="Base path for the export; writes OUTPUT.tle and (when OMM "
            "metadata exists) OUTPUT.json",
            show_default=False,
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing output files"),
    ] = False,
) -> None:
    """Export the entire database as re-ingestable TLE/OMM files.

    A logical backup: restore into any (empty, init-db'd) database with
    `thistle-db ingest OUTPUT.tle OUTPUT.json` — also works across dialects
    (e.g. SQLite to MariaDB). For physical backups of a live server, prefer
    the native tools (sqlite file copy / mariadb-dump / pg_dump).
    """
    config = load_config(ctx.obj)
    _setup_logging(config.logging.level)

    tle_path = Path(f"{output}.tle")
    omm_path = Path(f"{output}.json")
    existing = [p for p in (tle_path, omm_path) if p.exists()]
    if existing and not force:
        print(
            f"Error: output file(s) already exist: "
            f"{', '.join(str(p) for p in existing)} (use --force to overwrite)",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    session, engine = _open_session_or_exit(config)
    try:
        tle_count, omm_count = dump_db(session, tle_path, omm_path)
    finally:
        session.close()
        engine.dispose()

    print(f"Wrote {tle_path} ({tle_count} TLEs)")
    if omm_count:
        print(f"Wrote {omm_path} ({omm_count} OMM records)")
    else:
        print(f"No OMM metadata in the database; {omm_path} not written")


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

    session, engine = _open_session_or_exit(config)
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
