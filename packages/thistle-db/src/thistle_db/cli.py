import dataclasses
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
from thistle_db.progress import ProgressReporter, console

app = typer.Typer(
    name="thistle-db",
    help="TLE database management tool",
    no_args_is_help=True,
)

LOG_ROTATION = "10 MB"
LOG_RETENTION = 10


@dataclasses.dataclass
class CliContext:
    config_path: Path
    log_file: Optional[Path]
    no_progress: bool


def _setup_logging(level: str, log_file: Optional[Path] = None) -> None:
    logger.remove()
    # Console lines go through the shared rich console so they render above
    # a live progress bar instead of tearing it. Still stderr underneath.
    logger.add(
        # soft_wrap: never hard-wrap log lines at the console width (matters
        # for piped/redirected stderr, where rich assumes 80 columns).
        lambda message: console.print(
            message, end="", markup=False, highlight=False, soft_wrap=True
        ),
        level=level.upper(),
        colorize=False,
    )
    if log_file is not None:
        # Rotating file sink for cron use: caps disk usage without external
        # logrotate. Rotated files are kept (retention) then deleted.
        logger.add(
            str(log_file),
            level=level.upper(),
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
        )


def _progress(ctx_obj: CliContext) -> ProgressReporter:
    """Progress bar only when interactive: an explicit --no-progress wins,
    and non-TTY stderr (cron, redirects) auto-disables it."""
    return ProgressReporter(enabled=not ctx_obj.no_progress and console.is_terminal)


def _open_session_or_exit(config, *, readonly: bool = False):
    try:
        return open_session(config, readonly=readonly)
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

[ingest]
# Records that cannot be stored (unparseable text, elsets whose mandatory
# elements are not finite numbers, rows the database refuses) are copied
# here in their source format for later inspection, mirroring the source
# tree: ./incoming/20260824.tle -> ./rejects/incoming/20260824.tle (+ .log).
# Artifacts are overwritten each run and removed once a file ingests
# cleanly, so the directory always shows only what is currently broken and
# needs no retention policy. Comment out to drop rejects after logging.
reject_dir = "./rejects"

# Reject element sets whose epoch is more than this many days in the future
# (0 disables). A corrupted epoch field parses cleanly into 2000-2056 and,
# once stored, permanently degrades that object's incremental output. Real
# feeds run at most ~1 day ahead; raise this if you ingest predicted
# element sets. Past epochs are never bounded (archives ingest untouched).
# max_epoch_ahead_days = 30

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
# Generation is incremental: date files are rewritten when their rows are
# newer than the file, and new rows are appended to object files. Late
# deliveries are handled automatically.
lookback_days = 7    # object files: newly created rows considered per run
                     # (must exceed the ingest cron cadence)
# write_workers = 0  # threads used for output file writes; 0 = auto
                     # (CPU count x 4, capped), 1 = write serially

# Each [[output.files]] entry defines one generated output; `generate`
# requires at least one entry:
#   type   = "date"    -> one file per date (latest TLE per satellite)
#          = "object"  -> one file per satellite (all TLEs, epoch order)
#   format = "tle"     -> two-line element text
#          = "omm"     -> OMM CSV
#   dir    = destination directory (created if missing)
# Filename options:
#   object files: object_id = "int" (default) or "alpha5" (always 5 chars,
#                 e.g. E5693); zero_pad = true pads int IDs to 5 digits
#   date files:   date_format = strftime stem pattern (default "%Y%m%d")
#   any:          extension = override the ".tle"/".omm" default suffix
#                 filename = template arranging the pieces, default
#                   "{id}{ext}" / "{date}{ext}" — e.g. "tle_{id}.txt".
#                   Placeholders: {id} or {date}, {ext}, {format}, {type}.
# Entries may share a directory or use separate ones.

[[output.files]]
type = "date"
format = "tle"
dir = "./output"

[[output.files]]
type = "date"
format = "omm"
dir = "./output"

[[output.files]]
type = "object"
format = "tle"
dir = "./output"
# object_id = "alpha5"   # name files E5693.tle instead of 145693.tle
# zero_pad = true        # 00900.tle instead of 900.tle (with object_id = "int")

[[output.files]]
type = "object"
format = "omm"
dir = "./output"

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
    log: Annotated[
        Optional[Path],
        typer.Option(
            "--log",
            help="Also write logs to this file, rotated at 10 MB with the "
            "last 10 rotations kept (for cron use)",
        ),
    ] = None,
    no_progress: Annotated[
        bool,
        typer.Option(
            "--no-progress",
            help="Disable the interactive progress bar (it is also disabled "
            "automatically when stderr is not a terminal)",
        ),
    ] = False,
) -> None:
    ctx.obj = CliContext(config_path=config, log_file=log, no_progress=no_progress)


@app.command()
def init(ctx: typer.Context) -> None:
    """Scaffold config.toml and ~/.config/thistle-db.toml."""
    config_path: Path = ctx.obj.config_path
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
        # Credentials file: owner-only. No-op on Windows (chmod only
        # toggles the read-only bit there).
        secrets_path.chmod(0o600)
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
    defer_indexes: Annotated[
        bool,
        typer.Option(
            "--defer-indexes",
            help="With --drop: create the schema without the read-path "
            "indexes, so a bulk restore ingests without maintaining them "
            "row by row. Run `init-db` again after ingesting to build "
            "them (one fast sorted pass each). Dedup indexes are never "
            "deferred.",
        ),
    ] = False,
) -> None:
    """Create the database schema in the configured database.

    Idempotent: existing tables are left untouched and any missing model
    index is built (the finalize step after a --defer-indexes restore).
    With --drop, all tables are dropped and recreated (asks for
    confirmation unless --yes).
    """
    config = load_config(ctx.obj.config_path)
    _setup_logging(config.logging.level, ctx.obj.log_file)
    url = config.database.url.render_as_string(hide_password=True)

    if defer_indexes and not drop:
        print(
            "Error: --defer-indexes requires --drop (deferring is only "
            "meaningful when the tables start empty).",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

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

    result = init_db(config, drop=drop, defer_indexes=defer_indexes)

    print(f"Database: {url}")
    if result["dropped"]:
        print(f"Dropped:  {', '.join(result['dropped'])}")
    if result["created"]:
        print(f"Created:  {', '.join(result['created'])}")
    if result["existing"]:
        print(f"Existing: {', '.join(result['existing'])} (left untouched)")
    if result["indexes_created"]:
        print(f"Indexes built: {', '.join(result['indexes_created'])}")
    if result["indexes_deferred"]:
        print(f"Indexes deferred: {', '.join(result['indexes_deferred'])}")
        print(
            "Ingest your files, then run `thistle-db init-db` (no flags) "
            "to build the deferred indexes."
        )


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
    config = load_config(ctx.obj.config_path)
    _setup_logging(config.logging.level, ctx.obj.log_file)

    reject_dir = (
        Path(config.ingest.reject_dir) if config.ingest.reject_dir else None
    )

    session, engine = _open_session_or_exit(config)
    try:
        with _progress(ctx.obj) as progress:
            if files:
                task = progress.task("Ingesting files", total=len(files))
                total = 0
                failed = 0
                for file in files:
                    # Explicitly named files always parse; state is still recorded.
                    # No source root, so their rejects land flat under reject_dir.
                    status, count = ingest_source_file(
                        session,
                        file,
                        force=True,
                        reject_dir=reject_dir,
                        max_epoch_ahead_days=config.ingest.max_epoch_ahead_days,
                    )
                    if status == FileStatus.FAILED:
                        failed += 1
                    total += count
                    progress.advance(task)
                logger.info(
                    f"Ingested {total} new records from {len(files)} files"
                    + (f" ({failed} failed)" if failed else "")
                )
                if failed:
                    # Explicitly named files failing is an error the caller
                    # (cron included) must be able to detect. Directory
                    # scans stay tolerant: failed files there are retried
                    # on the next scan by design.
                    raise typer.Exit(code=1)
            else:
                total = ingest_sources(
                    session,
                    config.ingest.sources,
                    force=force,
                    reject_dir=reject_dir,
                    max_epoch_ahead_days=config.ingest.max_epoch_ahead_days,
                    progress=progress,
                )
                logger.info(f"Ingested {total} new records from configured sources")
    finally:
        session.close()
        engine.dispose()


@app.command()
def generate(
    ctx: typer.Context,
    rebuild_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Rebuild every output file from scratch (first run, disaster "
            "recovery, or after a generation gap longer than the lookback)",
        ),
    ] = False,
    lookback_days: Annotated[
        Optional[int],
        typer.Option(
            "--lookback-days",
            help="Override output.lookback_days: how many days of newly created "
            "rows to (re)consider for object files",
        ),
    ] = None,
    verify: Annotated[
        bool,
        typer.Option(
            "--verify",
            help="After generating, reconcile every object file's line count "
            "against the database and rewrite mismatches (reads all output "
            "files — intended for a periodic, e.g. weekly, cron run)",
        ),
    ] = False,
) -> None:
    """Generate output TLE/OMM files from the database.

    Incremental by default: date files are rewritten when the database has
    rows newer than the file, and object files are appended to. Cost scales
    with new data, not database size.
    """
    config = load_config(ctx.obj.config_path)
    _setup_logging(config.logging.level, ctx.obj.log_file)

    if not config.output.files:
        print(
            "Error: no outputs configured — add at least one "
            "[[output.files]] entry to the config",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    session, engine = _open_session_or_exit(config, readonly=True)
    try:
        with _progress(ctx.obj) as progress:
            generate_outputs(
                session,
                config.output,
                rebuild_all=rebuild_all,
                lookback_days=lookback_days,
                verify=verify,
                progress=progress,
            )
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
    config = load_config(ctx.obj.config_path)
    _setup_logging(config.logging.level, ctx.obj.log_file)

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

    session, engine = _open_session_or_exit(config, readonly=True)
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

    config = load_config(ctx.obj.config_path)
    _setup_logging(config.logging.level, ctx.obj.log_file)

    session, engine = _open_session_or_exit(config, readonly=True)
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
