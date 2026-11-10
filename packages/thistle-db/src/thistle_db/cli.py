import dataclasses
import datetime
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from pydantic import ValidationError
from sgp4.alpha5 import from_alpha5

from thistle_db.api import (
    DatabaseNotInitializedError,
    init_db,
    nearest_tles_for_date,
    open_session,
    tles_for_object,
)
from thistle_db.config import Settings, load_config
from thistle_db.export import EXPORT_FORMATS, export as export_store
from thistle_db.ingest import (
    DeltaEngine,
    ingest_paths,
    ingest_sources,
    readmit_matured_rejects,
)
from thistle_db.lock import LockHeldError, SingleWriterLock, lock_path_for
from thistle_db.outputs import rebuild_outputs
from thistle_db.progress import ProgressReporter, console
from thistle_db.rejects import export_unresolved, unresolved_summary
from thistle_db.runs import recent_runs, run_context, stuck_runs
from thistle_db.verify import verify_outputs

app = typer.Typer(
    name="thistle-db",
    help="Orbital element database manager",
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


def _load_config_or_exit(ctx_obj: CliContext) -> Settings:
    try:
        return load_config(ctx_obj.config_path)
    except ValidationError as err:
        print(f"Error: invalid configuration: {err}", file=sys.stderr)
        raise typer.Exit(code=2) from None


def _open_session_or_exit(config, *, readonly: bool = False):
    try:
        return open_session(config, readonly=readonly)
    except DatabaseNotInitializedError as err:
        print(f"Error: {err}", file=sys.stderr)
        raise typer.Exit(code=3) from None


def _require_outputs(config: Settings) -> None:
    if not config.output.files:
        print(
            "Error: no outputs configured — add at least one "
            "[[output.files]] entry to the config",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)


def _writer_lock(ctx_obj: CliContext) -> SingleWriterLock:
    return SingleWriterLock(lock_path_for(ctx_obj.config_path))


CONFIG_TEMPLATE = """\
# thistle-db configuration
# See: https://github.com/jolsten/thistle

# ----------------------------------------------
# Database connection — the canonical store
# ----------------------------------------------
[database]
# SQLAlchemy driver string. Examples:
#   "sqlite"          - local SQLite file (default; serverless)
#   "mysql+pymysql"   - MariaDB/MySQL via PyMySQL (pip install thistle-db[mysql])
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
# Ingest
# ----------------------------------------------
# Each [[ingest.sources]] entry defines a directory to scan for TLE/OMM files.
# File format is auto-detected by extension:
#   .tle / .txt / .3le  ->  Two-Line Element format
#   .json               ->  Space-Track OMM JSON
#   .csv                ->  OMM CSV
#   .xml                ->  OMM XML
#
# Records that cannot be stored are retained in the database `rejects`
# table with a reason code — inspect them with `thistle-db rejects`.

[ingest]
# Reject element sets whose epoch is more than this many days in the future
# (0 disables). A corrupted epoch field parses cleanly into 2000-2056 and,
# once stored, permanently degrades that object's incremental output. Real
# feeds run at most ~1 day ahead; raise this if you ingest predicted
# element sets. Past epochs are never bounded (archives ingest untouched).
# Rejected records re-admit automatically once inside the horizon.
# max_epoch_ahead_days = 30

# Reject TLE lines whose mod-10 checksum (column 69) does not match — the
# only corruption signal for records whose elements still parse. Lines
# shorter than 69 characters skip the check.
# verify_checksums = true

# Buffered records that trigger an early delta flush (memory budget,
# ~600 bytes per record; never a correctness knob).
# flush_records = 1000000

[[ingest.sources]]
path = "./incoming"
pattern = "*.tle"    # glob pattern for matching files

# Add more sources as needed:
# [[ingest.sources]]
# path = "/data/spacetrack/daily"
# pattern = "*.json"

# ----------------------------------------------
# Outputs — maintained incrementally by ingest
# ----------------------------------------------
# Each [[output.files]] entry defines one output; `ingest` requires at
# least one entry and maintains all of them in the same pass (there is no
# separate generate step — `rebuild` re-derives everything after a config
# change or for disaster recovery):
#   type   = "date"     -> one file per date (latest record per satellite)
#          = "object"   -> one file per satellite (all records, epoch order)
#   format = "tle"      -> verbatim two-line element text
#          = "omm-csv"  -> STK-compatible OMM CSV
#   dir    = destination directory (created if missing)
# Options:
#   omm-csv:      columns = "stk" (17-column STK projection, default)
#                 or "full" (every store column)
#   object files: object_id = "int" (default) or "alpha5" (always 5 chars,
#                 e.g. E5693); zero_pad = true pads int IDs to 5 digits
#   date files:   date_format = strftime stem pattern (default "%Y%m%d")
#   any:          extension = override the ".tle"/".csv" default suffix
#                 filename = template arranging the pieces, default
#                   "{id}{ext}" / "{date}{ext}" — e.g. "tle_{id}.txt".
#                   Placeholders: {id} or {date}, {ext}, {format}, {type}.

[[output.files]]
type = "date"
format = "tle"
dir = "./output"

[[output.files]]
type = "object"
format = "tle"
dir = "./output"
# object_id = "alpha5"   # name files E5693.tle instead of 145693.tle
# zero_pad = true        # 00900.tle instead of 900.tle (with object_id = "int")

[output]
# write_workers = 0  # threads for output file writes; 0 = auto
                     # (CPU count x 4, capped), 1 = write serially

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

    The only DDL path — on MariaDB/PostgreSQL, run it with an admin
    account and give the day-to-day account only DML grants. Idempotent:
    existing tables are left untouched and any missing model index is
    built (the finalize step after a --defer-indexes restore). With
    --drop, all tables are dropped and recreated (asks for confirmation
    unless --yes).
    """
    config = _load_config_or_exit(ctx.obj)
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
    """Ingest TLE/OMM files: validate, store, maintain every output.

    One pass does everything — there is no separate generate step. Scans
    are tolerant (exit 0; failed files retried next scan); explicitly
    named files exit 1 on any failure so cron wrappers can detect it.
    """
    config = _load_config_or_exit(ctx.obj)
    _setup_logging(config.logging.level, ctx.obj.log_file)
    _require_outputs(config)

    command = f"ingest {len(files)} file(s)" if files else "ingest"
    try:
        with _writer_lock(ctx.obj):
            session, engine_db = _open_session_or_exit(config)
            try:
                with _progress(ctx.obj) as progress:
                    with run_context(session, command) as (run, report):
                        with DeltaEngine(
                            session, config, report, run_id=run.id
                        ) as engine:
                            readmit_matured_rejects(session, engine)
                            if files:
                                ingest_paths(
                                    session, engine, files, progress=progress
                                )
                            else:
                                ingest_sources(
                                    session,
                                    engine,
                                    config.ingest.sources,
                                    force=force,
                                    progress=progress,
                                )
                logger.info(report.summary())
            finally:
                session.close()
                engine_db.dispose()
    except LockHeldError as err:
        print(f"Error: {err}", file=sys.stderr)
        raise typer.Exit(code=1) from None

    if files and report.files_failed:
        # Explicitly named files failing is an error the caller (cron
        # included) must be able to detect. Directory scans stay tolerant:
        # failed files there are retried on the next scan by design.
        raise typer.Exit(code=1)


@app.command()
def rebuild(ctx: typer.Context) -> None:
    """Re-derive every output file from a linear store scan.

    First run, disaster recovery, or after adding/changing an
    [[output.files]] entry.
    """
    config = _load_config_or_exit(ctx.obj)
    _setup_logging(config.logging.level, ctx.obj.log_file)
    _require_outputs(config)

    try:
        with _writer_lock(ctx.obj):
            session, engine_db = _open_session_or_exit(config)
            try:
                with _progress(ctx.obj) as progress:
                    with run_context(session, "rebuild") as (run, report):
                        written = rebuild_outputs(
                            session, config.output, progress=progress
                        )
                        report.output_writes.update(written)
                total = sum(written.values())
                logger.info(
                    f"Rebuilt {total} file(s) across {len(config.output.files)} "
                    "output(s)"
                )
            finally:
                session.close()
                engine_db.dispose()
    except LockHeldError as err:
        print(f"Error: {err}", file=sys.stderr)
        raise typer.Exit(code=1) from None


@app.command()
def verify(ctx: typer.Context) -> None:
    """Reconcile every output against the store (the authority).

    Per-object record counts, per-date winner counts, orphan files; also
    stamps rejects whose payload now exists in the store. Reads all
    output files — intended for a periodic (e.g. weekly) cron run, not
    every run. Exits 1 when mismatches are found.
    """
    config = _load_config_or_exit(ctx.obj)
    _setup_logging(config.logging.level, ctx.obj.log_file)
    _require_outputs(config)

    session, engine_db = _open_session_or_exit(config)
    try:
        report = verify_outputs(session, config.output)
    finally:
        session.close()
        engine_db.dispose()

    if report.rejects_resolved:
        print(f"Stamped {report.rejects_resolved} reject(s) as resolved")
    if not report.ok:
        print(
            f"verify: {len(report.issues)} issue(s) found — see warnings "
            "above; run `thistle-db rebuild` to re-derive outputs",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)
    print(f"verify: {report.outputs_checked} output(s) consistent with the store")


@app.command()
def export(
    ctx: typer.Context,
    output: Annotated[
        Path,
        typer.Argument(help="Destination file path", show_default=False),
    ],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help=f"Export format: {', '.join(EXPORT_FORMATS)}",
        ),
    ] = "csv",
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing output file"),
    ] = False,
) -> None:
    """Export the whole store to one file.

    csv is the full-fidelity backup/interchange form (re-ingestable into
    any dialect: `init-db` then `ingest FILE`); tle is a verbatim two-line
    concat for legacy consumers; omm-xml/omm-kvn are CCSDS OMM 2.0. For
    physical backups of a live server, prefer the native tools.
    """
    if fmt not in EXPORT_FORMATS:
        print(
            f"Error: unknown format {fmt!r} (choose from "
            f"{', '.join(EXPORT_FORMATS)})",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    config = _load_config_or_exit(ctx.obj)
    _setup_logging(config.logging.level, ctx.obj.log_file)

    if output.exists() and not force:
        print(
            f"Error: {output} already exists (use --force to overwrite)",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    session, engine_db = _open_session_or_exit(config, readonly=True)
    try:
        count = export_store(session, fmt, output)
    finally:
        session.close()
        engine_db.dispose()
    print(f"Wrote {output} ({count} records, format: {fmt})")


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
    date, within +/- DAYS days. Always exact — it queries the store.
    """
    parsed = _parse_target(target)

    config = _load_config_or_exit(ctx.obj)
    _setup_logging(config.logging.level, ctx.obj.log_file)

    session, engine_db = _open_session_or_exit(config, readonly=True)
    try:
        if isinstance(parsed, int):
            tles = tles_for_object(session, parsed)
        else:
            tles = nearest_tles_for_date(session, parsed, days)

        printed = 0
        for tle in tles:
            if tle.line1 and tle.line2:
                print(tle.line1)
                print(tle.line2)
                printed += 1
    finally:
        session.close()
        engine_db.dispose()

    if not printed:
        logger.warning(f"No TLEs found for {target}")
        raise typer.Exit(code=1)


@app.command()
def status(
    ctx: typer.Context,
    runs: Annotated[
        int,
        typer.Option("--runs", "-n", help="How many recent runs to show"),
    ] = 10,
) -> None:
    """Show recent runs, reject trends, and stuck runs."""
    config = _load_config_or_exit(ctx.obj)
    _setup_logging(config.logging.level, ctx.obj.log_file)

    session, engine_db = _open_session_or_exit(config, readonly=True)
    try:
        run_rows = recent_runs(session, limit=runs)
        stuck = stuck_runs(session)
        reject_rows = unresolved_summary(session)
    finally:
        session.close()
        engine_db.dispose()

    if not run_rows:
        print("No runs recorded yet.")
    else:
        print("Recent runs:")
        for run in run_rows:
            if run.finished is not None:
                duration = (run.finished - run.started).total_seconds()
                timing = f"{duration:8.1f}s"
            else:
                timing = "  running"
            started = run.started.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"  #{run.id:<5} {started}  {timing}  {run.status:<9} "
                f"{run.command:<20} "
                f"+{run.records_inserted} rows, {run.records_rejected} rejected"
                + (f"  [{run.error}]" if run.error else "")
            )

    if stuck:
        print(
            f"\nWARNING: {len(stuck)} run(s) stuck in 'running' "
            "(crashed? the advisory lock has already been released):"
        )
        for run in stuck:
            print(f"  #{run.id} {run.command} started {run.started}")

    if reject_rows:
        print("\nUnresolved rejects by reason:")
        for code, count, last_seen in reject_rows:
            print(f"  {code:<20} {count:>8}  (last seen {last_seen})")
    else:
        print("\nNo unresolved rejects.")


@app.command()
def rejects(
    ctx: typer.Context,
    export_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--export",
            help="Write unresolved payloads to this directory as "
            "re-ingestable files (rejects.tle / rejects.json)",
        ),
    ] = None,
) -> None:
    """Summarize the rejects table; optionally export payloads for repair.

    The repair workflow: `rejects --export DIR`, fix the records, then
    `ingest DIR/rejects.tle` — `verify` stamps the repaired ones resolved.
    """
    config = _load_config_or_exit(ctx.obj)
    _setup_logging(config.logging.level, ctx.obj.log_file)

    session, engine_db = _open_session_or_exit(config, readonly=True)
    try:
        rows = unresolved_summary(session)
        if not rows:
            print("No unresolved rejects.")
        else:
            print("Unresolved rejects by reason:")
            for code, count, last_seen in rows:
                print(f"  {code:<20} {count:>8}  (last seen {last_seen})")
        if export_dir is not None:
            tle_count, omm_count = export_unresolved(session, export_dir)
            print(
                f"Exported {tle_count} TLE and {omm_count} OMM payload(s) "
                f"to {export_dir}"
            )
    finally:
        session.close()
        engine_db.dispose()


def main() -> None:
    app()
