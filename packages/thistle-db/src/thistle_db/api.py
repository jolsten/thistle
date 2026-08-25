"""Programmatic query API for the thistle-db database."""

import datetime
import json
import pathlib

from sqlalchemy import Index, delete, func, insert, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from thistle_db.config import Settings, load_config
from thistle_db.model import TLE, Base, EpochDateState, date_from_sql


# Read-path indexes a bulk rebuild may defer. Row-by-row maintenance of
# these is what makes restoring a large catalog slow on MariaDB: modern
# MariaDB has no InnoDB change buffer (removed in 11), so every secondary-
# index insert touches a real page, and a full-catalog snapshot file
# scatters ~one page access per row across the whole (norad_cat_id, epoch)
# index — random I/O once the index outgrows the buffer pool. Building them
# once at the end is a sorted bulk build instead. Never deferrable: the
# unique dedup indexes (uq_tle_epoch_line_hash, omm tle_id, ingest_files
# path_hash), which ingest correctness depends on.
DEFERRABLE_INDEXES = ("ix_tle_norad_cat_id_epoch", "ix_tle_created")


def _model_indexes() -> dict[str, list[Index]]:
    """Model-declared (non-constraint) indexes, keyed by table name."""
    return {
        table.name: sorted(table.indexes, key=lambda ix: ix.name or "")
        for table in Base.metadata.sorted_tables
    }


def init_db(
    config: Settings, drop: bool = False, defer_indexes: bool = False
) -> dict[str, list[str]]:
    """Create the database schema, optionally dropping it first.

    The only intended DDL entry point. Idempotent when ``drop`` is False:
    existing tables are left untouched, and any *missing* model index is
    created — which makes a plain ``init_db`` the finalize step after a
    deferred-index rebuild.

    With ``drop=True``, all model tables are dropped first — destructive,
    callers must confirm with the user. ``defer_indexes=True`` (only valid
    with ``drop``) creates the schema without the ``DEFERRABLE_INDEXES``,
    for bulk restores: ingest everything, then run ``init_db`` again to
    build them in one pass each.

    Returns names as ``{"dropped": [...], "created": [...],
    "existing": [...], "indexes_created": [...], "indexes_deferred": [...]}``.
    """
    if defer_indexes and not drop:
        raise ValueError(
            "defer_indexes requires drop=True: deferring is only meaningful "
            "when the tables start empty"
        )
    engine = config.database.engine
    try:
        model_tables = set(Base.metadata.tables)
        before = set(inspect(engine).get_table_names()) & model_tables
        dropped: list[str] = []
        if drop:
            Base.metadata.drop_all(engine)
            dropped = sorted(before)
            before = set()
        Base.metadata.create_all(engine)
        after = set(inspect(engine).get_table_names()) & model_tables

        indexes_created: list[str] = []
        indexes_deferred: list[str] = []
        if defer_indexes:
            # create_all made them along with the (empty) tables; dropping
            # them now is instant, and ingest then maintains only the PK and
            # the unique dedup indexes.
            for table_name, indexes in _model_indexes().items():
                for index in indexes:
                    if index.name in DEFERRABLE_INDEXES:
                        index.drop(engine)
                        indexes_deferred.append(f"{table_name}.{index.name}")
        else:
            # Reconcile: build any model index the database lacks (the
            # finalize step after a deferred rebuild, or repair after a
            # hand-dropped index). A bulk CREATE INDEX is a sorted build —
            # minutes for tens of millions of rows, not hours of row-by-row
            # page churn.
            inspector = inspect(engine)
            for table_name, indexes in _model_indexes().items():
                existing = {ix["name"] for ix in inspector.get_indexes(table_name)}
                for index in indexes:
                    if index.name not in existing:
                        index.create(engine)
                        indexes_created.append(f"{table_name}.{index.name}")

        rebuild_epoch_date_state(engine)
        return {
            "dropped": dropped,
            "created": sorted(after - before),
            "existing": sorted(before),
            "indexes_created": sorted(indexes_created),
            "indexes_deferred": sorted(indexes_deferred),
        }
    finally:
        engine.dispose()


def rebuild_epoch_date_state(engine: Engine) -> int:
    """Recompute `epoch_date_state` from `tle`. Returns the row count.

    The table is derived (`date(epoch) -> MAX(created)`), maintained
    incrementally by ingest in the same transaction as the rows it
    describes. This recomputes it from scratch, which is always correct and
    is needed in three cases:

    - **Adoption.** On a database that predates the table it starts empty,
      and an empty table means no date file looks stale — generation would
      silently write nothing. This backfill is the migration path, so an
      existing deployment upgrades with one statement instead of a
      dump/restore cycle.
    - **Rows inserted outside `ingest`** (a bulk `COPY`/`mysqlimport`, an
      admin script): those dates' watermarks were never bumped, so their
      output files would never regenerate. This is the only realistic drift
      in normal operation.
    - **Bugs** in any future write path that forgets to bump a watermark.

    Crashes are deliberately *not* on that list: the watermark commits with
    its rows, so a partial ingest cannot leave the table behind.

    Runs as part of `init_db`, which is the only DDL path and is already
    idempotent — `generate` could not do this even if it wanted to, since
    the read tier opens read-only connections.
    """
    date_expr = func.date(TLE.epoch)
    with engine.begin() as conn:
        conn.execute(delete(EpochDateState))
        source = (
            select(date_expr, func.max(TLE.created))
            .where(TLE.epoch.is_not(None))
            .group_by(date_expr)
        )
        rows = [
            {"epoch_date": date_from_sql(raw), "last_created": last}
            for raw, last in conn.execute(source)
            if raw is not None
        ]
        if rows:
            conn.execute(insert(EpochDateState), rows)
    return len(rows)


class DatabaseNotInitializedError(RuntimeError):
    """The configured database is missing the thistle-db schema."""


def open_session(config: Settings, *, readonly: bool = False) -> tuple[Session, Engine]:
    """Open a session on the configured database.

    Never issues DDL — schema creation is `init_db`'s job. Raises
    DatabaseNotInitializedError if any model table is missing, so a typo'd
    database path/name fails loudly instead of quietly querying (or, for
    SQLite, creating) an empty database.

    With ``readonly=True`` the connection itself rejects writes (read-tier
    callers: queries, exports, thistle's fallback).
    """
    db = config.database
    # Check SQLite file existence before connecting: connecting alone would
    # create an empty file at the (possibly typo'd) path.
    if db.drivername.startswith("sqlite") and db.name and db.name != ":memory:":
        if not pathlib.Path(db.name).exists():
            raise DatabaseNotInitializedError(
                f"database not initialized: SQLite file {db.name} does not "
                "exist — run `thistle-db init-db`"
            )
    engine = config.database.readonly_engine if readonly else config.database.engine
    try:
        missing = set(Base.metadata.tables) - set(inspect(engine).get_table_names())
    except Exception:
        engine.dispose()
        raise
    if missing:
        engine.dispose()
        url = config.database.url.render_as_string(hide_password=True)
        raise DatabaseNotInitializedError(
            f"database not initialized: {url} is missing table(s) "
            f"{', '.join(sorted(missing))} — run `thistle-db init-db`"
        )
    session_factory = sessionmaker(bind=engine)
    return session_factory(), engine


def tles_for_object(session: Session, satnum: int) -> list[TLE]:
    """All TLEs for one satellite, ordered by epoch."""
    stmt = select(TLE).where(TLE.norad_cat_id == satnum).order_by(TLE.epoch)
    return list(session.execute(stmt).scalars().all())


def _boundary_tles(
    session: Session,
    start: datetime.datetime,
    end: datetime.datetime,
    latest: bool,
) -> list[TLE]:
    """Per object, the latest (or earliest) TLE with epoch in [start, end]."""
    order = (TLE.epoch.desc(), TLE.id.desc()) if latest else (TLE.epoch, TLE.id)
    rn = (
        func.row_number()
        .over(partition_by=TLE.norad_cat_id, order_by=order)
        .label("rn")
    )
    subq = (
        select(TLE.id.label("tle_id"), rn)
        .where(TLE.epoch >= start, TLE.epoch <= end, TLE.norad_cat_id.is_not(None))
        .subquery()
    )
    stmt = select(TLE).join(subq, TLE.id == subq.c.tle_id).where(subq.c.rn == 1)
    return list(session.execute(stmt).scalars().all())


def nearest_tles_for_date(
    session: Session, date: datetime.date, days: float
) -> list[TLE]:
    """Nearest TLE per satellite to 12:00 UTC on `date`, within +/- `days`.

    The per-object reduction happens in SQL (one candidate on each side of
    the center per object), so memory is O(objects), not O(rows in window).
    """
    center = datetime.datetime.combine(date, datetime.time(12))
    window = datetime.timedelta(days=days)

    # Latest at/before center, earliest after: the nearest is one of the two.
    before = _boundary_tles(session, center - window, center, latest=True)
    after = _boundary_tles(
        session,
        center + datetime.timedelta(microseconds=1),
        center + window,
        latest=False,
    )

    nearest: dict[int, TLE] = {}
    for tle in before + after:
        assert tle.norad_cat_id is not None
        best = nearest.get(tle.norad_cat_id)
        if best is None or abs(tle.epoch - center) < abs(best.epoch - center):
            nearest[tle.norad_cat_id] = tle
    return [nearest[satnum] for satnum in sorted(nearest)]


def dump_db(
    session: Session,
    tle_path: pathlib.Path,
    omm_path: pathlib.Path,
) -> tuple[int, int]:
    """Export the database as re-ingestable files (logical backup).

    Writes every element set to ``tle_path`` as plain two-line text —
    lossless, since rows store line1/line2 verbatim and the dedup key is
    that exact text. Rows with OMM metadata are additionally written to
    ``omm_path`` as Space-Track-style JSON records (TLE_LINE1/TLE_LINE2
    plus the metadata fields), which ``ingest`` reattaches to the same
    rows. ``omm_path`` is only created when metadata exists.

    Restore into any dialect: ``init-db`` then ``ingest`` both files.
    Returns (tle_count, omm_count).
    """
    from thistle_db.model import OmmMetadata
    from thistle_db.reader import write_tle

    order = (TLE.norad_cat_id, TLE.epoch, TLE.id)

    stmt = select(TLE).order_by(*order).execution_options(yield_per=5000)
    tle_count = 0

    def _pairs():
        nonlocal tle_count
        for tle in session.execute(stmt).scalars():
            tle_count += 1
            yield (tle.line1, tle.line2)

    write_tle(tle_path, _pairs())

    omm_stmt = (
        select(OmmMetadata, TLE)
        .join(TLE, OmmMetadata.tle_id == TLE.id)
        .order_by(*order)
        .execution_options(yield_per=5000)
    )
    # Stream the JSON array one record per line: the export must scale to
    # full-catalog dumps, so records are never all held in memory. The file
    # is only created once the first record arrives.
    omm_count = 0
    f = None
    try:
        for meta, tle in session.execute(omm_stmt):
            record = {
                "TLE_LINE1": tle.line1,
                "TLE_LINE2": tle.line2,
                "OBJECT_NAME": meta.object_name,
                "OBJECT_TYPE": meta.object_type,
                "COUNTRY_CODE": meta.country_code,
                "RCS_SIZE": meta.rcs_size,
                "LAUNCH_DATE": meta.launch_date,
                "SITE": meta.site,
                "DECAY_DATE": meta.decay_date,
                "ORIGINATOR": meta.originator,
                "GP_ID": meta.gp_id,
            }
            record = {k: v for k, v in record.items() if v is not None}
            if f is None:
                f = open(omm_path, "w")
                f.write("[\n")
            else:
                f.write(",\n")
            f.write(json.dumps(record))
            omm_count += 1
        if f is not None:
            f.write("\n]\n")
    finally:
        if f is not None:
            f.close()

    return tle_count, omm_count


def get_tles(satnum: int, config: Settings | None = None) -> list[tuple[str, str]]:
    """Return (line1, line2) pairs for one satellite, ordered by epoch.

    Opens a session on the configured database (config.toml / THISTLE_DB_*
    env vars when `config` is None) and disposes it before returning.
    """
    if config is None:
        config = load_config(None)
    session, engine = open_session(config, readonly=True)
    try:
        return [(tle.line1, tle.line2) for tle in tles_for_object(session, satnum)]
    finally:
        session.close()
        engine.dispose()
