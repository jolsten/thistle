"""Programmatic query API for the thistle-db database."""

import datetime
import pathlib

from sqlalchemy import func, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from thistle_db.config import Settings, load_config
from thistle_db.model import TLE, Base


def init_db(config: Settings, drop: bool = False) -> dict[str, list[str]]:
    """Create the database schema, optionally dropping it first.

    The only intended DDL entry point. Idempotent when ``drop`` is False:
    existing tables are left untouched. With ``drop=True``, all model tables
    are dropped first — destructive, callers must confirm with the user.

    Returns table names as ``{"dropped": [...], "created": [...],
    "existing": [...]}``.
    """
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
        return {
            "dropped": dropped,
            "created": sorted(after - before),
            "existing": sorted(before),
        }
    finally:
        engine.dispose()


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
    records = []
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
        records.append({k: v for k, v in record.items() if v is not None})

    if records:
        import json

        with open(omm_path, "w") as f:
            json.dump(records, f, indent=1)

    return tle_count, len(records)


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
