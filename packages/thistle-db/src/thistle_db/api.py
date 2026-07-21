"""Programmatic query API for the thistle-db database."""

import datetime

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from thistle_db.config import Settings, load_config
from thistle_db.model import TLE, Base


def open_session(config: Settings) -> tuple[Session, Engine]:
    """Create the schema if needed and open a session on the configured database."""
    engine = config.database.engine
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory(), engine


def tles_for_object(session: Session, satnum: int) -> list[TLE]:
    """All TLEs for one satellite, ordered by epoch."""
    stmt = select(TLE).where(TLE.norad_cat_id == satnum).order_by(TLE.epoch)
    return list(session.execute(stmt).scalars().all())


def nearest_tles_for_date(
    session: Session, date: datetime.date, days: float
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


def get_tles(satnum: int, config: Settings | None = None) -> list[tuple[str, str]]:
    """Return (line1, line2) pairs for one satellite, ordered by epoch.

    Opens a session on the configured database (config.toml / THISTLE_DB_*
    env vars when `config` is None) and disposes it before returning.
    """
    if config is None:
        config = load_config(None)
    session, engine = open_session(config)
    try:
        return [(tle.line1, tle.line2) for tle in tles_for_object(session, satnum)]
    finally:
        session.close()
        engine.dispose()
