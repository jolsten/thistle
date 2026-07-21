import datetime
import pathlib

from loguru import logger
from sgp4.api import Satrec
from sgp4.exporter import export_omm
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from thistle_db.config import OutputConfig
from thistle_db.model import TLE
from thistle_db.reader import TLETuple, write_omm_csv, write_tle


def _reconstruct_satrec(tle: TLE) -> Satrec:
    """Reconstruct a Satrec object from stored TLE lines."""
    return Satrec.twoline2rv(tle.line1, tle.line2)


def _safe_export_omm(sat: Satrec, name: str) -> dict | None:
    """Export OMM dict from Satrec, returning None if export fails."""
    try:
        return export_omm(sat, name)
    except (ValueError, AttributeError):
        return None


def _get_object_name(tle: TLE) -> str:
    """Get object name from OmmMetadata if available, else fall back to object_id."""
    if tle.omm_metadata and tle.omm_metadata.object_name:
        return tle.omm_metadata.object_name
    return tle.object_id


def generate_date_files(
    session: Session, output_dir: pathlib.Path, config: OutputConfig
) -> None:
    """Generate one file per date with the latest TLE per object."""
    # Get distinct dates (UTC date truncation of epoch)
    date_query = select(func.date(TLE.epoch).label("d")).distinct()
    dates = session.execute(date_query).scalars().all()

    logger.info(f"Generating date files for {len(dates)} dates")

    for raw_date in dates:
        if raw_date is None:
            continue

        # func.date() returns str on SQLite and datetime.date on MariaDB/MySQL.
        if isinstance(raw_date, datetime.date):
            date_val = raw_date
        else:
            date_val = datetime.date.fromisoformat(str(raw_date))

        # Subquery: max epoch per norad_cat_id on this date
        subq = (
            select(
                TLE.norad_cat_id,
                func.max(TLE.epoch).label("max_epoch"),
            )
            .where(func.date(TLE.epoch) == date_val)
            .group_by(TLE.norad_cat_id)
            .subquery()
        )

        # Join back to get full TLE rows
        stmt = (
            select(TLE)
            .join(
                subq,
                (TLE.norad_cat_id == subq.c.norad_cat_id)
                & (TLE.epoch == subq.c.max_epoch),
            )
            .order_by(TLE.norad_cat_id)
        )
        tles = session.execute(stmt).scalars().all()

        if not tles:
            continue

        filename_date = date_val.strftime("%Y%m%d")

        # TLE output
        if config.formats.tle:
            tle_path = output_dir / f"{filename_date}.tle"
            tle_tuples: list[TLETuple] = [(t.line1, t.line2) for t in tles]
            write_tle(tle_path, tle_tuples)

        # OMM output
        if config.formats.omm:
            omm_path = output_dir / f"{filename_date}.omm"
            omm_records = []
            for tle in tles:
                sat = _reconstruct_satrec(tle)
                name = _get_object_name(tle)
                record = _safe_export_omm(sat, name)
                if record is not None:
                    omm_records.append(record)
            write_omm_csv(omm_path, omm_records)


def generate_object_files(
    session: Session, output_dir: pathlib.Path, config: OutputConfig
) -> None:
    """Generate one file per satellite with all its TLEs."""
    # Get distinct NORAD IDs
    norad_ids = (
        session.execute(select(TLE.norad_cat_id).distinct().order_by(TLE.norad_cat_id))
        .scalars()
        .all()
    )

    logger.info(f"Generating object files for {len(norad_ids)} satellites")

    for norad_id in norad_ids:
        if norad_id is None:
            continue

        stmt = select(TLE).where(TLE.norad_cat_id == norad_id).order_by(TLE.epoch)
        tles = session.execute(stmt).scalars().all()

        if not tles:
            continue

        # TLE output
        if config.formats.tle:
            tle_path = output_dir / f"{norad_id}.tle"
            tle_tuples: list[TLETuple] = [(t.line1, t.line2) for t in tles]
            write_tle(tle_path, tle_tuples)

        # OMM output
        if config.formats.omm:
            omm_path = output_dir / f"{norad_id}.omm"
            omm_records = []
            for tle in tles:
                sat = _reconstruct_satrec(tle)
                name = _get_object_name(tle)
                record = _safe_export_omm(sat, name)
                if record is not None:
                    omm_records.append(record)
            write_omm_csv(omm_path, omm_records)


def generate(session: Session, config: OutputConfig) -> None:
    """Generate all configured output files."""
    output_dir = pathlib.Path(config.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.types.date_files:
        generate_date_files(session, output_dir, config)

    if config.types.object_files:
        generate_object_files(session, output_dir, config)

    logger.info(f"Output generation complete → {output_dir}")
