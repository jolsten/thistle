import enum
import hashlib
import pathlib
from typing import Sequence

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from thistle_db.config import IngestSource
from thistle_db.model import IngestFile, OmmMetadata, TLE, utcnow
from thistle_db.reader import (
    TLETuple,
    detect_format,
    read_omm_csv,
    read_omm_json,
    read_omm_xml,
    read_tle,
)

CHUNK_SIZE = 5000


def _bulk_insert_tles(session: Session, records: list[dict]) -> int:
    """Bulk insert TLE records with dialect-aware dedup.

    Returns count of newly inserted rows.
    """
    if not records:
        return 0

    dialect = session.bind.dialect.name  # type: ignore[union-attr]
    table = TLE.__table__
    total_inserted = 0

    for i in range(0, len(records), CHUNK_SIZE):
        chunk = records[i : i + CHUNK_SIZE]

        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = sqlite_insert(table).on_conflict_do_nothing(  # type: ignore[arg-type]
                index_elements=["line1", "line2"]
            )
            result = session.execute(stmt, chunk)
            total_inserted += result.rowcount  # type: ignore[attr-defined]
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(table).on_conflict_do_nothing(  # type: ignore[arg-type]
                index_elements=["line1", "line2"]
            )
            # psycopg3 doesn't populate rowcount for bulk inserts; use RETURNING.
            result = session.execute(stmt.returning(table.c.id), chunk)
            total_inserted += len(result.all())
        else:
            # mysql / mariadb (and any other INSERT IGNORE dialect)
            from sqlalchemy import insert

            stmt = insert(table).prefix_with("IGNORE")  # type: ignore[arg-type]
            result = session.execute(stmt, chunk)
            total_inserted += result.rowcount  # type: ignore[attr-defined]
        session.commit()

    return total_inserted


def ingest_tles(session: Session, tles: Sequence[TLETuple]) -> int:
    """Ingest TLE tuples into the database.

    Skips malformed records and duplicates. Returns count of newly inserted rows.
    """
    records = []
    for line1, line2 in tles:
        try:
            tle = TLE.from_twoline(line1, line2)
            now = utcnow()
            record = _tle_to_record(tle, now)
            records.append(record)
        except Exception:
            logger.warning(f"Skipping malformed TLE: {line1[:20]}...")

    return _bulk_insert_tles(session, records)


def _tle_to_record(tle: TLE, now=None) -> dict:
    """Convert a TLE ORM object to a dict for Core-level INSERT."""
    if now is None:
        now = utcnow()
    record = {
        col.name: getattr(tle, col.name)
        for col in TLE.__table__.columns
        if col.name != "id"
    }
    record["created"] = now
    record["modified"] = now
    return record


def ingest_omm(session: Session, omm_records: list[dict]) -> int:
    """Ingest OMM records into the database.

    Extracts TLE_LINE1/TLE_LINE2 from each record, inserts into TLE table,
    then populates OmmMetadata for newly inserted rows.
    Returns count of newly inserted TLE rows.
    """
    # Build TLE records from OMM data
    tle_records = []
    omm_by_lines: dict[tuple[str, str], dict] = {}

    for omm in omm_records:
        line1 = omm.get("TLE_LINE1", "").strip()
        line2 = omm.get("TLE_LINE2", "").strip()
        if not line1 or not line2:
            logger.warning(
                f"OMM record missing TLE lines: {omm.get('OBJECT_NAME', 'unknown')}"
            )
            continue

        try:
            tle = TLE.from_twoline(line1, line2)
            record = _tle_to_record(tle)
            tle_records.append(record)
            omm_by_lines[(line1, line2)] = omm
        except Exception:
            logger.warning(
                f"Skipping malformed OMM record: {omm.get('OBJECT_NAME', 'unknown')}"
            )

    # Bulk insert TLE rows
    inserted = _bulk_insert_tles(session, tle_records)

    if not omm_by_lines:
        return inserted

    # FK resolution: SELECT back by (line1, line2) to get IDs
    line_pairs = list(omm_by_lines.keys())
    for i in range(0, len(line_pairs), CHUNK_SIZE):
        chunk = line_pairs[i : i + CHUNK_SIZE]
        # Query TLE rows that match our line pairs and don't have metadata yet
        stmt = (
            select(TLE.id, TLE.line1, TLE.line2)
            .where(
                TLE.line1.in_([lp[0] for lp in chunk]),
                TLE.line2.in_([lp[1] for lp in chunk]),
            )
            .outerjoin(OmmMetadata, TLE.id == OmmMetadata.tle_id)
            .where(OmmMetadata.id.is_(None))
        )
        rows = session.execute(stmt).all()

        metadata_records = []
        for tle_id, line1, line2 in rows:
            omm = omm_by_lines.get((line1, line2))
            if omm is None:
                continue
            metadata_records.append(
                OmmMetadata(
                    tle_id=tle_id,
                    object_name=omm.get("OBJECT_NAME"),
                    object_type=omm.get("OBJECT_TYPE"),
                    country_code=omm.get("COUNTRY_CODE"),
                    rcs_size=omm.get("RCS_SIZE"),
                    launch_date=omm.get("LAUNCH_DATE"),
                    site=omm.get("SITE"),
                    decay_date=omm.get("DECAY_DATE"),
                    originator=omm.get("ORIGINATOR"),
                    gp_id=int(omm["GP_ID"]) if omm.get("GP_ID") else None,
                )
            )

        if metadata_records:
            session.add_all(metadata_records)
            session.commit()

    return inserted


def ingest_file(session: Session, path: pathlib.Path) -> int:
    """Detect format and ingest a single file. Returns count of newly inserted rows."""
    fmt = detect_format(path)
    logger.info(f"Ingesting {path} (format: {fmt})")

    if fmt == "tle":
        tles = read_tle(path)
        return ingest_tles(session, tles)
    elif fmt == "omm_json":
        records = read_omm_json(path)
        return ingest_omm(session, records)
    elif fmt == "omm_csv":
        records = read_omm_csv(path)
        return ingest_omm(session, records)
    elif fmt == "omm_xml":
        records = read_omm_xml(path)
        return ingest_omm(session, records)
    else:
        logger.warning(f"Unknown format for {path}, skipping")
        return 0


class FileStatus(enum.StrEnum):
    INGESTED = "ingested"
    """File was parsed and inserted (possibly 0 new rows via dedup)."""
    SKIPPED = "skipped"
    """Size and mtime match recorded state; file not opened."""
    REFRESHED = "refreshed"
    """Stat changed but content hash identical; state refreshed, no parse."""
    FAILED = "failed"
    """Stat/read/parse raised; logged, state not recorded (retried next scan)."""


def _path_key(path: pathlib.Path) -> tuple[str, str]:
    """Return (absolute resolved path, sha256 hex of that path string)."""
    resolved = str(path.resolve())
    return resolved, hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ingest_source_file(
    session: Session, path: pathlib.Path, *, force: bool = False
) -> tuple[FileStatus, int]:
    """Ingest one file with state tracking and error isolation.

    Unchanged files (per the ingest_files state table) are skipped without
    being opened. Returns (status, newly_inserted_row_count). Never raises for
    per-file problems — logs a warning and returns (FAILED, 0) instead, leaving
    the file's state unrecorded so it is retried on the next scan.
    """
    try:
        resolved, key = _path_key(path)
        # Stat before reading: a file modified mid-ingest looks changed on the
        # next scan; the redundant re-ingest is absorbed by DB dedup.
        stat = path.stat()

        row = session.execute(
            select(IngestFile).where(IngestFile.path_hash == key)
        ).scalar_one_or_none()

        if (
            not force
            and row is not None
            and row.size == stat.st_size
            and row.mtime_ns == stat.st_mtime_ns
        ):
            logger.debug(f"Unchanged (size/mtime), skipping: {path}")
            return FileStatus.SKIPPED, 0

        digest = _file_sha256(path)

        if not force and row is not None and row.sha256 == digest:
            row.size = stat.st_size
            row.mtime_ns = stat.st_mtime_ns
            session.commit()
            logger.debug(f"Content unchanged, refreshed state: {path}")
            return FileStatus.REFRESHED, 0

        count = ingest_file(session, path)

        if row is None:
            session.add(
                IngestFile(
                    path=resolved,
                    path_hash=key,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=digest,
                )
            )
        else:
            row.size = stat.st_size
            row.mtime_ns = stat.st_mtime_ns
            row.sha256 = digest
        session.commit()

        return FileStatus.INGESTED, count
    except Exception as exc:
        session.rollback()
        logger.warning(f"Failed to ingest {path}: {exc}")
        return FileStatus.FAILED, 0


def ingest_sources(
    session: Session, sources: list[IngestSource], *, force: bool = False
) -> int:
    """Scan configured source directories and ingest all matching files.

    Unchanged files are skipped via the ingest_files state table; a bad file
    is logged and never aborts the scan. Pass force=True to re-ingest
    regardless of recorded state.
    """
    total = 0
    for source in sources:
        source_path = pathlib.Path(source.path)
        if not source_path.is_dir():
            logger.warning(f"Source directory does not exist: {source_path}")
            continue

        files = sorted(source_path.glob(source.pattern))
        logger.info(
            f"Found {len(files)} files in {source_path} matching {source.pattern}"
        )

        for file in files:
            status, count = ingest_source_file(session, file, force=force)
            logger.info(f"  {file.name}: {status}, {count} new records")
            total += count

    return total
