"""SQLAlchemy models — the SQL store (schema v2, single-store architecture).

Five tables: ``tle`` (one immutable row per element set), ``omm_metadata``
(sparse enrichment sidecar — the one sanctioned UPDATE), ``ingest_files``
(source-file state), ``rejects`` (retained invalid input) and
``ingest_runs`` (run accounting). See PLAN.md §5 for the normative schema
and its conventions.
"""

import datetime
from math import isfinite, pi
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    ColumnElement,
    DateTime,
    Double,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    cast,
)
from sqlalchemy.dialects.mysql import BINARY as MYSQL_BINARY
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from thistle_db.record import GPRecord, implicit_semantics

# BIGINT surrogate keys: INSERT IGNORE burns auto-increment values on every
# duplicate row it skips, so id consumption grows with rows *attempted*, not
# rows stored — a 32-bit key can exhaust at a small fraction of 2^31 stored
# rows. SQLite needs plain INTEGER for the rowid-alias autoincrement.
BigIntPK = BigInteger().with_variant(Integer, "sqlite")

# Epochs carry microseconds; plain DATETIME on MariaDB/MySQL is fsp=0 and
# silently truncates to whole seconds, breaking epoch equality comparisons.
EpochDateTime = DateTime().with_variant(MYSQL_DATETIME(fsp=6), "mysql", "mariadb")

# Raw sha256 digest. The MariaDB variant matters: LargeBinary alone compiles
# to BLOB there, which cannot back a unique index without a prefix length;
# BINARY(32) keeps the dedup index entry fixed-width and half the hex size.
HashBinary = LargeBinary(32).with_variant(MYSQL_BINARY(32), "mysql", "mariadb")

# 4-byte float (~7 significant digits). The TLE text fields carry at most
# 7 significant digits everywhere except mean_motion (10), and the verbatim
# lines are the lossless source of truth — these columns are a convenience
# view, re-derivable at any time, so single precision loses nothing that
# matters. Never compare them for equality.


class _Float32(TypeDecorator[float]):
    """FLOAT storage read back as DOUBLE.

    MariaDB's text protocol formats FLOAT result columns with only 6
    significant digits, which does not round-trip a float32 (that needs 9)
    — the value Python receives can be a *different* float32 than the one
    stored, breaking incremental/rebuild output byte parity and line-less
    record identity. Casting the projection to DOUBLE is an exact widening
    on every dialect and makes the server emit full-precision text; the
    renderer's float32 re-rounding then recovers the stored value exactly.
    DDL is unaffected (storage stays 4-byte FLOAT).
    """

    impl = Float
    cache_ok = True

    def __init__(self) -> None:
        super().__init__(24)

    def column_expression(
        self, colexpr: ColumnElement[float]
    ) -> ColumnElement[float]:
        return cast(colexpr, Double())


Float32 = _Float32()

# WGS-72 constants (sgp4's default gravity model) for the derived-value
# convenience columns. Non-canonical: excluded from identity and export.
_MU_KM3_S2 = 398600.8
_EARTH_RADIUS_KM = 6378.135


class Base(DeclarativeBase):
    type_annotation_map = {
        # Bare Mapped[float] would otherwise compile to 4-byte FLOAT on
        # MariaDB, losing precision on round-trip.
        float: Float(53),
        datetime.datetime: EpochDateTime,
    }


def utcnow() -> datetime.datetime:
    """Naive UTC now — all stored datetimes are naive UTC so comparisons
    against loaded rows never mix aware and naive values."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def date_from_sql(raw) -> datetime.date:
    """Coerce a `func.date()` result to a `datetime.date`.

    SQLite returns it as a string, MariaDB and PostgreSQL as a date.
    """
    if isinstance(raw, datetime.date):
        return raw
    return datetime.date.fromisoformat(str(raw))


class Mixin:
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    created: Mapped[datetime.datetime] = mapped_column(default=utcnow)


class TLE(Base, Mixin):
    """One immutable row per element set (line-full or line-less).

    The table name is grandfathered — line-less OMM-native records live
    here too, but `tle` reads naturally in ad-hoc SQL and a rename buys
    nothing functional.
    """

    __tablename__ = "tle"

    # Index budget (3 B-trees total — exhaustive, see PLAN.md §5.1):
    #   PK(id)                          — clustered
    #   UNIQUE(epoch, record_hash)      — dedup, probed on every insert.
    #     Epoch-first: normal ingest only carries recent elsets, so probes
    #     land in the hot right edge of the index no matter how large the
    #     table grows. Also serves epoch range scans, and keeps bulk
    #     chronological restores cheap — do not change the column order.
    #   (norad_cat_id, epoch)           — the shape of every per-object query
    __table_args__ = (
        UniqueConstraint("epoch", "record_hash", name="uq_tle_epoch_record_hash"),
        Index("ix_tle_norad_cat_id_epoch", "norad_cat_id", "epoch"),
    )

    record_hash: Mapped[bytes] = mapped_column(HashBinary)
    """Raw sha256 digest of the record's identity (record.py §Uniqueness):
    sha256('line1\\nline2') when lines are present, the thistle-gp1
    semantic hash otherwise. Use ``HEX(record_hash)`` in ad-hoc SQL."""

    # Verbatim delivered bytes — the audit trail; null for line-less
    # OMM-native records. line0 is the 3LE name line when delivered.
    line0: Mapped[Optional[str]] = mapped_column(String(70))
    line1: Mapped[Optional[str]] = mapped_column(String(70))
    line2: Mapped[Optional[str]] = mapped_column(String(70))

    norad_cat_id: Mapped[int] = mapped_column(Integer)
    """NORAD Catalog Number. NOT NULL — records without one are rejected
    (they have no home in per-object outputs), so no read path branches
    on a missing ID."""
    object_id: Mapped[Optional[str]] = mapped_column(String(15))
    """International Designator (century-expanded; 'TBA' kept verbatim)."""
    classification_type: Mapped[str] = mapped_column(String(1), default="U")

    # Element set. Single precision (Float32) except mean_motion: the TLE
    # text carries 10 significant digits there, beyond FLOAT's ~7.
    epoch: Mapped[datetime.datetime]
    """Epoch of mean Keplerian elements. Naive UTC, microseconds."""
    mean_motion: Mapped[Optional[float]]
    """Mean motion [rev/day]"""
    eccentricity: Mapped[float] = mapped_column(Float32)
    inclination: Mapped[float] = mapped_column(Float32)
    """[deg]"""
    ra_of_asc_node: Mapped[float] = mapped_column(Float32)
    """[deg]"""
    arg_of_pericenter: Mapped[float] = mapped_column(Float32)
    """[deg]"""
    mean_anomaly: Mapped[float] = mapped_column(Float32)
    """[deg]"""
    bstar: Mapped[Optional[float]] = mapped_column(Float32)
    mean_motion_dot: Mapped[Optional[float]] = mapped_column(Float32)
    mean_motion_ddot: Mapped[Optional[float]] = mapped_column(Float32)

    ephemeris_type: Mapped[int] = mapped_column(SmallInteger, default=0)
    element_set_no: Mapped[Optional[int]]
    rev_at_epoch: Mapped[Optional[int]]
    mean_element_theory: Mapped[str] = mapped_column(String(16), default="SGP4")
    """Identity-bearing (part of the semantic hash), so stored per row.
    The other implicit-semantics fields (REF_FRAME/TIME_SYSTEM/CENTER_NAME)
    live as nullable overrides on omm_metadata."""

    # Derived convenience columns — NON-CANONICAL: recomputed at ingest,
    # excluded from identity and export, re-derivable at any time.
    semimajor_axis: Mapped[Optional[float]] = mapped_column(Float32)
    """[km]"""
    period: Mapped[Optional[float]] = mapped_column(Float32)
    """[min]"""
    apoapsis_alt: Mapped[Optional[float]] = mapped_column(Float32)
    """[km]"""
    periapsis_alt: Mapped[Optional[float]] = mapped_column(Float32)
    """[km]"""

    omm_metadata: Mapped[Optional["OmmMetadata"]] = relationship(
        back_populates="tle", uselist=False, lazy="select"
    )

    def __repr__(self) -> str:
        t = self.epoch.isoformat(sep="T", timespec="seconds")
        return f"TLE(norad_cat_id={self.norad_cat_id!r}, epoch={t!r})"


class OmmMetadata(Base, Mixin):
    """Sparse enrichment sidecar: metadata, provenance, semantics overrides.

    One row per `tle` row that has ever received an OMM delivery carrying
    more than elements. The §3.2 merge applies here — the one sanctioned
    UPDATE in the system; `tle` rows stay immutable.
    """

    __tablename__ = "omm_metadata"

    tle_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("tle.id"), unique=True, index=True
    )
    tle: Mapped["TLE"] = relationship(back_populates="omm_metadata")

    # Metadata — delivered, never inferred.
    object_name: Mapped[Optional[str]] = mapped_column(String(100))
    object_type: Mapped[Optional[str]] = mapped_column(String(20))
    rcs_size: Mapped[Optional[str]] = mapped_column(String(10))
    country_code: Mapped[Optional[str]] = mapped_column(String(10))
    launch_date: Mapped[Optional[str]] = mapped_column(String(10))
    site: Mapped[Optional[str]] = mapped_column(String(20))
    decay_date: Mapped[Optional[str]] = mapped_column(String(10))

    # Provenance — never backfillable except by OMM re-delivery.
    gp_id: Mapped[Optional[int]] = mapped_column()
    file_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    creation_date: Mapped[Optional[datetime.datetime]]
    originator: Mapped[Optional[str]] = mapped_column(String(20))

    # Delivered-semantics overrides. Null = implicit from the tle row's
    # ephemeris_type (record.implicit_semantics) — the common case costs
    # nothing per row.
    ref_frame: Mapped[Optional[str]] = mapped_column(String(16))
    time_system: Mapped[Optional[str]] = mapped_column(String(16))
    center_name: Mapped[Optional[str]] = mapped_column(String(16))


class IngestFile(Base, Mixin):
    """State of a previously ingested source file, used to skip unchanged
    files. A cache in spirit: losing a row only costs a re-read, absorbed
    by dedup."""

    __tablename__ = "ingest_files"

    path: Mapped[str] = mapped_column(String(1024))
    """Absolute resolved path (for humans; not the unique key)."""
    path_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    """sha256 of `path` — bounded unique key, safe for MySQL utf8mb4 index limits."""
    size: Mapped[int] = mapped_column(BigInteger)
    """File size in bytes at last successful ingest."""
    mtime_ns: Mapped[int] = mapped_column(BigInteger)
    """st_mtime_ns at last successful ingest (integer ns compares exactly on all dialects)."""
    sha256: Mapped[str] = mapped_column(String(64))
    """sha256 of file content at last successful ingest."""
    last_run_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    """The ingest_runs row that last recorded this file."""


class Reject(Base):
    """Retained invalid input — one row per *distinct* rejected payload.

    An archive, not a live view: rows are never deleted by a clean run
    (`resolved_at` carries the lifecycle), and the table is bounded by
    what is broken, not by time (re-rejection upserts `last_seen` /
    `times_seen`). Validity is table membership: this table and `tle`
    together are "all input rows". See PLAN.md §4/§5.3.
    """

    __tablename__ = "rejects"
    __table_args__ = (Index("ix_rejects_last_seen", "last_seen"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    record_hash: Mapped[bytes] = mapped_column(HashBinary, unique=True)
    """sha256 of the verbatim payload — the dedup key."""
    format: Mapped[str] = mapped_column(String(4))
    """'tle' | 'omm'"""
    payload: Mapped[str] = mapped_column(Text)
    """Verbatim rejected input ('line1\\nline2', or the record's canonical
    JSON), capped at rejects.PAYLOAD_CAP bytes."""
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    reason_code: Mapped[str] = mapped_column(String(20))
    """Closed vocabulary (rejects.ReasonCode); latest rejection wins."""
    reason: Mapped[str] = mapped_column(Text)
    """Free-text detail; latest rejection wins."""
    source_path: Mapped[Optional[str]] = mapped_column(String(1024))
    first_seen: Mapped[datetime.datetime] = mapped_column(default=utcnow)
    last_seen: Mapped[datetime.datetime] = mapped_column(default=utcnow)
    times_seen: Mapped[int] = mapped_column(Integer, default=1)
    norad_cat_id: Mapped[Optional[int]] = mapped_column(Integer)
    """Convenience column, populated when parseable."""
    epoch: Mapped[Optional[datetime.datetime]]
    """Convenience column, populated when parseable — drives future-epoch
    re-admission."""
    resolved_at: Mapped[Optional[datetime.datetime]]
    """Stamped when the same hash exists in `tle` (re-admission at ingest,
    reconciliation in `verify`)."""


class IngestRun(Base):
    """One row per run — written at start, finalized at completion.

    A row stuck in 'running' is itself a signal (crashed run), reported by
    `status`; it is never used as a lock (the advisory lockfile is).
    Index budget: PK only — status reads the insertion-ordered tail.
    """

    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    command: Mapped[str] = mapped_column(String(255))
    started: Mapped[datetime.datetime] = mapped_column(default=utcnow)
    finished: Mapped[Optional[datetime.datetime]]
    status: Mapped[str] = mapped_column(String(10), default="running")
    """'running' | 'succeeded' | 'failed'"""

    files_scanned: Mapped[int] = mapped_column(Integer, default=0)
    files_ingested: Mapped[int] = mapped_column(Integer, default=0)
    files_skipped: Mapped[int] = mapped_column(Integer, default=0)
    files_refreshed: Mapped[int] = mapped_column(Integer, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, default=0)
    records_seen: Mapped[int] = mapped_column(BigInteger, default=0)
    records_inserted: Mapped[int] = mapped_column(BigInteger, default=0)
    records_rejected: Mapped[int] = mapped_column(BigInteger, default=0)

    error: Mapped[Optional[str]] = mapped_column(Text)
    detail: Mapped[Optional[dict]] = mapped_column(JSON)
    """Extensible counters: per-output write counts, reject reason
    histogram. JSON maps to TEXT on SQLite/MariaDB via SQLAlchemy."""


# --- GPRecord <-> row conversion --------------------------------------------


def derived_values(
    mean_motion: Optional[float], eccentricity: float
) -> dict[str, Optional[float]]:
    """Derived convenience values from mean motion [rev/day] and e.

    WGS-72 constants; None wherever the inputs don't yield a positive
    finite result — a malformed elset must not cost the whole row here.
    """
    none: dict[str, Optional[float]] = {
        "semimajor_axis": None,
        "period": None,
        "apoapsis_alt": None,
        "periapsis_alt": None,
    }
    if mean_motion is None or not mean_motion > 0:
        return none
    n_rad_s = mean_motion * 2 * pi / 86400.0
    try:
        a_km = (_MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
        values = {
            "semimajor_axis": a_km,
            "period": 1440.0 / mean_motion,
            "apoapsis_alt": a_km * (1 + eccentricity) - _EARTH_RADIUS_KM,
            "periapsis_alt": a_km * (1 - eccentricity) - _EARTH_RADIUS_KM,
        }
    except (OverflowError, ZeroDivisionError):
        return none
    return {
        key: (value if isfinite(value) else None) for key, value in values.items()
    }


def tle_row_from_record(record: GPRecord, created: datetime.datetime) -> dict:
    """A GPRecord as a `tle` row dict for Core-level INSERT."""
    return {
        "created": created,
        "record_hash": bytes.fromhex(record.record_hash),
        "line0": record.tle_line0,
        "line1": record.tle_line1,
        "line2": record.tle_line2,
        "norad_cat_id": record.norad_cat_id,
        "object_id": record.object_id,
        "classification_type": (record.classification_type or "U")[:1],
        "epoch": record.epoch,
        "mean_motion": record.mean_motion,
        "eccentricity": record.eccentricity,
        "inclination": record.inclination,
        "ra_of_asc_node": record.ra_of_asc_node,
        "arg_of_pericenter": record.arg_of_pericenter,
        "mean_anomaly": record.mean_anomaly,
        "bstar": record.bstar,
        "mean_motion_dot": record.mean_motion_dot,
        "mean_motion_ddot": record.mean_motion_ddot,
        "ephemeris_type": record.ephemeris_type,
        "element_set_no": record.element_set_no,
        "rev_at_epoch": record.rev_at_epoch,
        "mean_element_theory": record.mean_element_theory[:16],
        **derived_values(record.mean_motion, record.eccentricity),
    }


# GPRecord field -> omm_metadata column (same names throughout).
METADATA_FIELDS = (
    "object_name",
    "object_type",
    "rcs_size",
    "country_code",
    "launch_date",
    "site",
    "decay_date",
    "gp_id",
    "file_id",
    "creation_date",
    "originator",
)
_SEMANTICS_FIELDS = ("ref_frame", "time_system", "center_name")

# Every column the enrichment merge may update — the one sanctioned UPDATE.
SIDECAR_FIELDS = METADATA_FIELDS + _SEMANTICS_FIELDS


def _semantics_overrides(record: GPRecord) -> dict:
    """Delivered semantics that differ from the implicit per-type values.

    Equal-to-implicit values store as NULL, so the common case costs
    nothing per row and projections fall back to the implicit table.
    """
    implicit = implicit_semantics(record.ephemeris_type)
    return {
        name: (
            getattr(record, name)
            if getattr(record, name) != implicit[name]
            else None
        )
        for name in _SEMANTICS_FIELDS
    }


def needs_metadata(record: GPRecord) -> bool:
    """Whether this record carries anything the omm_metadata sidecar stores."""
    if any(getattr(record, name) is not None for name in METADATA_FIELDS):
        return True
    return any(value is not None for value in _semantics_overrides(record).values())


def metadata_row_from_record(
    record: GPRecord, tle_id: int, created: datetime.datetime
) -> dict:
    """A GPRecord's sidecar as an `omm_metadata` row dict."""
    row = {"tle_id": tle_id, "created": created}
    for name in METADATA_FIELDS:
        row[name] = getattr(record, name)
    row.update(_semantics_overrides(record))
    return row


def record_from_row(tle: TLE, metadata: Optional[OmmMetadata]) -> GPRecord:
    """Reconstitute the canonical record from its row(s) — the read half of
    export and omm-csv output projection. Semantics fall back to the
    implicit per-type table when the sidecar holds no override."""
    implicit = implicit_semantics(tle.ephemeris_type)
    kwargs: dict[str, Any] = dict(
        record_hash=tle.record_hash.hex(),
        norad_cat_id=tle.norad_cat_id,
        epoch=tle.epoch,
        object_id=tle.object_id,
        mean_motion=tle.mean_motion,
        eccentricity=tle.eccentricity,
        inclination=tle.inclination,
        ra_of_asc_node=tle.ra_of_asc_node,
        arg_of_pericenter=tle.arg_of_pericenter,
        mean_anomaly=tle.mean_anomaly,
        bstar=tle.bstar,
        mean_motion_dot=tle.mean_motion_dot,
        mean_motion_ddot=tle.mean_motion_ddot,
        ephemeris_type=tle.ephemeris_type,
        classification_type=tle.classification_type,
        element_set_no=tle.element_set_no,
        rev_at_epoch=tle.rev_at_epoch,
        mean_element_theory=tle.mean_element_theory,
        ref_frame=implicit["ref_frame"],
        time_system=implicit["time_system"],
        center_name=implicit["center_name"],
        tle_line0=tle.line0,
        tle_line1=tle.line1,
        tle_line2=tle.line2,
    )
    if metadata is not None:
        for name in METADATA_FIELDS:
            kwargs[name] = getattr(metadata, name)
        for name in _SEMANTICS_FIELDS:
            value = getattr(metadata, name)
            if value is not None:
                kwargs[name] = value
    return GPRecord(**kwargs)
