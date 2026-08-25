import datetime
import hashlib
from math import degrees, isfinite, nan, pi, sqrt
from typing import Optional

from sgp4.alpha5 import to_alpha5
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Double,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import BINARY as MYSQL_BINARY
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

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
LineHash = LargeBinary(32).with_variant(MYSQL_BINARY(32), "mysql", "mariadb")

# 4-byte float (~7 significant digits). The TLE text fields carry at most
# 7 significant digits everywhere except mean_motion (10), and line1/line2
# are the lossless source of truth — these columns are a convenience view,
# re-derivable at any time, so single precision loses nothing that matters.
Float32 = Float(24)


def object_id(sat: Satrec) -> str:
    val = sat.intldesg
    if len(val) < 2:
        return val
    try:
        year = int(val[0:2])
    except ValueError:
        # Non-numeric designator (e.g. "TBA" placeholders): store verbatim
        # rather than rejecting an otherwise valid TLE as malformed.
        return val
    return ("20" if year < 57 else "19") + val


def period(sat: Satrec) -> float:
    """Calculate orbital period [min]

    Returns NaN rather than raising for a semi-major axis that is not a
    positive finite number — a malformed elset must not cost the whole row
    here; `_clean_floats` decides what a non-finite value means per column.
    """
    a_km = sat.a * sat.radiusearthkm
    if not a_km > 0:  # false for NaN, zero and negative alike
        return nan
    try:
        return 2 * pi * sqrt(a_km**3 / sat.mu)
    except OverflowError:  # a_km**3 beyond float range
        return nan


def mean_motion(sat: Satrec) -> float:
    """Mean Motion [revs/day]"""
    return sat.no_kozai * 60 * 24 / (2 * pi)


def compute_line_hash(line1: str, line2: str) -> bytes:
    """Raw sha256 digest of the exact TLE text — the dedup key.

    Semantically identical to a unique constraint on (line1, line2), but the
    index it backs is 32 bytes instead of 560. Use ``HEX(line_hash)`` when
    inspecting rows in ad-hoc SQL.
    """
    return hashlib.sha256(f"{line1}\n{line2}".encode("utf-8")).digest()


class Base(DeclarativeBase):
    type_annotation_map = {
        # Bare Mapped[float] would otherwise compile to 4-byte FLOAT on
        # MariaDB, losing precision on round-trip.
        float: Double(),
        datetime.datetime: EpochDateTime,
    }


def date_from_sql(raw) -> datetime.date:
    """Coerce a `func.date()` result to a `datetime.date`.

    SQLite returns it as a string, MariaDB and PostgreSQL as a date.
    """
    if isinstance(raw, datetime.date):
        return raw
    return datetime.date.fromisoformat(str(raw))


def utcnow() -> datetime.datetime:
    """Naive UTC now — all stored datetimes are naive UTC so comparisons
    against loaded rows never mix aware and naive values."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class Mixin:
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    created: Mapped[datetime.datetime] = mapped_column(default=utcnow)


class TLE(Base, Mixin):
    __tablename__ = "tle"

    # Index budget (3 B-trees total, vs 15 before):
    #   PK(id)                          — clustered
    #   UNIQUE(epoch, line_hash)        — dedup, probed on every insert.
    #     Epoch-first: normal ingest only carries recent elsets, so probes
    #     land in the hot right edge of the index no matter how large the
    #     table grows. Also serves epoch range scans (date files).
    #   (norad_cat_id, epoch)           — the shape of every per-object query
    #   (created)                       — incremental generate lookback
    __table_args__ = (
        UniqueConstraint("epoch", "line_hash", name="uq_tle_epoch_line_hash"),
        Index("ix_tle_norad_cat_id_epoch", "norad_cat_id", "epoch"),
        Index("ix_tle_created", "created"),
    )

    line1: Mapped[str] = mapped_column(String(70))
    line2: Mapped[str] = mapped_column(String(70))
    line_hash: Mapped[bytes] = mapped_column(LineHash)
    """Raw sha256 of 'line1\\nline2' — fixed 32-byte dedup key.

    Uniqueness is (epoch, line_hash); epoch is encoded in line1, so identical
    lines always have identical epochs and the pair is exactly as unique as
    the raw text."""

    object_id: Mapped[str] = mapped_column(String(15))
    """International Designator"""
    norad_cat_id: Mapped[Optional[int]]
    """NORAD Catalog Number ("Satellite Number")"""
    classification: Mapped[str] = mapped_column(String(1))
    """Classification Type"""

    # Element Set. Single precision (Float32) except mean_motion: the TLE
    # text carries 10 significant digits there, beyond FLOAT's ~7.
    epoch: Mapped[datetime.datetime]
    """Epoch of Mean Keplerian elements (mandatory). Naive UTC, microseconds."""
    mean_motion: Mapped[Optional[float]]
    """Keplerian Mean Motion [rev/day]"""
    eccentricity: Mapped[float] = mapped_column(Float32)
    """Eccentricity"""
    inclination: Mapped[float] = mapped_column(Float32)
    """Inclination [deg]"""
    ra_of_asc_node: Mapped[float] = mapped_column(Float32)
    """Right Ascension of Ascending Node [deg]"""
    arg_of_pericenter: Mapped[float] = mapped_column(Float32)
    """Argument of Pericenter [deg]"""
    mean_anomaly: Mapped[float] = mapped_column(Float32)
    """Mean Anomaly [deg]"""

    # Other TLE Details
    ephemeris_type: Mapped[int]
    """Element Set Type"""
    element_set_no: Mapped[int]
    """Element Set Number (for this satellite)
    Normally incremented sequentially, but my be out of sync.
    """
    rev_at_epoch: Mapped[Optional[int]]
    """Revolution Number"""
    bstar: Mapped[Optional[float]] = mapped_column(Float32)
    """Drag-like ballistic coefficient"""
    mean_motion_dot: Mapped[Optional[float]] = mapped_column(Float32)
    """First Time Derivative of Mean Motion (i.e., a drag term)"""
    mean_motion_ddot: Mapped[Optional[float]] = mapped_column(Float32)
    """Second Time Derivative of Mean Motion (i.e., a drag term)"""

    # Derived Values
    semimajor_axis: Mapped[Optional[float]] = mapped_column(Float32)
    """Semi-major Axis [km]"""
    period: Mapped[Optional[float]] = mapped_column(Float32)
    """Orbital Period [min]"""
    apoapsis_alt: Mapped[Optional[float]] = mapped_column(Float32)
    """Apoapsis Altitude [km]"""
    periapsis_alt: Mapped[Optional[float]] = mapped_column(Float32)
    """Periapsis Altitude [km]"""

    omm_metadata: Mapped[Optional["OmmMetadata"]] = relationship(
        back_populates="tle", uselist=False, lazy="select"
    )

    def __repr__(self) -> str:
        n = to_alpha5(self.norad_cat_id)
        t = self.epoch.isoformat(sep="T", timespec="seconds")
        e = self.element_set_no
        return f"TLE(norad_cat_id={n!r}, epoch={t!r}, element_set_no={e})"

    @classmethod
    def from_twoline(cls, *lines: str) -> "TLE":
        if len(lines) == 2:
            line1, line2 = lines
        elif len(lines) == 3:
            _, line1, line2 = lines
        else:
            msg = "Must provide an iterable of length 2 or 3"
            raise ValueError(msg)

        sat = Satrec.twoline2rv(line1, line2)
        fields = _fields_from_satrec(sat)

        return cls(
            line1=line1,
            line2=line2,
            line_hash=compute_line_hash(line1, line2),
            **fields,
        )


def epoch_from_lines(line1: str, line2: str) -> datetime.datetime:
    """Epoch of a TLE as stored in the database (naive UTC, microseconds).

    Byte-identical to the `epoch` column value for the same lines — used to
    compare on-disk output files against database rows.
    """
    sat = Satrec.twoline2rv(line1, line2)
    return sat_epoch_datetime(sat).replace(tzinfo=None)


class MalformedElsetError(ValueError):
    """A TLE parsed, but a mandatory element is not a finite number.

    Raised by `_clean_floats`; `ingest` catches it per record, so one such
    elset is quarantined and skipped without costing the rest of the file.
    """


# Nullability comes from the schema itself rather than a hand-maintained
# list, so a float column added later is covered without touching
# `_clean_floats`.
_NULLABLE = {col.name: col.nullable for col in TLE.__table__.columns}


def _clean_floats(fields: dict) -> dict:
    """Replace non-finite floats with None, or reject the elset.

    sgp4's parser is C `strtod`, which accepts the literal text "nan",
    "inf" and "nan(ind)" in a fixed-width field — exactly what a producer
    emits when it formats a missing value — and shifted field boundaries in
    corrupt text produce the same. Every float column can therefore arrive
    non-finite, and NaN reaches MariaDB as an unquoted `nan` token that the
    driver refuses *before* sending SQL, failing the whole insert chunk.

    Nullable columns take None: line1/line2 remain the lossless source of
    truth, so a missing derived value costs nothing. A non-finite *mandatory*
    element (eccentricity, inclination, RAAN, argument of pericenter, mean
    anomaly) means there is no orbit to store — the elset cannot be
    propagated or compared — so the record is rejected instead of being
    stored as a row that would displace good data in generated output.
    """
    unusable = []
    for name, value in fields.items():
        if isinstance(value, float) and not isfinite(value):
            if _NULLABLE.get(name, True):
                fields[name] = None
            else:
                unusable.append(f"{name}={value}")
    if unusable:
        raise MalformedElsetError(
            f"non-finite mandatory element(s): {', '.join(sorted(unusable))}"
        )
    return fields


def _fields_from_satrec(sat: Satrec) -> dict:
    """Extract orbital element fields from a populated Satrec object.

    Non-finite values are cleaned at this boundary — the one place sgp4
    output enters the model — so no other code path has to know about NaN.
    """
    epoch = sat_epoch_datetime(sat).replace(tzinfo=None)
    return _clean_floats(
        dict(
            object_id=object_id(sat),
            norad_cat_id=sat.satnum,
            classification=sat.classification,
            epoch=epoch,
            mean_motion=mean_motion(sat),
            eccentricity=sat.ecco,
            inclination=degrees(sat.inclo),
            ra_of_asc_node=degrees(sat.nodeo),
            arg_of_pericenter=degrees(sat.argpo),
            mean_anomaly=degrees(sat.mo),
            ephemeris_type=sat.ephtype,
            element_set_no=sat.elnum,
            rev_at_epoch=sat.revnum,
            bstar=sat.bstar,
            mean_motion_dot=sat.ndot,
            mean_motion_ddot=sat.nddot,
            semimajor_axis=sat.a * sat.radiusearthkm,
            period=period(sat),
            apoapsis_alt=sat.alta * sat.radiusearthkm,
            periapsis_alt=sat.altp * sat.radiusearthkm,
        )
    )


class OmmMetadata(Base, Mixin):
    __tablename__ = "omm_metadata"

    tle_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("tle.id"), unique=True, index=True
    )
    tle: Mapped["TLE"] = relationship(back_populates="omm_metadata")

    object_name: Mapped[Optional[str]] = mapped_column(String(100))
    object_type: Mapped[Optional[str]] = mapped_column(String(20))
    country_code: Mapped[Optional[str]] = mapped_column(String(10))
    rcs_size: Mapped[Optional[str]] = mapped_column(String(10))
    launch_date: Mapped[Optional[str]] = mapped_column(String(10))
    site: Mapped[Optional[str]] = mapped_column(String(20))
    decay_date: Mapped[Optional[str]] = mapped_column(String(10))
    originator: Mapped[Optional[str]] = mapped_column(String(20))
    gp_id: Mapped[Optional[int]] = mapped_column()


class EpochDateState(Base):
    """When each epoch date last received a row — the date pass's scope.

    Derived data: always recomputable as ``date(epoch) -> MAX(created)`` over
    ``tle`` (``init-db`` does exactly that). Maintained incrementally by
    ingest so the generator can ask "which date files are older than the rows
    they should hold?" without scanning the element-set table.

    One row per *date*, so it grows by one row a day rather than with the
    catalog — which is what lets the generator check every date file on every
    run instead of guessing with a trailing window.

    The watermark must be written in the **same transaction** as the rows it
    describes (see `ingest._bulk_insert_ignore`). If a chunk could commit
    without its watermark, that date's files would never regenerate and only
    a rebuild would notice.
    """

    __tablename__ = "epoch_date_state"

    epoch_date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    last_created: Mapped[datetime.datetime]
    """Newest `created` among rows with an epoch on this date."""


class IngestFile(Base, Mixin):
    """State of a previously ingested source file, used to skip unchanged files."""

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
