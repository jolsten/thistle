import datetime
from math import degrees, pi, sqrt
from typing import Optional

from sgp4.alpha5 import to_alpha5
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from sqlalchemy import BigInteger, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def object_id(sat: Satrec) -> str:
    val = sat.intldesg
    if len(val) < 2:
        return val
    else:
        return ("20" if int(val[0:2]) < 57 else "19") + val


def period(sat: Satrec) -> float:
    """Calculate orbital period [min]"""
    a_km = sat.a * sat.radiusearthkm
    return 2 * pi * sqrt(a_km**3 / sat.mu)


def mean_motion(sat: Satrec) -> float:
    """Mean Motion [revs/day]"""
    return sat.no_kozai * 60 * 24 / (2 * pi)


class Base(DeclarativeBase):
    pass


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


class Mixin:
    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
        index=True,
        unique=True,
    )
    created: Mapped[datetime.datetime] = mapped_column(
        default=utcnow,
        index=True,
    )
    modified: Mapped[datetime.datetime] = mapped_column(
        default=utcnow,
        onupdate=utcnow,
        index=True,
    )


class TLE(Base, Mixin):
    __tablename__ = "tle"

    __table_args__ = (UniqueConstraint("line1", "line2"),)

    line1: Mapped[str] = mapped_column(String(70), index=True)
    line2: Mapped[str] = mapped_column(String(70), index=True)

    object_id: Mapped[str] = mapped_column(String(15))
    """International Designator"""
    norad_cat_id: Mapped[Optional[int]] = mapped_column(index=True)
    """NORAD Catalog Number ("Satellite Number")"""
    classification: Mapped[str] = mapped_column(String(1))
    """Classification Type"""

    # Element Set
    epoch: Mapped[datetime.datetime] = mapped_column(DateTime, index=True)
    """Epoch of Mean Keplerian elements (mandatory)"""
    mean_motion: Mapped[Optional[float]] = mapped_column(index=True)
    """Keplerian Mean Motion [rev/day]"""
    eccentricity: Mapped[float] = mapped_column(index=True)
    """Eccentricity"""
    inclination: Mapped[float] = mapped_column(index=True)
    """Inclination [deg]"""
    ra_of_asc_node: Mapped[float] = mapped_column(index=True)
    """Right Ascension of Ascending Node [deg]"""
    arg_of_pericenter: Mapped[float] = mapped_column(index=True)
    """Argument of Pericenter [deg]"""
    mean_anomaly: Mapped[float] = mapped_column(index=True)
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
    bstar: Mapped[Optional[float]]
    """Drag-like ballistic coefficient"""
    mean_motion_dot: Mapped[Optional[float]]
    """First Time Derivative of Mean Motion (i.e., a drag term)"""
    mean_motion_ddot: Mapped[Optional[float]]
    """Second Time Derivative of Mean Motion (i.e., a drag term)"""

    # Derived Values
    semimajor_axis: Mapped[Optional[float]]
    """Semi-major Axis [km]"""
    period: Mapped[Optional[float]]
    """Orbital Period [min]"""
    apoapsis_alt: Mapped[Optional[float]]
    """Apoapsis Altitude [km]"""
    periapsis_alt: Mapped[Optional[float]]
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

        return cls(line1=line1, line2=line2, **fields)


def _fields_from_satrec(sat: Satrec) -> dict:
    """Extract orbital element fields from a populated Satrec object."""
    epoch = sat_epoch_datetime(sat)
    return dict(
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


class OmmMetadata(Base, Mixin):
    __tablename__ = "omm_metadata"

    tle_id: Mapped[int] = mapped_column(ForeignKey("tle.id"), unique=True, index=True)
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
