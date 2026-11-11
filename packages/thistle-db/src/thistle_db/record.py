"""The canonical GP record — the one unit of data in the target design.

An OMM-superset record modeled on Space-Track's published GP class, carrying
the verbatim TLE lines as fields when they exist. Everything in the system
produces, stores, or projects this record; see CLAUDE.md "The canonical GP
record" for the normative spec (field classes, identity, merge precedence).

This module is pure: no SQLAlchemy, no filesystem, no config. Element values
are derived from sgp4's parse (the same parser the rest of the system trusts)
and converted to the TLE-text/Space-Track units — the verbatim lines remain
the precision authority, exactly as for the database columns.
"""

import dataclasses
import datetime
import hashlib
from math import degrees, isfinite, pi
from typing import Optional, TypeGuard

from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

__all__ = [
    "GPRecord",
    "MalformedRecordError",
    "STORE_COLUMNS",
    "checksum_ok",
    "epoch_from_lines",
    "from_tle",
    "from_omm",
    "implicit_semantics",
    "merge",
    "line_hash",
    "semantic_hash",
    "tle_checksum",
]

# sgp4 stores mean-motion derivatives in radians/minute^n; the TLE text (and
# Space-Track's GP fields, which mirror it) use revs/day^n with the historical
# /2 and /6 factors already applied. xpdotp converts one way; these go back.
_XPDOTP = 1440.0 / (2.0 * pi)

# Implicit semantics for TLE-only deliveries, keyed by EPHEMERIS_TYPE — never
# hardcoded constants, so the first SGP4-XP (type 4) TLE that arrives is
# stored truthfully. Unknown types fall back to SGP4, matching how every
# consumer treats an unrecognized type character.
_THEORY_BY_EPHTYPE = {0: "SGP4", 2: "SGP4", 4: "SGP4-XP"}
_IMPLICIT_REF_FRAME = "TEME"
_IMPLICIT_TIME_SYSTEM = "UTC"
_IMPLICIT_CENTER_NAME = "EARTH"


class MalformedRecordError(ValueError):
    """A record parsed (or would have), but cannot be stored.

    A mandatory element is not a finite number or its TLE column span is
    blank/non-numeric (there is no orbit to store), or the record has no
    NORAD ID (no home in a per-object store). Callers catch this per
    record so one bad record never costs a file.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class GPRecord:
    # Identity
    record_hash: str
    """Lowercase hex sha256. Lines present: hash of 'line1\\nline2'.
    Line-less: the frozen thistle-gp1 semantic serialization."""

    # Parsed or delivered
    norad_cat_id: int
    epoch: datetime.datetime
    """Naive UTC, microseconds. Immutable — part of identity."""
    object_id: Optional[str] = None
    mean_motion: Optional[float] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    ra_of_asc_node: float = 0.0
    arg_of_pericenter: float = 0.0
    mean_anomaly: float = 0.0
    bstar: Optional[float] = None
    mean_motion_dot: Optional[float] = None
    mean_motion_ddot: Optional[float] = None
    ephemeris_type: int = 0
    classification_type: str = "U"
    element_set_no: Optional[int] = None
    rev_at_epoch: Optional[int] = None

    # Implicit semantics (materialized per EPHEMERIS_TYPE for TLE deliveries)
    mean_element_theory: str = "SGP4"
    ref_frame: str = _IMPLICIT_REF_FRAME
    time_system: str = _IMPLICIT_TIME_SYSTEM
    center_name: str = _IMPLICIT_CENTER_NAME

    # Metadata — null for TLE-only delivery, never inferred
    object_name: Optional[str] = None
    object_type: Optional[str] = None
    rcs_size: Optional[str] = None
    country_code: Optional[str] = None
    launch_date: Optional[str] = None
    site: Optional[str] = None
    decay_date: Optional[str] = None

    # Provenance — null for TLE-only delivery
    gp_id: Optional[int] = None
    file_id: Optional[int] = None
    creation_date: Optional[datetime.datetime] = None
    originator: Optional[str] = None

    # Verbatim rendering
    tle_line0: Optional[str] = None
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None
    """line1/line2 are immutable once set — for line-present records they
    define record_hash. line0 (the 3LE name line) is fill-if-null only;
    renames land in object_name via the merge rule."""


# --- Identity -------------------------------------------------------------


def line_hash(line1: str, line2: str) -> str:
    """Lowercase hex sha256 of the exact two-line text — the dedup key.

    The hex twin of model.compute_line_hash (which returns the raw digest for
    the database's BINARY(32) column); both spell the same identity.
    """
    return hashlib.sha256(f"{line1}\n{line2}".encode("utf-8")).hexdigest()


def _hex_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if value == 0.0:  # normalize -0.0
        value = 0.0
    return float(value).hex()


def semantic_hash(
    norad_cat_id: int,
    epoch: datetime.datetime,
    mean_element_theory: str,
    mean_motion: Optional[float],
    eccentricity: float,
    inclination: float,
    ra_of_asc_node: float,
    arg_of_pericenter: float,
    mean_anomaly: float,
    bstar: Optional[float],
    mean_motion_dot: Optional[float],
    mean_motion_ddot: Optional[float],
) -> str:
    """The frozen thistle-gp1 identity for line-less records.

    Hashes the *parsed doubles* (C99 hex-float form), so formatting variants
    of the same value ('.5', '0.5', '5.0E-1') hash equal while genuine
    last-ulp differences coexist as separate records. The leading tag
    domain-separates this hash space from line hashes and names the
    serialization version — changing anything below is a major store-format
    bump.
    """
    parts = [
        "thistle-gp1",
        str(int(norad_cat_id)),
        epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        mean_element_theory.upper(),
        _hex_float(mean_motion),
        _hex_float(eccentricity),
        _hex_float(inclination),
        _hex_float(ra_of_asc_node),
        _hex_float(arg_of_pericenter),
        _hex_float(mean_anomaly),
        _hex_float(bstar),
        _hex_float(mean_motion_dot),
        _hex_float(mean_motion_ddot),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# --- Validation -----------------------------------------------------------

# There is no orbit to store without these; a record carrying a non-finite
# mandatory element is rejected rather than stored (it cannot be propagated
# or compared, and would displace good data in outputs).
_MANDATORY_ELEMENTS = (
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
)
_NULLABLE_FLOATS = (
    "mean_motion",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
)


def _clean_floats(fields: dict) -> dict:
    """Non-finite nullable floats become None; non-finite mandatory rejects.

    The record-level twin of model._clean_floats: sgp4's C-strtod parser
    accepts literal 'nan'/'inf' text and corrupt fixed-width fields shift to
    the same effect, so every float can arrive non-finite.
    """
    unusable = []
    for name in _MANDATORY_ELEMENTS:
        value = fields.get(name)
        if not (isinstance(value, float) and isfinite(value)):
            unusable.append(f"{name}={value}")
    if unusable:
        raise MalformedRecordError(
            f"non-finite mandatory element(s): {', '.join(sorted(unusable))}"
        )
    for name in _NULLABLE_FLOATS:
        value = fields.get(name)
        if value is not None and not isfinite(value):
            fields[name] = None
    return fields


# --- Construction ---------------------------------------------------------


def implicit_semantics(ephemeris_type: int) -> dict:
    """Implicit semantics for a TLE-only delivery of this ephemeris type.

    Also the fallback the database store compares delivered OMM semantics
    against: values equal to these store as NULL overrides.
    """
    return dict(
        mean_element_theory=_THEORY_BY_EPHTYPE.get(ephemeris_type, "SGP4"),
        ref_frame=_IMPLICIT_REF_FRAME,
        time_system=_IMPLICIT_TIME_SYSTEM,
        center_name=_IMPLICIT_CENTER_NAME,
    )


_implicit_semantics = implicit_semantics  # internal alias


def epoch_from_lines(line1: str, line2: str) -> datetime.datetime:
    """Epoch of a TLE exactly as stored (naive UTC, microseconds).

    Byte-identical to the `epoch` column / GPRecord.epoch for the same
    lines — used to compare on-disk output files against records without
    risking a microsecond of parse skew.
    """
    sat = Satrec.twoline2rv(line1, line2)
    return sat_epoch_datetime(sat).replace(tzinfo=None)


def tle_checksum(line: str) -> int:
    """Mod-10 TLE checksum over columns 1-68: digits summed, '-' counts 1."""
    total = 0
    for char in line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def checksum_ok(line: str) -> bool:
    """Verify column 69 against the computed checksum.

    Lenient where the digit is absent: lines shorter than 69 characters
    pass (some tools emit trimmed lines; historical archives contain
    them). A full-length line whose 69th column is not a digit fails —
    that column can hold nothing else.
    """
    if len(line) < 69:
        return True
    check = line[68]
    if not check.isdigit():
        return False
    return int(check) == tle_checksum(line)


def _object_id_from_intldesg(intldesg: str) -> Optional[str]:
    """Expand the TLE's truncated international designator (YYNNNPPP form).

    Same windowing as model.object_id; non-numeric placeholders (e.g. 'TBA')
    are stored verbatim rather than rejecting an otherwise valid record.
    """
    val = intldesg.strip()
    if not val:
        return None
    if len(val) < 2:
        return val
    try:
        year = int(val[0:2])
    except ValueError:
        return val
    century = "20" if year < 57 else "19"
    return f"{century}{val[0:2]}-{val[2:]}"


def _elements_from_satrec(sat: Satrec) -> dict:
    return dict(
        norad_cat_id=sat.satnum,
        epoch=sat_epoch_datetime(sat).replace(tzinfo=None),
        object_id=_object_id_from_intldesg(sat.intldesg),
        mean_motion=sat.no_kozai * _XPDOTP,
        eccentricity=sat.ecco,
        inclination=degrees(sat.inclo),
        ra_of_asc_node=degrees(sat.nodeo),
        arg_of_pericenter=degrees(sat.argpo),
        mean_anomaly=degrees(sat.mo),
        bstar=sat.bstar,
        mean_motion_dot=sat.ndot * _XPDOTP * 1440.0,
        mean_motion_ddot=sat.nddot * _XPDOTP * 1440.0 * 1440.0,
        ephemeris_type=sat.ephtype,
        classification_type=sat.classification,
        element_set_no=sat.elnum,
        rev_at_epoch=sat.revnum,
    )


def _name_from_line0(line0: str) -> Optional[str]:
    name = line0.rstrip()
    if name.startswith("0 "):  # 3LE convention
        name = name[2:]
    return name or None


# twoline2rv is lenient by construction — C strtod over fixed columns — so
# a blank or truncated field parses as 0.0 and a damaged line can "succeed",
# silently storing a fictitious element (inclination 0°, mean motion 0,
# epoch at the day-number baseline). Require the identity- and element-
# bearing columns to be present and numeric before parsing. Spans are
# Python slices of the 1-indexed TLE column spec.
_REQUIRED_FIELDS: tuple[tuple[int, int, int, str], ...] = (
    (1, 18, 32, "epoch"),
    (2, 8, 16, "inclination"),
    (2, 17, 25, "ra_of_asc_node"),
    (2, 26, 33, "eccentricity"),  # digits only — decimal point implied
    (2, 34, 42, "arg_of_pericenter"),
    (2, 43, 51, "mean_anomaly"),
    (2, 52, 63, "mean_motion"),
)


def _finite_text(text: str) -> bool:
    try:
        return isfinite(float(text))
    except ValueError:
        return False


def _check_required_fields(line1: str, line2: str) -> None:
    if not line1[2:7].strip():
        raise MalformedRecordError("no NORAD ID (blank satellite number)")
    bad = []
    for line_no, start, stop, field in _REQUIRED_FIELDS:
        text = (line1 if line_no == 1 else line2)[start:stop].strip()
        ok = text.isdigit() if field == "eccentricity" else _finite_text(text)
        if not ok:
            bad.append(field)
    if bad:
        raise MalformedRecordError(
            f"blank or malformed mandatory field(s): {', '.join(bad)}"
        )


def from_tle(*lines: str) -> GPRecord:
    """Build a record from a delivered TLE (2 or 3 lines, verbatim).

    The sparse case: elements parsed, implicit semantics materialized per
    EPHEMERIS_TYPE, metadata and provenance null (OBJECT_NAME from a 3LE name
    line only). The delivered bytes are kept verbatim in tle_line1/2.
    """
    if len(lines) == 2:
        line0, (line1, line2) = None, lines
    elif len(lines) == 3:
        line0, line1, line2 = lines
    else:
        raise ValueError("Must provide 2 or 3 lines")

    _check_required_fields(line1, line2)
    sat = Satrec.twoline2rv(line1, line2)
    fields = _clean_floats(_elements_from_satrec(sat))
    return GPRecord(
        record_hash=line_hash(line1, line2),
        object_name=_name_from_line0(line0) if line0 is not None else None,
        tle_line0=line0.rstrip() if line0 is not None else None,
        tle_line1=line1,
        tle_line2=line2,
        **fields,
        **_implicit_semantics(fields["ephemeris_type"]),
    )


def _parse_datetime(text: Optional[str]) -> Optional[datetime.datetime]:
    if not text:
        return None
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1]
    dt = datetime.datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt


def _opt_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_hash(text: Optional[str]) -> "TypeGuard[str]":
    """A well-formed lowercase-hex sha256, as line_hash/semantic_hash emit."""
    if text is None or len(text) != 64:
        return False
    return all(c in "0123456789abcdef" for c in text)


def _opt_int(value) -> Optional[int]:
    text = _opt_str(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _opt_float(value) -> Optional[float]:
    text = _opt_str(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return float("nan")  # malformed text → cleaned like a parsed NaN


def from_omm(fields: dict) -> GPRecord:
    """Build a record from a Space-Track-style OMM field dict.

    Accepts the flat uppercase-keyed form every reader produces (JSON, CSV,
    XML), values as strings or already-typed. Near-passthrough: delivered
    metadata, provenance, and declared semantics are kept; embedded TLE lines
    (TLE_LINE1/2) make this a line-hashed record, otherwise identity is the
    thistle-gp1 semantic hash.
    """
    norad = _opt_int(fields.get("NORAD_CAT_ID"))
    if norad is None:
        raise MalformedRecordError(
            "no NORAD ID (records are stored per object)"
        )
    epoch = _parse_datetime(_opt_str(fields.get("EPOCH")))
    if epoch is None:
        raise MalformedRecordError("no EPOCH")

    ephemeris_type = _opt_int(fields.get("EPHEMERIS_TYPE")) or 0
    implicit = _implicit_semantics(ephemeris_type)
    elements = _clean_floats(
        dict(
            mean_motion=_opt_float(fields.get("MEAN_MOTION")),
            eccentricity=_opt_float(fields.get("ECCENTRICITY")),
            inclination=_opt_float(fields.get("INCLINATION")),
            ra_of_asc_node=_opt_float(fields.get("RA_OF_ASC_NODE")),
            arg_of_pericenter=_opt_float(fields.get("ARG_OF_PERICENTER")),
            mean_anomaly=_opt_float(fields.get("MEAN_ANOMALY")),
            bstar=_opt_float(fields.get("BSTAR")),
            mean_motion_dot=_opt_float(fields.get("MEAN_MOTION_DOT")),
            mean_motion_ddot=_opt_float(fields.get("MEAN_MOTION_DDOT")),
        )
    )

    line1 = _opt_str(fields.get("TLE_LINE1"))
    line2 = _opt_str(fields.get("TLE_LINE2"))
    theory = _opt_str(fields.get("MEAN_ELEMENT_THEORY")) or implicit[
        "mean_element_theory"
    ]
    if line1 and line2:
        record_hash = line_hash(line1, line2)
    elif _is_hash(delivered := _opt_str(fields.get("RECORD_HASH"))):
        # A line-less record re-ingested from thistle's own export: the
        # delivered identity wins, so export → restore round-trips exactly
        # even though single-precision element columns re-serialize with
        # fewer bits than the doubles the original semantic hash saw.
        # Line-present records never take this path — their identity is
        # always the line text.
        line1 = line2 = None
        record_hash = delivered  # type: ignore[assignment]
    else:
        line1 = line2 = None
        record_hash = semantic_hash(
            norad,
            epoch,
            theory,
            elements["mean_motion"],
            elements["eccentricity"],
            elements["inclination"],
            elements["ra_of_asc_node"],
            elements["arg_of_pericenter"],
            elements["mean_anomaly"],
            elements["bstar"],
            elements["mean_motion_dot"],
            elements["mean_motion_ddot"],
        )

    return GPRecord(
        record_hash=record_hash,
        norad_cat_id=norad,
        epoch=epoch,
        object_id=_opt_str(fields.get("OBJECT_ID")),
        ephemeris_type=ephemeris_type,
        classification_type=_opt_str(fields.get("CLASSIFICATION_TYPE")) or "U",
        element_set_no=_opt_int(fields.get("ELEMENT_SET_NO")),
        rev_at_epoch=_opt_int(fields.get("REV_AT_EPOCH")),
        mean_element_theory=theory,
        ref_frame=_opt_str(fields.get("REF_FRAME")) or implicit["ref_frame"],
        time_system=_opt_str(fields.get("TIME_SYSTEM"))
        or implicit["time_system"],
        center_name=_opt_str(fields.get("CENTER_NAME"))
        or implicit["center_name"],
        object_name=_opt_str(fields.get("OBJECT_NAME")),
        object_type=_opt_str(fields.get("OBJECT_TYPE")),
        rcs_size=_opt_str(fields.get("RCS_SIZE")),
        country_code=_opt_str(fields.get("COUNTRY_CODE")),
        launch_date=_opt_str(fields.get("LAUNCH_DATE")),
        site=_opt_str(fields.get("SITE")),
        decay_date=_opt_str(fields.get("DECAY_DATE")),
        gp_id=_opt_int(fields.get("GP_ID")),
        file_id=_opt_int(fields.get("FILE")),
        creation_date=_parse_datetime(_opt_str(fields.get("CREATION_DATE"))),
        originator=_opt_str(fields.get("ORIGINATOR")),
        tle_line0=_opt_str(fields.get("TLE_LINE0")),
        tle_line1=line1,
        tle_line2=line2,
        **elements,
    )


# --- Enrichment merge -----------------------------------------------------

_IMMUTABLE_FIELDS = frozenset(
    {"record_hash", "epoch", "tle_line1", "tle_line2"}
)
_FILL_ONLY_FIELDS = frozenset({"tle_line0"})


def merge(existing: GPRecord, new: GPRecord) -> GPRecord:
    """Merge a redelivery of the same record ('TLE first, OMM later').

    Field precedence (CLAUDE.md 'Enrichment merge'):
      1. Non-null beats null.
      2. Both non-null and unequal: newest CREATION_DATE wins field-wise;
         null CREATION_DATE loses to non-null; full tie → last-ingested
         (`new`) wins.
      3. record_hash, epoch, and the verbatim lines never change; line0 is
         fill-if-null (renames land in object_name).
    """
    if existing.record_hash != new.record_hash:
        raise ValueError(
            "merge requires identical record_hash: "
            f"{existing.record_hash} != {new.record_hash}"
        )

    if new.creation_date is None:
        prefer_new = existing.creation_date is None  # both null → last wins
    elif existing.creation_date is None:
        prefer_new = True
    else:
        prefer_new = new.creation_date >= existing.creation_date

    changes = {}
    for field in dataclasses.fields(GPRecord):
        name = field.name
        if name in _IMMUTABLE_FIELDS:
            continue
        old_value = getattr(existing, name)
        new_value = getattr(new, name)
        if name in _FILL_ONLY_FIELDS:
            if old_value is None and new_value is not None:
                changes[name] = new_value
            continue
        if old_value is None and new_value is not None:
            changes[name] = new_value
        elif (
            old_value is not None
            and new_value is not None
            and new_value != old_value
            and prefer_new
        ):
            changes[name] = new_value
    if not changes:
        return existing
    return dataclasses.replace(existing, **changes)


# --- Store CSV serialization ----------------------------------------------

# (field name, store column). Column vocabulary is CCSDS OMM keywords; the
# set is anchored to Space-Track's GP class (minus derived values, plus
# RECORD_HASH). Header rows are authoritative per file; additive-only within
# a store-format major version.
_FIELD_COLUMNS = [
    ("record_hash", "RECORD_HASH"),
    ("norad_cat_id", "NORAD_CAT_ID"),
    ("object_id", "OBJECT_ID"),
    ("epoch", "EPOCH"),
    ("mean_motion", "MEAN_MOTION"),
    ("eccentricity", "ECCENTRICITY"),
    ("inclination", "INCLINATION"),
    ("ra_of_asc_node", "RA_OF_ASC_NODE"),
    ("arg_of_pericenter", "ARG_OF_PERICENTER"),
    ("mean_anomaly", "MEAN_ANOMALY"),
    ("bstar", "BSTAR"),
    ("mean_motion_dot", "MEAN_MOTION_DOT"),
    ("mean_motion_ddot", "MEAN_MOTION_DDOT"),
    ("ephemeris_type", "EPHEMERIS_TYPE"),
    ("classification_type", "CLASSIFICATION_TYPE"),
    ("element_set_no", "ELEMENT_SET_NO"),
    ("rev_at_epoch", "REV_AT_EPOCH"),
    ("mean_element_theory", "MEAN_ELEMENT_THEORY"),
    ("ref_frame", "REF_FRAME"),
    ("time_system", "TIME_SYSTEM"),
    ("center_name", "CENTER_NAME"),
    ("object_name", "OBJECT_NAME"),
    ("object_type", "OBJECT_TYPE"),
    ("rcs_size", "RCS_SIZE"),
    ("country_code", "COUNTRY_CODE"),
    ("launch_date", "LAUNCH_DATE"),
    ("site", "SITE"),
    ("decay_date", "DECAY_DATE"),
    ("gp_id", "GP_ID"),
    ("file_id", "FILE"),
    ("creation_date", "CREATION_DATE"),
    ("originator", "ORIGINATOR"),
    ("tle_line0", "TLE_LINE0"),
    ("tle_line1", "TLE_LINE1"),
    ("tle_line2", "TLE_LINE2"),
]

STORE_COLUMNS = [column for _, column in _FIELD_COLUMNS]

_INT_FIELDS = {
    "norad_cat_id",
    "ephemeris_type",
    "element_set_no",
    "rev_at_epoch",
    "gp_id",
    "file_id",
}
_FLOAT_FIELDS = {
    "mean_motion",
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
}
_DATETIME_FIELDS = {"epoch", "creation_date"}


def to_csv_dict(record: GPRecord) -> dict:
    """Render as a store CSV row dict (column → string; None → '').

    Floats use repr (shortest round-trip) — the store carries the exact
    doubles; datetimes are ISO-8601 with microseconds.
    """
    row = {}
    for name, column in _FIELD_COLUMNS:
        value = getattr(record, name)
        if value is None:
            row[column] = ""
        elif name in _DATETIME_FIELDS:
            row[column] = value.strftime("%Y-%m-%dT%H:%M:%S.%f")
        elif name in _FLOAT_FIELDS:
            row[column] = repr(value)
        else:
            row[column] = str(value)
    return row


def from_csv_dict(row: dict) -> GPRecord:
    """Inverse of to_csv_dict. Unknown columns are ignored and missing
    nullable columns read as null — the reader half of the store's
    additive-only format versioning."""
    fields = {}
    for name, column in _FIELD_COLUMNS:
        text = row.get(column, "")
        if text is None or text == "":
            continue
        if name in _DATETIME_FIELDS:
            fields[name] = _parse_datetime(text)
        elif name in _FLOAT_FIELDS:
            fields[name] = float(text)
        elif name in _INT_FIELDS:
            fields[name] = int(text)
        else:
            fields[name] = text
    return GPRecord(**fields)


def to_omm_dict(record: GPRecord) -> dict:
    """Project onto the 17-column CelesTrak/STK OMM CSV form
    (reader.OMM_CSV_FIELDS order) — the STK-compatible output row. Every
    record is projectable: all 17 fields are populated even for TLE-only
    deliveries (OBJECT_NAME may be empty for a 2LE)."""
    full = to_csv_dict(record)
    return {
        "OBJECT_NAME": full["OBJECT_NAME"],
        "OBJECT_ID": full["OBJECT_ID"],
        "EPOCH": full["EPOCH"],
        "MEAN_MOTION": full["MEAN_MOTION"],
        "ECCENTRICITY": full["ECCENTRICITY"],
        "INCLINATION": full["INCLINATION"],
        "RA_OF_ASC_NODE": full["RA_OF_ASC_NODE"],
        "ARG_OF_PERICENTER": full["ARG_OF_PERICENTER"],
        "MEAN_ANOMALY": full["MEAN_ANOMALY"],
        "EPHEMERIS_TYPE": full["EPHEMERIS_TYPE"],
        "CLASSIFICATION_TYPE": full["CLASSIFICATION_TYPE"],
        "NORAD_CAT_ID": full["NORAD_CAT_ID"],
        "ELEMENT_SET_NO": full["ELEMENT_SET_NO"],
        "REV_AT_EPOCH": full["REV_AT_EPOCH"],
        "BSTAR": full["BSTAR"],
        "MEAN_MOTION_DOT": full["MEAN_MOTION_DOT"],
        "MEAN_MOTION_DDOT": full["MEAN_MOTION_DDOT"],
    }
