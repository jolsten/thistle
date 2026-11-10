"""Full-fidelity export — backup, interchange, and the migration path.

Every record reconstitutes losslessly from the store CSV (schema changes
ship as ``export`` → ``init-db --drop --defer-indexes`` → ``ingest`` →
``init-db``), so there is deliberately no migration framework. Four
formats:

- **csv** — the store-column projection (``record.STORE_COLUMNS``), one
  row per record, re-ingestable by ``ingest`` (it reads as OMM CSV).
  Derived values are excluded by construction: they are non-canonical.
- **tle** — verbatim two-line concat for legacy consumers (line-full
  records only).
- **omm-xml** / **omm-kvn** — CCSDS OMM for standards-adherent
  interchange, version 2.0 by default (STK's documented acceptance;
  Space-Track emits 3.0 — add a flag when someone needs it). Core
  keywords carry the elements; Space-Track extras and the verbatim lines
  ride in ``userDefinedParameters``, so the XML export is near
  full-fidelity too.
"""

import csv
import pathlib
from typing import Iterator, TextIO
from xml.sax.saxutils import escape

from sqlalchemy import select
from sqlalchemy.orm import Session

from thistle_db.model import OmmMetadata, TLE, record_from_row, utcnow
from thistle_db.record import GPRecord, STORE_COLUMNS, to_csv_dict

__all__ = ["EXPORT_FORMATS", "export"]

EXPORT_FORMATS = ("csv", "tle", "omm-xml", "omm-kvn")

_OMM_VERSION = "2.0"
_DEFAULT_ORIGINATOR = "THISTLE-DB"

# (OMM keyword, GPRecord attribute) for the CCSDS core sections.
_MEAN_ELEMENTS = (
    ("EPOCH", "epoch"),
    ("MEAN_MOTION", "mean_motion"),
    ("ECCENTRICITY", "eccentricity"),
    ("INCLINATION", "inclination"),
    ("RA_OF_ASC_NODE", "ra_of_asc_node"),
    ("ARG_OF_PERICENTER", "arg_of_pericenter"),
    ("MEAN_ANOMALY", "mean_anomaly"),
)
_TLE_PARAMETERS = (
    ("EPHEMERIS_TYPE", "ephemeris_type"),
    ("CLASSIFICATION_TYPE", "classification_type"),
    ("NORAD_CAT_ID", "norad_cat_id"),
    ("ELEMENT_SET_NO", "element_set_no"),
    ("REV_AT_EPOCH", "rev_at_epoch"),
    ("BSTAR", "bstar"),
    ("MEAN_MOTION_DOT", "mean_motion_dot"),
    ("MEAN_MOTION_DDOT", "mean_motion_ddot"),
)
# Space-Track extras + verbatim lines: not CCSDS core keywords, carried as
# USER_DEFINED parameters so the export stays near full-fidelity.
_USER_DEFINED = (
    ("RECORD_HASH", "record_hash"),
    ("OBJECT_TYPE", "object_type"),
    ("RCS_SIZE", "rcs_size"),
    ("COUNTRY_CODE", "country_code"),
    ("LAUNCH_DATE", "launch_date"),
    ("SITE", "site"),
    ("DECAY_DATE", "decay_date"),
    ("GP_ID", "gp_id"),
    ("FILE", "file_id"),
    ("TLE_LINE0", "tle_line0"),
    ("TLE_LINE1", "tle_line1"),
    ("TLE_LINE2", "tle_line2"),
)


def iter_store_records(session: Session) -> Iterator[GPRecord]:
    """Stream every record (with sidecar), ordered (norad, epoch, id)."""
    stmt = (
        select(TLE, OmmMetadata)
        .outerjoin(OmmMetadata, OmmMetadata.tle_id == TLE.id)
        .order_by(TLE.norad_cat_id, TLE.epoch, TLE.id)
        .execution_options(yield_per=5000)
    )
    for tle, metadata in session.execute(stmt):
        yield record_from_row(tle, metadata)


def export(session: Session, fmt: str, path: pathlib.Path) -> int:
    """Write the whole store to `path` in `fmt`. Returns the record count."""
    writers = {
        "csv": _export_csv,
        "tle": _export_tle,
        "omm-xml": _export_omm_xml,
        "omm-kvn": _export_omm_kvn,
    }
    return writers[fmt](session, path)


def _export_csv(session: Session, path: pathlib.Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STORE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in iter_store_records(session):
            writer.writerow(to_csv_dict(record))
            count += 1
    return count


def _export_tle(session: Session, path: pathlib.Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        for record in iter_store_records(session):
            if record.tle_line1 and record.tle_line2:
                f.write(f"{record.tle_line1}\n{record.tle_line2}\n")
                count += 1
    return count


def _kvn_value(record: GPRecord, attr: str):
    value = getattr(record, attr)
    if value is None:
        return None
    if attr in ("epoch", "creation_date"):
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f")
    if attr in ("mean_motion", "eccentricity", "inclination", "ra_of_asc_node",
                "arg_of_pericenter", "mean_anomaly", "bstar",
                "mean_motion_dot", "mean_motion_ddot"):
        return repr(value)
    return str(value)


def _record_header(record: GPRecord) -> tuple[str, str]:
    creation = record.creation_date or utcnow()
    return (
        creation.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        record.originator or _DEFAULT_ORIGINATOR,
    )


def _record_metadata(record: GPRecord) -> list[tuple[str, str]]:
    return [
        ("OBJECT_NAME", record.object_name or "UNKNOWN"),
        ("OBJECT_ID", record.object_id or "UNKNOWN"),
        ("CENTER_NAME", record.center_name),
        ("REF_FRAME", record.ref_frame),
        ("TIME_SYSTEM", record.time_system),
        ("MEAN_ELEMENT_THEORY", record.mean_element_theory),
    ]


def _export_omm_xml(session: Session, path: pathlib.Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<ndm>\n')
        for record in iter_store_records(session):
            _write_omm_xml_record(f, record)
            count += 1
        f.write("</ndm>\n")
    return count


def _write_omm_xml_record(f: TextIO, record: GPRecord) -> None:
    def tag(name: str, value) -> str:
        return f"      <{name}>{escape(str(value))}</{name}>\n"

    creation_date, originator = _record_header(record)
    f.write(f'  <omm id="CCSDS_OMM_VERS" version="{_OMM_VERSION}">\n')
    f.write("    <header>\n")
    f.write(f"      <CREATION_DATE>{creation_date}</CREATION_DATE>\n")
    f.write(f"      <ORIGINATOR>{escape(originator)}</ORIGINATOR>\n")
    f.write("    </header>\n    <body>\n    <segment>\n    <metadata>\n")
    for name, value in _record_metadata(record):
        f.write(tag(name, value))
    f.write("    </metadata>\n    <data>\n    <meanElements>\n")
    for name, attr in _MEAN_ELEMENTS:
        value = _kvn_value(record, attr)
        if value is not None:
            f.write(tag(name, value))
    f.write("    </meanElements>\n    <tleParameters>\n")
    for name, attr in _TLE_PARAMETERS:
        value = _kvn_value(record, attr)
        if value is not None:
            f.write(tag(name, value))
    f.write("    </tleParameters>\n")
    user_defined = [
        (name, _kvn_value(record, attr))
        for name, attr in _USER_DEFINED
        if getattr(record, attr) is not None
    ]
    if user_defined:
        f.write("    <userDefinedParameters>\n")
        for name, value in user_defined:
            f.write(
                f'      <USER_DEFINED parameter="{name}">'
                f"{escape(str(value))}</USER_DEFINED>\n"
            )
        f.write("    </userDefinedParameters>\n")
    f.write("    </data>\n    </segment>\n    </body>\n  </omm>\n")


def _export_omm_kvn(session: Session, path: pathlib.Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        for record in iter_store_records(session):
            if count:
                f.write("\n")
            _write_omm_kvn_record(f, record)
            count += 1
    return count


def _write_omm_kvn_record(f: TextIO, record: GPRecord) -> None:
    creation_date, originator = _record_header(record)
    lines = [
        ("CCSDS_OMM_VERS", _OMM_VERSION),
        ("CREATION_DATE", creation_date),
        ("ORIGINATOR", originator),
    ]
    lines.extend(_record_metadata(record))
    for name, attr in _MEAN_ELEMENTS + _TLE_PARAMETERS:
        value = _kvn_value(record, attr)
        if value is not None:
            lines.append((name, value))
    for name, attr in _USER_DEFINED:
        value = _kvn_value(record, attr)
        if value is not None:
            lines.append((f"USER_DEFINED_{name}", value))
    width = max(len(name) for name, _ in lines)
    for name, value in lines:
        f.write(f"{name.ljust(width)} = {value}\n")
