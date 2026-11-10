"""Schema v2: record<->row conversion, derived values, dedup constraint."""

import hashlib

import pytest
from sqlalchemy import exc, insert, select
from sqlalchemy.orm import Session

from .conftest import TLES
from thistle_db import record as rec
from thistle_db.model import (
    TLE,
    derived_values,
    metadata_row_from_record,
    needs_metadata,
    record_from_row,
    tle_row_from_record,
    utcnow,
)

L1, L2 = TLES[0]


def test_row_from_record_identity_and_columns():
    record = rec.from_tle(L1, L2)
    row = tle_row_from_record(record, utcnow())
    assert row["record_hash"] == hashlib.sha256(
        f"{L1}\n{L2}".encode()
    ).digest()
    assert row["line1"] == L1 and row["line2"] == L2
    assert row["norad_cat_id"] == 22
    assert row["epoch"] == record.epoch
    assert row["mean_element_theory"] == "SGP4"
    # Derived convenience values: LEO object, sane ranges.
    assert 85 < row["period"] < 120
    assert 6500 < row["semimajor_axis"] < 7500
    assert row["periapsis_alt"] < row["apoapsis_alt"]


def test_derived_values_guard_bad_mean_motion():
    none = {"semimajor_axis": None, "period": None, "apoapsis_alt": None,
            "periapsis_alt": None}
    assert derived_values(None, 0.1) == none
    assert derived_values(0.0, 0.1) == none
    assert derived_values(-1.0, 0.1) == none


def test_needs_metadata_false_for_bare_tle():
    record = rec.from_tle(L1, L2)
    assert not needs_metadata(record)


def test_needs_metadata_true_for_omm(omm_fields_minimal):
    record = rec.from_omm({**omm_fields_minimal, "OBJECT_NAME": "THING"})
    assert needs_metadata(record)


def test_semantics_equal_to_implicit_store_as_null(omm_fields_minimal):
    record = rec.from_omm(
        {**omm_fields_minimal, "OBJECT_NAME": "X", "REF_FRAME": "TEME"}
    )
    row = metadata_row_from_record(record, tle_id=1, created=utcnow())
    assert row["ref_frame"] is None  # equal to implicit -> NULL
    record2 = rec.from_omm(
        {**omm_fields_minimal, "OBJECT_NAME": "X", "REF_FRAME": "GCRF"}
    )
    row2 = metadata_row_from_record(record2, tle_id=1, created=utcnow())
    assert row2["ref_frame"] == "GCRF"  # genuine override is stored


def test_record_from_row_round_trip_line_full():
    record = rec.from_tle(L1, L2)
    row = tle_row_from_record(record, utcnow())
    row.pop("created")
    tle = TLE(**row)
    back = record_from_row(tle, None)
    assert back.record_hash == record.record_hash
    assert back.tle_line1 == L1 and back.tle_line2 == L2
    assert back.epoch == record.epoch
    assert back.norad_cat_id == record.norad_cat_id
    # Implicit semantics materialize identically.
    assert back.ref_frame == "TEME"
    assert back.time_system == "UTC"


@pytest.fixture
def omm_fields_minimal():
    return {
        "NORAD_CAT_ID": "900",
        "EPOCH": "2025-04-17T22:33:21.123456",
        "MEAN_MOTION": "13.73",
        "ECCENTRICITY": "0.0026",
        "INCLINATION": "90.2",
        "RA_OF_ASC_NODE": "60.7",
        "ARG_OF_PERICENTER": "151.6",
        "MEAN_ANOMALY": "208.6",
    }


def test_unique_epoch_record_hash(db_session: Session):
    record = rec.from_tle(L1, L2)
    row = tle_row_from_record(record, utcnow())
    db_session.execute(insert(TLE.__table__), [row])
    db_session.commit()
    with pytest.raises(exc.IntegrityError):
        db_session.execute(insert(TLE.__table__), [row])
        db_session.commit()
    db_session.rollback()
    count = len(db_session.execute(select(TLE)).scalars().all())
    assert count == 1
