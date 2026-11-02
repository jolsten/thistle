"""Unit tests for the canonical GP record (record.py).

The cross-check fixture is data/one.json — a real Space-Track OMM whose
embedded TLE lines let the same element set be built through both
constructors and compared.
"""

import dataclasses
import datetime
import json
import pathlib

import pytest

from thistle_db import record as rec
from thistle_db.reader import OMM_CSV_FIELDS

DATA = pathlib.Path(__file__).parent / "data"

L0 = "0 VANGUARD 2"
L1 = "1    11U 59001A   25109.79083689  .00003193  00000-0  16873-2 0  9997"
L2 = "2    11  32.8658 246.7104 1448126 168.7229 194.9782 11.89628311826759"


@pytest.fixture()
def omm_fields() -> dict:
    return json.loads((DATA / "one.json").read_text())


# --- from_tle -------------------------------------------------------------


def test_from_tle_parses_elements():
    r = rec.from_tle(L1, L2)
    assert r.norad_cat_id == 11
    assert r.object_id == "1959-001A"
    assert r.epoch == datetime.datetime(2025, 4, 19, 18, 58, 48, 307296)
    assert r.mean_motion == pytest.approx(11.89628311, abs=1e-8)
    assert r.eccentricity == pytest.approx(0.1448126)
    assert r.inclination == pytest.approx(32.8658)
    assert r.ra_of_asc_node == pytest.approx(246.7104)
    assert r.arg_of_pericenter == pytest.approx(168.7229)
    assert r.mean_anomaly == pytest.approx(194.9782)
    assert r.bstar == pytest.approx(0.0016873)
    assert r.mean_motion_dot == pytest.approx(3.193e-5, rel=1e-9)
    assert r.classification_type == "U"
    assert r.element_set_no == 999
    assert r.rev_at_epoch == 82675
    assert r.tle_line1 == L1 and r.tle_line2 == L2
    assert r.record_hash == rec.line_hash(L1, L2)


def test_from_tle_sparse_fields_are_null():
    r = rec.from_tle(L1, L2)
    for name in (
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
        "tle_line0",
    ):
        assert getattr(r, name) is None, name


def test_from_tle_3le_name():
    r = rec.from_tle(L0, L1, L2)
    assert r.object_name == "VANGUARD 2"
    assert r.tle_line0 == "0 VANGUARD 2"
    # the name line is not part of identity
    assert r.record_hash == rec.line_hash(L1, L2)


def test_implicit_semantics_by_ephemeris_type():
    r0 = rec.from_tle(L1, L2)
    assert (r0.mean_element_theory, r0.ref_frame, r0.time_system) == (
        "SGP4",
        "TEME",
        "UTC",
    )
    l1_t4 = L1[:62] + "4" + L1[63:]
    r4 = rec.from_tle(l1_t4, L2)
    assert r4.ephemeris_type == 4
    assert r4.mean_element_theory == "SGP4-XP"


# --- from_omm and TLE/OMM agreement ---------------------------------------


def test_from_omm_full_record(omm_fields):
    r = rec.from_omm(omm_fields)
    assert r.norad_cat_id == 11
    assert r.object_name == "VANGUARD 2"
    assert r.object_type == "PAYLOAD"
    assert r.rcs_size == "MEDIUM"
    assert r.country_code == "US"
    assert r.launch_date == "1959-02-17"
    assert r.site == "AFETR"
    assert r.decay_date is None  # null in the fixture
    assert r.gp_id == 285891879
    assert r.file_id == 4706888
    assert r.originator == "18 SPCS"
    assert r.creation_date == datetime.datetime(2025, 4, 19, 22, 26, 18)
    assert r.mean_element_theory == "SGP4"
    # identity comes from the embedded lines
    assert r.record_hash == rec.line_hash(
        omm_fields["TLE_LINE1"], omm_fields["TLE_LINE2"]
    )


def test_tle_and_omm_delivery_share_identity(omm_fields):
    """The same elset delivered both ways is one record."""
    from_lines = rec.from_tle(
        omm_fields["TLE_LINE0"],
        omm_fields["TLE_LINE1"],
        omm_fields["TLE_LINE2"],
    )
    from_message = rec.from_omm(omm_fields)
    assert from_lines.record_hash == from_message.record_hash
    assert from_lines.epoch == from_message.epoch
    assert from_lines.mean_motion == pytest.approx(
        from_message.mean_motion, abs=1e-8
    )


def test_from_omm_line_less_uses_semantic_hash(omm_fields):
    for key in ("TLE_LINE0", "TLE_LINE1", "TLE_LINE2"):
        del omm_fields[key]
    r = rec.from_omm(omm_fields)
    assert r.tle_line1 is None and r.tle_line2 is None
    assert r.record_hash == rec.semantic_hash(
        r.norad_cat_id,
        r.epoch,
        r.mean_element_theory,
        r.mean_motion,
        r.eccentricity,
        r.inclination,
        r.ra_of_asc_node,
        r.arg_of_pericenter,
        r.mean_anomaly,
        r.bstar,
        r.mean_motion_dot,
        r.mean_motion_ddot,
    )


def test_from_omm_rejects_missing_norad(omm_fields):
    omm_fields["NORAD_CAT_ID"] = None
    with pytest.raises(rec.MalformedRecordError, match="NORAD"):
        rec.from_omm(omm_fields)


def test_from_omm_rejects_non_finite_mandatory(omm_fields):
    omm_fields["ECCENTRICITY"] = "nan"
    with pytest.raises(rec.MalformedRecordError, match="eccentricity"):
        rec.from_omm(omm_fields)


def test_from_omm_nulls_non_finite_nullable(omm_fields):
    omm_fields["BSTAR"] = "nan"
    omm_fields["MEAN_MOTION_DOT"] = "not-a-number"
    r = rec.from_omm(omm_fields)
    assert r.bstar is None
    assert r.mean_motion_dot is None


# --- semantic hash --------------------------------------------------------


def _hash_kwargs(**overrides):
    base = dict(
        norad_cat_id=11,
        epoch=datetime.datetime(2025, 4, 19, 18, 58, 48, 307296),
        mean_element_theory="SGP4",
        mean_motion=11.89628311,
        eccentricity=0.1448126,
        inclination=32.8658,
        ra_of_asc_node=246.7104,
        arg_of_pericenter=168.7229,
        mean_anomaly=194.9782,
        bstar=0.0016873,
        mean_motion_dot=3.193e-05,
        mean_motion_ddot=0.0,
    )
    base.update(overrides)
    return base


def test_semantic_hash_ignores_formatting_variants():
    """'.5', '0.5' and '5.0E-1' parse to the same double → same identity."""
    a = rec.semantic_hash(**_hash_kwargs(eccentricity=float(".5")))
    b = rec.semantic_hash(**_hash_kwargs(eccentricity=float("0.50")))
    c = rec.semantic_hash(**_hash_kwargs(eccentricity=float("5.0E-1")))
    assert a == b == c


def test_semantic_hash_normalizes_negative_zero():
    a = rec.semantic_hash(**_hash_kwargs(mean_motion_ddot=0.0))
    b = rec.semantic_hash(**_hash_kwargs(mean_motion_ddot=-0.0))
    assert a == b


def test_semantic_hash_null_drag_term_is_distinct_from_zero():
    a = rec.semantic_hash(**_hash_kwargs(bstar=None))
    b = rec.semantic_hash(**_hash_kwargs(bstar=0.0))
    assert a != b


def test_semantic_hash_sensitive_to_values_and_theory():
    base = rec.semantic_hash(**_hash_kwargs())
    assert base != rec.semantic_hash(**_hash_kwargs(mean_motion=11.89628312))
    assert base != rec.semantic_hash(
        **_hash_kwargs(mean_element_theory="SGP4-XP")
    )


# --- merge ----------------------------------------------------------------


def test_merge_enriches_nulls(omm_fields):
    """The 'TLE first, OMM later' attach case."""
    sparse = rec.from_tle(omm_fields["TLE_LINE1"], omm_fields["TLE_LINE2"])
    full = rec.from_omm(omm_fields)
    merged = rec.merge(sparse, full)
    assert merged.record_hash == sparse.record_hash
    assert merged.object_name == "VANGUARD 2"
    assert merged.gp_id == 285891879
    assert merged.country_code == "US"
    assert merged.tle_line1 == sparse.tle_line1  # verbatim, unchanged


def test_merge_conflict_newest_creation_date_wins(omm_fields):
    old = rec.from_omm(omm_fields)
    reissue = dataclasses.replace(
        old,
        object_name="VANGUARD 2 (RENAMED)",
        gp_id=999999999,
        creation_date=old.creation_date + datetime.timedelta(days=1),
    )
    assert rec.merge(old, reissue).object_name == "VANGUARD 2 (RENAMED)"
    # ...and in the other delivery order the newer record still wins
    assert rec.merge(reissue, old).object_name == "VANGUARD 2 (RENAMED)"
    assert rec.merge(reissue, old).gp_id == 999999999


def test_merge_conflict_without_creation_date_last_ingested_wins():
    a = dataclasses.replace(rec.from_tle(L1, L2), object_name="A")
    b = dataclasses.replace(rec.from_tle(L1, L2), object_name="B")
    assert rec.merge(a, b).object_name == "B"
    assert rec.merge(b, a).object_name == "A"


def test_merge_null_never_overwrites_non_null(omm_fields):
    full = rec.from_omm(omm_fields)
    sparse = rec.from_tle(omm_fields["TLE_LINE1"], omm_fields["TLE_LINE2"])
    merged = rec.merge(full, sparse)
    assert merged.object_name == "VANGUARD 2"
    assert merged.gp_id == 285891879


def test_merge_requires_identical_hash():
    a = rec.from_tle(L1, L2)
    b = dataclasses.replace(a, record_hash="0" * 64)
    with pytest.raises(ValueError, match="record_hash"):
        rec.merge(a, b)


def test_merge_no_change_returns_existing_instance():
    a = rec.from_tle(L1, L2)
    assert rec.merge(a, rec.from_tle(L1, L2)) is a


# --- CSV serialization ----------------------------------------------------


def test_store_csv_round_trip(omm_fields):
    for original in (rec.from_omm(omm_fields), rec.from_tle(L0, L1, L2)):
        row = rec.to_csv_dict(original)
        assert list(row) == rec.STORE_COLUMNS
        assert rec.from_csv_dict(row) == original


def test_from_csv_dict_tolerates_unknown_and_missing_columns(omm_fields):
    """The reader half of additive-only format versioning."""
    row = rec.to_csv_dict(rec.from_omm(omm_fields))
    row["COVARIANCE_SOMEDAY"] = "1.0"  # unknown column: ignored
    del row["RCS_SIZE"]  # missing nullable column: null
    r = rec.from_csv_dict(row)
    assert r.rcs_size is None
    assert r.gp_id == 285891879


def test_omm_projection_matches_stk_columns(omm_fields):
    for r in (rec.from_omm(omm_fields), rec.from_tle(L1, L2)):
        projected = rec.to_omm_dict(r)
        assert list(projected) == OMM_CSV_FIELDS
        assert all(v != "" for k, v in projected.items() if k != "OBJECT_NAME")
    # 2LE delivery: no name, still projectable
    assert rec.to_omm_dict(rec.from_tle(L1, L2))["OBJECT_NAME"] == ""
