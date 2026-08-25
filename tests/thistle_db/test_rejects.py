"""Non-finite element handling and the quarantine directory.

sgp4's parser is C `strtod`, which accepts the literal text "nan"/"inf" in a
fixed-width field — what a producer emits when it formats a missing value —
so every float column can arrive non-finite. NaN reaches MariaDB as an
unquoted token its driver refuses before sending SQL, which historically
failed the whole 5000-row chunk and, with it, the rest of the file.

**Fixtures here must not depend on a text field parsing to NaN in a column
sgp4 reads with its own exponent-format reader.** Whether line 1's `bstar`
field reading "nan" yields NaN or 0.0 depends on whether numpy has been
imported into the process (it changes the CRT parse), so a fixture built on
it passes standalone and fails under a suite that pulls numpy in. The two
stable sources of non-finite values, used below, are a `nan` in line 2's
plain-float fields and a zero mean motion (infinite derived values by
arithmetic, not parsing). `_clean_floats` is unit-tested directly for the
per-column mapping.
"""

import json
import math

import pytest
from sqlalchemy.orm import Session

from thistle_db import rejects as rejects_mod
from thistle_db.ingest import ingest_file, ingest_omm, ingest_sources, ingest_tles
from thistle_db.config import IngestSource
from thistle_db.model import TLE, MalformedElsetError, _clean_floats
from thistle_db.rejects import RejectWriter, namespaces, reject_path

GOOD = (
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)

# inclination field (columns 9-16) reading "nan": mandatory column, no orbit.
NAN_INCLINATION = (GOOD[0], GOOD[1][:8] + "     nan" + GOOD[1][16:])

# Zero mean motion makes the derived values infinite, by arithmetic rather
# than parsing — the same coercion path, deterministic in any process.
ZERO_MEAN_MOTION = (GOOD[0], GOOD[1][:52] + "00.00000000" + GOOD[1][63:])


# ---------------------------------------------------------------------------
# Model layer: non-finite coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_clean_floats_nulls_nullable_columns(bad):
    """Every nullable float column takes NULL, whatever kind of non-finite."""
    for column in ("bstar", "mean_motion", "mean_motion_dot", "semimajor_axis"):
        cleaned = _clean_floats({column: bad, "eccentricity": 0.1})
        assert cleaned[column] is None
        assert cleaned["eccentricity"] == 0.1  # finite values untouched


@pytest.mark.parametrize(
    "column",
    [
        "eccentricity",
        "inclination",
        "ra_of_asc_node",
        "arg_of_pericenter",
        "mean_anomaly",
    ],
)
def test_clean_floats_rejects_mandatory_columns(column):
    with pytest.raises(MalformedElsetError, match=column):
        _clean_floats({column: math.nan})


def test_clean_floats_reads_nullability_from_the_schema():
    """A float column added later is covered without touching _clean_floats."""
    from thistle_db.model import _NULLABLE

    assert _NULLABLE["bstar"] is True
    assert _NULLABLE["inclination"] is False
    assert set(_NULLABLE) == {c.name for c in TLE.__table__.columns}


def test_infinite_derived_values_become_none():
    """A zero mean motion yields an infinite semi-major axis, not NaN."""
    tle = TLE.from_twoline(*ZERO_MEAN_MOTION)
    assert tle.semimajor_axis is None
    assert tle.period is None
    assert tle.apoapsis_alt is None
    assert tle.periapsis_alt is None


def test_nan_in_mandatory_column_is_rejected():
    with pytest.raises(MalformedElsetError, match="inclination"):
        TLE.from_twoline(*NAN_INCLINATION)


def test_no_stored_float_is_ever_non_finite():
    """The guarantee the MariaDB driver depends on."""
    tle = TLE.from_twoline(*ZERO_MEAN_MOTION)
    for column in TLE.__table__.columns:
        value = getattr(tle, column.name)
        if isinstance(value, float):
            assert math.isfinite(value), column.name


# ---------------------------------------------------------------------------
# Ingest layer
# ---------------------------------------------------------------------------


def test_non_finite_row_ingests_with_nulls(db_session: Session):
    assert ingest_tles(db_session, [ZERO_MEAN_MOTION]) == 1
    row = db_session.query(TLE).one()
    assert row.semimajor_axis is None
    assert row.period is None
    # The elset itself is preserved verbatim.
    assert (row.line1, row.line2) == ZERO_MEAN_MOTION


def test_unusable_elset_does_not_cost_the_batch(db_session: Session):
    """The reported symptom: one bad line must not lose the whole file."""
    batch = [GOOD, NAN_INCLINATION, ZERO_MEAN_MOTION]
    assert ingest_tles(db_session, batch) == 2
    assert db_session.query(TLE).count() == 2


def test_failed_chunk_falls_back_to_row_by_row(db_session: Session, monkeypatch):
    """A chunk the database refuses costs only the offending row.

    Simulates a driver that rejects a value while building the statement
    (pymysql does this for NaN), which fails the batch before any SQL is sent.
    """
    from thistle_db import ingest as ingest_mod

    real_executor = ingest_mod._insert_executor

    def poisoned_executor(session, table, index_elements):
        execute = real_executor(session, table, index_elements)

        def wrapper(rows):
            if any(r.get("line1", "").startswith("1 25544") for r in rows):
                raise RuntimeError("driver refused a value in this batch")
            return execute(rows)

        return wrapper

    monkeypatch.setattr(ingest_mod, "_insert_executor", poisoned_executor)

    other = (
        "1 00900U 64063C   24001.50000000  .00000106  00000-0  10270-3 0  9005",
        "2 00900  90.1970  25.0000 0025000 100.0000 260.0000 13.73000000563537",
    )
    inserted = ingest_tles(db_session, [GOOD, other])
    # The good row survives; only the poisoned one is lost.
    assert inserted == 1
    assert db_session.query(TLE).one().norad_cat_id == 900


# ---------------------------------------------------------------------------
# Quarantine directory
# ---------------------------------------------------------------------------


def _write_tle_file(path, tles):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{a}\n{b}\n" for a, b in tles))
    return path


def test_quarantine_writes_rejected_lines(db_session: Session, tmp_path):
    source = _write_tle_file(tmp_path / "in" / "day.tle", [GOOD, NAN_INCLINATION])
    reject_dir = tmp_path / "rejects"

    count = ingest_file(db_session, source, reject_dir=reject_dir)

    assert count == 1
    artifact = reject_dir / "day.tle"
    assert artifact.read_text().splitlines() == list(NAN_INCLINATION)
    # The reason survives log rotation, next to the record it explains.
    log = (reject_dir / "day.tle.log").read_text()
    assert "inclination" in log
    assert "rejected 1 of 2 records" in log


def test_quarantine_artifact_is_reingestable(db_session: Session, tmp_path):
    """A repaired reject file feeds straight back into ingest."""
    source = _write_tle_file(tmp_path / "in" / "day.tle", [NAN_INCLINATION])
    reject_dir = tmp_path / "rejects"
    ingest_file(db_session, source, reject_dir=reject_dir)

    artifact = reject_dir / "day.tle"
    _write_tle_file(artifact, [GOOD])  # the operator fixes the record
    assert ingest_file(db_session, artifact) == 1


def test_quarantine_self_cleans_when_source_is_fixed(db_session: Session, tmp_path):
    """The directory is a live view of what is currently broken."""
    source = tmp_path / "in" / "day.tle"
    _write_tle_file(source, [GOOD, NAN_INCLINATION])
    reject_dir = tmp_path / "rejects"
    ingest_file(db_session, source, reject_dir=reject_dir)
    assert (reject_dir / "day.tle").exists()

    _write_tle_file(source, [GOOD])
    ingest_file(db_session, source, reject_dir=reject_dir)
    assert not (reject_dir / "day.tle").exists()
    assert not (reject_dir / "day.tle.log").exists()


def test_quarantine_disabled_by_default(db_session: Session, tmp_path):
    source = _write_tle_file(tmp_path / "in" / "day.tle", [GOOD, NAN_INCLINATION])
    assert ingest_file(db_session, source) == 1
    assert list(tmp_path.glob("**/rejects*")) == []


def test_quarantine_caps_a_wholly_broken_file(
    db_session: Session, tmp_path, monkeypatch
):
    """Past the cap the file is not copied — only a marker describing it."""
    monkeypatch.setattr(rejects_mod, "REJECT_MAX_RECORDS", 2)
    source = _write_tle_file(tmp_path / "in" / "bad.tle", [NAN_INCLINATION] * 3)
    reject_dir = tmp_path / "rejects"

    ingest_file(db_session, source, reject_dir=reject_dir)

    assert not (reject_dir / "bad.tle").exists()
    marker = (reject_dir / "bad.tle.truncated").read_text()
    assert "records rejected: 3" in marker
    assert "inclination" in marker


def test_omm_rejects_quarantine_as_json(db_session: Session, tmp_path):
    reject_dir = tmp_path / "rejects"
    source = tmp_path / "in" / "gp.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            [
                {"TLE_LINE1": GOOD[0], "TLE_LINE2": GOOD[1], "OBJECT_NAME": "ISS"},
                {"OBJECT_NAME": "NO LINES"},
            ]
        )
    )

    assert ingest_file(db_session, source, reject_dir=reject_dir) == 1

    quarantined = json.loads((reject_dir / "gp.json.json").read_text())
    assert [r["OBJECT_NAME"] for r in quarantined] == ["NO LINES"]


def test_omm_record_without_lines_is_rejected(db_session: Session):
    assert ingest_omm(db_session, [{"OBJECT_NAME": "NO LINES"}]) == 0


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_reject_path_mirrors_the_source_tree(tmp_path):
    root = tmp_path / "daily"
    source = root / "2006" / "20060101.tle"
    target = reject_path(source, tmp_path / "rejects", root=root)
    assert target == tmp_path / "rejects" / "daily" / "2006" / "20060101.tle"


def test_reject_path_without_a_root_is_flat(tmp_path):
    source = tmp_path / "somewhere" / "day.tle"
    assert reject_path(source, tmp_path / "r") == tmp_path / "r" / "day.tle"


def test_namespaces_disambiguate_same_named_roots(tmp_path):
    a = tmp_path / "one" / "daily"
    b = tmp_path / "two" / "daily"
    unique = tmp_path / "archive"
    result = namespaces([a, b, unique])
    assert result[str(unique)] == "archive"
    assert result[str(a)] != result[str(b)]
    assert all(name.startswith("daily-") for name in (result[str(a)], result[str(b)]))


def test_scan_quarantines_under_the_source_namespace(db_session: Session, tmp_path):
    incoming = tmp_path / "incoming"
    _write_tle_file(incoming / "day.tle", [GOOD, NAN_INCLINATION])
    reject_dir = tmp_path / "rejects"

    ingest_sources(
        db_session,
        [IngestSource(path=str(incoming), pattern="*.tle")],
        reject_dir=reject_dir,
    )

    assert (reject_dir / "incoming" / "day.tle").exists()


def test_quarantine_failure_does_not_fail_ingest(db_session: Session, tmp_path):
    """A broken reject_dir must not turn a working ingest into a failed one."""
    source = _write_tle_file(tmp_path / "in" / "day.tle", [GOOD, NAN_INCLINATION])
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")

    assert ingest_file(db_session, source, reject_dir=blocked) == 1


def test_writer_summary_reports_the_ratio(tmp_path):
    writer = RejectWriter(tmp_path / "s.tle", tmp_path / "t.tle", omm=False)
    writer.seen(30000)
    for _ in range(3):
        writer.reject("unparseable: nope", line1="1 x", line2="2 x")
    assert writer.summary().startswith("3 of 30000 records rejected")
    writer.close()


def test_scan_ignores_files_under_the_quarantine_directory(
    db_session: Session, tmp_path
):
    """A reject_dir nested in a source dir must not re-ingest its own output."""
    incoming = tmp_path / "incoming"
    reject_dir = incoming / "rejects"
    _write_tle_file(incoming / "day.tle", [GOOD, NAN_INCLINATION])
    source = [IngestSource(path=str(incoming), pattern="**/*.tle")]

    ingest_sources(db_session, source, reject_dir=reject_dir)
    quarantined = reject_dir / "incoming" / "day.tle"
    assert quarantined.exists()

    # Second scan: the quarantine artifact is in glob range but must be skipped,
    # so it neither re-ingests nor nests a level deeper.
    ingest_sources(db_session, source, reject_dir=reject_dir, force=True)
    assert not (reject_dir / "incoming" / "rejects").exists()


# ---------------------------------------------------------------------------
# Future-epoch guard
# ---------------------------------------------------------------------------

# Epoch 56365.5 = 2056-12-30: what a corrupted epoch field looks like after
# the two-digit-year mapping (00-56 -> 20xx). Parses cleanly.
FUTURE_2056 = (
    "1 25544U 98067A   56365.50000000  .00016717  00000-0  10270-3 0  9005",
    GOOD[1],
)


def _tle_days_ahead(days: float) -> tuple[str, str]:
    """A synthetic elset whose epoch is `days` ahead of now."""
    import datetime

    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        days=days
    )
    field = f"{future.year % 100:02d}{future.timetuple().tm_yday:03d}.50000000"
    return (GOOD[0][:18] + field + GOOD[0][32:], GOOD[1])


def test_far_future_epoch_rejected_by_default(db_session: Session):
    assert ingest_tles(db_session, [GOOD, FUTURE_2056]) == 1
    assert [row.epoch.year for row in db_session.query(TLE).all()] == [2024]


def test_epoch_within_horizon_ingests(db_session: Session):
    """Feeds legitimately run hours to a day ahead; the guard must not
    touch them."""
    assert ingest_tles(db_session, [_tle_days_ahead(2)]) == 1


def test_epoch_guard_boundary(db_session: Session):
    assert ingest_tles(db_session, [_tle_days_ahead(40)]) == 0
    assert (
        ingest_tles(db_session, [_tle_days_ahead(40)], max_epoch_ahead_days=60) == 1
    )


def test_epoch_guard_disabled_with_zero(db_session: Session):
    assert ingest_tles(db_session, [FUTURE_2056], max_epoch_ahead_days=0) == 1


def test_past_epochs_are_never_bounded(db_session: Session):
    """One-sided by design: historical archives must ingest untouched even
    with a tight horizon."""
    assert ingest_tles(db_session, [GOOD], max_epoch_ahead_days=1) == 1  # 2024


def test_future_epoch_is_quarantined_with_reason(db_session: Session, tmp_path):
    source = _write_tle_file(tmp_path / "in" / "day.tle", [GOOD, FUTURE_2056])
    reject_dir = tmp_path / "rejects"

    assert ingest_file(db_session, source, reject_dir=reject_dir) == 1

    assert (reject_dir / "day.tle").read_text().splitlines() == list(FUTURE_2056)
    log = (reject_dir / "day.tle.log").read_text()
    assert "epoch more than 30 days in the future" in log


def test_omm_future_epoch_rejected(db_session: Session):
    inserted = ingest_omm(
        db_session,
        [{"TLE_LINE1": FUTURE_2056[0], "TLE_LINE2": FUTURE_2056[1]}],
    )
    assert inserted == 0
    assert db_session.query(TLE).count() == 0


def test_guard_config_default_and_validation():
    from pydantic import ValidationError

    from thistle_db.config import IngestConfig

    assert IngestConfig().max_epoch_ahead_days == 30
    with pytest.raises(ValidationError):
        IngestConfig(max_epoch_ahead_days=-1)
