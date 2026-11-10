"""The rejects table: retention, dedup, the fuse, payload cap, export."""

import pathlib

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .conftest import TLES, engine_ingest, make_settings
from thistle_db import rejects as rejects_mod
from thistle_db.model import Reject
from thistle_db.rejects import (
    DbRejectWriter,
    PAYLOAD_CAP,
    ReasonCode,
    export_unresolved,
    unresolved_summary,
)

L1, L2 = TLES[0]
BAD_L1 = L1[:30] + ("9" if L1[30] != "9" else "8") + L1[31:]


def _reject_count(session: Session) -> int:
    return session.execute(select(func.count(Reject.id))).scalar_one()


def test_sink_upserts_and_dedups(db_session: Session):
    with DbRejectWriter(db_session, source_path=pathlib.Path("a.tle")) as sink:
        sink.seen()
        sink.reject(ReasonCode.CHECKSUM, "mismatch", line1=BAD_L1, line2=L2)
    assert _reject_count(db_session) == 1

    # Same payload from another file, different reason: one row, bumped.
    with DbRejectWriter(db_session, source_path=pathlib.Path("b.tle")) as sink:
        sink.seen()
        sink.reject(ReasonCode.UNPARSEABLE, "other reason", line1=BAD_L1, line2=L2)
    row = db_session.execute(select(Reject)).scalar_one()
    assert row.times_seen == 2
    assert row.reason_code == "unparseable"  # latest wins
    assert row.source_path == "b.tle"
    assert row.first_seen <= row.last_seen


def test_within_file_duplicates_fold(db_session: Session):
    with DbRejectWriter(db_session, source_path=None) as sink:
        for _ in range(3):
            sink.seen()
            sink.reject(ReasonCode.CHECKSUM, "mismatch", line1=BAD_L1, line2=L2)
    row = db_session.execute(select(Reject)).scalar_one()
    assert row.times_seen == 3


def test_convenience_columns_parsed(db_session: Session):
    with DbRejectWriter(db_session, source_path=None) as sink:
        sink.seen()
        sink.reject(ReasonCode.CHECKSUM, "mismatch", line1=BAD_L1, line2=L2)
    row = db_session.execute(select(Reject)).scalar_one()
    assert row.norad_cat_id == 22
    assert row.epoch is not None and row.epoch.year == 2025


def test_fuse_stores_no_payloads(db_session: Session, monkeypatch):
    monkeypatch.setattr(rejects_mod, "REJECT_MAX_RECORDS", 5)
    with DbRejectWriter(db_session, source_path=None) as sink:
        for i in range(10):
            sink.seen()
            # Distinct payloads (mis-encoded files produce unbounded
            # distinct garbage) — exactly what the fuse exists for.
            sink.reject(
                ReasonCode.UNPARSEABLE, "garbage", line1=f"junk-{i}", line2="x"
            )
    assert sink.rejected == 10
    assert _reject_count(db_session) == 0  # over the fuse: nothing stored


def test_payload_cap_truncates(db_session: Session):
    huge = "1 " + "X" * (PAYLOAD_CAP + 100)
    with DbRejectWriter(db_session, source_path=None) as sink:
        sink.seen()
        sink.reject(ReasonCode.UNPARSEABLE, "huge", line1=huge, line2="2 X")
    row = db_session.execute(select(Reject)).scalar_one()
    assert row.truncated is True
    assert len(row.payload.encode()) <= PAYLOAD_CAP


def test_summary_and_export(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    bad = tmp_path / "bad.tle"
    bad.write_text(f"{BAD_L1}\n{L2}\n")
    engine_ingest(db_session, settings, files=[bad])

    summary = unresolved_summary(db_session)
    assert [(code, count) for code, count, _ in summary] == [("checksum", 1)]

    out = tmp_path / "repair"
    tle_count, omm_count = export_unresolved(db_session, out)
    assert (tle_count, omm_count) == (1, 0)
    exported = (out / "rejects.tle").read_text()
    assert BAD_L1 in exported  # verbatim, ready for repair + re-ingest


def test_reject_write_failure_never_fails_ingest(
    db_session: Session, tmp_path, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("quarantine down")

    monkeypatch.setattr(rejects_mod, "upsert_rejects", boom)
    monkeypatch.setattr("thistle_db.ingest.upsert_rejects", boom)
    settings = make_settings(tmp_path)
    mixed = tmp_path / "mixed.tle"
    mixed.write_text(f"{BAD_L1}\n{L2}\n{L1}\n{L2}\n")
    report = engine_ingest(db_session, settings, files=[mixed])
    # The good record still ingests; the quarantine failure is contained.
    assert report.files_failed == 0
    assert report.records_inserted == 1
