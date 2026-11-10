"""Run accounting: runs are observable as data."""

import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from thistle_db.model import IngestRun
from thistle_db.runs import recent_runs, run_context, stuck_runs


def test_run_context_success(db_session: Session):
    with run_context(db_session, "ingest") as (run, report):
        assert run.id is not None  # row committed at start
        report.records_seen = 10
        report.records_inserted = 7
        report.records_rejected = 3
        report.reject_reasons["checksum"] = 3
    row = db_session.execute(select(IngestRun)).scalar_one()
    assert row.status == "succeeded"
    assert row.finished is not None
    assert row.records_inserted == 7
    assert row.detail == {"reject_reasons": {"checksum": 3}}


def test_run_context_failure_records_and_reraises(db_session: Session):
    with pytest.raises(RuntimeError):
        with run_context(db_session, "ingest") as (run, report):
            raise RuntimeError("boom")
    row = db_session.execute(select(IngestRun)).scalar_one()
    assert row.status == "failed"
    assert "boom" in (row.error or "")
    assert row.finished is not None


def test_stuck_runs_detects_crashed_rows(db_session: Session):
    with run_context(db_session, "a") as (run, report):
        pass
    # A crashed run: inserted, never finalized, started long ago.
    old = IngestRun(
        command="crashed",
        status="running",
        started=datetime.datetime(2020, 1, 1),
    )
    db_session.add(old)
    db_session.commit()
    stuck = stuck_runs(db_session)
    assert [r.command for r in stuck] == ["crashed"]
    assert len(recent_runs(db_session)) == 2
