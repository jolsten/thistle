"""Run accounting — runs are observable as data, not just log lines.

Every writing command inserts its ``ingest_runs`` row first and finalizes
it last; structured report objects flow from the engine to the CLI, the
run row, and any wrapper (cron mail, an orchestrator) — all consuming the
same numbers. A row stuck in ``running`` is itself a signal (crashed
run), surfaced by ``status``.
"""

import contextlib
import dataclasses
import datetime
from typing import Iterator, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from thistle_db.model import IngestRun, utcnow

__all__ = ["IngestReport", "run_context", "recent_runs", "stuck_runs"]


@dataclasses.dataclass
class IngestReport:
    """Counters for one run — the structured return of the delta engine."""

    files_scanned: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_refreshed: int = 0
    files_failed: int = 0
    records_seen: int = 0
    records_inserted: int = 0
    records_rejected: int = 0
    records_readmitted: int = 0
    lineless_skipped: int = 0
    """Line-less records skipped by tle outputs (one WARNING per run)."""
    reject_reasons: dict[str, int] = dataclasses.field(default_factory=dict)
    output_writes: dict[str, int] = dataclasses.field(default_factory=dict)
    """Files written per output entry (key: '<type>:<format>:<dir>')."""

    def add_reject_reasons(self, reasons: dict[str, int]) -> None:
        for reason, count in reasons.items():
            self.reject_reasons[reason] = (
                self.reject_reasons.get(reason, 0) + count
            )

    def detail(self) -> dict:
        """The extensible JSON detail stored on the run row."""
        detail: dict = {}
        if self.reject_reasons:
            detail["reject_reasons"] = self.reject_reasons
        if self.output_writes:
            detail["output_writes"] = self.output_writes
        if self.lineless_skipped:
            detail["lineless_skipped"] = self.lineless_skipped
        if self.records_readmitted:
            detail["records_readmitted"] = self.records_readmitted
        return detail

    def summary(self) -> str:
        return (
            f"{self.records_inserted} new records "
            f"({self.records_seen} seen, {self.records_rejected} rejected) "
            f"from {self.files_ingested} file(s) "
            f"({self.files_skipped} skipped, {self.files_refreshed} "
            f"refreshed, {self.files_failed} failed)"
        )


@contextlib.contextmanager
def run_context(session: Session, command: str) -> Iterator[tuple[IngestRun, IngestReport]]:
    """Insert the run row (status 'running'), finalize on exit.

    On an exception the row finishes 'failed' with the error recorded and
    the exception propagates. The session is rolled back first so a
    mid-transaction failure cannot block the bookkeeping write.
    """
    report = IngestReport()
    run = IngestRun(command=command[:255], started=utcnow(), status="running")
    session.add(run)
    session.commit()
    try:
        yield run, report
    except BaseException as exc:
        session.rollback()
        _finalize(session, run, report, status="failed", error=repr(exc))
        raise
    else:
        _finalize(session, run, report, status="succeeded", error=None)


def _finalize(
    session: Session,
    run: IngestRun,
    report: IngestReport,
    *,
    status: str,
    error: Optional[str],
) -> None:
    try:
        run.finished = utcnow()
        run.status = status
        run.error = error
        run.files_scanned = report.files_scanned
        run.files_ingested = report.files_ingested
        run.files_skipped = report.files_skipped
        run.files_refreshed = report.files_refreshed
        run.files_failed = report.files_failed
        run.records_seen = report.records_seen
        run.records_inserted = report.records_inserted
        run.records_rejected = report.records_rejected
        run.detail = report.detail() or None
        session.add(run)
        session.commit()
    except Exception:
        # Bookkeeping must never mask the run's own outcome.
        session.rollback()


def recent_runs(session: Session, limit: int = 10) -> list[IngestRun]:
    stmt = select(IngestRun).order_by(IngestRun.id.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


def stuck_runs(
    session: Session, older_than: datetime.timedelta = datetime.timedelta(hours=1)
) -> list[IngestRun]:
    """Rows still 'running' past a grace period — crashed runs."""
    cutoff = utcnow() - older_than
    stmt = (
        select(IngestRun)
        .where(IngestRun.status == "running", IngestRun.started < cutoff)
        .order_by(IngestRun.id.desc())
    )
    return list(session.execute(stmt).scalars().all())
