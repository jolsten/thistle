"""Count reconciliation of every output against the store (the authority).

Catches damage the incremental tail guard can't see: mid-file truncation,
deleted files for quiet objects, orphan files for objects the store no
longer knows. Reads every output file, so this is a periodic (e.g.
weekly) cron run, not an every-run check. Also stamps
``rejects.resolved_at`` for payloads whose exact hash now exists in
``tle`` — the reconciliation half of the rejects lifecycle.

Line-less records (OMM-native) are invisible to tle outputs by design;
expected counts for tle outputs therefore consider only line-full rows.
In the rare edge where an object's newest record for a date is line-less,
the incremental date path keeps the older line-full entry while a rebuild
drops the object for that date — verify may flag such a date file; a
`rebuild` settles it.
"""

import dataclasses
import datetime
import pathlib
from typing import Optional

from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from thistle_db.config import OutputConfig, OutputFile
from thistle_db.model import Reject, TLE, date_from_sql, utcnow
from thistle_db.outputs import _read_csv_rows, _read_tle_pairs, output_key

__all__ = ["VerifyIssue", "VerifyReport", "stamp_resolved_rejects", "verify_outputs"]


@dataclasses.dataclass
class VerifyIssue:
    output: str
    """output_key of the entry."""
    kind: str
    """'missing' | 'count' | 'orphan'"""
    path: str
    expected: Optional[int] = None
    actual: Optional[int] = None

    def describe(self) -> str:
        if self.kind == "orphan":
            return f"{self.output}: orphan file {self.path}"
        if self.kind == "missing":
            return f"{self.output}: missing {self.path} (expected {self.expected})"
        return (
            f"{self.output}: {self.path} holds {self.actual} record(s), "
            f"store says {self.expected}"
        )


@dataclasses.dataclass
class VerifyReport:
    issues: list[VerifyIssue] = dataclasses.field(default_factory=list)
    outputs_checked: int = 0
    rejects_resolved: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


def _count_file_records(entry: OutputFile, path: pathlib.Path) -> int:
    if entry.format == "tle":
        return len(_read_tle_pairs(path))
    return len(_read_csv_rows(path))


def _expected_per_object(session: Session, lines_only: bool) -> dict[int, int]:
    stmt = select(TLE.norad_cat_id, func.count(TLE.id)).group_by(TLE.norad_cat_id)
    if lines_only:
        stmt = stmt.where(TLE.line1.is_not(None), TLE.line2.is_not(None))
    return {norad: count for norad, count in session.execute(stmt)}


def _expected_per_date(session: Session, lines_only: bool) -> dict[datetime.date, int]:
    date_expr = func.date(TLE.epoch)
    stmt = select(
        date_expr, func.count(func.distinct(TLE.norad_cat_id))
    ).group_by(date_expr)
    if lines_only:
        stmt = stmt.where(TLE.line1.is_not(None), TLE.line2.is_not(None))
    return {
        date_from_sql(raw): count
        for raw, count in session.execute(stmt)
        if raw is not None
    }


def _verify_object_entry(
    session: Session, entry: OutputFile, report: VerifyReport
) -> None:
    key = output_key(entry)
    expected = _expected_per_object(session, lines_only=entry.format == "tle")
    directory = pathlib.Path(entry.dir)

    for norad, count in expected.items():
        path = entry.object_path(norad)
        if not path.exists():
            report.issues.append(
                VerifyIssue(key, "missing", str(path), expected=count)
            )
            continue
        actual = _count_file_records(entry, path)
        if actual != count:
            report.issues.append(
                VerifyIssue(key, "count", str(path), expected=count, actual=actual)
            )

    # Orphan scan: files this output's naming claims, for objects the
    # store doesn't know. Date files sharing the directory parse to None
    # and never look like orphans.
    if directory.is_dir():
        for path in sorted(directory.glob(entry.object_glob())):
            norad = entry.parse_object_name(path.name)
            if norad is not None and norad not in expected:
                report.issues.append(VerifyIssue(key, "orphan", str(path)))


def _verify_date_entry(
    session: Session, entry: OutputFile, report: VerifyReport
) -> None:
    key = output_key(entry)
    expected = _expected_per_date(session, lines_only=entry.format == "tle")
    for day, count in expected.items():
        path = entry.date_path(day)
        if not path.exists():
            report.issues.append(
                VerifyIssue(key, "missing", str(path), expected=count)
            )
            continue
        actual = _count_file_records(entry, path)
        if actual != count:
            report.issues.append(
                VerifyIssue(key, "count", str(path), expected=count, actual=actual)
            )


def verify_outputs(session: Session, output: OutputConfig) -> VerifyReport:
    """Reconcile every output against the store; stamp resolved rejects."""
    report = VerifyReport()
    for entry in output.files:
        if entry.type == "object":
            _verify_object_entry(session, entry, report)
        else:
            _verify_date_entry(session, entry, report)
        report.outputs_checked += 1

    report.rejects_resolved = stamp_resolved_rejects(session)

    for issue in report.issues:
        logger.warning(issue.describe())
    if report.ok:
        logger.info(f"verify: {report.outputs_checked} output(s) consistent")
    return report


def stamp_resolved_rejects(session: Session) -> int:
    """Mark unresolved rejects whose exact payload hash now exists in `tle`.

    The future-epoch case: the same text, redelivered or re-admitted after
    its epoch came inside the horizon, now lives in the store.
    """
    now = utcnow()
    in_store = select(TLE.id).where(TLE.record_hash == Reject.record_hash).exists()
    result = session.execute(
        update(Reject)
        .where(Reject.resolved_at.is_(None), in_store)
        .values(resolved_at=now)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return int(getattr(result, "rowcount", 0) or 0)
