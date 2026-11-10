"""Retained invalid input — the database-backed reject sink.

A rejected record is one that never reaches the ``tle`` table: text sgp4
cannot parse, an elset whose mandatory elements are not finite numbers, a
checksum mismatch, a future epoch beyond the horizon, a record with no
NORAD ID, or a row the database itself refuses. Each distinct rejected
payload is retained as a row in the ``rejects`` table (see
``model.Reject``), so nothing is silently lost and "what is broken, and
since when" is one query. Validity is table membership — ``tle`` ∪
``rejects`` is all input rows.

Two load-bearing properties:

- **Bounded by what is broken, not by time.** Re-rejection of a known
  payload upserts ``last_seen``/``times_seen`` instead of inserting.
- **Fused per file.** Past ``REJECT_MAX_RECORDS`` rejections from one
  source file in one run, no payloads are stored at all — only counters
  and a WARNING with a reason histogram. Beyond the cap a file is not a
  good feed with bad lines but a wrong file (mis-encoded, still gzipped,
  truncated), and even deduplicated rows would be unbounded garbage.

A reject-write failure is logged and never propagates: it must not turn a
working ingest into a failed one.
"""

import datetime
import enum
import hashlib
import json
import pathlib
from typing import Iterable, Optional, Protocol, cast

from loguru import logger
from sqlalchemy import Table, func, select
from sqlalchemy.orm import Session

from thistle_db.model import Reject, utcnow

# Copy at most this many records from one source file per run. Deliberately
# a constant, not config: it is a fuse, not a tuning knob.
REJECT_MAX_RECORDS = 10_000

# Verbatim payload cap per reject row. Garbage "lines" from mis-encoded
# files can be arbitrarily long; 4 KB keeps the row bounded while holding
# any legitimate TLE pair or OMM record whole.
PAYLOAD_CAP = 4096

_REJECTS_TABLE = cast(Table, Reject.__table__)


class ReasonCode(enum.StrEnum):
    """Closed vocabulary for ``rejects.reason_code``."""

    UNPARSEABLE = "unparseable"
    MALFORMED_ELEMENT = "malformed_element"
    FUTURE_EPOCH = "future_epoch"
    NO_NORAD_ID = "no_norad_id"
    CHECKSUM = "checksum"
    DB_REFUSED = "db_refused"


class RejectSink(Protocol):
    """Where ingest reports the records it could not store.

    ``seen`` is called once per record attempted, so the aggregate log line
    can report rejections as a fraction of the file rather than a bare
    count — the ratio is what distinguishes a few bad lines in a good file
    from an entirely wrong file.
    """

    def seen(self, n: int = 1) -> None: ...

    def reject(
        self,
        code: ReasonCode,
        reason: str,
        *,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        record: Optional[dict] = None,
    ) -> None: ...


class _NullSink:
    """Inert sink — the default for library callers."""

    def seen(self, n: int = 1) -> None:
        pass

    def reject(self, code: ReasonCode, reason: str, **kwargs) -> None:
        pass


NO_REJECTS: RejectSink = _NullSink()
"""Shared inert sink: rejects are only logged."""


# --- payload construction ---------------------------------------------------


def payload_for_lines(line1: Optional[str], line2: Optional[str]) -> str:
    return f"{line1 or ''}\n{line2 or ''}"


def payload_for_record(record: dict) -> str:
    """Canonical JSON for an OMM record — deterministic, so the same
    delivered record always hashes identically."""
    return json.dumps(record, sort_keys=True, default=str)


def payload_hash(payload: str) -> bytes:
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _norad_from_line1(line1: str) -> Optional[int]:
    """Best-effort NORAD ID from line 1 (convenience column only)."""
    try:
        from sgp4.alpha5 import from_alpha5

        return from_alpha5(line1[2:7].replace(" ", "0"))
    except Exception:
        return None


def _epoch_from_line1(line1: str) -> Optional[datetime.datetime]:
    """Best-effort epoch from line 1's YYDDD.ffff field.

    Convenience-column precision (drives the future-epoch re-admission
    horizon, which is measured in days) — not the exact stored epoch.
    """
    try:
        raw = float(line1[18:32].replace(" ", "0"))
        year2, doy = divmod(raw, 1000.0)
        year = int(year2) + (2000 if year2 < 57 else 1900)
        return datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
    except Exception:
        return None


def _convenience_fields(
    line1: Optional[str], record: Optional[dict]
) -> tuple[Optional[int], Optional[datetime.datetime]]:
    if record is not None:
        try:
            norad = int(str(record.get("NORAD_CAT_ID", "")).strip() or "x")
        except ValueError:
            norad = None
        epoch = None
        raw_epoch = str(record.get("EPOCH", "")).strip()
        if raw_epoch:
            try:
                epoch = datetime.datetime.fromisoformat(raw_epoch.rstrip("Z"))
                if epoch.tzinfo is not None:
                    epoch = epoch.astimezone(datetime.timezone.utc).replace(
                        tzinfo=None
                    )
            except ValueError:
                epoch = None
        if norad is not None or epoch is not None:
            return norad, epoch
    if line1:
        return _norad_from_line1(line1), _epoch_from_line1(line1)
    return None, None


# --- the sink ----------------------------------------------------------------


class DbRejectWriter:
    """Collects one source file's rejections and upserts them at close.

    Use as a context manager; the flush happens on exit. Shares the
    caller's session but contains its own failures: a quarantine failure
    rolls back and logs, never propagates.
    """

    def __init__(
        self,
        session: Session,
        *,
        source_path: Optional[pathlib.Path] = None,
        fmt: str = "tle",
    ):
        self.session = session
        self.source_path = str(source_path) if source_path is not None else None
        self.fmt = "omm" if fmt.startswith("omm") else "tle"
        self.total = 0
        self.rejected = 0
        self.reasons: dict[str, int] = {}
        # record_hash -> row dict; within-file duplicates fold into
        # times_seen so the upsert sees each payload once.
        self._rows: dict[bytes, dict] = {}

    def __enter__(self) -> "DbRejectWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- collection ------------------------------------------------------

    def seen(self, n: int = 1) -> None:
        self.total += n

    def reject(
        self,
        code: ReasonCode,
        reason: str,
        *,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        record: Optional[dict] = None,
    ) -> None:
        self.rejected += 1
        self.reasons[str(code)] = self.reasons.get(str(code), 0) + 1
        if self.rejected > REJECT_MAX_RECORDS:
            # Past the fuse: keep counting (the WARNING reports totals and
            # the histogram), stop buffering — a wholly corrupt file must
            # not be held in memory or written to the table.
            return

        if record is not None:
            payload = payload_for_record(record)
            fmt = "omm"
        else:
            payload = payload_for_lines(line1, line2)
            fmt = self.fmt
        digest = payload_hash(payload)
        truncated = len(payload.encode("utf-8")) > PAYLOAD_CAP
        if truncated:
            payload = payload.encode("utf-8")[:PAYLOAD_CAP].decode(
                "utf-8", errors="replace"
            )
        norad, epoch = _convenience_fields(line1, record)

        existing = self._rows.get(digest)
        if existing is not None:
            existing["times_seen"] += 1
            existing["reason_code"] = str(code)
            existing["reason"] = reason
            return
        self._rows[digest] = {
            "record_hash": digest,
            "format": fmt,
            "payload": payload,
            "truncated": truncated,
            "reason_code": str(code),
            "reason": reason,
            "source_path": self.source_path,
            "times_seen": 1,
            "norad_cat_id": norad,
            "epoch": epoch,
        }

    # -- output ----------------------------------------------------------

    def close(self) -> None:
        try:
            self._flush()
        except Exception as exc:
            self.session.rollback()
            logger.warning(
                f"Could not record rejects for {self.source_path}: {exc}"
            )
        finally:
            self._rows = {}

    def _flush(self) -> None:
        if self.rejected == 0:
            return
        if self.rejected > REJECT_MAX_RECORDS:
            # The fuse blew: the file is wrong, not a feed with bad lines.
            # No payload rows — the WARNING below carries the histogram,
            # and the run row carries the counters.
            logger.warning(
                f"{self.source_path or 'input'}: {self.rejected} of "
                f"{self.total} records rejected — over the "
                f"{REJECT_MAX_RECORDS}-record fuse, payloads not stored; "
                f"inspect the source file itself. Reasons: "
                + ", ".join(
                    f"{count}× {reason}"
                    for reason, count in sorted(
                        self.reasons.items(), key=lambda kv: -kv[1]
                    )
                )
            )
            return

        now = utcnow()
        rows = []
        for row in self._rows.values():
            rows.append({**row, "first_seen": now, "last_seen": now})
        upsert_rejects(self.session, rows)
        self.session.commit()

    # -- reporting -------------------------------------------------------

    def summary(self) -> Optional[str]:
        """One-line report of what was rejected, or None when nothing was."""
        if self.rejected == 0:
            return None
        return (
            f"{self.rejected} of {self.total} records rejected "
            f"-> rejects table"
        )


def upsert_rejects(session: Session, rows: list[dict]) -> None:
    """Dialect-aware upsert on rejects.record_hash.

    New payloads insert; known payloads bump last_seen and add the batch's
    times_seen, and take the latest reason/source. resolved_at is left
    alone — resolution is stamped by re-admission and verify, not here.
    """
    if not rows:
        return
    table = _REJECTS_TABLE
    dialect = session.bind.dialect.name  # type: ignore[union-attr]

    if dialect in ("sqlite", "postgresql"):
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        else:
            from sqlalchemy.dialects.postgresql import insert as dialect_insert

        stmt = dialect_insert(table)
        stmt = stmt.on_conflict_do_update(
            index_elements=["record_hash"],
            set_={
                "last_seen": stmt.excluded.last_seen,
                "times_seen": table.c.times_seen + stmt.excluded.times_seen,
                "reason_code": stmt.excluded.reason_code,
                "reason": stmt.excluded.reason,
                "source_path": stmt.excluded.source_path,
            },
        )
    else:
        from sqlalchemy.dialects.mysql import insert as mysql_insert

        my_stmt = mysql_insert(table)
        stmt = my_stmt.on_duplicate_key_update(
            last_seen=my_stmt.inserted.last_seen,
            times_seen=table.c.times_seen + my_stmt.inserted.times_seen,
            reason_code=my_stmt.inserted.reason_code,
            reason=my_stmt.inserted.reason,
            source_path=my_stmt.inserted.source_path,
        )
    session.execute(stmt, rows)


# --- queries (status / rejects CLI) ------------------------------------------


def unresolved_summary(session: Session) -> list[tuple[str, int, datetime.datetime]]:
    """(reason_code, count, newest last_seen) per reason, unresolved only."""
    stmt = (
        select(
            Reject.reason_code,
            func.count(Reject.id),
            func.max(Reject.last_seen),
        )
        .where(Reject.resolved_at.is_(None))
        .group_by(Reject.reason_code)
        .order_by(func.count(Reject.id).desc())
    )
    return [(code, count, last) for code, count, last in session.execute(stmt)]


def export_unresolved(
    session: Session, directory: pathlib.Path
) -> tuple[int, int]:
    """Write unresolved, untruncated payloads as re-ingestable files.

    The repair workflow: export, fix, ingest. TLE payloads concatenate
    into ``rejects.tle``; OMM payloads into ``rejects.json`` (an array of
    the original records). Returns (tle_count, omm_count).
    """
    directory.mkdir(parents=True, exist_ok=True)
    stmt = (
        select(Reject.format, Reject.payload)
        .where(Reject.resolved_at.is_(None), Reject.truncated.is_(False))
        .order_by(Reject.id)
    )
    tle_lines: list[str] = []
    omm_records: list = []
    for fmt, payload in session.execute(stmt):
        if fmt == "tle":
            tle_lines.append(payload.rstrip("\n"))
        else:
            try:
                omm_records.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    tle_count = 0
    if tle_lines:
        path = directory / "rejects.tle"
        path.write_text("\n".join(tle_lines) + "\n", encoding="utf-8")
        tle_count = len(tle_lines)
    if omm_records:
        path = directory / "rejects.json"
        path.write_text(json.dumps(omm_records, indent=1), encoding="utf-8")
    return tle_count, len(omm_records)


def iter_readmission_candidates(
    session: Session, horizon: datetime.datetime
) -> Iterable[Reject]:
    """Unresolved future-epoch rejects whose epoch has come inside the
    horizon — eligible for re-admission (their source file will never be
    re-read, so this is the only path back in). Truncated payloads cannot
    be reconstructed and are skipped."""
    stmt = select(Reject).where(
        Reject.resolved_at.is_(None),
        Reject.reason_code == str(ReasonCode.FUTURE_EPOCH),
        Reject.truncated.is_(False),
        Reject.epoch.is_not(None),
        Reject.epoch <= horizon,
    )
    return session.execute(stmt).scalars().all()
