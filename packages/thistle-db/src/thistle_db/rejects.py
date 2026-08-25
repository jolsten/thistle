"""Quarantine for element sets that could not be ingested.

A rejected record is one that never reaches the database: text sgp4 cannot
parse, an elset whose mandatory elements are not finite numbers (see
``model.MalformedElsetError``), or a row the database itself refuses. Its
raw text would otherwise survive only in a log line that rotates away, so
with ``[ingest] reject_dir`` configured each such record is copied to a
quarantine directory for later inspection.

The layout mirrors the source tree, so provenance is visible in the path
and a repaired file is directly re-ingestable::

    /data/spacetrack/daily/2006/20060101.tle   (source)
    rejects/daily/2006/20060101.tle            (rejected records, verbatim)
    rejects/daily/2006/20060101.tle.log        (one reason per record)

Artifacts are written in the source file's own format, so
``thistle-db ingest rejects/daily/2006/20060101.tle`` works once the
records are fixed. OMM sources quarantine as JSON (``<name>.json``), which
``detect_format`` reads back as Space-Track OMM.

The directory is a **live view of what is currently broken**, not an
append-only archive: each ingest of a source file overwrites that file's
artifacts, and a run with no rejects deletes them. Total volume is
therefore bounded by the set of currently-broken records rather than
growing with time, so no retention policy is needed.
"""

import datetime
import hashlib
import json
import pathlib
from typing import Iterable, Optional, Protocol

from loguru import logger

# Copy at most this many records from one source file. Beyond it, a file is
# not "a good feed with some bad lines" but a wrong file (mis-encoded, still
# gzipped, truncated), and copying it would duplicate the whole delivery
# into the quarantine directory. A TLE record is ~142 bytes, so this caps
# one file's quarantine at ~1.4 MB; past the cap only a marker is written.
# Deliberately a constant, not config: it is a fuse, not a tuning knob.
REJECT_MAX_RECORDS = 10_000


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
        reason: str,
        *,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        record: Optional[dict] = None,
    ) -> None: ...


class _NullSink:
    """Inert sink — the default for library callers and unconfigured runs."""

    def seen(self, n: int = 1) -> None:
        pass

    def reject(self, reason: str, **kwargs) -> None:
        pass


NO_REJECTS: RejectSink = _NullSink()
"""Shared inert sink: quarantine disabled, rejects are only logged."""


def reject_path(
    source: pathlib.Path,
    reject_dir: pathlib.Path,
    *,
    root: Optional[pathlib.Path] = None,
    namespace: Optional[str] = None,
) -> pathlib.Path:
    """Where `source`'s quarantine artifacts live under `reject_dir`.

    Files under a configured source directory mirror their path below a
    namespace directory (the source root's name, or `namespace` when two
    configured roots share a basename). Files named directly on the command
    line have no root and land flat, under their own name.
    """
    if root is not None:
        try:
            relative = source.resolve().relative_to(root.resolve())
        except ValueError:
            relative = pathlib.Path(source.name)
        return reject_dir / (namespace or root.name) / relative
    return reject_dir / source.name


def namespaces(roots: Iterable[pathlib.Path]) -> dict[str, str]:
    """Namespace directory name per configured source root, keyed by path str.

    The root's basename, except where two configured roots share one — those
    get an 8-hex suffix of the resolved path so their rejects cannot collide
    in the quarantine tree.
    """
    roots = list(roots)
    counts: dict[str, int] = {}
    for root in roots:
        counts[root.name] = counts.get(root.name, 0) + 1
    result: dict[str, str] = {}
    for root in roots:
        if counts[root.name] > 1:
            digest = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
            result[str(root)] = f"{root.name}-{digest[:8]}"
        else:
            result[str(root)] = root.name
    return result


class RejectWriter:
    """Collects one source file's rejected records and writes its artifacts.

    Use as a context manager: artifacts are written — or removed, when the
    file now ingests cleanly — on exit. Filesystem errors are contained: a
    quarantine failure is logged and never propagates, since it must not
    turn an otherwise working ingest into a failed one.
    """

    def __init__(self, source: pathlib.Path, target: pathlib.Path, *, omm: bool):
        self.source = source
        # The artifact must carry an extension matching its *content* so
        # detect_format reads it back correctly on re-ingest.
        self.target = target.with_name(target.name + ".json") if omm else target
        self.log_path = self.target.with_name(self.target.name + ".log")
        self.marker_path = target.with_name(target.name + ".truncated")
        self.omm = omm
        self.total = 0
        self.rejected = 0
        self.reasons: dict[str, int] = {}
        self._buffer: list[
            tuple[str, Optional[str], Optional[str], Optional[dict]]
        ] = []

    def __enter__(self) -> "RejectWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- collection ------------------------------------------------------

    def seen(self, n: int = 1) -> None:
        self.total += n

    def reject(
        self,
        reason: str,
        *,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        record: Optional[dict] = None,
    ) -> None:
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        # Past the cap, keep counting (the marker reports totals) but stop
        # buffering, so a wholly corrupt file cannot be held in memory.
        if self.rejected <= REJECT_MAX_RECORDS:
            self._buffer.append((reason, line1, line2, record))

    # -- output ----------------------------------------------------------

    def close(self) -> None:
        try:
            self._write()
        except OSError as exc:
            logger.warning(f"Could not write quarantine for {self.source}: {exc}")

    def _write(self) -> None:
        if self.rejected == 0:
            # A previously broken file that now ingests cleanly: drop stale
            # artifacts so the directory shows only what is currently broken.
            self._unlink(self.target, self.log_path, self.marker_path)
            return

        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.rejected > REJECT_MAX_RECORDS:
            self._unlink(self.target, self.log_path)
            self._write_marker()
            return

        self._unlink(self.marker_path)
        if self.omm:
            records = [rec for _, _, _, rec in self._buffer if rec is not None]
            with open(self.target, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=1)
        else:
            with open(self.target, "w", encoding="utf-8") as f:
                f.write(
                    "".join(
                        f"{line1}\n{line2}\n"
                        for _, line1, line2, _ in self._buffer
                        if line1 is not None and line2 is not None
                    )
                )
        self._write_log()

    def _write_log(self) -> None:
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# source: {self.source}\n")
            f.write(f"# written: {stamp}\n")
            f.write(f"# rejected {self.rejected} of {self.total} records\n")
            for reason, line1, _, record in self._buffer:
                ident = line1 if line1 is not None else _record_label(record)
                f.write(f"{reason}\t{ident}\n")

    def _write_marker(self) -> None:
        with open(self.marker_path, "w", encoding="utf-8") as f:
            f.write(f"source: {self.source}\n")
            f.write(f"records attempted: {self.total}\n")
            f.write(f"records rejected: {self.rejected}\n")
            f.write(
                f"not quarantined: over the {REJECT_MAX_RECORDS}-record cap "
                "- inspect the source file itself\n"
            )
            f.write("reasons:\n")
            for reason, count in sorted(
                self.reasons.items(), key=lambda item: item[1], reverse=True
            ):
                f.write(f"  {count}\t{reason}\n")

    @staticmethod
    def _unlink(*paths: pathlib.Path) -> None:
        for path in paths:
            path.unlink(missing_ok=True)

    # -- reporting -------------------------------------------------------

    def summary(self) -> Optional[str]:
        """One-line report of what was rejected, or None when nothing was."""
        if self.rejected == 0:
            return None
        capped = self.rejected > REJECT_MAX_RECORDS
        where = self.marker_path if capped else self.target
        return f"{self.rejected} of {self.total} records rejected -> {where}"


def _record_label(record: Optional[dict]) -> str:
    """Short identifier for an OMM record with no usable TLE lines."""
    if not record:
        return "<unknown record>"
    return str(record.get("OBJECT_NAME") or record.get("NORAD_CAT_ID") or record)
