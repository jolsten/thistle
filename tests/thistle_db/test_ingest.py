"""The delta engine: idempotence, enrichment, state tracking, crash replay."""

import json
import pathlib

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .conftest import TLES, engine_ingest, make_settings, with_checksum
from thistle_db.config import IngestSource
from thistle_db.ingest import CHUNK_SIZE
from thistle_db.model import IngestFile, OmmMetadata, Reject, TLE
from thistle_db.outputs import OutputApplier
from thistle_db.reader import read_tle, unique

DATA = pathlib.Path(__file__).parent / "data"
BIG_FILE = DATA / "20250418.txt"

ONE = json.loads((DATA / "one.json").read_text())


def _tle_count(session: Session) -> int:
    return session.execute(select(func.count(TLE.id))).scalar_one()


def _write_tle(path: pathlib.Path, *pairs: tuple[str, str]) -> pathlib.Path:
    path.write_text("".join(f"{l1}\n{l2}\n" for l1, l2 in pairs))
    return path


def test_multi_chunk_ingest_and_dedup(db_session: Session, tmp_path):
    """The fixture holds >CHUNK_SIZE records, so multi-chunk stays exercised."""
    expected = len(unique(list(read_tle(BIG_FILE))))
    assert expected > CHUNK_SIZE
    settings = make_settings(tmp_path)
    report = engine_ingest(db_session, settings, files=[BIG_FILE])
    assert report.records_inserted == expected
    assert _tle_count(db_session) == expected


def test_double_ingest_is_noop(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    _write_tle(source_dir / "a.tle", TLES[0], TLES[1])
    sources = [IngestSource(path=str(source_dir), pattern="*.tle")]

    first = engine_ingest(db_session, settings, sources=sources)
    assert first.records_inserted == 2
    second = engine_ingest(db_session, settings, sources=sources)
    assert second.records_inserted == 0
    assert second.files_skipped == 1  # unchanged: not even opened
    assert _tle_count(db_session) == 2

    # --force re-reads; dedup still absorbs everything.
    third = engine_ingest(db_session, settings, sources=sources, force=True)
    assert third.files_ingested == 1
    assert third.records_inserted == 0


def test_refresh_on_touch(db_session: Session, tmp_path):
    """mtime changes but content doesn't: state refreshes without a parse."""
    settings = make_settings(tmp_path)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    path = _write_tle(source_dir / "a.tle", TLES[0])
    sources = [IngestSource(path=str(source_dir), pattern="*.tle")]
    engine_ingest(db_session, settings, sources=sources)

    stat = path.stat()
    import os

    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    report = engine_ingest(db_session, settings, sources=sources)
    assert report.files_refreshed == 1
    assert report.records_seen == 0  # never parsed


def test_failed_file_retried_next_scan(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    bad = source_dir / "bad.json"
    bad.write_text("{ not json")
    sources = [IngestSource(path=str(source_dir), pattern="*.json")]

    report = engine_ingest(db_session, settings, sources=sources)
    assert report.files_failed == 1
    assert (
        db_session.execute(select(func.count(IngestFile.id))).scalar_one() == 0
    )
    # Repaired in place: next scan ingests it.
    bad.write_text(json.dumps([ONE]))
    report2 = engine_ingest(db_session, settings, sources=sources)
    assert report2.files_ingested == 1
    assert report2.records_inserted == 1


def test_tle_then_omm_enriches_same_row(db_session: Session, tmp_path):
    """'TLE first, identical-lined OMM later' lands on one row + sidecar."""
    settings = make_settings(tmp_path)
    tle_file = _write_tle(
        tmp_path / "one.tle", (ONE["TLE_LINE1"], ONE["TLE_LINE2"])
    )
    engine_ingest(db_session, settings, files=[tle_file])
    assert _tle_count(db_session) == 1
    assert (
        db_session.execute(select(func.count(OmmMetadata.id))).scalar_one() == 0
    )

    engine_ingest(db_session, settings, files=[DATA / "one.json"])
    assert _tle_count(db_session) == 1  # deduped on the line hash
    metadata = db_session.execute(select(OmmMetadata)).scalar_one()
    assert metadata.object_name == ONE["OBJECT_NAME"]
    assert metadata.gp_id == int(ONE["GP_ID"])


def test_omm_reissue_merge_newest_creation_date_wins(
    db_session: Session, tmp_path
):
    settings = make_settings(tmp_path)
    old = dict(ONE)
    old["OBJECT_NAME"] = "OLD NAME"
    old["CREATION_DATE"] = "2025-01-01T00:00:00"
    new = dict(ONE)
    new["OBJECT_NAME"] = "NEW NAME"
    new["CREATION_DATE"] = "2025-06-01T00:00:00"

    (tmp_path / "old.json").write_text(json.dumps([old]))
    (tmp_path / "new.json").write_text(json.dumps([new]))

    engine_ingest(db_session, settings, files=[tmp_path / "old.json"])
    engine_ingest(db_session, settings, files=[tmp_path / "new.json"])
    metadata = db_session.execute(select(OmmMetadata)).scalar_one()
    assert metadata.object_name == "NEW NAME"

    # A *stale* re-delivery does not regress the newer metadata.
    engine_ingest(
        db_session, settings, files=[tmp_path / "old.json"], force=True
    )
    metadata = db_session.execute(select(OmmMetadata)).scalar_one()
    assert metadata.object_name == "NEW NAME"
    assert _tle_count(db_session) == 1


def test_checksum_mismatch_rejected_and_retained(db_session: Session, tmp_path):
    line1, line2 = TLES[0]
    corrupted = line1[:30] + ("9" if line1[30] != "9" else "8") + line1[31:]
    settings = make_settings(tmp_path)
    path = _write_tle(tmp_path / "bad.tle", (corrupted, line2))

    report = engine_ingest(db_session, settings, files=[path])
    assert report.records_inserted == 0
    assert report.records_rejected == 1
    reject = db_session.execute(select(Reject)).scalar_one()
    assert reject.reason_code == "checksum"
    assert corrupted in reject.payload
    assert reject.norad_cat_id == 22
    assert reject.resolved_at is None

    # Re-rejection dedups: times_seen bumps, no second row.
    engine_ingest(db_session, settings, files=[path])
    reject = db_session.execute(select(Reject)).scalar_one()
    assert reject.times_seen == 2


def test_checksum_verification_can_be_disabled(db_session: Session, tmp_path):
    line1, line2 = TLES[0]
    corrupted = line1[:30] + ("9" if line1[30] != "9" else "8") + line1[31:]
    settings = make_settings(tmp_path, verify_checksums=False)
    path = _write_tle(tmp_path / "bad.tle", (corrupted, line2))
    report = engine_ingest(db_session, settings, files=[path])
    assert report.records_inserted == 1
    assert report.records_rejected == 0


def test_future_epoch_reject_and_readmission(db_session: Session, tmp_path):
    line1, line2 = TLES[0]
    future1 = with_checksum(line1[:18] + "56107.90268499" + line1[32:])
    path = _write_tle(tmp_path / "future.tle", (future1, line2))

    settings = make_settings(tmp_path)  # 30-day horizon
    report = engine_ingest(db_session, settings, files=[path])
    assert report.records_inserted == 0
    reject = db_session.execute(select(Reject)).scalar_one()
    assert reject.reason_code == "future_epoch"
    assert reject.epoch is not None and reject.epoch.year == 2056

    # Horizon widened (stand-in for time passing): the re-admission pass
    # promotes the reject into the store and stamps it resolved — its
    # source file is state-tracked and will never be re-read.
    wide = make_settings(tmp_path, max_epoch_ahead_days=20000)
    report2 = engine_ingest(db_session, wide, sources=[])
    assert report2.records_readmitted == 1
    assert _tle_count(db_session) == 1
    reject = db_session.execute(select(Reject)).scalar_one()
    assert reject.resolved_at is not None

    # Subsequent runs have no candidates left.
    report3 = engine_ingest(db_session, wide, sources=[])
    assert report3.records_readmitted == 0


def test_unparseable_and_no_norad_rejects(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    garbage = tmp_path / "garbage.tle"
    garbage.write_text(
        "1 THIS IS NOT A TLE AT ALL, JUST TEXT THAT STARTS RIGHT\n"
        "2 AND A SECOND LINE TO PAIR WITH IT                    \n"
    )
    report = engine_ingest(db_session, settings, files=[garbage])
    assert report.records_rejected == 1
    assert db_session.execute(select(Reject)).scalar_one().reason_code in (
        "unparseable",
        "malformed_element",
    )

    no_norad = tmp_path / "nonorad.json"
    omm = {k: v for k, v in ONE.items() if k != "NORAD_CAT_ID"}
    omm.pop("TLE_LINE1", None)
    omm.pop("TLE_LINE2", None)
    no_norad.write_text(json.dumps([omm]))
    report2 = engine_ingest(db_session, settings, files=[no_norad])
    assert report2.records_rejected == 1
    rows = db_session.execute(select(Reject)).scalars().all()
    assert "no_norad_id" in {r.reason_code for r in rows}


def test_replay_after_lost_state_converges(db_session: Session, tmp_path):
    """Crash-safety: replaying a file whose state never committed changes
    nothing — every sink absorbs the duplicates."""
    settings = make_settings(tmp_path)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    _write_tle(source_dir / "a.tle", TLES[0], TLES[1], TLES[2])
    sources = [IngestSource(path=str(source_dir), pattern="*.tle")]

    engine_ingest(db_session, settings, sources=sources)
    objects_dir = tmp_path / "objects"
    reference = {
        p.name: p.read_bytes() for p in sorted(objects_dir.iterdir())
    }
    count = _tle_count(db_session)

    # Simulate the crash window: rows + outputs landed, file state did not.
    db_session.execute(delete(IngestFile))
    db_session.commit()

    report = engine_ingest(db_session, settings, sources=sources)
    assert report.files_ingested == 1  # re-read in full
    assert report.records_inserted == 0
    assert _tle_count(db_session) == count
    replayed = {p.name: p.read_bytes() for p in sorted(objects_dir.iterdir())}
    assert replayed == reference  # byte-identical outputs


def test_crash_between_store_and_outputs_converges(
    db_session: Session, tmp_path, monkeypatch
):
    """Kill between flush steps 1 and 2, then re-run: byte-identical end
    state. The store committed, the outputs and file state did not."""
    settings = make_settings(tmp_path)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    _write_tle(source_dir / "a.tle", TLES[0], TLES[1])
    sources = [IngestSource(path=str(source_dir), pattern="*.tle")]

    original_apply = OutputApplier.apply

    def exploding_apply(self, delta):
        raise RuntimeError("simulated crash before outputs")

    monkeypatch.setattr(OutputApplier, "apply", exploding_apply)
    try:
        engine_ingest(db_session, settings, sources=sources)
    except RuntimeError:
        pass
    monkeypatch.setattr(OutputApplier, "apply", original_apply)

    assert _tle_count(db_session) == 2  # store committed first
    assert (
        db_session.execute(select(func.count(IngestFile.id))).scalar_one() == 0
    )
    assert not (tmp_path / "objects").exists() or not list(
        (tmp_path / "objects").iterdir()
    )

    report = engine_ingest(db_session, settings, sources=sources)
    assert report.records_inserted == 0  # dedup absorbed the replay
    assert report.files_ingested == 1
    # Both records are object 22: one file, two records, epoch order.
    obj = tmp_path / "objects" / "22.tle"
    assert obj.read_text().count("\n") == 4
