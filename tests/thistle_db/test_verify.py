"""verify: count reconciliation, orphans, reject resolution stamping."""


from sqlalchemy import select
from sqlalchemy.orm import Session

from .conftest import TLES, engine_ingest, make_settings
from thistle_db.model import Reject
from thistle_db.verify import stamp_resolved_rejects, verify_outputs

L1, L2 = TLES[0]
BAD_L1 = L1[:30] + ("9" if L1[30] != "9" else "8") + L1[31:]


def _ingested(db_session, tmp_path):
    settings = make_settings(tmp_path)
    path = tmp_path / "in.tle"
    path.write_text("".join(f"{a}\n{b}\n" for a, b in TLES))
    engine_ingest(db_session, settings, files=[path])
    return settings


def test_consistent_after_ingest(db_session: Session, tmp_path):
    settings = _ingested(db_session, tmp_path)
    report = verify_outputs(db_session, settings.output)
    assert report.ok
    assert report.outputs_checked == 2


def test_detects_truncation(db_session: Session, tmp_path):
    settings = _ingested(db_session, tmp_path)
    obj = tmp_path / "objects" / "22.tle"
    lines = obj.read_text().splitlines()
    obj.write_text("\n".join(lines[:-2]) + "\n")  # drop the last record
    report = verify_outputs(db_session, settings.output)
    issues = [i for i in report.issues if i.kind == "count"]
    assert len(issues) == 1
    assert issues[0].expected == 2 and issues[0].actual == 1


def test_detects_missing_and_orphan(db_session: Session, tmp_path):
    settings = _ingested(db_session, tmp_path)
    (tmp_path / "objects" / "81069.tle").unlink()
    (tmp_path / "objects" / "99999.tle").write_text(f"{L1}\n{L2}\n")
    report = verify_outputs(db_session, settings.output)
    kinds = sorted(i.kind for i in report.issues)
    assert kinds == ["missing", "orphan"]


def test_stamps_resolved_when_payload_reaches_store(
    db_session: Session, tmp_path
):
    # Rejected under checksum verification...
    strict = make_settings(tmp_path)
    bad = tmp_path / "bad.tle"
    bad.write_text(f"{BAD_L1}\n{L2}\n")
    engine_ingest(db_session, strict, files=[bad])
    reject = db_session.execute(select(Reject)).scalar_one()
    assert reject.resolved_at is None

    # ...later ingested with verification off: same payload hash lands in
    # the store, and verify stamps the reject resolved.
    lenient = make_settings(tmp_path, verify_checksums=False)
    engine_ingest(db_session, lenient, files=[bad], force=False)
    # force-free named ingest always re-reads
    stamped = stamp_resolved_rejects(db_session)
    assert stamped == 1
    reject = db_session.execute(select(Reject)).scalar_one()
    assert reject.resolved_at is not None
