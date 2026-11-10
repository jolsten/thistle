"""Export: full-fidelity CSV round-trip, TLE concat, CCSDS OMM forms."""

import json
import pathlib

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from .conftest import TLES, engine_ingest, make_settings
from thistle_db.export import export
from thistle_db.model import Base, Reject, TLE

DATA = pathlib.Path(__file__).parent / "data"
ONE = json.loads((DATA / "one.json").read_text())


def _fresh_sqlite(tmp_path, name):
    engine = create_engine(f"sqlite:///{tmp_path / name}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _populate(db_session, tmp_path) -> int:
    settings = make_settings(tmp_path)
    path = tmp_path / "in.tle"
    path.write_text("".join(f"{a}\n{b}\n" for a, b in TLES))
    engine_ingest(db_session, settings, files=[path, DATA / "one.json"])
    return db_session.execute(select(func.count(TLE.id))).scalar_one()


def test_csv_round_trip_restores_everything(db_session: Session, tmp_path):
    count = _populate(db_session, tmp_path)
    csv_path = tmp_path / "export.csv"
    assert export(db_session, "csv", csv_path) == count

    restored = _fresh_sqlite(tmp_path, "restore.db")
    settings = make_settings(tmp_path / "restore_out")
    report = engine_ingest(restored, settings, files=[csv_path])
    assert report.records_inserted == count

    # Identity survives: the two stores hold identical hash sets.
    source_hashes = set(db_session.execute(select(TLE.record_hash)).scalars())
    restored_hashes = set(restored.execute(select(TLE.record_hash)).scalars())
    assert restored_hashes == source_hashes
    assert (
        restored.execute(select(func.count(Reject.id))).scalar_one() == 0
    )


def test_csv_round_trip_preserves_lineless_identity(
    db_session: Session, tmp_path
):
    lineless = {k: v for k, v in ONE.items() if not k.startswith("TLE_LINE")}
    path = tmp_path / "lineless.json"
    path.write_text(json.dumps([lineless]))
    settings = make_settings(tmp_path)
    engine_ingest(db_session, settings, files=[path])
    original_hash = db_session.execute(select(TLE.record_hash)).scalar_one()

    csv_path = tmp_path / "export.csv"
    export(db_session, "csv", csv_path)
    restored = _fresh_sqlite(tmp_path, "restore.db")
    engine_ingest(
        restored, make_settings(tmp_path / "out2"), files=[csv_path]
    )
    # The exported RECORD_HASH wins over a recomputed semantic hash, so
    # identity is stable across export -> restore even though element
    # columns re-serialize at single precision.
    assert (
        restored.execute(select(TLE.record_hash)).scalar_one() == original_hash
    )

    # And re-ingesting the export into the SAME store is a no-op.
    report = engine_ingest(db_session, settings, files=[csv_path])
    assert report.records_inserted == 0


def test_tle_export_is_verbatim(db_session: Session, tmp_path):
    _populate(db_session, tmp_path)
    out = tmp_path / "export.tle"
    count = export(db_session, "tle", out)
    text = out.read_text()
    for line1, line2 in TLES:
        assert line1 in text and line2 in text
    assert count == text.count("\n") / 2


def test_omm_xml_and_kvn(db_session: Session, tmp_path):
    _populate(db_session, tmp_path)
    xml_path = tmp_path / "export.xml"
    export(db_session, "omm-xml", xml_path)
    xml = xml_path.read_text()
    assert 'version="2.0"' in xml
    assert "<MEAN_MOTION>" in xml
    assert "VANGUARD 2" in xml
    assert 'parameter="TLE_LINE1"' in xml  # verbatim lines ride user-defined
    import xml.etree.ElementTree as ET

    ET.fromstring(xml)  # well-formed

    kvn_path = tmp_path / "export.kvn"
    export(db_session, "omm-kvn", kvn_path)
    kvn = kvn_path.read_text()
    assert "CCSDS_OMM_VERS" in kvn
    assert "MEAN_ELEMENT_THEORY" in kvn
