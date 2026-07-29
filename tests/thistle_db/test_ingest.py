from sqlalchemy.orm import Session

from .conftest import TLES
from thistle_db.ingest import ingest_tles
from thistle_db.model import TLE


def test_ingest_tles(db_session: Session):
    count = ingest_tles(db_session, TLES)
    assert count == len(TLES)
    assert db_session.query(TLE).count() == len(TLES)


def test_ingest_tles_idempotent(db_session: Session):
    count1 = ingest_tles(db_session, TLES)
    assert count1 == len(TLES)

    count2 = ingest_tles(db_session, TLES)
    assert count2 == 0
    assert db_session.query(TLE).count() == len(TLES)


def test_ingest_tles_malformed(db_session: Session):
    bad_tles = [("1 INVALID LINE", "2 ALSO INVALID")]
    count = ingest_tles(db_session, bad_tles)
    assert count == 0


def test_ingest_tles_mixed_good_and_bad(db_session: Session):
    mixed = list(TLES) + [("1 INVALID", "2 INVALID")]
    count = ingest_tles(db_session, mixed)
    assert count == len(TLES)


def test_omm_after_tle_attaches_metadata_without_duplicating(db_session: Session):
    """Same element set delivered as TLE first, then OMM: one row, metadata
    attached to it (dedup via line_hash)."""
    from thistle_db.ingest import ingest_omm
    from thistle_db.model import OmmMetadata

    ingest_tles(db_session, TLES[:1])

    omm = {
        "TLE_LINE1": TLES[0][0],
        "TLE_LINE2": TLES[0][1],
        "OBJECT_NAME": "EXPLORER 7",
        "GP_ID": "12345",
    }
    inserted = ingest_omm(db_session, [omm])
    assert inserted == 0  # TLE row already present
    assert db_session.query(TLE).count() == 1
    meta = db_session.query(OmmMetadata).one()
    assert meta.object_name == "EXPLORER 7"
    assert meta.tle_id == db_session.query(TLE).one().id

    # Idempotent: re-delivery attaches nothing new.
    assert ingest_omm(db_session, [omm]) == 0
    assert db_session.query(OmmMetadata).count() == 1
