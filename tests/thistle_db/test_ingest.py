from sqlalchemy.orm import Session

from tests.conftest import TLES
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
