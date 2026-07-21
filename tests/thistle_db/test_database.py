import pathlib

import pytest
from sqlalchemy import exc
from sqlalchemy.orm import Session

from .conftest import TLE_FILES, TLES
from thistle_db.model import TLE
from thistle_db.reader import TLETuple, read_tle, unique


@pytest.mark.parametrize("tle", TLES)
def test_add(tle: TLETuple, db_session: Session):
    line1, line2 = tle
    item = TLE.from_twoline(line1, line2)
    db_session.add(item)
    print(db_session.query(TLE).all())
    db_session.commit()


@pytest.mark.parametrize("tle", TLES)
def test_add_twice(tle: TLETuple, db_session: Session):
    item = TLE.from_twoline(*tle)
    db_session.add(item)
    db_session.commit()

    with pytest.raises(exc.IntegrityError):
        item = TLE.from_twoline(*tle)
        db_session.add(item)
        try:
            db_session.commit()
        except exc.IntegrityError as err:
            db_session.rollback()
            raise err


@pytest.mark.parametrize("file", TLE_FILES)
class TestAddAll:
    def test_add_all(self, file: pathlib.Path, db_session: Session):
        tles = read_tle(file)
        tles = unique(tles)
        items = [TLE.from_twoline(line1, line2) for line1, line2 in tles]
        db_session.add_all(items)
        db_session.commit()


def test_add_many_twice(db_session: Session):
    for line1, line2 in TLES:
        item = TLE.from_twoline(line1, line2)
        db_session.add(item)
        db_session.commit()

    for line1, line2 in TLES:
        item = TLE.from_twoline(line1, line2)
        db_session.add(item)
        try:
            db_session.commit()
        except exc.IntegrityError:
            db_session.rollback()
