import pathlib

import pytest

from thistle_db.reader import detect_format, read_omm_json


JSON_FILES = list(pathlib.Path("tests/thistle_db/data/").glob("*.json"))


def test_detect_format_tle():
    assert detect_format("foo.tle") == "tle"
    assert detect_format("foo.txt") == "tle"
    assert detect_format("foo.3le") == "tle"


def test_detect_format_omm():
    assert detect_format("foo.json") == "omm_json"
    assert detect_format("foo.csv") == "omm_csv"
    assert detect_format("foo.xml") == "omm_xml"


def test_detect_format_unknown():
    assert detect_format("foo.xyz") == "tle"


@pytest.mark.parametrize("file", JSON_FILES)
def test_read_omm_json(file: pathlib.Path):
    records = read_omm_json(file)
    assert isinstance(records, list)
    assert len(records) > 0
    # All records should be dicts
    for record in records:
        assert isinstance(record, dict)


def test_read_omm_json_has_tle_lines():
    """Verify Space-Track JSON includes TLE_LINE1/TLE_LINE2."""
    records = read_omm_json("tests/thistle_db/data/one.json")
    assert len(records) >= 1
    record = records[0]
    assert "TLE_LINE1" in record
    assert "TLE_LINE2" in record
    assert record["TLE_LINE1"].startswith("1 ")
    assert record["TLE_LINE2"].startswith("2 ")
