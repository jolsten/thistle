import pathlib

import pytest

from thistle.reader import TLEReader, _parse_tle_file

@pytest.mark.parametrize("file", list(pathlib.Path("tests/data").glob("*.tle")))
class TestTleFileParser:
    def test_tuple_count(self, file: str) -> None:
        text = pathlib.Path(file).read_text().splitlines()
        tles = _parse_tle_file(file)
        assert len(text) / 2 == len(tles)

    def test_line_length(self, file: str) -> None:
        tles = _parse_tle_file(file)
        for tle in tles:
            assert len(tle[0]) == 69
            assert len(tle[1]) == 69


def test_loader_iss():
    file = "tests/data/25544.tle"
    text = pathlib.Path(file).read_text().splitlines()

    loader = TLEReader([25544])
    loader.read(file)
    satrecs = loader.select(25544)
    assert len(satrecs) == len(text) / 2


@pytest.mark.parametrize("satnum", [
    25544,
    31113,
    45556,
])
def test_loader_leo(satnum: int):
    file = "tests/data/leo.tle"
    loader = TLEReader()
    loader.read(file)
    satrecs = loader.select(satnum)
    assert len(satrecs) > 0
    for sat in satrecs:
        assert sat.satnum == satnum
