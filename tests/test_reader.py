import pathlib

from thistle.reader import read_tle_file

def test_reader_pair_count():
    file = "tests/data/25544.tle"
    text = pathlib.Path(file).read_text().splitlines()
    tles = read_tle_file(file)
    assert len(text) / 2 == len(tles)
    
def test_reader_line_length():
    file = "tests/data/25544.tle"
    tles = read_tle_file(file)
    for tle in tles:
        assert len(tle[0]) == 69
        assert len(tle[1]) == 69
