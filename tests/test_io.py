import pytest

from thistle.io import read_tle, read_tles, write_tle, write_tles
from thistle.utils import tle_epoch, tle_satnum

from .conftest import DAILY_FILES, OBJECT_FILES


@pytest.mark.parametrize("file", DAILY_FILES + OBJECT_FILES)
def test_read_one(file):
    tles = read_tle(file)
    assert len(tles)


def test_read_many():
    tles = read_tles(DAILY_FILES + OBJECT_FILES)
    assert len(tles)


# --- read_tle edge cases ---


def test_read_empty_file(tmp_path):
    f = tmp_path / "empty.tle"
    f.write_text("")
    assert read_tle(f) == []


def test_read_blank_lines(tmp_path):
    content = (
        "\n"
        "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996\n"
        "\n"
        "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05\n"
        "\n"
    )
    f = tmp_path / "blanks.tle"
    f.write_text(content)
    tles = read_tle(f)
    assert len(tles) == 1


def test_read_orphan_line1(tmp_path):
    """A line-1 without a matching line-2 should be skipped."""
    content = (
        "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996\n"
    )
    f = tmp_path / "orphan.tle"
    f.write_text(content)
    assert read_tle(f) == []


def test_read_three_line_format(tmp_path):
    """Three-line TLE format (name + line1 + line2) should parse correctly."""
    content = (
        "ISS (ZARYA)\n"
        "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996\n"
        "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05\n"
    )
    f = tmp_path / "threeline.tle"
    f.write_text(content)
    tles = read_tle(f)
    assert len(tles) == 1
    assert tles[0][0].startswith("1 ")
    assert tles[0][1].startswith("2 ")


# --- write_tle / write_tles ---


def test_write_roundtrip(tmp_path):
    """Write TLEs and read them back; content should match."""
    original = read_tle(OBJECT_FILES[0])
    out = tmp_path / "out.tle"
    write_tle(out, original)
    result = read_tle(out)
    assert result == original


def test_write_unique(tmp_path):
    """Duplicate TLEs should be removed when unique=True."""
    line1 = "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996"
    line2 = "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05"
    tles = [(line1, line2), (line1, line2), (line1, line2)]
    out = tmp_path / "unique.tle"
    write_tle(out, tles, unique=True)
    result = read_tle(out)
    assert len(result) == 1
    assert result[0] == (line1, line2)


def test_write_sorted(tmp_path):
    """TLEs should be sorted by satnum then epoch when sort=True."""
    original = read_tle(OBJECT_FILES[0])
    reversed_tles = list(reversed(original))
    out = tmp_path / "sorted.tle"
    write_tle(out, reversed_tles, sort=True)
    result = read_tle(out)
    for a, b in zip(result, result[1:]):
        sn_a, sn_b = tle_satnum(a), tle_satnum(b)
        if sn_a == sn_b:
            assert tle_epoch(a) <= tle_epoch(b)


def test_write_tles_multiple_files(tmp_path):
    """write_tles should write each file independently."""
    tles_a = read_tle(OBJECT_FILES[0])
    tles_b = read_tle(OBJECT_FILES[1])
    file_a = tmp_path / "a.tle"
    file_b = tmp_path / "b.tle"
    write_tles({file_a: tles_a, file_b: tles_b}, unique=False)
    assert read_tle(file_a) == tles_a
    assert read_tle(file_b) == tles_b
