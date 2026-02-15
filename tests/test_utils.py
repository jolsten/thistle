"""Tests for thistle.utils."""

import datetime
import pathlib
from typing import Callable, Iterable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sgp4.conveniences import jday_datetime

from thistle.utils import (
    DATETIME64_MAX,
    DATETIME64_MIN,
    DATETIME_MAX,
    DATETIME_MIN,
    TIME_SCALE,
    datetime_to_dt64,
    datetime_to_tle_epoch,
    dt64_to_datetime,
    ensure_alpha5,
    from_alpha5,
    group_by,
    jday_datetime64,
    read_tle,
    tle_date,
    tle_epoch,
    tle_satnum,
    to_alpha5,
    unique,
)

from .conftest import DAILY_FILES, OBJECT_FILES


# ---------------------------------------------------------------------------
# datetime / dt64 conversion
# ---------------------------------------------------------------------------


@given(st.datetimes(min_value=DATETIME_MIN, max_value=DATETIME_MAX))
def test_convert_1(time_in: datetime.datetime):
    assert time_in == dt64_to_datetime(datetime_to_dt64(time_in))


@given(
    st.integers(
        min_value=DATETIME64_MIN.astype(int), max_value=DATETIME64_MAX.astype(int)
    )
)
def test_convert_2(integer: int):
    dt64 = np.datetime64(integer, TIME_SCALE)
    assert dt64 == datetime_to_dt64(dt64_to_datetime(dt64))


# ---------------------------------------------------------------------------
# datetime_to_tle_epoch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dt, yy, days",
    [
        (datetime.datetime(2000, 1, 1, 0, 0, 0), 0, 1.0),
        (datetime.datetime(2000, 1, 1, 12, 0, 0), 0, 1.5),
        (datetime.datetime(2000, 1, 2, 0, 0, 0), 0, 2.0),
        (datetime.datetime(2001, 1, 1, 0, 0, 0), 1, 1.0),
        (datetime.datetime(1957, 1, 1, 0, 0, 0), 57, 1.0),
    ],
)
def test_datetime_to_tle_epoch(dt: datetime.datetime, yy: int, days: float):
    got_yy, got_days = datetime_to_tle_epoch(dt.replace(tzinfo=datetime.timezone.utc))
    assert got_yy == yy
    assert got_days == days


# ---------------------------------------------------------------------------
# jday_datetime64
# ---------------------------------------------------------------------------


@given(
    st.lists(
        st.datetimes(
            min_value=DATETIME_MIN,
            max_value=DATETIME_MAX,
            timezones=st.sampled_from([datetime.timezone.utc]),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_jday_datetime64(dt_list: list[datetime.datetime]) -> None:
    exp_jd, exp_fr = [], []
    for dt in dt_list:
        jd, fr = jday_datetime(dt)
        exp_jd.append(jd)
        exp_fr.append(fr)
    exp_jd = np.array(exp_jd, dtype="f8")
    exp_fr = np.array(exp_fr, dtype="f8")

    times = np.array([datetime_to_dt64(dt) for dt in dt_list], dtype="datetime64[us]")
    jd, fr = jday_datetime64(times)

    assert jd == pytest.approx(exp_jd.tolist())
    assert fr == pytest.approx(exp_fr.tolist())


# ---------------------------------------------------------------------------
# TLE field helpers (tle_epoch, tle_date, tle_satnum)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epoch_str, date, epoch, satnum",
    [
        ("57001.00000000", "19570101", 1957001.0, "25544"),
        ("25032.00000000", "20250201", 2025032.0, "25544"),
        ("56366.00000000", "20561231", 2056366.0, "A0001"),
    ],
)
class TestTLEFuncs:
    def test_tle_epoch(self, epoch_str: str, date: str, epoch: float, satnum: str):
        line1 = f"1 25544U 98067A   {epoch_str}  .00020137  00000-0  16538-3 0  9993"
        line2 = "2 25544  51.6335 344.7760 0007976 126.2523 325.9359 15.70406856328906"
        tle = (line1, line2)
        assert epoch == tle_epoch(tle)

    def test_tle_date(self, epoch_str: str, date: str, epoch: float, satnum: str):
        line1 = f"1 25544U 98067A   {epoch_str}  .00020137  00000-0  16538-3 0  9993"
        line2 = "2 25544  51.6335 344.7760 0007976 126.2523 325.9359 15.70406856328906"
        tle = (line1, line2)
        assert tle_date(tle) == date

    def test_tle_satnum(self, epoch_str: str, date: str, epoch: float, satnum: str):
        line1 = f"1 {satnum:5}U 98067A   25077.00000000  .00020137  00000-0  16538-3 0  9993"
        line2 = f"2 {satnum:5}  51.6335 344.7760 0007976 126.2523 325.9359 15.70406856328906"
        tle = (line1, line2)
        assert tle_satnum(tle) == satnum


# ---------------------------------------------------------------------------
# unique / group_by
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, exp",
    [
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZZZZZZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ([1, 1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ],
)
def test_unique(value: Iterable, exp: Iterable):
    value, exp = list(value), list(exp)
    assert unique(value) == exp


@pytest.mark.parametrize(
    "value, exp, key",
    [
        (
            [1, 1.5, 2, 2.5, 3, 3.5],
            {1: [1, 1.5], 2: [2, 2.5], 3: [3, 3.5]},
            lambda x: x // 1,
        ),
        ("AaBb", {"A": ["A", "a"], "B": ["B", "b"]}, lambda x: x.upper()),
    ],
)
def test_group_by(value: Iterable, exp: dict, key: Callable):
    assert group_by(value, key=key) == exp


# ---------------------------------------------------------------------------
# read_tle
# ---------------------------------------------------------------------------


class TestReadTLE:
    """Tests for read_tle."""

    @pytest.mark.parametrize("file", DAILY_FILES + OBJECT_FILES)
    def test_read_one(self, file: pathlib.Path):
        tles = read_tle(file)
        assert len(tles) > 0

    @pytest.mark.parametrize("file", DAILY_FILES + OBJECT_FILES)
    def test_returns_list_of_tuples(self, file: pathlib.Path):
        tles = read_tle(file)
        for tle in tles:
            assert isinstance(tle, tuple)
            assert len(tle) == 2

    @pytest.mark.parametrize("file", DAILY_FILES + OBJECT_FILES)
    def test_line_prefixes(self, file: pathlib.Path):
        """Line 1 starts with '1', line 2 starts with '2'."""
        tles = read_tle(file)
        for line1, line2 in tles:
            assert line1[0] == "1"
            assert line2[0] == "2"

    def test_empty_file(self, tmp_path: pathlib.Path):
        f = tmp_path / "empty.tle"
        f.write_text("")
        assert read_tle(f) == []

    def test_blank_lines(self, tmp_path: pathlib.Path):
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

    def test_orphan_line1(self, tmp_path: pathlib.Path):
        """A line-1 without a matching line-2 should be skipped."""
        content = "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996\n"
        f = tmp_path / "orphan.tle"
        f.write_text(content)
        assert read_tle(f) == []

    def test_orphan_line2(self, tmp_path: pathlib.Path):
        """A line-2 without a preceding line-1 should be skipped."""
        content = "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05\n"
        f = tmp_path / "orphan2.tle"
        f.write_text(content)
        assert read_tle(f) == []

    def test_three_line_format(self, tmp_path: pathlib.Path):
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

    def test_multiple_tles(self, tmp_path: pathlib.Path):
        """File with two TLEs returns both."""
        content = (
            "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996\n"
            "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05\n"
            "1 25544U 98067A   98325.28472222 -.00003657  11563-4  00000+0 0  9996\n"
            "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05\n"
        )
        f = tmp_path / "two.tle"
        f.write_text(content)
        assert len(read_tle(f)) == 2

    def test_no_trailing_whitespace(self, tmp_path: pathlib.Path):
        """Lines should have trailing whitespace stripped."""
        content = (
            "1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0  9996   \n"
            "2 25544 051.5908 168.3788 0125362 086.4185 359.7454 16.05064833    05   \n"
        )
        f = tmp_path / "trailing.tle"
        f.write_text(content)
        tles = read_tle(f)
        assert not tles[0][0].endswith(" ")
        assert not tles[0][1].endswith(" ")


# ---------------------------------------------------------------------------
# Alpha-5 encoding/decoding
# ---------------------------------------------------------------------------


class TestAlpha5:
    """Tests for to_alpha5, from_alpha5, ensure_alpha5."""

    @pytest.mark.parametrize(
        "objnum, alpha5",
        [
            (0, "00000"),
            (1, "00001"),
            (99999, "99999"),
            (100000, "A0000"),
            (148493, "E8493"),
            (182931, "J2931"),
            (234018, "P4018"),
            (301928, "W1928"),
            (339999, "Z9999"),
        ],
    )
    def test_to_alpha5(self, objnum: int, alpha5: str):
        assert to_alpha5(objnum) == alpha5

    @pytest.mark.parametrize(
        "objnum, alpha5",
        [
            (0, "00000"),
            (99999, "99999"),
            (100000, "A0000"),
            (148493, "E8493"),
            (182931, "J2931"),
            (234018, "P4018"),
            (301928, "W1928"),
            (339999, "Z9999"),
        ],
    )
    def test_from_alpha5(self, objnum: int, alpha5: str):
        assert from_alpha5(alpha5) == objnum

    @pytest.mark.parametrize(
        "objnum, alpha5",
        [
            (100000, "A0000"),
            (339999, "Z9999"),
        ],
    )
    def test_roundtrip(self, objnum: int, alpha5: str):
        """to_alpha5 and from_alpha5 are inverses."""
        assert from_alpha5(to_alpha5(objnum)) == objnum
        assert to_alpha5(from_alpha5(alpha5)) == alpha5

    def test_ensure_alpha5_from_int(self):
        assert ensure_alpha5(100000) == "A0000"

    def test_ensure_alpha5_from_str(self):
        assert ensure_alpha5("A0000") == "A0000"

    def test_ensure_alpha5_passthrough(self):
        """String input is returned unchanged, even if numeric."""
        assert ensure_alpha5("25544") == "25544"

    @pytest.mark.parametrize("satnum", [-1, 340000, 999999])
    def test_invalid_satnum_raises(self, satnum: int):
        with pytest.raises(ValueError):
            to_alpha5(satnum)

    def test_ensure_alpha5_bad_type_raises(self):
        with pytest.raises(TypeError):
            ensure_alpha5(3.14)  # type: ignore[arg-type]

    def test_zero_padded(self):
        """Numbers < 100,000 are zero-padded to 5 digits."""
        assert to_alpha5(42) == "00042"
        assert len(to_alpha5(0)) == 5
