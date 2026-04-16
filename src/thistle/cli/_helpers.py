"""Shared constants and helpers for the thistle CLI."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from typing import TextIO

from sgp4.api import Satrec

MU = 398600.4418  # Earth gravitational parameter (km^3/s^2)
RE = 6371.0  # Earth mean radius (km)

DEFAULT_COLUMNS = [
    # (name, width, description, numeric?)
    ("norad", 5, "NORAD catalog number", True),
    ("intl", 8, "International designator", False),
    ("epoch", 19, "TLE epoch (ISO 8601)", False),
    ("sma", 8, "Semi-major axis (km)", True),
    ("peri", 8, "Perigee altitude (km)", True),
    ("apo", 8, "Apogee altitude (km)", True),
    ("mm", 6, "Mean motion (rev/day)", True),
    ("ecc", 7, "Eccentricity", True),
    ("inc", 7, "Inclination (deg)", True),
    ("raan", 6, "RAAN (deg)", True),
    ("aop", 6, "Arg of perigee (deg)", True),
    ("elnum", 5, "Element set number", True),
    ("revnum", 6, "Revolution number at epoch", True),
]

ALL_COLUMNS = [
    (1, "norad", "NORAD catalog number"),
    (2, "class", "Classification U/C/S (use --all to include)"),
    (3, "intl", "International designator"),
    (4, "epoch", "TLE epoch (ISO 8601)"),
    (5, "sma", "Semi-major axis (km)"),
    (6, "peri", "Perigee altitude (km)"),
    (7, "apo", "Apogee altitude (km)"),
    (8, "mm", "Mean motion (rev/day)"),
    (9, "ecc", "Eccentricity"),
    (10, "inc", "Inclination (deg)"),
    (11, "raan", "RAAN (deg)"),
    (12, "aop", "Argument of perigee (deg)"),
    (13, "ma", "Mean anomaly (deg) (use --all to include)"),
    (14, "bstar", "B* drag coefficient (use --all to include)"),
    (15, "elnum", "Element set number"),
    (16, "revnum", "Revolution number at epoch"),
]

ALL_GROUPS = [
    "eci",
    "ecef",
    "lla",
    "keplerian",
    "equinoctial",
    "sunlight",
    "beta",
    "lst",
    "mag_enu",
    "mag_total",
    "mag_ecef",
]


def warn_if_tty(stream: TextIO, name: str) -> None:
    if hasattr(stream, "isatty") and stream.isatty():
        print(
            f"thistle {name}: reading from stdin (pipe data or press Ctrl-D to end)",
            file=sys.stderr,
        )


def epoch_to_datetime(epochyr: int, epochdays: float) -> datetime:
    year = 2000 + epochyr if epochyr < 57 else 1900 + epochyr
    return datetime(year, 1, 1) + timedelta(days=epochdays - 1)


def epoch_to_iso(epochyr: int, epochdays: float) -> str:
    return epoch_to_datetime(epochyr, epochdays).strftime("%Y-%m-%dT%H:%M:%S")


def parse_tle_pair(line1: str, line2: str, include_all: bool = False) -> list[str]:
    sat = Satrec.twoline2rv(line1, line2)

    epoch_iso = epoch_to_iso(sat.epochyr, sat.epochdays)
    mean_motion_revday = sat.no_kozai * 1440.0 / (2.0 * math.pi)
    mm_rad_per_sec = sat.no_kozai / 60.0
    sma_km = (MU / (mm_rad_per_sec**2)) ** (1.0 / 3.0)
    perigee_alt = sma_km * (1 - sat.ecco) - RE
    apogee_alt = sma_km * (1 + sat.ecco) - RE

    if include_all:
        return [
            str(sat.satnum),
            sat.classification,
            sat.intldesg.strip(),
            epoch_iso,
            f"{sma_km:.2f}",
            f"{perigee_alt:.2f}",
            f"{apogee_alt:.2f}",
            f"{mean_motion_revday:.8f}",
            f"{sat.ecco:.7f}",
            f"{math.degrees(sat.inclo):.4f}",
            f"{math.degrees(sat.nodeo):.4f}",
            f"{math.degrees(sat.argpo):.4f}",
            f"{math.degrees(sat.mo):.4f}",
            f"{sat.bstar:.6e}",
            str(sat.elnum),
            str(sat.revnum),
        ]

    return [
        str(sat.satnum),
        sat.intldesg.strip(),
        epoch_iso,
        f"{sma_km:.2f}",
        f"{perigee_alt:.2f}",
        f"{apogee_alt:.2f}",
        f"{mean_motion_revday:.3f}",
        f"{sat.ecco:.5f}",
        f"{math.degrees(sat.inclo):.4f}",
        f"{math.degrees(sat.nodeo):.2f}",
        f"{math.degrees(sat.argpo):.2f}",
        str(sat.elnum),
        str(sat.revnum),
    ]


def parse_site(
    spec: str,
) -> tuple[str, tuple[float, float] | tuple[float, float, float]]:
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid site spec '{spec}' (use name:lat:lon or name:lat:lon:alt)"
        )
    name = parts[0]
    if len(parts) == 3:
        return name, (float(parts[1]), float(parts[2]))
    return name, (float(parts[1]), float(parts[2]), float(parts[3]))


def parse_tle_epochs(source: TextIO) -> list[tuple[str, str, Satrec]]:
    """Parse TLE pairs from a stream, return (line1, line2, satrec) triples."""
    results: list[tuple[str, str, Satrec]] = []
    line1: str | None = None
    for raw_line in source:
        line = raw_line.rstrip()
        if not line:
            continue
        if line[0] == "1":
            line1 = line
        elif line[0] == "2" and line1 is not None:
            try:
                sat = Satrec.twoline2rv(line1, line)
                results.append((line1, line, sat))
            except Exception as e:
                print(f"Warning: skipping TLE: {e}", file=sys.stderr)
            line1 = None
    return results


class AlignedWriter:
    """Buffers rows and flushes with aligned columns."""

    def __init__(self, delimiter: str, numeric: list[bool] | None = None):
        self._delim = delimiter
        self._rows: list[list[str]] = []
        self._widths: list[int] = []
        self._numeric = numeric

    def add_row(self, fields: list[str]) -> None:
        for i, f in enumerate(fields):
            w = len(f)
            if i < len(self._widths):
                if w > self._widths[i]:
                    self._widths[i] = w
            else:
                self._widths.append(w)
        self._rows.append(fields)

    def flush(self) -> None:
        for row in self._rows:
            parts: list[str] = []
            for i, f in enumerate(row):
                w = self._widths[i] if i < len(self._widths) else len(f)
                if self._numeric and i < len(self._numeric) and self._numeric[i]:
                    parts.append(f.rjust(w))
                else:
                    parts.append(f.ljust(w))
            sys.stdout.write(self._delim.join(parts).rstrip() + "\n")
        self._rows.clear()


def read_and_emit_tles(
    source: TextIO,
    writer: AlignedWriter,
    include_all: bool = False,
) -> None:
    line1: str | None = None

    for raw_line in source:
        line = raw_line.rstrip()
        if not line:
            continue

        if line[0] == "1":
            line1 = line
        elif line[0] == "2" and line1 is not None:
            try:
                fields = parse_tle_pair(line1, line, include_all)
                writer.add_row(fields)
            except Exception as e:
                print(f"Warning: skipping TLE: {e}", file=sys.stderr)
            line1 = None
