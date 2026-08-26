"""Shared constants and helpers for the thistle CLI."""

from __future__ import annotations

import contextlib
import math
import sys
import warnings
from datetime import datetime, timedelta
from typing import TextIO

import numpy as np
from sgp4.api import Satrec

MU = 398600.4418  # Earth gravitational parameter (km^3/s^2)
RE = 6371.0  # Earth mean radius (km)
J2 = 1.08262668e-3  # Earth J2 zonal harmonic

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


@contextlib.contextmanager
def report_tle_warnings():
    """Present TLEExtrapolationWarnings as clean one-line CLI warnings.

    The first warning per satellite is printed to stderr immediately in
    "Warning: ..." style with the CLI remedy; repeats are suppressed while
    their counts accumulate, and an accurate per-satellite total is printed
    at the end when repeats occurred.
    """
    from thistle.propagator import (
        _REMEDY_STEM,
        TLEExtrapolationWarning,
        _fmt_satnum,
        _fmt_timedelta,
    )

    stats: dict[int, dict] = {}

    with warnings.catch_warnings():
        # Guarantee repeats reach showwarning so tallies stay accurate.
        warnings.simplefilter("always", TLEExtrapolationWarning)
        default_show = warnings.showwarning

        def _show(message, category, filename, lineno, file=None, line=None):
            if not isinstance(message, TLEExtrapolationWarning):
                default_show(message, category, filename, lineno, file, line)
                return
            s = stats.get(message.satnum)
            if s is None:
                stats[message.satnum] = {
                    "n_bad": message.n_bad or 0,
                    "n_total": message.n_total or 0,
                    "max_offset": message.max_offset,
                    "threshold": message.threshold,
                    "repeats": 0,
                }
                print(
                    f"Warning: {message.body} {_REMEDY_STEM}"
                    "--warn-days N (0 disables).",
                    file=sys.stderr,
                )
            else:
                s["repeats"] += 1
                s["n_bad"] += message.n_bad or 0
                s["n_total"] += message.n_total or 0
                s["max_offset"] = max(s["max_offset"], message.max_offset)

        warnings.showwarning = _show
        try:
            yield
        finally:
            for satnum, s in stats.items():
                if s["repeats"] and s["n_total"]:
                    print(
                        f"Warning: satellite {_fmt_satnum(satnum)}: in total, "
                        f"{s['n_bad']:,} of {s['n_total']:,} propagation times "
                        f"were more than {_fmt_timedelta(s['threshold'])} from "
                        "the nearest TLE epoch "
                        f"(worst: {_fmt_timedelta(s['max_offset'])}).",
                        file=sys.stderr,
                    )


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


def nodal_rate_rev_per_day(sat: Satrec) -> float:
    """Nodal revolution rate: mean motion plus the J2 apsidal precession rate.

    Revolutions are counted at ascending-node crossings, so the relevant rate
    is d(argument of latitude)/dt = n + argp_dot, not the mean-anomaly rate.
    """
    n0 = sat.no_kozai * 1440.0 / (2.0 * math.pi)  # rev/day
    mm_rad_per_sec = sat.no_kozai / 60.0
    sma_km = (MU / (mm_rad_per_sec**2)) ** (1.0 / 3.0)
    p = sma_km * (1.0 - sat.ecco**2)
    cos_i = math.cos(sat.inclo)
    argp_dot = 0.75 * n0 * J2 * (RE / p) ** 2 * (5.0 * cos_i**2 - 1.0)  # rev/day
    return n0 + argp_dot


def ndot_rev_per_day2(sat: Satrec) -> float:
    """The TLE line-1 first-derivative field (already n-dot/2) in rev/day^2.

    sgp4 stores ndot in rad/min^2; this inverts its read-time conversion.
    """
    return sat.ndot * 1440.0**2 / (2.0 * math.pi)


def _rev_phase(sat: Satrec) -> float:
    """Fraction of the current revolution completed at the TLE epoch."""
    u0 = (sat.argpo + sat.mo) % (2.0 * math.pi)  # mean argument of latitude
    return u0 / (2.0 * math.pi)


def revnum_at(sat: Satrec, dt: datetime) -> float:
    """Estimated fractional revolution number at ``dt``.

    Integer part is the current rev (increments at the ascending node);
    the fraction is the portion of that rev completed.
    """
    epoch = epoch_to_datetime(sat.epochyr, sat.epochdays)
    dt_days = (dt - epoch).total_seconds() / 86400.0
    n = nodal_rate_rev_per_day(sat)
    c = ndot_rev_per_day2(sat)
    return sat.revnum + _rev_phase(sat) + n * dt_days + c * dt_days**2


def rev_bounds(sat: Satrec, revnum: int) -> tuple[datetime, datetime]:
    """Estimated (start, stop) ascending-node crossing times of rev ``revnum``."""
    epoch = epoch_to_datetime(sat.epochyr, sat.epochdays)
    n = nodal_rate_rev_per_day(sat)
    c = ndot_rev_per_day2(sat)

    def invert(drev: float) -> datetime:
        # Solve c*t^2 + n*t - drev = 0 for elapsed days t.
        disc = n**2 + 4.0 * c * drev
        if abs(c) < 1e-12 or disc < 0.0:
            dt_days = drev / n
        else:
            dt_days = (-n + math.sqrt(disc)) / (2.0 * c)
        return epoch + timedelta(days=dt_days)

    drev_start = (revnum - sat.revnum) - _rev_phase(sat)
    return invert(drev_start), invert(drev_start + 1.0)


_MIN_TLES_FOR_DETECTION = 8


def detect_maneuvers(
    times: list[datetime], values: list[float], threshold: float
) -> list[datetime]:
    """Epochs immediately after discrete jumps in a per-TLE series.

    A jump is a consecutive delta exceeding ``threshold`` robust sigmas
    (1.4826 x MAD of the signed deltas, which ignores the jumps themselves).
    Flagged deltas closer together than max(2 days, 5x the median TLE
    cadence) are merged into one event at the largest delta, since a single
    burn perturbs several successive TLE fits. Detection is skipped for
    short or constant series where the scale is meaningless.

    Note this detects impulsive changes only: continuous low-thrust (e.g.
    electric orbit raising) appears as a smooth trend with mutually
    consistent deltas and is largely invisible to a jump detector.
    """
    if len(values) < _MIN_TLES_FOR_DETECTION:
        return []
    deltas = np.diff(np.asarray(values, dtype=float))
    sigma = 1.4826 * float(np.median(np.abs(deltas - np.median(deltas))))
    if sigma < 1e-12:
        return []
    flagged = np.nonzero(np.abs(deltas) > threshold * sigma)[0]
    if len(flagged) == 0:
        return []

    spacings = [
        (times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)
    ]
    merge_gap = max(2.0 * 86400.0, 5.0 * float(np.median(spacings)))

    events: list[datetime] = []
    cluster = [flagged[0]]
    for i in flagged[1:]:
        if (times[i + 1] - times[cluster[-1] + 1]).total_seconds() <= merge_gap:
            cluster.append(i)
        else:
            best = max(cluster, key=lambda j: abs(deltas[j]))
            events.append(times[best + 1])
            cluster = [i]
    best = max(cluster, key=lambda j: abs(deltas[j]))
    events.append(times[best + 1])
    return events


def maneuver_epochs(sats: list[Satrec], threshold: float) -> list[datetime]:
    """Detected maneuver epochs for an epoch-sorted single-object TLE list.

    Runs :func:`detect_maneuvers` on the mean-motion series.
    """
    times = [epoch_to_datetime(s.epochyr, s.epochdays) for s in sats]
    mm_series = [s.no_kozai * 1440.0 / (2.0 * math.pi) for s in sats]
    return detect_maneuvers(times, mm_series, threshold)


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
