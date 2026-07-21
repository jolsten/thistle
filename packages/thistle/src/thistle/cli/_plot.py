"""Figure construction for the ``thistle plot`` subcommand.

Everything except :func:`detect_maneuvers` requires matplotlib, which is an
optional dependency (``pip install 'thistle[plot]'``); import of this module
is deferred to the command body.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from sgp4.api import Satrec

from thistle.cli._helpers import MU, RE, epoch_to_datetime


def _sma_km(sat: Satrec) -> float:
    mm_rad_per_sec = sat.no_kozai / 60.0
    return (MU / (mm_rad_per_sec**2)) ** (1.0 / 3.0)


def _mm_revday(sat: Satrec) -> float:
    return sat.no_kozai * 1440.0 / (2.0 * math.pi)


def _per_sat(extract: Callable[[Satrec], float]) -> Callable[[list[Satrec]], np.ndarray]:
    return lambda sats: np.array([extract(s) for s in sats], dtype=float)


def _epoch_times(sats: list[Satrec]):
    """Skyfield array Time at the TLE epochs (lazy _core import loads de421)."""
    from thistle._core import ts
    from thistle.utils import dt64_to_time

    dts = np.array(
        [
            np.datetime64(epoch_to_datetime(s.epochyr, s.epochdays).isoformat(), "ns")
            for s in sats
        ],
        dtype="datetime64[ns]",
    )
    return dt64_to_time(dts, ts)


def _lon_series(sats: list[Satrec]) -> np.ndarray:
    """Approximate subsatellite longitude at epoch (deg, [-180, 180)).

    Mean longitude minus GMST; accurate to a fraction of a degree for
    near-circular, low-inclination (i.e. GEO) orbits.
    """
    t = _epoch_times(sats)
    gmst_deg = np.asarray(t.gmst, dtype=float) * 15.0
    mean_lon = np.array(
        [math.degrees(s.nodeo + s.argpo + s.mo) for s in sats], dtype=float
    )
    return (mean_lon - gmst_deg + 180.0) % 360.0 - 180.0


def _ltan_series(sats: list[Satrec]) -> np.ndarray:
    """Local time of ascending node (hours, [0, 24))."""
    from thistle._core import eph

    t = _epoch_times(sats)
    ra_sun_hours = np.asarray(
        eph["earth"].at(t).observe(eph["sun"]).apparent().radec()[0].hours,
        dtype=float,
    )
    raan_hours = np.array(
        [math.degrees(s.nodeo) / 15.0 for s in sats], dtype=float
    )
    return (raan_hours - ra_sun_hours + 12.0) % 24.0


# field name -> (axis label, series function over the TLE list)
FIELDS: dict[str, tuple[str, Callable[[list[Satrec]], np.ndarray]]] = {
    "sma": ("SMA (km)", _per_sat(_sma_km)),
    "mm": ("Mean motion (rev/day)", _per_sat(_mm_revday)),
    "ecc": ("Eccentricity", _per_sat(lambda s: s.ecco)),
    "inc": ("Inclination (deg)", _per_sat(lambda s: math.degrees(s.inclo))),
    "raan": ("RAAN (deg)", _per_sat(lambda s: math.degrees(s.nodeo))),
    "aop": ("Arg of perigee (deg)", _per_sat(lambda s: math.degrees(s.argpo))),
    "bstar": ("B* (1/Re)", _per_sat(lambda s: s.bstar)),
    "peri": ("Perigee alt (km)", _per_sat(lambda s: _sma_km(s) * (1 - s.ecco) - RE)),
    "apo": ("Apogee alt (km)", _per_sat(lambda s: _sma_km(s) * (1 + s.ecco) - RE)),
    "revnum": ("Rev number", _per_sat(lambda s: float(s.revnum))),
    "lon": ("Longitude (deg E)", _lon_series),
    "ltan": ("LTAN (hours)", _ltan_series),
}

DEFAULT_FIELDS = ["sma", "mm", "ecc", "inc", "raan", "aop"]

# preset name -> panel list, tuned per orbit regime
PRESETS: dict[str, list[str]] = {
    "leo": ["sma", "mm", "peri", "apo", "inc", "bstar"],
    "geo": ["lon", "mm", "sma", "ecc", "inc"],
    "sunsync": ["ltan", "sma", "mm", "ecc", "inc"],
    "heo": ["peri", "apo", "ecc", "inc", "aop", "mm"],
}

# Angles that precess secularly; unwrap so drift doesn't sawtooth at 0/360.
_UNWRAP_FIELDS = {"raan", "aop"}

_SERIES_COLOR = "#1f77b4"
_MANEUVER_COLOR = "#d62728"


def make_figure(
    sats: list[Satrec],
    fields: list[str],
    maneuvers: list[datetime],
    title: Optional[str] = None,
):
    """Build a stacked element-history figure; caller saves or shows it."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times = [epoch_to_datetime(s.epochyr, s.epochdays) for s in sats]

    fig, axes = plt.subplots(
        nrows=len(fields),
        ncols=1,
        sharex=True,
        figsize=(10, 1.0 + 1.6 * len(fields)),
        squeeze=False,
    )

    for ax, field in zip(axes[:, 0], fields):
        label, series_fn = FIELDS[field]
        values = series_fn(sats)
        if field in _UNWRAP_FIELDS:
            values = np.degrees(np.unwrap(np.radians(values)))
        ax.plot(
            times,
            values,
            color=_SERIES_COLOR,
            linewidth=1.2,
            marker=".",
            markersize=4,
        )
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for t in maneuvers:
            ax.axvline(
                t,
                color=_MANEUVER_COLOR,
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )

    if maneuvers:
        axes[0, 0].axvline(
            maneuvers[0],
            color=_MANEUVER_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="detected maneuver",
        )
        axes[0, 0].legend(loc="best", fontsize=8, frameon=False)

    locator = mdates.AutoDateLocator()
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    if title is None and sats:
        title = f"Object {sats[0].satnum} ({sats[0].intldesg.strip()})"
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig
