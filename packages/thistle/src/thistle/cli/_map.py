"""Ground-trace map construction for the ``thistle groundtrack`` subcommand.

Spec parsing and trace generation need no plotting libraries; cartopy and
matplotlib (the ``plot`` extra) are imported only inside :func:`make_map`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

# spec key -> matplotlib kwarg (value converter applied where not str)
_KWARG_KEYS = {
    "color": ("color", str),
    "style": ("linestyle", str),
    "width": ("linewidth", float),
    "alpha": ("alpha", float),
    "marker": ("marker", str),
    "label": ("label", str),
}


@dataclass
class TraceSpec:
    start: datetime
    stop: datetime
    sat: Optional[str] = None
    plot_kwargs: dict = field(default_factory=dict)


def parse_spec_line(line: str) -> Optional[TraceSpec]:
    """Parse ``START STOP [key=val ...]`` into a TraceSpec.

    Returns None for blank lines and '#' comments (silently) and for
    malformed lines (with a stderr warning).
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    tokens = stripped.split()
    if len(tokens) < 2:
        print(f"Warning: skipping spec line (need START STOP): {stripped}", file=sys.stderr)
        return None

    try:
        start = datetime.fromisoformat(tokens[0])
        stop = datetime.fromisoformat(tokens[1])
    except ValueError:
        print(f"Warning: skipping spec line (bad time): {stripped}", file=sys.stderr)
        return None

    if stop <= start:
        print(f"Warning: skipping spec line (stop <= start): {stripped}", file=sys.stderr)
        return None

    spec = TraceSpec(start=start, stop=stop)
    for token in tokens[2:]:
        key, sep, val = token.partition("=")
        if not sep or not val:
            print(f"Warning: ignoring malformed option: {token}", file=sys.stderr)
            continue
        if key == "sat":
            spec.sat = val
        elif key in _KWARG_KEYS:
            kwarg, conv = _KWARG_KEYS[key]
            try:
                spec.plot_kwargs[kwarg] = conv(val)
            except ValueError:
                print(f"Warning: ignoring bad value: {token}", file=sys.stderr)
        else:
            print(f"Warning: ignoring unknown option: {key}", file=sys.stderr)
    return spec


def trace_lla(
    propagator, start: datetime, stop: datetime, step_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground-trace (lons, lats, alts) sampled every ``step_s`` seconds."""
    from thistle import generate

    t0 = np.datetime64(start.isoformat(), "ns")
    t1 = np.datetime64(stop.isoformat(), "ns")
    step = np.timedelta64(int(step_s * 1e9), "ns")
    times = np.arange(t0, t1 + step, step)

    data = generate(times, propagator, ["lla"])
    return data["lon"], data["lat"], data["alt"]


def make_map(
    traces: list[tuple[np.ndarray, np.ndarray, dict]],
    sites: dict[str, tuple],
    ring_alt_m: Optional[float],
    min_el: float,
    title: Optional[str] = None,
):
    """Build the map figure; caller saves or shows it.

    traces: (lons, lats, plot_kwargs) per spec line.
    sites: name -> (lat, lon) or (lat, lon, alt_m).
    ring_alt_m: satellite altitude for visibility rings; None = markers only.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    from thistle import visibility_circle

    fig, ax = plt.subplots(
        figsize=(12, 7), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    any_label = False
    for lons, lats, kwargs in traces:
        kwargs.setdefault("linewidth", 1.2)
        any_label = any_label or "label" in kwargs
        ax.plot(lons, lats, transform=ccrs.Geodetic(), **kwargs)

    for name, coords in sites.items():
        lat, lon = coords[0], coords[1]
        site_alt = coords[2] if len(coords) > 2 else 0.0
        ax.plot(lon, lat, "k+", markersize=10, transform=ccrs.PlateCarree())
        ax.text(
            lon, lat, f" {name}", fontsize=8, transform=ccrs.PlateCarree()
        )
        if ring_alt_m is not None:
            ring_lats, ring_lons = visibility_circle(
                lat, lon, site_alt, sat_alt=ring_alt_m, min_el=min_el
            )
            ax.plot(
                [*ring_lons, ring_lons[0]],
                [*ring_lats, ring_lats[0]],
                color="gray",
                linestyle="--",
                linewidth=0.8,
                transform=ccrs.Geodetic(),
            )

    if any_label:
        ax.legend(loc="lower left", fontsize=8)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig
