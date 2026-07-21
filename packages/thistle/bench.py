"""Benchmark: Epoch/Midpoint switching vs TCA switching for 10 days of ISS TLEs, 1s steps."""

import datetime
import time

import numpy as np
from skyfield.api import EarthSatellite, load

from thistle.utils import read_tle
from thistle.propagator import (
    EpochSwitchStrategy,
    MidpointSwitchStrategy,
    TCASwitchStrategy,
    _slices_by_transitions,
    merge_geos,
)
from thistle.utils import time_to_dt64

UTC = datetime.timezone.utc

# --- Load TLEs and pick ~1 per day for 10 days ---
all_tles = read_tle("tests/data/25544.tle")
ts = load.timescale()
sats = [EarthSatellite(a, b, ts=ts) for a, b in all_tles]
sats.sort(key=lambda s: s.epoch.tt)

sats_daily: list[EarthSatellite] = []
last_date = None
for s in sats:
    d = s.epoch.utc_datetime().date()
    if d != last_date:
        sats_daily.append(s)
        last_date = d
sats_10 = sats_daily[:10]
print(f"Using {len(sats_10)} TLEs")

# --- Build time array: span of the 10 TLEs at 1-second steps ---
first_epoch = sats_10[0].epoch.utc_datetime().replace(tzinfo=None)
last_epoch = sats_10[-1].epoch.utc_datetime().replace(tzinfo=None)
span = last_epoch - first_epoch
n_points = int(span.total_seconds())
print(f"Span: {span.days} days, {n_points:,} time points (1s steps)")

start_dt = first_epoch.replace(tzinfo=UTC)
start_jd = ts.from_datetime(start_dt).tt
step_jd = 1.0 / 86400.0
tt_jd = np.linspace(start_jd, start_jd + n_points * step_jd, n_points)
tt = ts.tt_jd(tt_jd)

# --- Shared: convert skyfield Time -> datetime64 ---
t0 = time.perf_counter()
dt64 = time_to_dt64(tt)
t_convert = time.perf_counter() - t0
print(f"\ntime_to_dt64: {t_convert:.3f}s ({n_points:,} points)")


def bench_strategy(name, strategy):
    """Run the full propagation pipeline for a switching strategy and return timing."""
    # Compute transitions
    t0 = time.perf_counter()
    strategy.compute_transitions()
    t_transitions = time.perf_counter() - t0

    # Slice assignment
    t0 = time.perf_counter()
    indices = _slices_by_transitions(strategy.transitions, dt64)
    t_slicing = time.perf_counter() - t0

    # Propagation
    t0 = time.perf_counter()
    geos = []
    for idx, slice_ in indices:
        g = strategy.satellites[idx].at(tt[slice_])
        geos.append(g)
    geo = merge_geos(geos, ts)
    t_propagation = time.perf_counter() - t0

    t_total = t_transitions + t_slicing + t_propagation

    print(f"\n{'=' * 50}")
    print(f"Strategy: {name}")
    print(f"  Transitions:  {t_transitions:.3f}s")
    print(f"  Slicing:      {t_slicing:.3f}s")
    print(f"  Propagation:  {t_propagation:.3f}s")
    print(f"  Total:        {t_total:.3f}s")
    print(f"  Segments:     {len(indices)}")

    return t_total, geo


# --- Benchmark each strategy ---
epoch_strat = EpochSwitchStrategy(list(sats_10))
midpoint_strat = MidpointSwitchStrategy(list(sats_10))
tca_strat = TCASwitchStrategy(list(sats_10), ts=ts)

t_epoch, geo_epoch = bench_strategy("Epoch", epoch_strat)
t_midpoint, geo_midpoint = bench_strategy("Midpoint", midpoint_strat)
t_tca, geo_tca = bench_strategy("TCA", tca_strat)

# --- Compare position differences ---
print(f"\n{'=' * 50}")
print("Position differences (km)")

diff_em = np.linalg.norm(geo_epoch.xyz.km - geo_midpoint.xyz.km, axis=0)
diff_et = np.linalg.norm(geo_epoch.xyz.km - geo_tca.xyz.km, axis=0)
diff_mt = np.linalg.norm(geo_midpoint.xyz.km - geo_tca.xyz.km, axis=0)

print(f"  Epoch vs Midpoint:  max={diff_em.max():.3f}  mean={diff_em.mean():.3f}")
print(f"  Epoch vs TCA:       max={diff_et.max():.3f}  mean={diff_et.mean():.3f}")
print(f"  Midpoint vs TCA:    max={diff_mt.max():.3f}  mean={diff_mt.mean():.3f}")

# --- Summary ---
print(f"\n{'=' * 50}")
print("Summary (excluding shared time_to_dt64 overhead)")
print(f"  Epoch:    {t_epoch:.3f}s")
print(f"  Midpoint: {t_midpoint:.3f}s")
print(f"  TCA:      {t_tca:.3f}s")
print(f"  TCA / Epoch overhead:    {t_tca / t_epoch:.2f}x")
print(f"  TCA / Midpoint overhead: {t_tca / t_midpoint:.2f}x")
