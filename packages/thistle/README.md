# Thistle

Satellite orbit propagation and data generation built on [SGP4](https://pypi.org/project/sgp4/) and [Skyfield](https://rhodesmill.org/skyfield/). Handles automatic TLE switching for long-duration propagation, and generates orbit data, ground site ranges, and satellite events from a single propagation pass.

## Installation

```
pip install thistle
```

Requires Python >= 3.9.

## Quick start

```python
import numpy as np
from thistle import Propagator, read_tle, generate

tles = read_tle("satellite_data.tle")
prop = Propagator(tles, method="midpoint")

times = np.arange(
    np.datetime64("2024-01-15T00:00"),
    np.datetime64("2024-01-16T00:00"),
    np.timedelta64(60, "s"),
)

data = generate(times, prop, ["eci", "lla"])
# data["eci_x"], data["lat"], data["lon"], ...
```

## Propagator

A single TLE degrades in accuracy as you propagate further from its epoch. For tracking a satellite over days or weeks you need multiple TLEs and a strategy for switching between them.

```python
from thistle import Propagator, read_tle

tles = read_tle("satellite_data.tle")
prop = Propagator(tles, method="midpoint")
```

### Switching strategies

| Strategy | Description |
|---|---|
| `"epoch"` | Uses the most recent TLE at or before the target time |
| `"midpoint"` | Transitions at the midpoint between consecutive TLE epochs |
| `"tca"` | Transitions at the time of closest approach between neighboring TLEs |

You can also pass a strategy instance directly:

```python
from thistle import Propagator, EpochSwitchStrategy, read_tle

tles = read_tle("satellite_data.tle")
prop = Propagator(tles, method=EpochSwitchStrategy([]))
```

This is useful for subclassing `SwitchingStrategy` to implement custom switching logic.

### Lookup methods

```python
time = np.datetime64("2024-01-15T12:00:00")

satellite = prop.find_satellite(time)  # Skyfield EarthSatellite
satrec = prop.find_satrec(time)        # sgp4 Satrec
tle = prop.find_tle(time)              # (line1, line2) strings
```

### Direct propagation

The Propagator can be used as a drop-in replacement for a Skyfield `EarthSatellite`:

```python
from skyfield.api import load
ts = load.timescale()

geo = prop.at(ts.utc(2024, 1, 15, 12, 0, 0))
```

## Data generation

`generate()` propagates once and extracts multiple data groups in a single pass.

```python
from thistle import generate

data = generate(times, prop, ["eci", "lla", "keplerian", "sunlight"])
```

### Groups

| Group | Keys | Units |
|---|---|---|
| `eci` | `eci_x, eci_y, eci_z, eci_vx, eci_vy, eci_vz` | m, m/s |
| `ecef` | `ecef_x, ecef_y, ecef_z, ecef_vx, ecef_vy, ecef_vz` | m, m/s |
| `lla` | `lat, lon, alt` | deg, deg, m |
| `keplerian` | `sma, ecc, inc, raan, aop, ta, ma, ea, arglat, tlon, mlon, lonper, mm` | m, -, deg (mm: deg/day) |
| `equinoctial` | `p, f, g, h, k, L` | m, -, -, -, -, deg |
| `sunlight` | `sun` | int8 (0=umbra, 1=penumbra, 2=sunlit) |
| `beta` | `beta` | deg |
| `lst` | `lst` | hours [0, 24) |
| `mag_enu` | `Be, Bn, Bu` | nT |
| `mag_total` | `Bt` | nT |
| `mag_ecef` | `Bx, By, Bz` | nT |

All values are returned as NumPy arrays. Angles and dimensionless quantities are float32; positions, velocities, and range data are float64.

### Ground site range

Pass `sites` to `generate()` to compute slant range and range rate without a second propagation:

```python
data = generate(times, prop, ["lla"], sites={"ksc": (28.57, -80.65)})
# data["range_ksc"]       -> slant range in meters
# data["range_rate_ksc"]  -> range rate in m/s
```

Sites can also be a list (keys become `range_0`, `range_1`, ...):

```python
data = generate(times, prop, ["lla"], sites=[(28.57, -80.65), (34.05, -118.24)])
```

Each site tuple is `(lat, lon)` or `(lat, lon, alt_m)`.

The standalone `generate_range()` function is also available:

```python
from thistle import generate_range

rng = generate_range(times, prop, sites={"ksc": (28.57, -80.65)})
```

### Doppler shift

```python
from thistle import doppler_shift

doppler_hz = doppler_shift(data["range_rate_ksc"], freq=437e6)
```

## Events

Find satellite events within a time window. All event functions accept either an `EarthSatellite` or a `Propagator` and return lists of dicts.

### Passes

```python
from thistle import find_passes

passes = find_passes(start, stop, prop, lat=28.57, lon=-80.65, min_elevation=10.0)
for p in passes:
    print(p["start"], p["stop"], p["peak_elevation"])
```

Returns dicts with keys: `start`, `stop`, `peak_time`, `peak_elevation`.

### Node crossings

```python
from thistle import find_node_crossings

crossings = find_node_crossings(start, stop, prop)
for c in crossings:
    print(c["start"], c["longitude"], c["ascending"])
```

Returns dicts with keys: `start`, `stop`, `longitude`, `ascending`.

### Sunlit and eclipse periods

```python
from thistle import find_sunlit_periods, find_eclipse_periods

sunlit = find_sunlit_periods(start, stop, prop)
eclipse = find_eclipse_periods(start, stop, prop)
```

### Ascending and descending periods

```python
from thistle import find_ascending_periods, find_descending_periods

ascending = find_ascending_periods(start, stop, prop)
descending = find_descending_periods(start, stop, prop)
```

All period functions return dicts with keys: `start`, `stop`.

## Visibility circle

Compute the ground footprint where a satellite at a given altitude is visible above a minimum elevation angle:

```python
from thistle import visibility_circle

lats, lons = visibility_circle(28.57, -80.65, alt=0.0, sat_alt=408_000, min_el=10.0)
```

## Accuracy

### TLE propagation

SGP4 propagation from TLEs is inherently limited to roughly 1-10 km near epoch, degrading to 10-100 km over days. For sub-kilometer accuracy, use precision ephemerides (SP3) instead of TLEs.

### Time scale

Thistle treats UTC as UT1, introducing ~0.2-0.9 seconds of time offset (~1-7 m position error for LEO). This is negligible compared to TLE propagation errors and avoids the ~12x overhead of proper UTC/leap-second handling.

### Coordinate system

ECI outputs use the ICRF (International Celestial Reference Frame), equivalent to J2000 for most applications.

## License

MIT. See [LICENSE](LICENSE) for details.
