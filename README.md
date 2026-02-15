# Thistle

A wrapper for [sgp4](https://pypi.org/project/sgp4/) and [skyfield](https://rhodesmill.org/skyfield/) that handles automatic TLE switching for accurate long-duration satellite orbit propagation.

## The Problem

A single Two-Line Element (TLE) set degrades in accuracy as you propagate further from its epoch. For tracking a satellite over days or weeks you need multiple TLEs, and you need to decide when to switch between them.

## The Solution

Thistle manages multiple TLEs for a satellite and automatically selects the most appropriate one based on the propagation time.

## Installation

```
pip install thistle
```

## Usage

```python
from thistle import Propagator, read_tle

# Load TLEs (multiple epochs for one satellite)
tles = read_tle("satellite_data.tle")

# Create a propagator with a switching strategy
prop = Propagator(tles, method="midpoint")

# Propagate — Thistle picks the best TLE for each time
geo = prop.at(times)
```

## Switching Strategies

| Strategy | Description |
|---|---|
| `"epoch"` | Uses the most recent TLE at or before the target time (conservative) |
| `"midpoint"` | Uses the nearest TLE by epoch; transitions at midpoints between consecutive epochs |
| `"tca"` | Transitions at the time of closest approach between neighboring TLEs |

## Lookup Methods

You can look up the active TLE for any point in time:

```python
import numpy as np

time = np.datetime64("2024-01-15T12:00:00")

# Get the EarthSatellite (skyfield)
satellite = prop.find_satellite(time)

# Get the Satrec (sgp4)
satrec = prop.find_satrec(time)

# Get the TLE lines as a (line1, line2) tuple
tle = prop.find_tle(time)
```

## Requirements

Python >= 3.9
