# Changelog

## [0.4.1]

### Fixed

- The CLI (`thistle` entry point and the `cli` extra) now requires Python 3.10+.
  On 3.9, typer's argument parsing misbehaved on several subcommand invocations,
  so the CLI never worked there. The `typer` dependency in the `cli` extra now
  carries a `python_version >= '3.10'` marker, and the `thistle` entry point
  prints a clear message and exits 1 when invoked on Python 3.9.
- The library itself continues to support Python 3.9.

## [0.4.0]

### Added

- Command-line interface (`thistle` entry point) with six subcommands:
  - `inspect` -- parse TLE files and display orbital parameters as a table
  - `find-tle` -- find the correct TLE for given timestamps via a switching
    strategy
  - `summary` -- summarize TLE data quality for a single object (epoch range,
    interval stats, gap detection)
  - `catalog` -- summarize TLE data across a directory (aggregate interval
    stats and gap detection)
  - `filter` -- filter TLEs by epoch time, NORAD catalog number, or orbital
    elements (inclination, eccentricity, period, SMA, perigee, apogee,
    revolution number)
  - `propagate` -- propagate TLEs and generate orbital data from stdin
    timestamps
- `cli` optional dependency group (`pip install 'thistle[cli]'`) pulling in
  `typer`. The CLI fails gracefully with a helpful install hint when the
  extra is not installed.

## [0.3.0]

### Fixed

- `jday_datetime64()` now normalizes input arrays to `datetime64[us]` before
  conversion. Previously, `datetime64[ns]` arrays produced wildly incorrect
  Julian dates because NumPy promoted the subtraction to nanoseconds while the
  divisor still assumed microseconds.

### Added

- `generate()` now includes the input time array as `"times"` — the first key
  in the returned dict.
- Tests for `generate()` with `datetime64[s]`, `[ms]`, `[us]`, and `[ns]`
  input arrays, covering both the `EarthSatellite` and `Propagator` code paths.
