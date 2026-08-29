# Changelog

## [0.10.0] - 2026-08-29

### Changed

- The bundled 17 MB `de421.bsp` ephemeris is no longer shipped inside the
  thistle wheel (now ~58 KB). It comes from the new `skyfield-data`
  dependency instead, which bundles the same file in *its* wheel — so
  installs remain fully offline, nothing is ever downloaded at runtime,
  and the ephemeris data is identical. `skyfield-data` also carries
  `finals2000A.all` (Earth rotation data) and warns when its bundled
  files expire; thistle never reads that file (timescales use skyfield's
  builtin tables — verified offline against an empty data directory), so
  the expiration warning is suppressed at import rather than surfaced to
  every caller. de421.bsp itself is a frozen product with coverage
  through 2053.

## [0.9.0]

### Added

- `Propagator` now warns when requested propagation times are more than
  `warn_threshold` (default: 7 days) from the nearest loaded TLE epoch --
  before the first TLE, after the last, or inside a coverage gap -- since
  SGP4 results that far from epoch are questionable. The new
  `thistle.TLEExtrapolationWarning` carries the offending counts, worst
  offset, threshold, and epoch span as attributes; `warn_threshold` accepts
  a timedelta or a number of days, and `None`/`0` disables. `at()`,
  `segment_times()` (and therefore `generate()`/`generate_range()`),
  `find_satellite()`/`find_satrec()`/`find_tle()`, and the event-finding
  functions all participate; a new `Propagator.check_coverage(start, stop)`
  checks a whole window. Event functions emit at most one window-phrased
  warning per call.
- CLI: `propagate`, `find-tle`, and `groundtrack` gained `--warn-days N`
  (default 7, 0 disables) and present the warning as a clean one-line
  `Warning: ...` on stderr -- printed once per satellite, with an accurate
  end-of-run total when more warnings were suppressed. stdout is unaffected.

### Fixed

- `generate()` with a `Propagator` returned its data dict in a
  nondeterministic key order (it iterated a `set`), so `thistle propagate`
  column order could differ between runs. Keys now follow the requested
  group order.

## [0.8.1]

### Changed

- The `plot` extra now caps `matplotlib<3.13`: cartopy (through 0.25, the
  current release) reads `Formatter.locs`, which Matplotlib deprecated in
  3.11 and removes in 3.13 — `thistle map` axis rendering would crash. The
  cap will be lifted once a cartopy release stops using it.

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
