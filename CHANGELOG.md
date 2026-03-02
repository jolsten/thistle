# Changelog

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
