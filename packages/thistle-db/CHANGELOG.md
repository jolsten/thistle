# Changelog

## 0.9.0 (unreleased)

### Added

- Interactive **progress bars** for `ingest` and `generate` (rich, on
  stderr, anchored at the bottom with log lines rendering above). Enabled
  only when stderr is a terminal; `--no-progress` (global option) forces
  them off. Cron runs are unaffected — non-TTY stderr auto-disables.
- **`--log FILE`** (global option): additionally write logs to a rotating
  file (10 MB per file, last 10 rotations kept) — crons no longer need
  shell redirection for logging.

### Changed

- Per-file "skipped (unchanged)" ingest lines moved from INFO to **DEBUG**;
  steady-state scans no longer flood the log. Each source directory now
  gets an INFO summary line (`N new records (X ingested, Y skipped, ...)`).

## 0.8.0 (2026-07-29)

Complete storage-layer redesign for scale. At 10M+ rows on MariaDB, ingest
no longer degrades with table size and `generate` cost scales with new data
instead of total history.

### Breaking — migration required

The database schema is incompatible with 0.7 and there is no in-place
migration path (thistle-db deliberately has no migration framework; the
dump/restore cycle is the supported path).

**Migrate by dumping with the OLD version installed, before upgrading:**

```bash
# 1. With thistle-db 0.7.x still installed:
thistle-db dump /backups/pre-0.8

# 2. Upgrade the package, then rebuild and restore:
thistle-db init-db --drop --yes
thistle-db ingest /backups/pre-0.8.tle /backups/pre-0.8.json
thistle-db generate --all
```

The dump is lossless (verbatim TLE text plus OMM metadata), so no data is
lost. The final `generate --all` is required: restoring resets `created`
timestamps, which incremental generation depends on.

Schema changes:

- `tle` carries 4 B-trees instead of 15: `PRIMARY KEY(id)`,
  `UNIQUE(epoch, line_hash)`, `(norad_cat_id, epoch)`, `(created)`.
- Dedup is enforced via `line_hash` — the raw sha256 of the exact
  `line1`/`line2` text (`BINARY(32)` on MariaDB) — keyed with `epoch` so
  routine ingest probes only the hot edge of the index. Semantics are
  unchanged: exact-text uniqueness.
- Surrogate keys are `BIGINT` (guards against `INSERT IGNORE`
  auto-increment exhaustion).
- `epoch` and `created` are `DATETIME(6)` on MariaDB (microseconds now
  round-trip; previously truncated to whole seconds).
- Element/derived float columns are single precision (`FLOAT`);
  `mean_motion` stays `DOUBLE`. The TLE text remains the lossless source
  of truth.
- The `modified` column is removed from all tables (rows are immutable).

### Changed

- `generate` is incremental: date files are rewritten for a trailing epoch
  window (`output.window_days`, default 60) plus any date receiving new
  rows; object files are appended behind a tail guard, with automatic full
  rewrites on late deliveries, epoch ties, or damaged files. New flags:
  `--all`, `--verify`, `--window-days`, `--lookback-days`; new config keys
  `output.window_days` / `output.lookback_days`.
- `get-tle` date queries reduce per-object in SQL (bounded memory).
- OMM ingest resolves TLE foreign keys via `line_hash` and bulk-inserts
  metadata (was per-row ORM inserts with a pathological lookup query).
- TLE file ingest streams instead of materializing whole files; full
  catalog restores run in constant memory.
- Database engines are cached per process with `pool_pre_ping`; read-only
  sessions use a dedicated engine.

### Added

- `generate --verify`: weekly-cron integrity sweep reconciling object files
  against the database (repairs truncation, deletion, and mid-file damage).
- MariaDB deployment tuning guidance in the README
  (`innodb_buffer_pool_size` sizing is a must; the 128 MB default is the
  classic cause of large-catalog ingest collapse).
