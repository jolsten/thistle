# Changelog

## [Unreleased]

### Breaking — schema and output config

- New `epoch_date_state` table (see below). **Run `thistle-db init-db`
  after upgrading**: it creates the table and backfills it from the
  element sets, so no dump/restore is needed. Until it is populated, no
  date files are generated — `generate` warns rather than silently doing
  nothing. Re-run `init-db` any time rows are inserted outside `ingest`
  (a bulk `COPY`, an admin script), since those bypass the watermark.
- `[output] window_days` and `generate --window-days` are removed. Configs
  that still set `window_days` fail validation with a pointer at the key
  (`[output]` is `extra="forbid"`); there is no replacement to set,
  because date scoping is now exact rather than windowed.

### Fixed

- **A malformed element set no longer costs an entire file's ingest.** sgp4
  parses TLE fields with C `strtod`, which accepts the literal text `nan`,
  `inf` and `nan(ind)` — what a producer emits when it formats a missing
  value with `printf` — so any float field could arrive non-finite. On
  MariaDB, pymysql refuses NaN while building the statement, *before* any
  SQL is sent, which failed the whole 5000-row insert chunk and propagated
  out of `ingest_tles`, losing the rest of the file. On SQLite it was
  silent: NaN was stored as NULL and `inf` was stored as `inf`.

  Non-finite values are now cleaned where sgp4 output enters the model,
  driven by the schema's own nullability. Nullable columns (`bstar`,
  `mean_motion`, `mean_motion_dot`, `mean_motion_ddot`, `semimajor_axis`,
  `period`, `apoapsis_alt`, `periapsis_alt`) take NULL — `line1`/`line2`
  remain the lossless source of truth. A non-finite *mandatory* element
  (eccentricity, inclination, RAAN, argument of pericenter, mean anomaly)
  means there is no orbit to store, so that record is rejected rather than
  stored as a row that would displace the object's good elset from a date
  file. In a 20,000-case fuzz of corrupted lines, ~2.5% of non-finite
  results fell in the mandatory group; the rest are now stored.

  No schema change: all affected columns were already nullable.

- **A row the database refuses now costs only itself.** `_bulk_insert_ignore`
  retries a failed chunk row by row, so one unforeseen bad value can no
  longer take 4999 good records with it.

- `period()` returns NaN (cleaned to NULL) instead of raising `ValueError`
  for a non-positive semi-major axis, which previously discarded an
  otherwise storable row.

### Added

- **`filename` on `[[output.files]]` — a template for generated filenames.**
  Previously only the extension and the stem's rendering were configurable;
  a template lets the stem carry prefixes and multi-part suffixes:

  ```toml
  [[output.files]]
  type = "object"
  format = "tle"
  filename = "tle_{id}.txt"    # -> tle_00900.txt
  zero_pad = true
  ```

  Placeholders are `{id}` (object outputs) or `{date}` (date outputs),
  plus `{ext}`, `{format}` and `{type}`. The template only *arranges* the
  pieces — `object_id`/`zero_pad`, `date_format` and `extension` still
  decide how each renders. It defaults to `"{id}{ext}"` / `"{date}{ext}"`,
  so entries that omit it name files exactly as before.

  Templates are validated when the config loads rather than mid-generate:
  an unknown placeholder, a missing `{id}`/`{date}`, a format spec, or a
  path separator fails immediately, since a naming error raised partway
  through a run would leave a half-written output tree behind. `--verify`'s
  orphan scan derives its glob and its filename parser from the same
  template, so it tracks whatever naming is configured.

- **`[ingest] max_epoch_ahead_days` — reject elsets whose epoch is too far
  in the future (default 30 days).** TLE epochs carry two-digit years
  (00-56 -> 2000-2056), so a corrupted epoch field parses cleanly into a
  date decades out. Once stored, such a row permanently poisons its object
  file's tail guard — every later legitimate elset triggers a full rewrite
  instead of an append — and plants a junk date file nothing cleans up.
  Rejected elsets are quarantined like any other (see `reject_dir` below).

  The guard is one-sided: past epochs are never bounded, so historical
  archives ingest exactly as before. Real feeds run at most about a day
  ahead of now, so the default is generous; raise it for predicted-elset
  workflows, or set `max_epoch_ahead_days = 0` to disable.

  **Default-on behavior change**: such elsets were previously stored. If
  one already reached your database, `--verify` will not remove it (it is
  a valid row); rejecting it going forward starts at the next ingest.

- **`[ingest] reject_dir` — a quarantine directory for records that could
  not be ingested.** Each rejected record is copied there in its source
  file's own format, mirroring the source tree, so provenance is visible in
  the path and a repaired file is directly re-ingestable:

  ```
  /data/incoming/2006/20060101.tle   (source)
  rejects/incoming/2006/20060101.tle (rejected records, verbatim)
  rejects/incoming/2006/20060101.tle.log  (one reason per record)
  ```

  The directory is a live view of what is currently broken, not an archive:
  artifacts are overwritten each run and deleted once a file ingests
  cleanly, so volume is bounded by what is currently broken and no
  retention policy is needed. Past 10,000 rejected records from one file,
  only a marker with counts and a reason histogram is written — beyond that
  the file is not a good feed with bad lines but a wrong file, and copying
  it would duplicate the whole delivery.

  Unset by default, so existing deployments are unchanged; the `init`
  scaffold ships it enabled at `./rejects`.

### Changed

- **`generate` is 4-6x faster.** On a synthetic 5k-object / 40-day catalog
  across four outputs, a full rebuild went from 38.2s to 10.0s and a
  steady-state incremental run from 13.5s to 2.2s. Four changes, in order
  of what they bought:

  - **Date files are rewritten only when they are older than their rows.**
    Previously every date in the trailing `window_days` (default 60) was
    re-queried and rewritten on every run, whether or not it had new data —
    10.2s of the 13.5s incremental run, and O(catalog x window) rather than
    the O(new rows) the design calls for.

    Scope now comes from a new `epoch_date_state` table: one row per epoch
    date holding the newest `created` among that date's elsets, maintained
    by ingest in the same transaction as the rows it describes. A run
    compares each date's watermark against that date's file mtime — one
    `stat` per date per output — and rewrites only what is genuinely stale.

    **`[output] window_days` and `--window-days` are gone**, and
    `lookback_days` no longer affects date files at all. Because the table
    grows by a row a day rather than with the catalog, every date file is
    checked on every run, so a generation gap of any length, a restore that
    resets `created`, and a delivery for a date years old all self-correct
    on the next ordinary run — no `generate --all` rule for date files.

    **Behavior change worth noting**: a date file that is silently
    corrupted — truncated or hand-edited, so its mtime is *newer* than its
    watermark — is no longer repaired by the next run. `--verify` now
    covers date files for exactly this reason; if you relied on the window
    rewrite as self-healing, add `--verify` to a periodic cron entry.

  - **Output writes run on a thread pool**, sized by the new
    `[output] write_workers` (0 = auto, 1 = serial). Per-file `open`/close
    was 44% of a full rebuild — blocked time that releases the GIL.
  - **Rows load as Core column tuples instead of ORM entities**, which was
    ~7.6s of the full rebuild; `omm_metadata` is joined only when an omm
    output exists.
  - **Cheaper serialization**: one write per TLE file instead of two
    `print` calls per record, and `csv.writer` over pre-ordered rows
    instead of `DictWriter`.

  Output content is unchanged — OMM export still goes through sgp4's
  `export_omm`, so generated files are byte-identical.

- `--verify` now reconciles **date files** as well as object files, using
  per-date distinct object counts against each tle date file's line count.

- `scripts/bench_generate.py` builds a synthetic catalog and times both
  generation shapes, so the numbers above stay reproducible.

- Individual rejected records now log at DEBUG. Each file instead gets one
  WARNING reporting rejections **over the total attempted**
  (`3 of 30000 records rejected -> ...`) — the ratio distinguishes a few bad
  lines from an entirely wrong file, and a persistently broken feed no
  longer emits thousands of warning lines per run.

## [0.12.0] - 2026-08-19

### Breaking — output config redesigned

`[output]` no longer takes `dir`, `[output.formats]`, or `[output.types]`.
Outputs are now declared as `[[output.files]]` entries — each one names a
type, a format, a destination directory, and a filename scheme, so several
outputs can be generated side by side (different directories per format,
alpha-5 and integer naming conventions at once, custom extensions):

```toml
[output]
window_days = 60
lookback_days = 7

[[output.files]]
type = "object"          # "date" | "object"
format = "tle"           # "tle" | "omm"
dir = "./output/tle"
object_id = "alpha5"     # "int" (default) | "alpha5"  -> E5693.tle
# zero_pad = true        # with object_id = "int": 00900.tle
# extension = ".txt"     # override the ".tle"/".omm" default
# date_format = "%Y%m%d" # date files: strftime stem pattern
```

Old configs fail validation with a pointer at the removed key (`dir`,
`formats`, `types`) rather than being silently reinterpreted, and there is
no implicit default: `generate` with no `[[output.files]]` entries prints
an error and exits 2 (the library `generate()` raises `ValueError`). To
reproduce the previous behavior, declare the four entries the `init`
scaffold ships (date + object × tle + omm under `./output`). No database
change — regenerate outputs with `thistle-db generate --all` after
switching naming schemes (old files with the previous naming are left
behind; `--verify` reports them as orphans).

### Changed

- Each output now self-heals independently in the incremental object pass:
  a missing or damaged file in one output is rewritten from the database
  without forcing a rewrite of the sibling format (outputs needing a
  rewrite share one fetch). A deleted `.omm` file is now restored with full
  history on the next run that touches the object (previously it was
  recreated with only the pending rows until a `--verify` sweep).

## [0.11.0] - 2026-07-31

### Fixed

- **Credential resolution order now matches the documentation**: env vars >
  user secrets (`~/.config/thistle-db.toml`) > system secrets file >
  config.toml values. Previously the order was effectively reversed —
  `THISTLE_DB_DATABASE__USERNAME`/`__PASSWORD` env vars lost to any
  file-provided value, and the system secrets file beat the user one. A
  scaffolded-but-unfilled user secrets file (empty strings) no longer masks
  credentials from lower layers.

  **Check your deployment before upgrading**: if credentials differ between
  layers (e.g. a stale env var alongside current file credentials, or values
  in both secrets files), the credential actually used may change with this
  release.
- `load_config(None)` (library callers, e.g. `thistle_db.get_tles`) now
  honors `$THISTLE_DB_CONFIG` and falls back to `./config.toml`, matching
  the CLI's config discovery.
- A TLE with a non-numeric international designator (e.g. a `TBA`
  placeholder) is now ingested with the designator stored verbatim instead
  of being skipped as malformed.
- An `.omm` output file whose last epoch has zero microseconds is no longer
  misread as damaged (which caused a harmless but pointless rewrite).
- TLE parsing in `ingest` now requires the standard `"1 "`/`"2 "` line
  prefixes (matching the generator's tail guard), so 3LE name lines
  beginning with a digit can no longer be misread as element lines.

### Changed

- `ingest FILE...` (explicitly named files) now exits non-zero when any
  named file fails to ingest, so cron wrappers can detect the failure.
  Directory scans (`ingest` with no arguments) remain tolerant: failed
  files are logged and retried on the next scan, exit code 0.
- `dump` streams the OMM JSON export instead of building it in memory —
  full-catalog dumps now run in constant memory (the `.tle` side already
  did). The JSON is written one record per line (still a valid array;
  `ingest` reads it unchanged).
- OMM ingest resolves TLE foreign keys with an index-backed
  `(epoch, line_hash)` lookup; previously the lookup was by `line_hash`
  alone, which required a full-table scan per chunk on large MariaDB
  catalogs.
- `init` now creates `~/.config/thistle-db.toml` with owner-only
  permissions (`0600`) on POSIX systems.

### Documentation

- **Ingest and generate must not run concurrently** — the incremental
  generator's mtime watermark assumes it. This constraint is now explicit,
  and the README cron examples with separate entries serialize them with a
  shared `flock`. (Chained `ingest && generate` entries were always safe;
  the weekly `--verify` sweep remains the backstop.)

## [0.9.0] - 2026-07-30

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

## [0.8.0] - 2026-07-29

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
