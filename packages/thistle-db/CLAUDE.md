# CLAUDE.md

This file is the **target design spec** for thistle-db. Where the code and this
document disagree, this document wins — update the code to match, not the other
way around. Sections marked **[NOT YET IMPLEMENTED]** are known gaps.

## What thistle-db is

thistle-db is a database manager for orbital element sets. It ingests TLE
(Two-Line Element) and OMM (Orbit Mean-Elements Message) files from configured
directories into a SQL database with uniqueness guarantees, and generates
organized output files (by date and by satellite) from that database.

## Core workflow

1. TLE/OMM files are delivered into directories defined in `config.toml`
   (`[[ingest.sources]]` entries: a `path` plus a glob `pattern`).
   The schema is created once with `thistle-db init-db` — the **only** DDL
   path; other commands never create tables and fail with a clear
   "run `thistle-db init-db`" error (exit 3) against an uninitialized
   database.
2. `thistle-db ingest` scans those directories, auto-detects each file's format,
   parses the element sets, and inserts them into the database. Duplicates are
   silently skipped — ingest is **idempotent**: running it twice over the same
   files must never create duplicate rows or fail.
3. `thistle-db generate` writes output files from the database.

### File formats

Format is auto-detected per file (`reader.detect_format`):

| Format | Extensions | Reader |
|---|---|---|
| TLE (2-line or 3-line) | `.tle`, `.txt`, `.3le` | `read_tle` |
| OMM JSON (Space-Track) | `.json` | `read_omm_json` |
| OMM CSV | `.csv` | `read_omm_csv` |
| OMM XML | `.xml` | `read_omm_xml` |

### Delivery patterns to handle

- **New file per delivery** (e.g. daily `YYYYMMDD.txt`): straightforward scan-and-ingest.
- **In-place updates**: a provider may append to or rewrite an existing file
  rather than delivering a delta. This must be handled elegantly:
  - Correctness comes from DB-level dedup — re-reading a whole updated file and
    inserting only the new element sets is always safe.
  - Efficiency comes from **file-state tracking**: each successfully ingested
    file's path, size, mtime, and content hash is recorded in the
    `ingest_files` table (`IngestFile` model). On scan
    (`ingest.ingest_source_file`), unchanged files (size + mtime match) are
    skipped without being opened; a changed file whose content hash still
    matches just refreshes its state; otherwise the file is re-ingested (dedup
    absorbs the already-seen records). A file that fails to parse gets no
    state recorded, so it is retried next scan. `--force` bypasses skipping.

### Malformed records and quarantine

sgp4's parser is C `strtod`, which accepts the literal text `nan`, `inf` and
`nan(ind)` in a fixed-width field — exactly what a producer emits when it
formats a missing value with `printf` — and corrupt text shifts field
boundaries to the same effect. **Every float column can therefore arrive
non-finite**, and NaN reaches MariaDB as an unquoted `nan` token that pymysql
refuses *before* sending SQL, failing the whole insert chunk.

`model._clean_floats` handles this at the one boundary where sgp4 output
enters the model, driven by the schema's own nullability so a float column
added later is covered without touching it:

- non-finite in a **nullable** column → `NULL`. `line1`/`line2` remain the
  lossless source of truth, so a missing derived value costs nothing.
- non-finite in a **mandatory** element (eccentricity, inclination, RAAN,
  argument of pericenter, mean anomaly) → `MalformedElsetError`, and the
  record is rejected. There is no orbit to store: it cannot be propagated or
  compared, and a row carrying it would be selected by
  `_latest_per_object_for_date`, *displacing* the object's good elset from
  that day's output file.

One caveat when writing tests or reproducing a report: whether a `nan` text
field actually parses to NaN depends on the column *and the process*. Line
2's plain-float fields are stable, but line 1's `bstar`/`ndot`/`nddot` use
sgp4's own exponent-format reader, which yields `0.0` instead of NaN once
numpy has been imported (numpy changes the CRT parse on Windows). A fixture
built on a `nan` bstar therefore passes standalone and fails under a suite
that pulls numpy in. Prefer a zero mean motion, whose infinite derived
values come from arithmetic rather than parsing, and unit-test
`_clean_floats` directly for the per-column mapping.

No stored float is ever non-finite. Do not relax this by making the
mandatory columns nullable and flagging invalid rows — that would require
threading a validity predicate through every read path (both generator
queries, `tles_for_object`, `nearest_tles_for_date`, `verify_object_files`,
and thistle's db fallback), and missing one yields silent NaN positions
downstream.

A second rejection class is the **future-epoch guard**: an elset whose
epoch runs more than `[ingest] max_epoch_ahead_days` (default 30; 0
disables) ahead of now is rejected. TLE epochs carry two-digit years
(00-56 → 2000-2056), so a corrupted epoch field parses *cleanly* into a
date decades out — and once stored, such a row does two kinds of permanent
damage: it becomes its object file's tail epoch, so every later legitimate
elset compares `epoch <= tail` and triggers a full rewrite instead of an
append, forever; and it plants a junk date file (plus a watermark row)
that nothing cleans up. Real feeds deliver elsets at most hours to a day
ahead, so 30 days is generous by orders of magnitude; raise it for
predicted-elset workflows. The guard is deliberately one-sided — epochs in
the past are never bounded, so historical archives ingest untouched.

Rejected records are preserved outside the database instead. With
`[ingest] reject_dir` set, each one is copied to a quarantine directory
(`rejects.py`) in its source file's own format, mirroring the source tree so
provenance is visible in the path and a repaired file is directly
re-ingestable:

    /data/spacetrack/daily/2006/20060101.tle   (source)
    rejects/daily/2006/20060101.tle            (rejected records, verbatim)
    rejects/daily/2006/20060101.tle.log        (one reason per record)

Two properties are load-bearing:

- **Live view, not an archive.** Each ingest of a source file overwrites
  that file's artifacts, and a run with no rejects deletes them. Volume is
  bounded by what is *currently* broken rather than growing with time, so no
  retention policy is needed. Partitioning by ingest date would break this
  (a persistently broken daily feed would write a fresh copy every day, and
  a backfill would dump every reject into one directory).
- **Bounded per file.** Past `REJECT_MAX_RECORDS` (10k, a constant — a fuse,
  not a tuning knob) only a marker with counts and a reason histogram is
  written. Beyond that cap a file is not a good feed with bad lines but a
  wrong file (mis-encoded, still gzipped, truncated), and copying it would
  duplicate the entire delivery.

A quarantine failure is logged and never propagates: it must not turn a
working ingest into a failed one.

Chunk-level insert failures fall back to row-by-row inserts
(`ingest._insert_individually`), so a record the database refuses costs only
itself rather than the 5000-row chunk it was batched with — and, since the
exception previously propagated out of `ingest_tles`, the rest of the file.

## Data model (`model.py`)

Single canonical element-set table plus an OMM metadata sidecar. Do **not**
split OMM into a fully separate element-set table.

- **`tle`** — one row per unique element set. Stores the raw `line1`/`line2`
  text, a `line_hash` (sha256 hex of the exact text — the dedup key), parsed
  fields (norad_cat_id, epoch, Keplerian elements, drag terms) and derived
  values (semimajor axis, period, apoapsis/periapsis altitude). Parsing goes
  through `sgp4.Satrec`.
- **`omm_metadata`** — OMM-only fields (object name/type, country code, RCS
  size, launch/decay date, site, originator, GP_ID), one-to-one with a `tle`
  row via `tle_id`.
- **`ingest_files`** — per-file ingest state (path, size, mtime_ns, sha256)
  used to skip unchanged source files on scan; keyed by a sha256 of the
  resolved path (bounded length for cross-dialect unique indexes).
- **`epoch_date_state`** — one row per epoch date holding the newest
  `created` among that date's element sets; the date pass's scope. Derived
  data (`date(epoch) -> MAX(created)`), maintained incrementally by ingest
  and recomputable at any time by `init-db` (`api.rebuild_epoch_date_state`).
  Two rules are load-bearing: the watermark must be written **in the same
  transaction** as the rows it describes, so a crash can never leave a date
  permanently unregenerated; and it holds one row per *date*, not per elset,
  so checking every date file every run stays affordable as the catalog
  grows. Rows arriving outside `ingest` (a bulk `COPY`, an admin script)
  bypass the watermark and need a rebuild — the only realistic drift.

Schema conventions (deliberate, sized for hundreds of millions of rows on
MariaDB — do not regress them casually):

- Surrogate keys are **BIGINT** (`INTEGER` on SQLite for rowid autoincrement):
  `INSERT IGNORE` burns auto-increment values on every duplicate it skips, so
  id consumption tracks rows *attempted*, not stored.
- All stored datetimes are **naive UTC with microseconds** (`DATETIME(6)` on
  MariaDB); `epoch` must round-trip exactly for the generator's tail guard.
- Element/derived float columns are **single precision** (`Float32` =
  FLOAT(24)): the TLE text carries at most 7 significant digits everywhere
  except `mean_motion` (10 digits — kept DOUBLE), and `line1`/`line2` are
  the lossless source of truth, so these columns are a re-derivable
  convenience view. Never compare them for equality; recompute from the
  lines when full precision matters.
- Rows are immutable once inserted — there is no `modified` column.
- The `tle` **index budget is intentionally minimal** (PK,
  `UNIQUE(epoch, line_hash)`, `(norad_cat_id, epoch)`, `(created)`). Every
  additional secondary index taxes every insert; add one only with a query
  that needs it.

### Uniqueness

The dedup key is the **exact text of `(line1, line2)`**, enforced as
`UNIQUE(epoch, line_hash)` where `line_hash` is the raw 32-byte sha256 of
`line1 + "\n" + line2` (`BINARY(32)` on MariaDB — a plain BLOB could not
back the unique index; use `HEX(line_hash)` in ad-hoc SQL). This is
semantically identical to a unique constraint on the raw text (the epoch is
encoded in line1, so identical lines always share an epoch) but keeps the
index 32 bytes wide instead of 560. Epoch-first ordering means
normal "recent elsets" ingest probes only the hot right edge of the index
regardless of table size. Textual near-duplicates from different providers
coexist as separate rows — this is deliberate. Do not change the key to
`(norad_cat_id, epoch)` or similar without an explicit decision.

If a TLE is delivered first and the OMM version of the same element set arrives
later (with identical lines), the TLE row is **not** duplicated — the OMM
delivery attaches an `omm_metadata` row to the existing `tle` row (resolved by
`line_hash`). Both representations are thereby preserved: the TLE lines in
`tle`, the OMM extras in `omm_metadata`.

Dedup is enforced at the database with dialect-aware upserts
(`ingest._bulk_insert_ignore`): `ON CONFLICT DO NOTHING` for SQLite/PostgreSQL,
`INSERT IGNORE` for MySQL/MariaDB. Inserts are chunked (`CHUNK_SIZE = 5000`).
Any new bulk-write path must follow this pattern and work on all three dialects.

## Configuration (`config.py`)

- `config.toml` (path via `-c`, default `./config.toml`), parsed into
  pydantic-settings models: `[database]`, `[ingest]` (`reject_dir`,
  `max_epoch_ahead_days`, plus `[[ingest.sources]]` entries), `[output]` (with `[[output.files]]`
  entries — see the `generate` command), `[logging]`.
  `[output]` rejects unknown keys (`extra="forbid"`), so pre-0.12 configs
  (`dir`, `formats`, `types`) fail validation with a pointer at the key
  instead of being silently ignored.
- Database credentials are **never** stored in `config.toml`. Resolution order
  (highest priority first):
  1. Env vars `THISTLE_DB_DATABASE__USERNAME` / `THISTLE_DB_DATABASE__PASSWORD`
  2. User secrets file `~/.config/thistle-db.toml`
  3. System secrets file (`database.secrets_file` in config)
- Supported databases: SQLite (default/dev), MariaDB/MySQL (operational,
  `thistle-db[mysql]` extra), PostgreSQL (tested in CI).

## CLI (`cli.py`)

`thistle-db [-c CONFIG] [--log FILE] [--no-progress] <command>` — the
config path defaults to the `THISTLE_DB_CONFIG` env var when set (shared
with thistle's db fallback), else `./config.toml`. Global options:

- `--log FILE` — add a rotating loguru file sink (10 MB rotation, last 10
  kept) alongside console logging; the cron-friendly alternative to shell
  redirection.
- `--no-progress` — force the progress bar off. The bar (rich, stderr,
  `progress.py`) only appears when stderr is an interactive terminal, so
  crons never see it; log lines render above the live bar via the shared
  rich console. Command *output* (get-tle TLEs, dump messages) is stdout
  and must never route through the progress console.

Commands are privilege-tiered: `init-db` is the only DDL path, `ingest` the
only DML path, and the read commands (`get-tle`, `generate`, `dump`, plus
thistle's `get_tles` fallback) open **read-only connections** — writes are
rejected at the connection level (`PRAGMA query_only` / `SET SESSION
TRANSACTION READ ONLY` / read-only transaction characteristics) as defense
in depth on top of DB grants. Deployments can therefore give the cron
account DML-only rights and readers SELECT-only rights:

- `init` — scaffold `config.toml` and `~/.config/thistle-db.toml`.
- `init-db [--drop] [--yes] [--defer-indexes]` — create the database
  schema; the intended DDL entry point (run once with an admin account when
  using MariaDB/PostgreSQL). Also recomputes `epoch_date_state` from the
  element sets (the adoption path for an existing database and the repair
  for rows inserted outside `ingest`), and builds any *missing* model
  index — so a plain `init-db` is also the finalize step after a deferred
  rebuild. Idempotent without `--drop`.
  `--defer-indexes` (requires `--drop`) creates the schema without the
  read-path indexes (`ix_tle_norad_cat_id_epoch`, `ix_tle_created` —
  `api.DEFERRABLE_INDEXES`) for bulk restores; the unique dedup indexes are
  never deferred, so ingest idempotency holds throughout. Flow:
  `init-db --drop --defer-indexes` → ingest everything → `init-db`. `--drop` destroys and recreates all tables:
  it prompts for confirmation, and in non-interactive use fails closed
  unless `--yes` is passed.
- `ingest [FILES...] [--force]` — ingest specific files (always parsed, even
  if recorded as unchanged), or scan all configured sources when no files are
  given; `--force` re-ingests scanned files regardless of recorded state.
  With named files, any failure yields exit code 1 (after all files are
  attempted) so cron wrappers can detect it; scans stay tolerant (exit 0,
  failed files retried next scan).
- `get-tle TARGET [--days N]` — print TLEs to stdout. `TARGET` is either an
  8-digit date `YYYYMMDD` (→ the nearest TLE per object to 12:00 UTC on that
  date, within ±N days, default 7) or an alpha-5-compatible NORAD ID
  (e.g. `25544`, `00022`, `E5693` → every TLE for that object, epoch order).
  Exits 1 if nothing matches.
- `dump OUTPUT [--force]` — logical backup: write every element set to
  `OUTPUT.tle` (lossless, verbatim line text) and rows with OMM metadata to
  `OUTPUT.json` (Space-Track form, only created when metadata exists).
  Restore = `init-db` + `ingest` both files (for large catalogs use the
  deferred-index flow under "Schema changes"); works across dialects. Refuses
  to overwrite existing outputs without `--force`. Physical backups of live
  servers belong to native tools (`mariadb-dump`, `pg_dump`, SQLite file
  copy), not this command.
- `generate [--all] [--window-days N] [--lookback-days N] [--verify]` —
  write output files per `[output]` config. Outputs are declared as
  `[[output.files]]` entries; each names a `type`, a `format` (`tle` =
  two-line text, `omm` = CSV), a destination `dir` (created if missing;
  entries may share or split directories), and a filename scheme. At least
  one entry is required: with none configured, `generate` prints an error
  and exits 2 (misconfiguration, not a silent no-op); the library entry
  point raises `ValueError`. The `init` scaffold ships the standard four
  entries (date + object files in both formats under `./output`).
  - `type = "date"`: one file per epoch date — latest element set per
    satellite for that date. Filename stem from `date_format` (strftime,
    default `"%Y%m%d"`).
  - `type = "object"`: one file per satellite — all element sets, ordered
    by epoch. Filename stem is the NORAD ID, rendered per
    `object_id = "int"` (default) or `"alpha5"` (always 5 characters, e.g.
    `00900`, `E5693`); `zero_pad = true` pads int IDs to 5 digits.
  - `extension` overrides the `.tle`/`.omm` default suffix on any entry.
  - `filename` is a template arranging those pieces, defaulting to
    `"{id}{ext}"` (object) and `"{date}{ext}"` (date) — an entry that omits
    it names files exactly as before. Placeholders are `{id}`/`{date}`
    (rendered per `object_id`/`zero_pad` and `date_format`), `{ext}`,
    `{format}` and `{type}`; e.g. `filename = "tle_{id}.txt"` →
    `tle_00900.txt`. Templates are validated at **config load**, not
    mid-generate: an unknown placeholder, a missing `{id}`/`{date}` (every
    file would render to one name and overwrite the last), a format spec,
    or anything containing a path separator or `..` fails there, since a
    `KeyError` raised while naming the 40,000th file would leave a
    half-written output tree behind. Directories come from `dir` only.
    `OutputFile.parse_object_name`/`object_glob` derive the inverse from
    the same template, so `--verify`'s orphan scan tracks whatever naming
    is configured.
  - **Incremental by default** — steady-state cost must stay O(new rows),
    independent of catalog size, with no persistent generator state:
    - Date files: rewritten when the database holds rows **newer than the
      file**. Scope comes from `epoch_date_state`, a one-row-per-date
      watermark table ingest maintains in the same transaction as the rows
      it describes; each date's watermark is compared against that date's
      file mtime. Because the table grows by a row a day rather than with
      the catalog, **every** date file is checked on every run — one `stat`
      per date per output — so there is no trailing window and no lookback
      for dates. A generation gap of any length, a restore that resets
      `created`, and a delivery for a date years old all self-correct on
      the next ordinary run, with no `--all`. The tradeoffs: silent
      corruption of a file whose watermark is older than it is caught by
      `--verify` rather than the normal run, and the mtime comparison
      carries the same requirement as the object pass (ingest and generate
      must not overlap).
    - Object files: rows created within `output.lookback_days` (default 7)
      are streamed in keyset batches ordered by `(norad_cat_id, epoch, id)`
      and **appended** to each object's file in every object output behind
      a tail guard (only rows strictly newer than the file's last epoch are
      appended; the file mtime distinguishes overlapping-lookback re-runs
      from genuinely late rows). Each output decides independently against
      its own file's tail and mtime.
      The mtime watermark assumes ingest and generate never run
      concurrently — deployments must serialize them (cron chaining with
      `&&`, or a shared `flock`); a row committed mid-generate could
      otherwise be misclassified as already on disk until the next
      `--verify` sweep.
      Any anomaly — late delivery, epoch tie, torn last line, missing or
      inconsistent file — falls back to a full rewrite of that object's
      file from the database, which is always authoritative (outputs
      needing a rewrite share one fetch). An omm file may legitimately hold
      only a subset of rows (sgp4 cannot export every elset), so its tail
      may lag; re-appending a previously seen but unexportable row exports
      nothing and stays harmless.
    - `--all` rebuilds everything (first run, disaster recovery, or after a
      generation gap longer than the lookback). `lookback_days` must exceed
      the ingest cron cadence.
    - `--verify` follows the normal run with a count-reconciliation sweep
      of **both** file kinds, rewriting whatever disagrees:
      - object files: per-object database row counts (one aggregated index
        query) against each tle object output's line count;
      - date files: per-date distinct object counts (one aggregated query)
        against each tle date file's line count — the compensating control
        for the normal run trusting watermarks rather than re-reading
        files.
      It catches damage the tail guard can't see (mid-file
      truncation/edits, files deleted for quiet objects). Reads all output
      files, so it belongs on a periodic (e.g. weekly) cron entry, not
      every run. Requires a tle output of the kind being verified — tle
      files hold every row verbatim and are the verification authority.

### Generation performance

Two rules keep generation from regressing:

- **Rows are read as Core column tuples, never ORM entities.** Materializing
  `TLE` objects cost ~7.6s of a 38s full rebuild and held the whole batch as
  Python objects. `generator._elset_columns` selects exactly the columns
  generation reads, and joins `omm_metadata` only when an omm output exists.
- **Filesystem work runs on a thread pool** (`writer.WritePool`,
  `output.write_workers`, default auto). Per-file `open`/close was 44% of a
  full rebuild — pure blocked time, and the GIL is released for it. The pool
  only ever receives pure filesystem work; the SQLAlchemy `Session` is not
  thread-safe, so **every query stays on the calling thread** and tasks are
  handed data already fetched. Work is submitted a chunk of objects at a
  time, so memory stays bounded and each task owns a distinct path.

`scripts/bench_generate.py` builds a synthetic catalog and times both
shapes; on 5k objects x 40 days across four outputs, `--all` runs in ~10s
and a steady-state incremental run in ~2.2s (from 38s and 13.5s).

## Module map

| Module | Responsibility |
|---|---|
| `cli.py` | typer CLI, subcommand dispatch, logging setup |
| `config.py` | pydantic-settings config + layered secrets resolution |
| `model.py` | SQLAlchemy ORM models (`TLE`, `OmmMetadata`), TLE parsing via sgp4 |
| `reader.py` | format detection + file readers (TLE, OMM JSON/CSV/XML) |
| `ingest.py` | bulk insert with dedup, source-directory scanning |
| `rejects.py` | quarantine for records that could not be ingested |
| `generator.py` | output file generation from the database |
| `writer.py` | thread pool for output file writes |
| `progress.py` | shared rich console (stderr) + no-op-able progress reporter |

## Error handling philosophy

Ingest is tolerant: a malformed TLE or OMM record is skipped (and
quarantined when `reject_dir` is set) — one bad record must never abort a
file, and one bad file must never abort a scan. A missing source directory
is a warning, not an error.

Logging levels: routine no-ops (unchanged files skipped on scan) and
individual rejected records log at DEBUG; per-file actions and per-directory
summaries at INFO; data problems at WARNING. Rejections are reported as one
WARNING per file giving the count **over the total attempted** — the ratio
is what separates a few bad lines in a good file from an entirely wrong
file, and a per-record warning would emit thousands of lines for a
persistently broken feed. The INFO log of a steady-state cron run should
stay short.

## Development

- Python ≥ 3.11. Package manager: `uv`. Build: hatchling + hatch-vcs
  (version comes from git tags — never hardcode it).
- Run tests: `uv run pytest tests/thistle_db` from the workspace root. Lint:
  `uv run ruff check`. Types: `uv run pyright`.
- Tests cover SQLite by default; set `THISTLE_DB_TEST_MARIADB=1` /
  `THISTLE_DB_TEST_POSTGRES=1` (requires Docker) to run the MariaDB and
  PostgreSQL backends via testcontainers — CI runs all three. Dialect-specific
  behavior (upserts) must be tested against all three.
- Test fixtures live in `tests/thistle_db/data/` (real TLE text files and
  Space-Track OMM JSON).

### Schema changes (no migration framework — deliberate)

There is intentionally **no Alembic or other migration tooling**. The
database is a derived store: every row reconstitutes losslessly from a
`dump`, so schema changes ship as breaking releases migrated by
`dump` → `init-db --drop --defer-indexes` → `ingest` → `init-db` (plus one
`generate --all`, since a restore resets `created` timestamps and the
incremental generator's state derives from them).

**Restore performance (MariaDB).** Restoring a large catalog is index-bound,
not parse- or round-trip-bound: ingest issues ~40 server queries per 30k-row
file (pymysql batches the chunks), but modern MariaDB has no InnoDB change
buffer (removed in 11), so every secondary-index insert touches a real
page. A full-catalog snapshot file scatters one `(norad_cat_id, epoch)`
page access per row across the whole index; once that index outgrows
`innodb_buffer_pool_size`, each becomes random storage I/O and per-file
time is set by storage latency, not by thistle-db. Two remedies, in order:
defer the read-path indexes during the restore (`--defer-indexes` — the
final `CREATE INDEX` is one sorted build each), and size the buffer pool
to hold the `tle` indexes. The unique dedup index stays cheap during
chronological restores because a snapshot file's epochs land in one hot
region of the epoch-first index — another reason its column order must not
change. Do not add a migration framework without an explicit
decision — the trigger to revisit is when restore downtime becomes
unacceptable (roughly 100M+ rows). Purely additive changes (a new index, a
nullable column) may instead ship with hand-written `ALTER` statements for
all three dialects in the release notes.
