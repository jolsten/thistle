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
  pydantic-settings models: `[database]`, `[[ingest.sources]]`, `[output]`,
  `[logging]`.
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
- `init-db [--drop] [--yes]` — create the database schema; the intended DDL
  entry point (run once with an admin account when using MariaDB/PostgreSQL).
  Idempotent without `--drop`. `--drop` destroys and recreates all tables:
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
  Restore = `init-db` + `ingest` both files; works across dialects. Refuses
  to overwrite existing outputs without `--force`. Physical backups of live
  servers belong to native tools (`mariadb-dump`, `pg_dump`, SQLite file
  copy), not this command.
- `generate [--all] [--window-days N] [--lookback-days N] [--verify]` —
  write output files per `[output]` config:
  - `date_files`: `YYYYMMDD.{tle,omm}` — latest element set per satellite for
    that date.
  - `object_files`: `NORAD_ID.{tle,omm}` — all element sets for a satellite,
    ordered by epoch.
  - Formats toggleable via `[output.formats]` (tle = two-line text, omm = CSV).
  - **Incremental by default** — steady-state cost must stay O(new rows),
    independent of catalog size, with no persistent generator state:
    - Date files: rewritten for a trailing epoch window
      (`output.window_days`, default 60) plus any date that received new rows
      within the lookback — so arbitrarily late deliveries land in the right
      old date file.
    - Object files: rows created within `output.lookback_days` (default 7)
      are streamed in keyset batches ordered by `(norad_cat_id, epoch, id)`
      and **appended** to each object's files behind a tail guard (only rows
      strictly newer than the file's last epoch are appended; the file mtime
      distinguishes overlapping-lookback re-runs from genuinely late rows).
      The mtime watermark assumes ingest and generate never run
      concurrently — deployments must serialize them (cron chaining with
      `&&`, or a shared `flock`); a row committed mid-generate could
      otherwise be misclassified as already on disk until the next
      `--verify` sweep.
      Any anomaly — late delivery, epoch tie, torn last line, missing or
      inconsistent file — falls back to a full rewrite of that one object
      from the database, which is always authoritative. The `.tle` file is
      the tail authority; the `.omm` file may legitimately hold only a
      subset (sgp4 cannot export every elset).
    - `--all` rebuilds everything (first run, disaster recovery, or after a
      generation gap longer than the lookback). `lookback_days` must exceed
      the ingest cron cadence.
    - `--verify` follows the normal run with a count-reconciliation sweep:
      per-object database row counts (one aggregated index query) against
      each `.tle` file's line count, rewriting mismatches — catches damage
      the tail guard can't see (mid-file truncation/edits, files deleted for
      quiet objects). Reads all output files, so it belongs on a periodic
      (e.g. weekly) cron entry, not every run. Requires tle output; the
      `.tle` file is the verification authority.

## Module map

| Module | Responsibility |
|---|---|
| `cli.py` | typer CLI, subcommand dispatch, logging setup |
| `config.py` | pydantic-settings config + layered secrets resolution |
| `model.py` | SQLAlchemy ORM models (`TLE`, `OmmMetadata`), TLE parsing via sgp4 |
| `reader.py` | format detection + file readers (TLE, OMM JSON/CSV/XML) |
| `ingest.py` | bulk insert with dedup, source-directory scanning |
| `generator.py` | output file generation from the database |
| `progress.py` | shared rich console (stderr) + no-op-able progress reporter |

## Error handling philosophy

Ingest is tolerant: a malformed TLE or OMM record is logged (loguru, WARNING)
and skipped — one bad record must never abort a file, and one bad file must
never abort a scan. A missing source directory is a warning, not an error.

Logging levels: routine no-ops (unchanged files skipped on scan) log at
DEBUG; per-file actions and per-directory summaries at INFO; data problems
at WARNING. The INFO log of a steady-state cron run should stay short.

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
`dump` → `init-db --drop` → `ingest` (plus one `generate --all`, since a
restore resets `created` timestamps and the incremental generator's state
derives from them). Do not add a migration framework without an explicit
decision — the trigger to revisit is when restore downtime becomes
unacceptable (roughly 100M+ rows). Purely additive changes (a new index, a
nullable column) may instead ship with hand-written `ALTER` statements for
all three dialects in the release notes.
