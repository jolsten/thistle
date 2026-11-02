# CLAUDE.md

This file is the **target design spec** for thistle-db. Where the code and this
document disagree, this document wins — update the code to match, not the other
way around. Sections marked **[NOT YET IMPLEMENTED]** are known gaps.

**Architecture status (2026-09):** this spec describes the GP-record
architecture adopted from `PLAN-gp-store.md` (kept beside this file as the
design rationale). The shipping code still implements the previous
architecture — dual `database`/`files` backends, TLE-line stores, a
steady-state `generate` command with `epoch_date_state` watermarks; see this
file's git history for its documentation. That machinery is scheduled for
replacement per the build order at the bottom: fix bugs in it, but do not
extend it. `record.py` (the canonical record) is implemented and tested.

## What thistle-db is

thistle-db is a database manager for orbital element sets. It ingests TLE
(Two-Line Element) and OMM (Orbit Mean-Elements Message) files from configured
directories into a canonical store — a per-object file tree or a SQL
database — and maintains organized output files (TLE by object and by date;
STK-compatible OMM CSV) directly from ingest.

## Shape of the system [NOT YET IMPLEMENTED]

```
sources ──► readers ──► validation ──► delta buffer (GP records, by object)
(TLE/OMM)               (rejects,           │  flush at end-of-scan / flush_records
                         epoch guard,       ▼
                         quarantine)   1. store.add(delta)         dedup by RECORD_HASH
                                       2. output.apply(delta) ×N   merge-dedup / tail guard / date upsert
                                       3. commit ingest_files state
```

Three ideas, one per layer:

1. **One canonical record** — the GP record (`record.py`), an OMM-superset
   modeled on Space-Track's published GP class, carrying verbatim TLE lines
   as fields. Everything produces, stores, or projects this record.
2. **One engine** — the single-pass delta flush. Ingest validates records,
   buffers them grouped by object, and at flush time hands the same delta to
   every sink. Source-file state commits only after all sinks succeed.
   **Every sink is idempotent against replay** (store dedups on
   `RECORD_HASH`; object outputs dedup on merge and guard appends by tail
   epoch; date outputs upsert on `epoch >=`), so that one ordering rule is
   the entire crash-safety story: a crash replays at most the files whose
   state was never recorded, and every sink absorbs the duplicates. Outputs
   consume the *validated buffer, pre-dedup* — they never ask the store
   which rows were new, which is what eliminates watermark/lookback
   machinery for both store types.
3. **One store interface, two implementations** — filesystem or SQL, chosen
   per deployment. Outputs are lossy projections, identical in behavior
   regardless of store.

There is no steady-state `generate`. Ingest maintains the outputs. The cron
entry is `thistle-db ingest`.

## Core workflow

1. TLE/OMM files are delivered into directories defined in `config.toml`
   (`[[ingest.sources]]` entries: a `path` plus a glob `pattern`). For the
   database store, the schema is created once with `thistle-db init-db` — the
   **only** DDL path; other commands never create tables and fail with a
   clear "run `thistle-db init-db`" error (exit 3) against an uninitialized
   database.
2. `thistle-db ingest` scans those directories, auto-detects each file's
   format, parses and validates the element sets, adds them to the store,
   and updates every configured output — all in one pass. Duplicates are
   silently absorbed — ingest is **idempotent**: running it twice over the
   same files must never create duplicate records or fail.
3. `rebuild`, `verify`, and `export` cover disaster recovery, periodic
   reconciliation, and interchange (see CLI).

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
  rather than delivering a delta. Correctness comes from store-level dedup —
  re-reading a whole updated file is always safe. Efficiency comes from
  **file-state tracking**: each successfully ingested file's path, size,
  mtime, and content hash is recorded (`ingest_files`). On scan, unchanged
  files (size + mtime match) are skipped without being opened; a changed
  file whose content hash still matches just refreshes its state; otherwise
  the file is re-ingested (dedup absorbs the already-seen records). A file
  that fails to parse gets no state recorded, so it is retried next scan.
  `--force` bypasses skipping.

## The canonical GP record (`record.py`)

One flattened GP record per element set. Column vocabulary is CCSDS OMM
keywords; the column set is anchored to Space-Track's published GP class
(minus derived values, plus `RECORD_HASH`), so schema evolution has an
external trigger rather than being a thistle design activity.

| Class | Columns | Rule |
|---|---|---|
| Identity | `RECORD_HASH` (first column) | See Uniqueness |
| Parsed or delivered | `NORAD_CAT_ID`, `OBJECT_ID`, `EPOCH` (ISO-8601, µs), `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`, `MEAN_MOTION_DOT`, `MEAN_MOTION_DDOT`, `EPHEMERIS_TYPE`, `CLASSIFICATION_TYPE`, `ELEMENT_SET_NO`, `REV_AT_EPOCH` | From the OMM when delivered as OMM; parsed from the lines when TLE-only (Space-Track/TLE-text units for the derivatives) |
| Implicit semantics | `MEAN_ELEMENT_THEORY`, `REF_FRAME`, `TIME_SYSTEM`, `CENTER_NAME` | For TLE-only records, materialized from a **per-`EPHEMERIS_TYPE` table** (type 0 → SGP4/TEME/UTC/EARTH; type 4 → SGP4-XP), never hardcoded — the first SGP4-XP TLE that arrives is stored truthfully. python-sgp4 (2.27) parses type-4 lines fine but would silently *propagate* them with classic SGP4 — thistle-db stores and never propagates, but don't hand type-4 records to sgp4's propagator in tests or tooling |
| Metadata | `OBJECT_NAME`, `OBJECT_TYPE`, `RCS_SIZE`, `COUNTRY_CODE`, `LAUNCH_DATE`, `SITE`, `DECAY_DATE` | From OMM; null for TLE-only (except `OBJECT_NAME` from a 3LE name line). Never inferred |
| Provenance | `GP_ID`, `FILE`, `CREATION_DATE`, `ORIGINATOR` | From OMM; null for TLE-only. Never backfillable except by OMM re-delivery |
| Verbatim rendering | `TLE_LINE0`, `TLE_LINE1`, `TLE_LINE2` | Delivered bytes (TLE delivery) or the provider's embedded lines (OMM delivery); null for an OMM-native record with no lines |

Decisions with rationale:

- **Element columns are stored even when redundant with the lines**: uniform
  readers, grep/spreadsheet usability, projection without running sgp4.
  ~600 B/record instead of ~350; accepted.
- **Derived values (`SEMIMAJOR_AXIS`, `PERIOD`, `APOAPSIS`, `PERIAPSIS`) are
  not canonical** — recomputed where needed. For today's type-0 records the
  OMM element fields carry no more precision than the TLE rendering anyway
  (`BSTAR "0.00168730000000"` is `16873-2` expanded).
- **TLE lines stay first-class and byte-verbatim.** TLE → OMM → TLE does not
  round-trip byte-exactly (space- vs zero-padded catalog numbers, sign
  rendering, `00000-0`/`00000+0` bstar forms). Delivered bytes are the audit
  trail and make TLE output a passthrough.

### Uniqueness

`RECORD_HASH` (lowercase hex sha256):

- **Lines present**: hash of `line1 + "\n" + line2` (`record.line_hash`; the
  database's `line_hash` column stores the same digest raw). Textually
  distinct near-duplicates from different providers deliberately coexist.
  Do not change this key to `(norad_cat_id, epoch)` or similar without an
  explicit decision.
- **Line-less** (OMM-native records unrepresentable as TLE): semantic
  identity — a re-issue with changed metadata or a new `GP_ID` enriches the
  existing record rather than coexisting. The frozen `thistle-gp1`
  serialization (`record.semantic_hash`): sha256 over the UTF-8 `"\n"`-join
  of the literal `thistle-gp1`; `NORAD_CAT_ID` as unpadded decimal; the
  stored µs epoch as `YYYY-MM-DDTHH:MM:SS.ffffff`; `MEAN_ELEMENT_THEORY`
  uppercase; then `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`,
  `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`,
  `MEAN_MOTION_DOT`, `MEAN_MOTION_DDOT` as Python `float.hex()` of the
  stored doubles (`-0.0` normalized to `0.0`, empty string for null drag
  terms). Hex floats are exact and reimplementable anywhere; hashing the
  *parsed doubles* means formatting variants of one value hash equal while
  last-ulp parse differences coexist. Changing this serialization changes
  identity → major store-format bump; the tag makes the bump self-announcing.

### Enrichment merge (`record.merge`)

"TLE first, identical-lined OMM later" updates the existing record's null
columns (in the files store this lands on the merge-rewrite path, which a
duplicate delivery forces anyway; in the database store it is the one
sanctioned UPDATE). Precedence:

1. Non-null beats null.
2. Both non-null and unequal (a re-issue: same lines, new `GP_ID`): newest
   `CREATION_DATE` wins field-wise; null `CREATION_DATE` loses to non-null;
   full tie (both equal or both absent) → **last-ingested wins**. Chosen over
   an ingest-timestamp ordinal: matches the date-file tie rule and keeps
   thistle state out of the canonical record. Accepted consequence: replaying
   *genuinely conflicting* redeliveries in a different order can flip such
   fields (replay of identical data cannot).
3. `RECORD_HASH`, `EPOCH`, and `TLE_LINE1/2` never change; `TLE_LINE0` is
   fill-if-null (renames land in `OBJECT_NAME` via rule 2).

Immutability convention: **identity and lines immutable; other fields
enrichable.** Epoch-tie ordering in files: existing records before new.

### Malformed records and quarantine

sgp4's parser is C `strtod`, which accepts the literal text `nan`, `inf` and
`nan(ind)` in a fixed-width field — exactly what a producer emits when it
formats a missing value with `printf` — and corrupt text shifts field
boundaries to the same effect. **Every float can therefore arrive
non-finite.** Cleaning happens at the one boundary where parsed values enter
the record (`record._clean_floats`, mirrored by `model._clean_floats` for
the database rows):

- non-finite in a **nullable** field → `NULL`. Verbatim lines remain the
  lossless source of truth, so a missing value costs nothing.
- non-finite in a **mandatory** element (eccentricity, inclination, RAAN,
  argument of pericenter, mean anomaly) → `MalformedRecordError`, and the
  record is rejected. There is no orbit to store, and such a record would
  displace the object's good elset in date outputs.
- a record with **no NORAD ID** has no home in a per-object store and is
  rejected (quarantined as usual).

No stored float is ever non-finite. Do not relax this by flagging invalid
records instead — a validity predicate would have to thread through every
read path, and missing one yields silent NaN positions downstream.

One caveat for tests: whether `nan` text parses to NaN depends on the column
*and the process*. Line 2's plain-float fields are stable, but line 1's
`bstar`/`ndot`/`nddot` use sgp4's exponent-format reader, which yields `0.0`
instead of NaN once numpy has been imported (numpy changes the CRT parse on
Windows). Prefer a zero mean motion (infinite *derived* values from
arithmetic) and unit-test the cleaning directly.

The **future-epoch guard**: an elset whose epoch runs more than
`[ingest] max_epoch_ahead_days` (default 30; 0 disables) ahead of now is
rejected. TLE epochs carry two-digit years (00-56 → 2000-2056), so a
corrupted epoch parses *cleanly* into a date decades out — and once stored it
becomes its object file's tail epoch (forcing merge-rewrites forever) and
plants a junk date file. Real feeds deliver at most hours to a day ahead;
raise it for predicted-elset workflows. Deliberately one-sided — past epochs
are never bounded, so historical archives ingest untouched.

Rejected records are preserved outside the store. With `[ingest] reject_dir`
set, each is copied to a quarantine directory (`rejects.py`) in its source
file's own format, mirroring the source tree so provenance is visible in the
path and a repaired file is directly re-ingestable:

    /data/spacetrack/daily/2006/20060101.tle   (source)
    rejects/daily/2006/20060101.tle            (rejected records, verbatim)
    rejects/daily/2006/20060101.tle.log        (one reason per record)

Two properties are load-bearing:

- **Live view, not an archive.** Each ingest of a source file overwrites that
  file's artifacts, and a run with no rejects deletes them. Volume is bounded
  by what is *currently* broken, so no retention policy is needed.
- **Bounded per file.** Past `REJECT_MAX_RECORDS` (10k — a fuse, not a tuning
  knob) only a marker with counts and a reason histogram is written; beyond
  that cap a file is not a good feed with bad lines but a wrong file.

A quarantine failure is logged and never propagates: it must not turn a
working ingest into a failed one.

## The store [NOT YET IMPLEMENTED]

One interface, two implementations, chosen in config:

```toml
[storage]
type = "files"                    # or "database"
state = "./thistle-state.sqlite"  # files store only: source-file tracking
flush_records = 1000000           # memory budget per flush
```

Interface: `add(delta)` (idempotent), linear scan (rebuild/export),
per-object read, date-candidate read, per-object and per-date counts
(verify). Outputs are **lossy projections** (TLE has no metadata; the STK
CSV has no provenance), so the store must be full-fidelity in either
implementation — never reconstructible from outputs. Moving between stores
goes through `export` + `ingest`.

### `type = "files"` — GP-CSV object tree

One epoch-ordered CSV file per object (naming via the `object_id` /
`zero_pad` / `filename` templating), one record per line
(`record.STORE_COLUMNS` order), plus the small state SQLite (a cache,
created on first use, safe to delete).

- Append-or-merge per object: all buffered epochs strictly newer than the
  file's tail → append; anything else (late, tie, re-delivery, torn tail,
  enrichment) → merge-rewrite (tmp + atomic rename). Torn CSV tail = drop
  the unparseable last line, healed by the next merge.
- ~500–600 B/record; a 100k-elset worst-case object ≈ 60 MB. If size ever
  matters, per-object zstd with merge-time recompression is the escape hatch
  (never gzip-member appends).
- **Format versioning** (the price of normalization, paid deliberately):
  every file begins with a header row, authoritative for that file's
  columns; readers tolerate unknown columns and treat missing nullable
  columns as null (`record.from_csv_dict` implements this), so additive
  change touches zero existing files. A tree-root `store.toml` records
  format version and store-defining rules. Only additive changes within a
  major version.
- `get-tle YYYYMMDD` reads date outputs when configured (candidates are each
  day's winners — a bounded approximation); exact nearest-to-noon needs the
  database store.

### `type = "database"` — SQL

The two-table shape survives on merits: `tle` (one row per record, verbatim
lines + parsed columns) + `omm_metadata` (sparse sidecar, unique FK — now
also carrying provenance: `GP_ID`, `FILE`, `CREATION_DATE`, `ORIGINATOR`
**[NOT YET IMPLEMENTED: FILE/CREATION_DATE columns]**), with `ingest_files`
in the same database. Enrichment is an UPDATE of the metadata row — the one
sanctioned mutation; `tle` rows stay immutable.

Schema conventions (deliberate, sized for hundreds of millions of rows on
MariaDB — do not regress casually):

- Surrogate keys are **BIGINT** (`INTEGER` on SQLite): `INSERT IGNORE` burns
  auto-increment values on every duplicate skipped, so id consumption tracks
  rows *attempted*.
- All stored datetimes are **naive UTC with microseconds** (`DATETIME(6)` on
  MariaDB); `epoch` must round-trip exactly for the tail guard.
- Element/derived float columns are **single precision** (`Float32`) except
  `mean_motion` (10 significant digits — DOUBLE); the lines are the lossless
  truth, so these are a re-derivable convenience view. Never compare them
  for equality.
- The `tle` **index budget is minimal** (PK, `UNIQUE(epoch, line_hash)`,
  `(norad_cat_id, epoch)`, `(created)`). Epoch-first uniqueness keeps normal
  ingest probing the hot right edge of the index regardless of table size,
  and keeps the index 32 bytes wide instead of 560 (`BINARY(32)`; use
  `HEX(line_hash)` in ad-hoc SQL). Every added secondary index taxes every
  insert; add one only with a query that needs it.
- Dedup is enforced with dialect-aware upserts (`ON CONFLICT DO NOTHING` for
  SQLite/PostgreSQL, `INSERT IGNORE` for MySQL/MariaDB), chunked
  (`CHUNK_SIZE = 5000`); chunk failures fall back to row-by-row inserts so a
  refused record costs only itself. Any new bulk-write path must follow this
  pattern on all three dialects.

This store's reason to exist: exact `nearest_tles_for_date` over all elsets,
ad-hoc SQL, fast metadata queries. Its schema changes via `export` →
`init-db --drop` → `ingest`; there is intentionally **no migration
framework** (see Schema changes below).

## Outputs [NOT YET IMPLEMENTED]

Declarative `[[output.files]]` entries, all maintained incrementally by the
flush engine, identical behavior on either store. Each entry names a `type`
(`object` / `date`), a `format`, a destination `dir`, and a filename scheme.
At least one entry is required (misconfiguration exits 2, not a silent
no-op).

- **`format = "tle"`**: verbatim passthrough of stored lines — the
  compatibility surface for TLE-only consumers (GMAT, legacy pipelines).
  Records with null lines are **always skipped and counted** (one WARNING
  per run): TLE outputs never contain a byte thistle synthesized, keeping
  exporter fidelity out of the correctness surface. Object files: all
  records, epoch order. Date files: latest record per object per epoch date.
- **`format = "omm-csv"`**: header row + the CelesTrak/STK 17-column
  projection (`reader.OMM_CSV_FIELDS` order; `record.to_omm_dict`) —
  verified against live CelesTrak output and STK's documented "OMM-based
  CSV" acceptance. All 17 fields are populated even for TLE-only deliveries,
  so every record is exportable — no subset caveat. `columns = "stk"`
  (default) | `"full"` (all store columns). Acceptance check: a generated
  file loads in STK's SGP4 propagator.
- Filename templating as currently implemented: `extension` overrides the
  default suffix; `filename` arranges `{id}`/`{date}`/`{ext}`/`{format}`/
  `{type}`, validated at **config load** (unknown placeholder, missing
  `{id}`/`{date}`, format specs, path separators, `..` all fail there — a
  `KeyError` while naming the 40,000th file would leave a half-written
  tree). Directories come from `dir` only. `parse_object_name`/`object_glob`
  derive the inverse for verify's orphan scan.
- An `object_id = "alpha5"` output meeting `NORAD_CAT_ID` > 339,999 **falls
  back to the plain integer stem for that object** with one WARNING per run —
  ingest never fails or drops valid data over an output naming choice;
  `parse_object_name`/`object_glob` must accept both stems.
- Date outputs are upserted from the delta ("latest per object that day" can
  only change when a newer-epoch record for that (object, date) arrives);
  replace on `epoch >=`, so newest-ingested wins ties.

Filesystem work runs on the write pool (`writer.WritePool`,
`output.write_workers`): per-file open/close was 44% of a full rebuild —
pure blocked time with the GIL released. The pool only ever receives pure
filesystem work; sessions/queries stay on the calling thread; work is
submitted a chunk of objects at a time so memory stays bounded and each task
owns a distinct path. Rows are read as Core column tuples, never ORM
entities (materializing ORM objects cost ~7.6s of a 38s rebuild).
`scripts/bench_store.py` benches store ingest (files vs SQLite, chrono vs
reverse delivery); a rebuild/output bench is still to be written.
Ingest scans **sort source files chronologically** (by name — delivery
files are date-named) so historical backfills hit the append path, not
merge-rewrites [NOT YET IMPLEMENTED].

## Configuration (`config.py`)

- `config.toml` (path via `-c`, default `./config.toml`; the
  `THISTLE_DB_CONFIG` env var overrides the default — shared with thistle's
  db fallback), parsed into pydantic-settings models: `[database]`,
  `[storage]`, `[ingest]` (`reject_dir`, `max_epoch_ahead_days`,
  `[[ingest.sources]]`), `[output]` with `[[output.files]]`, `[logging]`.
  `[output]` rejects unknown keys (`extra="forbid"`) so stale configs fail
  validation with a pointer at the key instead of being silently ignored.
- Database credentials are **never** stored in `config.toml`. Resolution
  order (highest first):
  1. Env vars `THISTLE_DB_DATABASE__USERNAME` / `THISTLE_DB_DATABASE__PASSWORD`
  2. User secrets file `~/.config/thistle-db.toml`
  3. System secrets file (`database.secrets_file` in config)
- Supported databases: SQLite (default/dev), MariaDB/MySQL (operational,
  `thistle-db[mysql]` extra), PostgreSQL (tested in CI).

## CLI (`cli.py`)

`thistle-db [-c CONFIG] [--log FILE] [--no-progress] <command>`. Global
options:

- `--log FILE` — rotating loguru file sink (10 MB rotation, last 10 kept)
  alongside console logging; the cron-friendly alternative to redirection.
- `--no-progress` — force the progress bar off. The bar (rich, stderr,
  `progress.py`) only appears when stderr is an interactive terminal; log
  lines render above the live bar via the shared console. Command *output*
  (get-tle TLEs, export messages) is stdout and must never route through the
  progress console.

Commands **[NOT YET IMPLEMENTED: rebuild/verify/export as named; ingest does
not yet maintain outputs]**:

- `init` — scaffold `config.toml` and `~/.config/thistle-db.toml`.
- `init-db [--drop] [--yes] [--defer-indexes]` — database store: create the
  schema (run once with an admin account on MariaDB/PostgreSQL); idempotent
  without `--drop`; `--drop` prompts, and fails closed non-interactively
  without `--yes`. `--defer-indexes` (requires `--drop`) creates the schema
  without the read-path indexes for bulk restores — the unique dedup index
  is never deferred, so idempotency holds throughout; a later plain
  `init-db` builds missing indexes (the finalize step). Files store: create
  the state cache.
- `ingest [FILES...] [--force]` — everything: validate, store, outputs.
  Scans stay tolerant (exit 0; failed files retried next scan); named files
  exit 1 on any failure so cron wrappers can detect it.
- `rebuild` — linear store scan re-deriving all outputs (first run, disaster
  recovery, new output entry).
- `verify` — count reconciliation of every output against the **store** (the
  authority): per-object record counts, per-date winner counts. Catches
  damage the tail guard can't see (mid-file truncation, deleted files for
  quiet objects). Reads all output files → periodic cron (e.g. weekly), not
  every run.
- `export OUTPUT [--format tle|csv|omm-xml|omm-kvn] [--force]` — full-fidelity
  CSV is the store-to-store migration and backup form; TLE concat for
  legacy; CCSDS OMM XML/KVN (core keywords + extras in
  `userDefinedParameters`) for standards-adherent interchange, emitting OMM
  version 2.0 by default (STK's documented acceptance; Space-Track emits
  3.0 — add a flag when someone needs it). Refuses to overwrite without
  `--force`. Physical backups of live servers belong to native tools.
- `get-tle TARGET [--days N]` — print TLEs to stdout. `TARGET` is an 8-digit
  date `YYYYMMDD` (nearest TLE per object to 12:00 UTC, ±N days, default 7)
  or an alpha-5-compatible NORAD ID (every TLE for that object, epoch
  order). Exits 1 if nothing matches. Exact on the database store; bounded
  approximation on files.

Privilege tiering: `init-db` = DDL; `ingest` = DML **and output writes** (the
ingest account owns the output tree); read commands open **read-only
connections** (`PRAGMA query_only` / read-only transaction characteristics)
as defense in depth on top of DB grants. **Single-writer ingest is
universal** — output-file guards assume serialized ingest on either store
(`flock` / cron chaining).

## Error handling philosophy

Ingest is tolerant: a malformed record is skipped (and quarantined when
`reject_dir` is set) — one bad record must never abort a file, and one bad
file must never abort a scan. A missing source directory is a warning.

Logging levels: routine no-ops (unchanged files skipped) and individual
rejected records at DEBUG; per-file actions and per-directory summaries at
INFO; data problems at WARNING. Rejections are one WARNING per file giving
the count **over the total attempted** — the ratio separates a few bad lines
from an entirely wrong file. Line-less records skipped by tle outputs are
one WARNING per run with the count. The INFO log of a steady-state cron run
stays short.

## Module map

| Module | Responsibility | Status |
|---|---|---|
| `record.py` | canonical GP record: construction, identity, merge, CSV/STK serialization | implemented |
| `cli.py` | typer CLI, subcommand dispatch, logging setup | legacy commands |
| `config.py` | pydantic-settings config + layered secrets resolution | legacy `[storage] backend` |
| `model.py` | SQLAlchemy models for the database store | needs provenance columns |
| `reader.py` | format detection + file readers (TLE, OMM JSON/CSV/XML) | current |
| `ingest.py` | validation, scanning, bulk insert | to become the delta engine |
| `filestore.py` | TLE-line files backend | to become the GP-CSV store |
| `rejects.py` | quarantine for rejected records | current |
| `generator.py` | steady-state output generation | to be deleted (rebuild/verify remain) |
| `writer.py` | thread pool for output file writes | current |
| `progress.py` | shared rich console (stderr) + no-op-able progress reporter | current |

## Development

- Python ≥ 3.11. Package manager: `uv`. Build: hatchling + hatch-vcs
  (version comes from git tags — never hardcode it).
- Run tests: `uv run pytest tests/thistle_db` from the workspace root. Lint:
  `uv run ruff check`. Types: `uv run pyright`.
- Tests cover SQLite by default; set `THISTLE_DB_TEST_MARIADB=1` /
  `THISTLE_DB_TEST_POSTGRES=1` (requires Docker) for MariaDB/PostgreSQL via
  testcontainers — CI runs all three. Dialect-specific behavior (upserts)
  must be tested against all three.
- Test fixtures live in `tests/thistle_db/data/` — real TLE text and
  Space-Track OMM JSON, truncated to a few hundred records each (realism
  from genuine provider output, not volume). `20250418.txt` keeps 6,001
  records — above `CHUNK_SIZE` — so multi-chunk ingest stays exercised.
  A test needing volume should generate it (see `scripts/bench_store.py`
  for the synthesis pattern).
  `one.json` is the canonical TLE/OMM cross-check fixture.

### Schema changes (no migration framework — deliberate)

There is intentionally **no Alembic or other migration tooling.**

- **Files store**: header-row versioning means additive changes touch zero
  existing files; anything non-additive is a major store-format bump with an
  explicit upgrade path.
- **Database store**: every record reconstitutes losslessly from a
  full-fidelity `export`, so schema changes ship as breaking releases
  migrated by `export` → `init-db --drop --defer-indexes` → `ingest` →
  `init-db`. Restore performance on MariaDB is index-bound, not parse-bound:
  modern MariaDB has no InnoDB change buffer, so every secondary-index
  insert touches a real page — defer the read-path indexes and size the
  buffer pool to hold the `tle` indexes. The unique dedup index stays cheap
  during chronological restores because a snapshot's epochs land in one hot
  region of the epoch-first index — a reason its column order must not
  change. Purely additive changes may ship as hand-written `ALTER`
  statements for all three dialects in the release notes.

## Build order (remaining)

Per `PLAN-gp-store.md` (steps 1–3 done: spec adopted, spikes run, record
module implemented):

4. Delta engine + GP-CSV files store + tle/omm-csv outputs (the serverless
   deployment, end to end).
5. Database store behind the same interface; exact `get-tle`.
6. `rebuild`, `verify`, `export`; retire the legacy code paths and remove
   the "legacy" rows from the module map above.
