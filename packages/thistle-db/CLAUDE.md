# CLAUDE.md

This file is the **target design spec** for thistle-db. Where the code and this
document disagree, this document wins — update the code to match, not the other
way around.

**Architecture status (2026-11):** the GP-record, single-SQL-store
architecture is **implemented** — adopted from `PLAN-gp-store.md`
(2026-09), amended 2026-11-09 (files store dropped; DB-resident `rejects`
and `ingest_runs` tables; checksum validation; see `PLAN.md` for the full
design + rationale), and built the same day. The legacy dual-backend
code (`filestore.py`, `generator.py`, `[storage]`, the filesystem
quarantine, `epoch_date_state` watermarks) is deleted; see git history
for its documentation.

## What thistle-db is

thistle-db is a database manager for orbital element sets. It ingests TLE
(Two-Line Element) and OMM (Orbit Mean-Elements Message) files from configured
directories into a SQL database — the canonical store — and maintains
organized output files (TLE by object and by date; STK-compatible OMM CSV)
directly from ingest.

## Shape of the system

```
sources ──► readers ──► validation ──► delta buffer (GP records, by object)
(TLE/OMM)                   │                │  flush at end-of-scan / flush_records
                            │ rejected       ▼
                            ▼           1. store.add(delta)         dedup by RECORD_HASH
                       rejects table    2. output.apply(delta) ×N   merge-dedup / tail guard / date upsert
                       (reason-coded,   3. commit ingest_files state
                        deduped)
                                        run accounting ──► ingest_runs (one row per run)
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
   machinery.
3. **One store — a SQL database.** SQLite for dev and small/serverless
   deployments; MariaDB/MySQL or PostgreSQL operationally. The former
   "one interface, two implementations" design (GP-CSV object tree as an
   alternative store) was **dropped 2026-11-09**: SQLite already covers the
   serverless case, the dual-store interface taxed every feature with a
   lowest-common-denominator spec (exact vs approximate `get-tle`,
   merge-as-UPDATE vs merge-rewrite, no natural home for the rejects and
   runs tables), and the files store's bulk-ingest throughput edge only
   mattered for rare full-catalog restores, which the deferred-index flow
   already serves. Outputs remain files; the *store* is the database. The
   delta engine calls the store module directly — no store abstraction
   layer is maintained.

There is no steady-state `generate`. Ingest maintains the outputs. The cron
entry is `thistle-db ingest`.

## Core workflow

1. TLE/OMM files are delivered into directories defined in `config.toml`
   (`[[ingest.sources]]` entries: a `path` plus a glob `pattern`). The schema
   is created once with `thistle-db init-db` — the **only** DDL path; other
   commands never create tables and fail with a clear "run `thistle-db
   init-db`" error (exit 3) against an uninitialized database.
2. `thistle-db ingest` scans those directories, auto-detects each file's
   format, parses and validates the element sets, adds them to the store,
   and updates every configured output — all in one pass. Duplicates are
   silently absorbed — ingest is **idempotent**: running it twice over the
   same files must never create duplicate records or fail. Records that
   fail validation land in the `rejects` table; nothing is silently lost.
3. `rebuild`, `verify`, `export`, and `status` cover disaster recovery,
   periodic reconciliation, interchange, and observability (see CLI).

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
  database's `record_hash` column stores the same digest raw). Textually
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
  One carve-out: a line-less record delivered **with a well-formed
  `RECORD_HASH` field** (thistle's own export) keeps that delivered
  identity, so export → restore round-trips exactly even though
  single-precision element columns re-serialize with fewer bits than the
  doubles the original hash saw. Line-present records never take this
  path — their identity is always the line text.

### Enrichment merge (`record.merge`)

"TLE first, identical-lined OMM later" updates the existing record's null
columns — in the database this is an UPDATE of the metadata row, the one
sanctioned mutation; `tle` rows stay immutable. Precedence:

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
enrichable.**

### Validation and the rejects table

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
- a record with **no NORAD ID** has no home in a per-object output and is
  rejected.
- **checksum mismatch** → rejected (`verify_checksums = true`, the default).
  Each full-length TLE line carries a mod-10 checksum in column 69 that sgp4
  never verifies. A single flipped digit changes an element value *and*
  breaks the checksum — for a record whose elements still parse finite, the
  checksum is the only corruption signal validation has. Lenient where the
  digit is absent: lines shorter than 69 characters (some tools emit trimmed
  lines; historical archives contain them) skip the check rather than
  rejecting whole archives.
- the **future-epoch guard**: an elset whose epoch runs more than
  `[ingest] max_epoch_ahead_days` (default 30; 0 disables) ahead of now is
  rejected. TLE epochs carry two-digit years (00-56 → 2000-2056), so a
  corrupted epoch parses *cleanly* into a date decades out — and once stored
  it becomes its object file's tail epoch (forcing merge-rewrites forever)
  and plants a junk date file. Real feeds deliver at most hours to a day
  ahead; raise it for predicted-elset workflows. Deliberately one-sided —
  past epochs are never bounded, so historical archives ingest untouched.

No stored float is ever non-finite. **Do not relax this by flagging invalid
records in `tle` instead** — a validity predicate would have to thread
through every read path, and missing one yields silent NaN positions
downstream. The sanctioned mechanism for retaining invalid input is the
`rejects` table below: validity is **table membership**, never a column.
"All input rows" means `tle` ∪ `rejects`; a `UNION ALL` view may present the
flag-style picture for ad-hoc analysis without weakening the store.

One caveat for tests: whether `nan` text parses to NaN depends on the column
*and the process*. Line 2's plain-float fields are stable, but line 1's
`bstar`/`ndot`/`nddot` use sgp4's exponent-format reader, which yields `0.0`
instead of NaN once numpy has been imported (numpy changes the CRT parse on
Windows). Prefer a zero mean motion (infinite *derived* values from
arithmetic) and unit-test the cleaning directly.

#### The `rejects` table

Rejected records are **retained in the database**, replacing the old
filesystem quarantine (decided 2026-11-09 — once the store is always a
database, rejects become queryable rows instead of files only visible on
the ingest host). One row per *distinct* rejected payload:

- `record_hash` — sha256 of the verbatim payload (`line1\nline2` for TLE;
  the raw record JSON for OMM), raw `BINARY(32)`, **unique** — the dedup
  key, same convention as `tle.record_hash`.
- `format` (`tle` | `omm`) and `payload` — the verbatim rejected input,
  capped (a few KB) with a `truncated` flag; garbage "lines" from
  mis-encoded files can be arbitrarily long.
- `reason_code` — closed vocabulary: `unparseable`, `malformed_element`,
  `future_epoch`, `no_norad_id`, `checksum`, `db_refused` — plus a
  free-text `reason` detail. Latest rejection wins both.
- `source_path`, `first_seen`, `last_seen`, `times_seen` — provenance and
  recurrence; re-rejection of a known payload is an upsert bumping
  `last_seen`/`times_seen`, so the table is bounded by *distinct* broken
  records, not by time. Append-only otherwise: a run with no rejects
  deletes nothing (unlike the old live-view quarantine — the table is an
  archive; `resolved_at` carries the lifecycle instead).
- `norad_cat_id`, `epoch` — nullable convenience columns populated when
  parseable, so "which objects have a bad feed" is one query.
- `resolved_at` — stamped by `verify` when the same `record_hash` exists in
  `tle` (the payload later ingested cleanly).

Index budget: PK, `UNIQUE(record_hash)`, `(last_seen)` for status trends.
Dialect-aware upserts, same pattern as every other write path.

Two behaviors the table enables, both cheap:

- **Future-epoch re-admission**: each ingest run re-validates unresolved
  `future_epoch` rows whose epoch has come inside the horizon and promotes
  them into the normal delta (the file-state tracker would otherwise never
  re-read their source file). The table is small, a scan is fine.
- **Per-file fuse preserved**: past `REJECT_MAX_RECORDS` per source file per
  run (10k — a fuse, not a tuning knob; a file beyond it is not a good feed
  with bad lines but a wrong file, and even deduped rows would be unbounded
  garbage), individual payloads are not stored; the run's counters and a
  WARNING with a reason histogram record the event. Inspect the source file
  itself.

A reject-write failure is logged and never propagates: it must not turn a
working ingest into a failed one. One bad record never aborts a file; one
bad file never aborts a scan.

## The store

A SQL database — SQLite (default/dev/serverless), MariaDB/MySQL
(operational), PostgreSQL (tested in CI). The two-table element shape
survives on merits: `tle` (one row per record — `norad_cat_id` NOT NULL,
nullable verbatim `line0/line1/line2`, parsed columns,
`mean_element_theory`) + `omm_metadata` (sparse sidecar, unique FK,
carrying metadata, provenance `gp_id`/`file_id`/`creation_date`/
`originator`, and nullable `ref_frame`/`time_system`/`center_name`
overrides — null means implicit per ephemeris type), with `ingest_files`,
`rejects`, and `ingest_runs` in the same database — one backup artifact,
one place every question is answerable in SQL. Enrichment is an UPDATE of
the metadata row — the one sanctioned mutation; `tle` rows stay immutable.

Outputs are **lossy projections** (TLE has no metadata; the STK CSV has no
provenance), so the store must be full-fidelity — never reconstructible
from outputs.

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
  for equality. `Float32` casts its projection to DOUBLE
  (`column_expression`): MariaDB's text protocol renders FLOAT results with
  only 6 significant digits, which does not round-trip a float32 (needs 9)
  — without the cast, read-back returns a *different* float32, breaking
  rebuild byte parity and line-less identity. Any raw-SQL read of these
  columns on MariaDB must `CAST(col AS DOUBLE)` for the same reason.
- The `tle` **index budget is minimal and exhaustive** — 3 B-trees: PK,
  `UNIQUE(epoch, record_hash)`, `(norad_cat_id, epoch)`. Epoch-first
  uniqueness keeps normal ingest probing the hot right edge of the index
  regardless of table size, and keeps the index 32 bytes wide instead of
  560 (`BINARY(32)`; use `HEX(record_hash)` in ad-hoc SQL). The legacy
  `(created)` index is gone with `generate`'s lookback. Every added
  secondary index taxes every insert; add one only with a query that
  needs it.
- Dedup is enforced with dialect-aware upserts (`ON CONFLICT DO NOTHING` for
  SQLite/PostgreSQL, `INSERT IGNORE` for MySQL/MariaDB), chunked
  (`CHUNK_SIZE = 5000`); chunk failures fall back to row-by-row inserts so a
  refused record costs only itself. Any new bulk-write path must follow this
  pattern on all three dialects.

Exact `nearest_tles_for_date` over all elsets, ad-hoc SQL, and fast metadata
queries come standard. Schema changes go via `export` → `init-db --drop` →
`ingest`; there is intentionally **no migration framework** (see Schema
changes below).

## Outputs

Declarative `[[output.files]]` entries, all maintained incrementally by the
flush engine. Each entry names a `type` (`object` / `date`), a `format`, a
destination `dir`, and a filename scheme. At least one entry is required
(misconfiguration exits 2, not a silent no-op).

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
  file loads in STK's SGP4 propagator. CSV rows render at **stored
  precision** (single-precision element columns rounded before
  formatting), so an incremental write and a later `rebuild` — which
  reads the rounded columns back — produce byte-identical files.
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
- Output files are projections, always rebuildable from the store
  (`rebuild`); a damaged output is an inconvenience, never data loss. The
  incremental machinery (tail-epoch append guard, merge-dedup rewrite,
  atomic tmp+rename) is an efficiency layer on top of that guarantee.

Filesystem work runs on the write pool (`writer.WritePool`,
`output.write_workers`): per-file open/close was 44% of a full rebuild —
pure blocked time with the GIL released. The pool only ever receives pure
filesystem work; sessions/queries stay on the calling thread; work is
submitted a chunk of objects at a time so memory stays bounded and each task
owns a distinct path. `rebuild` streams rows with `yield_per` (bounded
memory); switching its reads from ORM entities to Core column tuples is
the known perf lever if full-rebuild time ever matters at catalog scale
(the legacy generator measured ~7.6s of a 38s rebuild in ORM
materialization). Ingest scans **sort source files chronologically** (by
name — delivery files are date-named) so historical backfills hit the
append path, not merge-rewrites.

## Observability

Runs are **observable as data**, not just log lines:

- **`ingest_runs` table**: one row per run — command, started/finished,
  status (`running` / `succeeded` / `failed`), counters (files scanned /
  ingested / skipped / refreshed / failed; records seen / inserted /
  rejected), and an error summary on failure. Written at run start, updated
  at completion; a row stuck in `running` is itself a signal (crashed run).
  Index budget: PK only (insertion-ordered reads suffice).
- **Structured reports**: `ingest_sources` and friends return a report
  object carrying those counters, not a bare int — the CLI prints it, the
  run row stores it, and any wrapper (cron mail, orchestrator) consumes the
  same numbers.
- **`thistle-db status`**: reads `ingest_runs` and `rejects` — last run
  outcome and duration, reject trends by reason, unresolved reject counts.
  The cron-and-terminal dashboard.

**Orchestration stays out of the core.** No Dagster (or other orchestrator)
imports in thistle-db: the pipeline is three linear nodes (fetch → ingest →
verify) with its own incrementality, and **single-writer ingest is a
universal invariant** — an orchestrator that can launch a backfill beside a
scheduled run violates it unless explicitly serialized. If orchestration is
ever adopted, it lives in a separate package that shells into / imports the
library API and reads `ingest_runs`/`rejects` for metadata; the core never
knows it exists. Until then: cron / Task Scheduler + `status` + exit codes.

## Configuration (`config.py`)

- `config.toml` (path via `-c`, default `./config.toml`; the
  `THISTLE_DB_CONFIG` env var overrides the default — shared with thistle's
  db fallback), parsed into pydantic-settings models: `[database]`,
  `[ingest]` (`max_epoch_ahead_days`, `verify_checksums`, `flush_records`,
  `[[ingest.sources]]`), `[output]` with `[[output.files]]`, `[logging]`.
  `[output]` rejects unknown keys (`extra="forbid"`) so stale configs fail
  validation with a pointer at the key instead of being silently ignored.
- **Gone with the files store**: the `[storage]` section (`backend`,
  `state`, `flush_records`) and `[ingest] reject_dir`. `flush_records`
  (the delta-buffer memory budget, default 1,000,000) lives in
  `[ingest]`; rejects live in the database. `[ingest]` and `[output]`
  both reject unknown keys, so a stale config fails loudly at load.
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

Commands:

- `init` — scaffold `config.toml` and `~/.config/thistle-db.toml`.
- `init-db [--drop] [--yes] [--defer-indexes]` — create the schema (run once
  with an admin account on MariaDB/PostgreSQL); idempotent without `--drop`;
  `--drop` prompts, and fails closed non-interactively without `--yes`.
  `--defer-indexes` (requires `--drop`) creates the schema without the
  read-path indexes for bulk restores — the unique dedup index is never
  deferred, so idempotency holds throughout; a later plain `init-db` builds
  missing indexes (the finalize step).
- `ingest [FILES...] [--force]` — everything: validate, store, outputs, run
  accounting. Scans stay tolerant (exit 0; failed files retried next scan);
  named files exit 1 on any failure so cron wrappers can detect it.
- `rebuild` — linear store scan re-deriving all outputs (first run, disaster
  recovery, new output entry).
- `verify` — count reconciliation of every output against the **store** (the
  authority): per-object record counts, per-date winner counts. Also stamps
  `rejects.resolved_at` for payloads now present in `tle`. Catches damage
  the tail guard can't see (mid-file truncation, deleted files for quiet
  objects). Reads all output files → periodic cron (e.g. weekly), not every
  run.
- `export OUTPUT [--format tle|csv|omm-xml|omm-kvn] [--force]` — full-fidelity
  CSV is the backup/interchange form; TLE concat for legacy; CCSDS OMM
  XML/KVN (core keywords + extras in `userDefinedParameters`) for
  standards-adherent interchange, emitting OMM version 2.0 by default
  (STK's documented acceptance; Space-Track emits 3.0 — add a flag when
  someone needs it). Refuses to overwrite without `--force`. Physical
  backups of live servers belong to native tools.
- `get-tle TARGET [--days N]` — print TLEs to stdout. `TARGET` is an 8-digit
  date `YYYYMMDD` (nearest TLE per object to 12:00 UTC, ±N days, default 7)
  or an alpha-5-compatible NORAD ID (every TLE for that object, epoch
  order). Exits 1 if nothing matches. Always exact — it queries the store.
- `status` — last runs, reject trends, unresolved reject counts (see
  Observability).
- `rejects [--export DIR]` — summarize the rejects table; `--export` writes
  unresolved payloads back out as re-ingestable files (repair workflow:
  export, fix, ingest).

Privilege tiering: `init-db` = DDL; `ingest` = DML **and output writes** (the
ingest account owns the output tree); read commands open **read-only
connections** (`PRAGMA query_only` / read-only transaction characteristics)
as defense in depth on top of DB grants. **Single-writer ingest is
universal** — output-file guards assume serialized ingest (`flock` / cron
chaining).

## Error handling philosophy

Ingest is tolerant: a malformed record is skipped and recorded in the
`rejects` table — one bad record must never abort a file, and one bad file
must never abort a scan. A missing source directory is a warning.

Logging levels: routine no-ops (unchanged files skipped) and individual
rejected records at DEBUG; per-file actions and per-directory summaries at
INFO; data problems at WARNING. Rejections are one WARNING per file giving
the count **over the total attempted** — the ratio separates a few bad lines
from an entirely wrong file. Line-less records skipped by tle outputs are
one WARNING per run with the count. The INFO log of a steady-state cron run
stays short.

## Module map

| Module | Responsibility |
|---|---|
| `record.py` | canonical GP record: construction, identity (incl. checksum helpers, exact epochs), merge, CSV/STK serialization |
| `model.py` | SQLAlchemy models (schema v2) + GPRecord↔row conversion, derived values |
| `config.py` | pydantic-settings config + layered secrets resolution, output naming/templating |
| `reader.py` | format detection + file readers (TLE, OMM JSON/CSV/XML) |
| `ingest.py` | the delta engine: validation, scanning, buffer/flush, store sink, re-admission |
| `outputs.py` | output application from the delta (tail guard, merge, date upsert) + rebuild |
| `rejects.py` | DB-backed reject sink (fuse, payload cap, upserts) + reject queries/export |
| `runs.py` | run accounting (`ingest_runs`, `IngestReport`, status queries) |
| `verify.py` | output↔store reconciliation, orphan scan, reject resolution stamping |
| `export.py` | full-fidelity CSV / TLE / CCSDS OMM XML+KVN export |
| `lock.py` | single-writer advisory lockfile |
| `api.py` | init_db (the only DDL path), sessions, query API, `get_tles` |
| `cli.py` | typer CLI, subcommand dispatch, logging setup |
| `writer.py` | thread pool for output file writes |
| `progress.py` | shared rich console (stderr) + no-op-able progress reporter |

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
  A test needing volume should generate it, never commit it.
  `one.json` is the canonical TLE/OMM cross-check fixture.

### Schema changes (no migration framework — deliberate)

There is intentionally **no Alembic or other migration tooling.** Every
record reconstitutes losslessly from a full-fidelity `export`, so schema
changes ship as breaking releases migrated by `export` → `init-db --drop
--defer-indexes` → `ingest` → `init-db`. Restore performance on MariaDB is
index-bound, not parse-bound: modern MariaDB has no InnoDB change buffer, so
every secondary-index insert touches a real page — defer the read-path
indexes and size the buffer pool to hold the `tle` indexes. The unique dedup
index stays cheap during chronological restores because a snapshot's epochs
land in one hot region of the epoch-first index — a reason its column order
must not change. Purely additive changes may ship as hand-written `ALTER`
statements for all three dialects in the release notes.

## Build order

Complete. `PLAN-gp-store.md` steps 1–3 (spec, spikes, `record.py`) landed
2026-09; `PLAN.md` milestones M4–M5 (schema v2, delta engine, outputs,
rejects/runs/checksums, rebuild/verify/export/status/rejects CLI, legacy
deletion) landed 2026-11-09. `PLAN.md` remains the detailed design
rationale for the implemented system.
