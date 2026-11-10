# thistle-db — comprehensive design & implementation plan

Status: **drafted and implemented 2026-11-09** (milestones M4–M5 in §15
are complete; CLAUDE.md is synced). Retained as the detailed design
rationale. This was written as the forward-looking plan,
incorporating the GP-record architecture (adopted 2026-09-02,
`PLAN-gp-store.md` — retained as rationale) and the 2026-11-09 amendments
(single SQL store; DB-resident rejects; checksum validation; run
accounting). `CLAUDE.md` is the normative spec; on adoption of this plan,
sync it with the decisions new here (notably: `line_hash` → `record_hash`
column, `norad_cat_id` NOT NULL, the `(created)` index dropped, implicit
semantics placement).

**Clean slate — backwards compatibility is out of scope.** No migration
tooling, no deprecation phases, no config shims. An existing deployment
adopts this by re-ingesting its source files, or by `export` from the old
code and `ingest` into the new. Legacy code paths are deleted, not
deprecated.

---

## 1. Purpose and scope

thistle-db ingests TLE (Two-Line Element) and OMM (Orbit Mean-Elements
Message) files from configured directories into a SQL database — the
canonical, full-fidelity store — and maintains organized output files
(TLE by object and by date; STK-compatible OMM CSV) directly from ingest.
It is operated by cron: the steady-state entry point is `thistle-db
ingest`, run on a schedule, single-writer.

Non-goals:

- **No propagation.** thistle-db stores and projects element sets; orbit
  propagation lives in `thistle`.
- **No orchestrator in the core.** No Dagster/Prefect/Airflow imports;
  see §11.
- **No migration framework.** Schema changes ship as breaking releases
  migrated by export → re-ingest; see §12.
- **No data synthesis.** TLE outputs are verbatim passthrough; metadata is
  never inferred; provenance is never backfilled except by OMM re-delivery.

Scale targets: hundreds of millions of rows on MariaDB; tens of millions
comfortably on SQLite (dev and small/serverless deployments).

## 2. Shape of the system

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
   buffers them grouped by object, and at flush time hands the same delta
   to every sink, in a fixed order: store, then outputs, then source-file
   state. **Every sink is idempotent against replay**, so that one ordering
   rule is the entire crash-safety story: a crash replays at most the files
   whose state was never recorded, and every sink absorbs the duplicates.
   Outputs consume the *validated buffer, pre-dedup* — they never ask the
   store which rows were new, which is what eliminates watermark/lookback
   machinery.
3. **One store — a SQL database.** SQLite (default, dev, serverless),
   MariaDB/MySQL (operational), PostgreSQL (tested in CI). There is no
   store abstraction layer and no alternative backend; the delta engine
   calls the store module directly. Outputs remain files; the *store* is
   the database.

There is no steady-state `generate` command. Ingest maintains the outputs.

## 3. The canonical GP record (`record.py` — implemented)

One flattened record per element set. Column vocabulary is CCSDS OMM
keywords; the column set is anchored to Space-Track's published GP class
(minus derived values, plus `RECORD_HASH`), so schema evolution has an
external trigger rather than being a thistle design activity.

| Class | Columns | Rule |
|---|---|---|
| Identity | `RECORD_HASH` | see §3.1 |
| Parsed or delivered | `NORAD_CAT_ID`, `OBJECT_ID`, `EPOCH` (µs), `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`, `MEAN_MOTION_DOT`, `MEAN_MOTION_DDOT`, `EPHEMERIS_TYPE`, `CLASSIFICATION_TYPE`, `ELEMENT_SET_NO`, `REV_AT_EPOCH` | From the OMM when delivered as OMM; parsed from the lines when TLE-only (Space-Track/TLE-text units for the drag derivatives) |
| Implicit semantics | `MEAN_ELEMENT_THEORY`, `REF_FRAME`, `TIME_SYSTEM`, `CENTER_NAME` | For TLE-only records, materialized from a **per-`EPHEMERIS_TYPE` table** (type 0 → SGP4/TEME/UTC/EARTH; type 4 → SGP4-XP), never hardcoded. python-sgp4 (2.27) parses type-4 lines but would silently propagate them with classic SGP4 — thistle-db stores and never propagates |
| Metadata | `OBJECT_NAME`, `OBJECT_TYPE`, `RCS_SIZE`, `COUNTRY_CODE`, `LAUNCH_DATE`, `SITE`, `DECAY_DATE` | From OMM; null for TLE-only (except `OBJECT_NAME` from a 3LE name line). Never inferred |
| Provenance | `GP_ID`, `FILE`, `CREATION_DATE`, `ORIGINATOR` | From OMM; null for TLE-only. Never backfillable except by OMM re-delivery |
| Verbatim rendering | `TLE_LINE0`, `TLE_LINE1`, `TLE_LINE2` | Delivered bytes (TLE delivery) or the provider's embedded lines (OMM delivery); null for an OMM-native record with no lines |

Decisions with rationale:

- **Element columns are stored even when redundant with the lines**:
  uniform readers, grep/spreadsheet usability, projection without running
  sgp4. ~600 B/record instead of ~350; accepted.
- **Derived values (`SEMIMAJOR_AXIS`, `PERIOD`, `APOAPSIS`, `PERIAPSIS`)
  are not canonical** — excluded from identity and export, recomputed on
  ingest. They *are* kept as nullable convenience columns in the database
  (§5.1) because "period < 200" in ad-hoc SQL is worth 16 B/row.
- **TLE lines stay first-class and byte-verbatim.** TLE → OMM → TLE does
  not round-trip byte-exactly (space- vs zero-padded catalog numbers, sign
  rendering, `00000-0`/`00000+0` bstar forms). Delivered bytes are the
  audit trail and make TLE output a passthrough.

### 3.1 Identity — `RECORD_HASH`

Lowercase hex sha256 in the record; raw 32 bytes in the database.

- **Lines present**: hash of `line1 + "\n" + line2` (`record.line_hash`).
  Verbatim text is the identity of a delivery; textually distinct
  near-duplicates from different providers deliberately coexist. Do not
  change this key to `(norad_cat_id, epoch)` or similar without an
  explicit decision.
- **Line-less** (OMM-native records unrepresentable as TLE): semantic
  identity — a re-issue with changed metadata or a new `GP_ID` enriches
  the existing record rather than coexisting. The frozen `thistle-gp1`
  serialization (`record.semantic_hash`): sha256 over the UTF-8
  `"\n"`-join of the literal `thistle-gp1`; `NORAD_CAT_ID` as unpadded
  decimal; the stored µs epoch as `YYYY-MM-DDTHH:MM:SS.ffffff`;
  `MEAN_ELEMENT_THEORY` uppercase; then `MEAN_MOTION`, `ECCENTRICITY`,
  `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`,
  `BSTAR`, `MEAN_MOTION_DOT`, `MEAN_MOTION_DDOT` as Python `float.hex()`
  of the stored doubles (`-0.0` normalized to `0.0`, empty string for null
  drag terms). Hex floats are exact and reimplementable anywhere; hashing
  the *parsed doubles* means formatting variants of one value hash equal
  while last-ulp parse differences coexist. Changing this serialization
  changes identity → major version bump; the tag makes it self-announcing.

### 3.2 Enrichment merge (`record.merge`)

"TLE first, identical-lined OMM later" fills the existing record's null
columns. In the database this is an UPDATE of the metadata sidecar — the
one sanctioned mutation; element rows stay immutable. Precedence:

1. Non-null beats null.
2. Both non-null and unequal (a re-issue: same lines, new `GP_ID`): newest
   `CREATION_DATE` wins field-wise; null `CREATION_DATE` loses to
   non-null; full tie → **last-ingested wins**. Chosen over an
   ingest-timestamp ordinal: matches the date-file tie rule and keeps
   thistle state out of the canonical record. Accepted consequence:
   replaying *genuinely conflicting* redeliveries in a different order can
   flip such fields (replay of identical data cannot).
3. `RECORD_HASH`, `EPOCH`, and `TLE_LINE1/2` never change; `TLE_LINE0` is
   fill-if-null (renames land in `OBJECT_NAME` via rule 2).

Convention: **identity and lines immutable; other fields enrichable.**

## 4. Validation and the rejects table

Validation is the gate between "input row" and "store row". Everything
that fails it is **retained** in the `rejects` table (§5.3) — nothing is
silently lost; validity is **table membership**, never a flag column.
"All input rows" = `tle` ∪ `rejects`; an optional `UNION ALL` view
presents the flag-style picture for ad-hoc analysis without weakening the
store.

**Do not add a validity flag to `tle`.** A validity predicate would have
to thread through every read path — outputs, `get-tle`, exports, ad-hoc
SQL — and missing one yields silent NaN positions downstream. Invalid
input also cannot populate the schema honestly (no finite eccentricity,
no epoch, arbitrary garbage text) without making every column nullable
for all rows.

Rules, applied per record at the one boundary where parsed values enter
the system (`record._clean_floats` and the `from_tle`/`from_omm`
constructors):

| Condition | Outcome | `reason_code` |
|---|---|---|
| Text sgp4 cannot parse / OMM record with no epoch | reject | `unparseable` |
| Non-finite **mandatory** element (eccentricity, inclination, RAAN, arg of pericenter, mean anomaly) | reject — there is no orbit to store, and the row would displace good elsets in date outputs | `malformed_element` |
| Non-finite **nullable** field (bstar, derivatives, derived values) | store with `NULL` — the verbatim lines remain the lossless truth | — |
| No NORAD catalog number | reject — no home in per-object outputs | `no_norad_id` |
| TLE checksum mismatch (either line) | reject | `checksum` |
| Epoch more than `max_epoch_ahead_days` ahead of now (default 30; 0 disables) | reject | `future_epoch` |
| Database refuses the row (driver/constraint error on the row-by-row retry path) | reject | `db_refused` |

**No stored float is ever non-finite.** sgp4's parser is C `strtod`, which
accepts literal `nan`/`inf`/`nan(ind)` in a fixed-width field — exactly
what a producer emits when it `printf`s a missing value — and corrupt text
shifts field boundaries to the same effect, so every float can arrive
non-finite.

**Checksums** (`verify_checksums = true`, default): each full-length TLE
line carries a mod-10 checksum in column 69 (digits summed, `-` counts 1,
everything else 0) that sgp4 never verifies. A single flipped digit
changes an element value *and* breaks the checksum — for a record whose
elements still parse finite, the checksum is the only corruption signal
validation has. Lenient where the digit is absent: lines shorter than 69
characters skip the check (some tools emit trimmed lines; historical
archives contain them).

**Future-epoch guard**: TLE epochs carry two-digit years (00-56 → 20xx),
so a corrupted epoch parses *cleanly* into a date decades out — and once
stored it becomes its object file's tail epoch (forcing merge-rewrites
forever) and plants a junk date file. Real feeds deliver at most hours to
a day ahead. Deliberately one-sided: past epochs are never bounded, so
historical archives ingest untouched.

**Future-epoch re-admission**: each ingest run begins by re-validating
unresolved `future_epoch` rejects whose epoch has come inside the horizon,
promoting survivors into the normal delta and stamping `resolved_at` after
the flush succeeds. Without this, a rejected elset whose source file never
changes would be lost forever (file-state tracking skips the file). The
table is small — a scan is fine.

**Per-file fuse**: past `REJECT_MAX_RECORDS` per source file per run
(10,000 — a constant, not config; a fuse, not a tuning knob), individual
payloads are not stored: a file beyond it is not a good feed with bad
lines but a wrong file (mis-encoded, still gzipped, truncated), and even
deduped rows would be unbounded garbage. The run's counters and one
WARNING with a reason histogram record the event; inspect the source file
itself.

**Containment**: a reject-write failure is logged and never propagates —
it must not turn a working ingest into a failed one. One bad record never
aborts a file; one bad file never aborts a scan.

## 5. Database schema

Conventions (deliberate, sized for hundreds of millions of rows on
MariaDB — do not regress casually):

- Surrogate keys are **BIGINT** (`INTEGER` on SQLite for rowid aliasing):
  `INSERT IGNORE` burns auto-increment values on every duplicate skipped,
  so id consumption tracks rows *attempted*, not stored.
- All stored datetimes are **naive UTC with microseconds** (`DATETIME(6)`
  on MariaDB — plain DATETIME truncates to whole seconds and breaks epoch
  equality).
- Hashes are raw **`BINARY(32)`** (LargeBinary elsewhere): a BLOB cannot
  back a unique index on MariaDB without a prefix length, and 32 bytes is
  half the hex size. Use `HEX(...)` in ad-hoc SQL.
- Element/derived float columns are **single precision** (`FLOAT(24)`)
  except `mean_motion` (10 significant digits in TLE text — DOUBLE); the
  lines are the lossless truth, so these are a re-derivable convenience
  view. Never compare them for equality. Projections cast to DOUBLE
  (`Float32.column_expression`): MariaDB's text protocol renders FLOAT
  results with 6 significant digits, which does not round-trip float32 —
  ad-hoc SQL on MariaDB needs `CAST(col AS DOUBLE)` likewise.
- Closed vocabularies (`reason_code`, run `status`) are **VARCHAR with the
  vocabulary enforced in code**, not SQL ENUMs (dialect-portable).
- Dedup is enforced with **dialect-aware upserts** (`ON CONFLICT DO
  NOTHING` for SQLite/PostgreSQL, `INSERT IGNORE` for MySQL/MariaDB),
  chunked (`CHUNK_SIZE = 5000`); a refused chunk is retried row by row so
  a bad record costs only itself (and is quarantined as `db_refused`).
  Any new bulk-write path must follow this pattern on all three dialects.
- **Every added secondary index taxes every insert**; add one only with a
  query that needs it. Index budgets below are exhaustive.

### 5.1 `tle` — one row per element set

The name is grandfathered (rows can be line-less OMM-native records, but
`tle` reads naturally in ad-hoc SQL and a rename buys nothing functional).
Rows are **immutable** once inserted.

| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT PK | |
| `created` | DATETIME(6) | ingest time, one value per batch |
| `record_hash` | BINARY(32) | raw digest of §3.1 (line hash or semantic hash) |
| `line0`, `line1`, `line2` | VARCHAR(70) NULL | verbatim delivered bytes; all three null for line-less OMM-native records |
| `norad_cat_id` | INT **NOT NULL** | no-ID records are rejected (§4), so the column is non-nullable — queries and outputs never branch on it |
| `object_id` | VARCHAR(15) NULL | international designator, century-expanded; non-numeric designators (`TBA`) stored verbatim |
| `classification` | CHAR(1) | |
| `epoch` | DATETIME(6) NOT NULL | |
| `mean_motion` | DOUBLE | rev/day |
| `eccentricity`, `inclination`, `ra_of_asc_node`, `arg_of_pericenter`, `mean_anomaly` | FLOAT(24) NOT NULL | the mandatory elements; degrees |
| `bstar`, `mean_motion_dot`, `mean_motion_ddot` | FLOAT(24) NULL | TLE-text units |
| `ephemeris_type` | SMALLINT | |
| `element_set_no`, `rev_at_epoch` | INT / INT NULL | |
| `mean_element_theory` | VARCHAR(16) | delivered (OMM) or materialized from the per-type table (TLE); part of the semantic hash, so stored |
| `semimajor_axis`, `period`, `apoapsis_alt`, `periapsis_alt` | FLOAT(24) NULL | **non-canonical** convenience columns: recomputed at ingest, excluded from export and identity |

Index budget — **3 B-trees total**:

- PK (clustered).
- `UNIQUE(epoch, record_hash)` — the dedup key, probed on every insert.
  Epoch-first: normal ingest carries recent elsets, so probes land in the
  hot right edge of the index regardless of table size; it also serves
  epoch range scans. This column order is also what keeps chronological
  bulk restores cheap — do not change it.
- `(norad_cat_id, epoch)` — the shape of every per-object query.

The legacy `(created)` index is **dropped**: it existed for `generate`'s
incremental lookback, which no longer exists.

### 5.2 `omm_metadata` — sparse enrichment sidecar

One row per `tle` row that has ever received OMM delivery.
`tle_id` BIGINT FK, UNIQUE + indexed. The **one sanctioned UPDATE** in the
system applies the §3.2 merge here.

Columns: `object_name` VARCHAR(100), `object_type` VARCHAR(20),
`country_code` VARCHAR(10), `rcs_size` VARCHAR(10), `launch_date`
VARCHAR(10), `site` VARCHAR(20), `decay_date` VARCHAR(10) (delivered
strings, never computed on); provenance `gp_id` INT, `file` BIGINT,
`creation_date` DATETIME(6), `originator` VARCHAR(20); delivered-semantics
overrides `ref_frame`, `time_system`, `center_name` VARCHAR(16) NULL —
null means "implicit from `ephemeris_type`" (§3), so the common case costs
nothing per row; projections use delivered values when present, else the
implicit table.

### 5.3 `rejects` — retained invalid input

One row per **distinct** rejected payload; bounded by what is broken, not
by time. An archive, not a live view: rows are never deleted by a clean
run; `resolved_at` carries the lifecycle.

| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT PK | |
| `record_hash` | BINARY(32) UNIQUE | sha256 of the verbatim payload (`line1\nline2` for TLE; canonical JSON of the record for OMM) |
| `format` | VARCHAR(4) | `tle` / `omm` |
| `payload` | TEXT | verbatim input, capped at 4096 bytes |
| `truncated` | BOOLEAN | payload hit the cap (garbage "lines" from mis-encoded files can be arbitrarily long) |
| `reason_code` | VARCHAR(20) | closed vocabulary of §4; latest rejection wins |
| `reason` | TEXT | free-text detail |
| `source_path` | VARCHAR(1024) | latest source file |
| `first_seen`, `last_seen` | DATETIME(6) | |
| `times_seen` | INT | re-rejection is an upsert bumping `last_seen`/`times_seen` |
| `norad_cat_id`, `epoch` | INT NULL, DATETIME(6) NULL | populated when parseable — "which objects have a bad feed" is one query |
| `resolved_at` | DATETIME(6) NULL | stamped when the same hash exists in `tle` (re-admission at ingest; reconciliation in `verify`) |

Index budget: PK, `UNIQUE(record_hash)`, `(last_seen)` (status trends).

### 5.4 `ingest_files` — source-file state

As currently implemented: `path` VARCHAR(1024) (for humans),
`path_hash` CHAR(64) UNIQUE (sha256 of the resolved path — bounded key,
safe for utf8mb4 index limits), `size` BIGINT, `mtime_ns` BIGINT (integer
ns compares exactly on all dialects), `sha256` CHAR(64) (content hash),
plus `last_run_id` BIGINT NULL linking to the run that recorded it.

### 5.5 `ingest_runs` — run accounting

One row per run, written at start (`status = 'running'`), finalized at
completion. A row stuck in `running` is itself a signal (crashed run) —
reported by `status`, never used as a lock.

Columns: `id` BIGINT PK; `command` VARCHAR(255) (subcommand + salient
flags); `started`, `finished` DATETIME(6) (finished NULL while running);
`status` VARCHAR(10) (`running`/`succeeded`/`failed`); counters
`files_scanned`, `files_ingested`, `files_skipped`, `files_refreshed`,
`files_failed` INT and `records_seen`, `records_inserted`,
`records_rejected` BIGINT; `error` TEXT NULL; `detail` JSON NULL
(extensible: per-output write counts, reject reason histogram — JSON maps
to TEXT on SQLite/MariaDB via SQLAlchemy, queryable where supported).
Index budget: PK only — status reads the insertion-ordered tail.

## 6. The ingest engine

### 6.1 Scan

For each `[[ingest.sources]]` entry (a `path` + glob `pattern`): list
matching files and **sort by name ascending** — delivery files are
date-named, so historical backfills arrive chronologically and hit the
hot edge of the dedup index and the outputs' append path. A missing
source directory is a WARNING, never an error.

File-state tracking (`ingest_files`): unchanged files (size + mtime match)
are skipped without being opened; a changed file whose content sha256
still matches just refreshes its stat state; otherwise the file is
re-ingested and dedup absorbs the already-seen records. A file that fails
to stat/read/parse gets no state recorded and is retried next scan.
`--force` bypasses skipping. Format is auto-detected per file
(`reader.detect_format`):

| Format | Extensions | Reader |
|---|---|---|
| TLE (2- or 3-line) | `.tle`, `.txt`, `.3le` | `read_tle` |
| OMM JSON (Space-Track) | `.json` | `read_omm_json` |
| OMM CSV | `.csv` | `read_omm_csv` |
| OMM XML | `.xml` | `read_omm_xml` |

### 6.2 Delta buffer and flush

Validated records accumulate in an in-memory buffer grouped by
`norad_cat_id` (~600 B/record). The buffer flushes at end-of-scan, or
early when it reaches `flush_records` (default 1,000,000 — purely a
memory budget, never a correctness knob: flushes are idempotent, so an
early flush only means touching hot output files more than once per run).

Flush order — the single ordering rule the whole crash story rests on:

1. **`store.add(delta)`** — chunked dialect upsert into `tle` on
   `UNIQUE(epoch, record_hash)`; then the OMM metadata pass: select back
   ids for the batch's `(epoch, record_hash)` pairs (epoch must lead the
   lookup — it is the unique index's leading column; hash-only would be a
   full scan), insert missing sidecar rows, apply the §3.2 merge UPDATE to
   existing ones.
2. **`output.apply(delta)`** for every `[[output.files]]` entry (§7) —
   consuming the same validated, pre-dedup delta.
3. **Commit `ingest_files` state** for every file fully consumed by this
   flush. A file split across flushes commits its state only with the
   flush containing its last record.

A crash between any two steps replays at most the files whose state never
committed; every sink is idempotent against that replay (store dedups on
the unique key; object outputs dedup on merge and guard appends by tail
epoch; date outputs upsert on `epoch >=`).

### 6.3 Run accounting and concurrency

Every run inserts its `ingest_runs` row first and finalizes it last;
structured report objects (not bare ints) flow from the engine to the CLI,
the run row, and any wrapper. **Single-writer ingest is a universal
invariant** — output-file guards assume serialized ingest. Enforcement is
an advisory OS lock on a lockfile beside the config (`flock` /
`msvcrt.locking`); a second concurrent `ingest` exits 1 immediately. The
`running` run row is *not* the lock (a crash would wedge it); it is a
diagnostic.

## 7. Outputs

Declarative `[[output.files]]` entries, all maintained incrementally by
the flush engine. Each entry: `type` (`object` / `date`), `format`
(`tle` / `omm-csv`), destination `dir`, and a filename scheme. At least
one entry is required (misconfiguration exits 2, not a silent no-op).
Outputs are **lossy projections** (TLE has no metadata; the STK CSV has no
provenance) — always rebuildable from the store, never the reverse. A
damaged output is an inconvenience, never data loss; the incremental
machinery below is an efficiency layer on top of `rebuild`.

- **`format = "tle"`**: verbatim passthrough of stored lines — the
  compatibility surface for TLE-only consumers (GMAT, legacy pipelines).
  Line-less records are **always skipped and counted** (one WARNING per
  run): TLE outputs never contain a byte thistle synthesized. Object
  files: all records, epoch order, epoch-tie ordering existing-before-new.
  Date files: latest record per object per epoch date.
- **`format = "omm-csv"`**: header row + the CelesTrak/STK 17-column
  projection (`OBJECT_NAME, OBJECT_ID, EPOCH, MEAN_MOTION, ECCENTRICITY,
  INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY,
  EPHEMERIS_TYPE, CLASSIFICATION_TYPE, NORAD_CAT_ID, ELEMENT_SET_NO,
  REV_AT_EPOCH, BSTAR, MEAN_MOTION_DOT, MEAN_MOTION_DDOT`) — verified
  against live CelesTrak output and STK's documented acceptance. All 17
  fields populate even for TLE-only deliveries, so every record is
  exportable. `columns = "stk"` (default) | `"full"` (all store columns).
  Acceptance check: a generated file loads in STK's SGP4 propagator.
- **Incremental maintenance**: object files append when every buffered
  epoch for that object is strictly newer than the file's tail epoch (read
  from the file itself); anything else — late delivery, tie, re-delivery,
  torn tail — triggers a merge-rewrite (tmp + atomic rename, dedup on
  record hash during merge). Date files are upserted from the delta:
  "latest per object that day" can only change when a newer-epoch record
  for that (object, date) arrives; replace on `epoch >=`, so
  newest-ingested wins ties (consistent with the merge tie rule).
- **Filename templating**: `extension` overrides the default suffix;
  `filename` arranges `{id}`/`{date}`/`{ext}`/`{format}`/`{type}`,
  validated at **config load** (unknown placeholder, missing required
  placeholder, format specs, path separators, `..` all fail there — a
  `KeyError` while naming the 40,000th file would leave a half-written
  tree). Directories come from `dir` only. `object_id = "int" | "alpha5"`,
  `zero_pad`, `date_format` control rendering;
  `parse_object_name`/`object_glob` derive the inverse for `verify`'s
  orphan scan.
- An `object_id = "alpha5"` output meeting `NORAD_CAT_ID` > 339,999
  **falls back to the plain integer stem** for that object with one
  WARNING per run — ingest never fails or drops valid data over an output
  naming choice; the inverse parsers accept both stems.
- **Write pool** (`writer.WritePool`, `output.write_workers`, 0 = auto):
  per-file open/close was 44% of a full rebuild — blocked time with the
  GIL released. The pool receives *pure filesystem work only*;
  sessions/queries stay on the calling thread; work is submitted a chunk
  of objects at a time so memory stays bounded and each task owns a
  distinct path. Rows are read as Core column tuples, never ORM entities
  (materializing ORM objects cost ~7.6 s of a 38 s rebuild).

## 8. Queries and library API

- `tles_for_object(session, satnum)` — all elsets, epoch order.
- `nearest_tles_for_date(session, date, days)` — nearest elset per object
  to 12:00 UTC, ±N days; the per-object reduction happens in SQL (window
  functions, one candidate each side of noon), so memory is O(objects).
  Always exact — this query is a core reason the store is a database.
- `get_tles(satnum, config=None)` — the `thistle` integration point:
  opens a read-only session on the configured database, returns
  `(line1, line2)` pairs.
- Read-tier calls open **read-only connections** (`PRAGMA query_only` /
  `SET SESSION TRANSACTION READ ONLY`) — defense in depth on top of DB
  grants; grants are the real enforcement.

## 9. CLI

`thistle-db [-c CONFIG] [--log FILE] [--no-progress] <command>`

Global: `--log FILE` — rotating loguru sink (10 MB, keep 10), the
cron-friendly alternative to redirection. `--no-progress` — force the rich
progress bar off (it only appears on interactive stderr; log lines render
above the live bar; command *output* is stdout and never routes through
the progress console).

| Command | Does |
|---|---|
| `init` | scaffold `config.toml` + `~/.config/thistle-db.toml` |
| `init-db [--drop] [--yes] [--defer-indexes]` | create the schema — the **only** DDL path; idempotent without `--drop` (also builds any missing model index — the finalize step after a deferred restore); `--drop` prompts and fails closed non-interactively without `--yes`; `--defer-indexes` (requires `--drop`) omits `(norad_cat_id, epoch)` for bulk restores — the unique dedup index is never deferred, so idempotency holds throughout |
| `ingest [FILES...] [--force]` | everything: validate, store, outputs, run accounting. Directory scans stay tolerant (exit 0; failed files retried next scan); named files exit 1 on any failure so cron wrappers can detect it |
| `rebuild` | linear store scan (ordered `norad_cat_id, epoch, id`) re-deriving all outputs — first run, disaster recovery, new output entry |
| `verify` | count reconciliation of every output against the store (the authority): per-object record counts, per-date winner counts, orphan-file scan; stamps `rejects.resolved_at` for payloads now in `tle`. Reads all output files → periodic cron (weekly), not every run |
| `export OUTPUT [--format tle\|csv\|omm-xml\|omm-kvn] [--force]` | full-fidelity CSV (`STORE_COLUMNS`, minus derived) is the backup/interchange form; TLE concat for legacy; CCSDS OMM XML/KVN (core keywords + extras in `userDefinedParameters`), OMM version 2.0 by default (STK's acceptance; Space-Track emits 3.0 — flag when needed). Refuses to overwrite without `--force` |
| `get-tle TARGET [--days N]` | TLEs to stdout. `TARGET` = `YYYYMMDD` (nearest per object to noon, ±N days, default 7) or an alpha-5-compatible NORAD ID (full history). Exits 1 if nothing matches |
| `status` | last runs (outcome, duration, counters), reject trends by reason, unresolved reject counts, stuck `running` rows |
| `rejects [--export DIR]` | summarize the rejects table; `--export` writes unresolved payloads as re-ingestable files (repair workflow: export, fix, ingest) |

Exit codes: 0 success; 1 runtime failure / no match; 2 invalid config;
3 database not initialized (commands never create schema implicitly —
`init-db` can run under an admin account while the day-to-day account
holds only DML grants).

Privilege tiering: `init-db` = DDL; `ingest` = DML + output-tree writes;
read commands = read-only connections.

## 10. Configuration

`config.toml` (path via `-c`, default `./config.toml`; `THISTLE_DB_CONFIG`
overrides the default — shared with thistle's db fallback). Parsed by
pydantic-settings; `extra="forbid"` on `[output]` and `[ingest]` so stale
keys fail loudly with a pointer at the key. Target shape:

```toml
[database]
drivername = "sqlite"            # or mysql+pymysql / postgresql+psycopg
name = "thistle-db.db"
# host / port / secrets_file for server databases

[ingest]
max_epoch_ahead_days = 30
verify_checksums = true
flush_records = 1000000          # delta-buffer memory budget

[[ingest.sources]]
path = "/data/spacetrack/daily"
pattern = "*.txt"

[[output.files]]
type = "object"                  # or "date"
format = "tle"                   # or "omm-csv"
dir = "./output/objects"
object_id = "int"                # or "alpha5"; zero_pad, date_format,
                                 # extension, filename as specced in §7

[output]
write_workers = 0                # 0 = auto

[logging]
level = "INFO"
```

Gone (clean slate): the `[storage]` section entirely (`backend`, `state`,
`flush_records` — the budget moves to `[ingest]`), `[ingest] reject_dir`
(rejects live in the database), `[output] lookback_days` (no generate).

Credentials are **never** stored in `config.toml`. Resolution order
(highest first): env vars (`THISTLE_DB_DATABASE__USERNAME` / `__PASSWORD`;
env ranks above init kwargs in the settings sources), user secrets
`~/.config/thistle-db.toml`, system secrets (`database.secrets_file`),
then config values. Empty strings in a scaffolded secrets file never mask
a lower layer.

## 11. Observability and operations

- **Runs are data**: `ingest_runs` + structured reports + `status` (§5.5,
  §6.3, §9). The cron recipe is: scheduled `fetch` (e.g.
  `scripts/download_tles.py` or any delivery mechanism) → `thistle-db
  ingest --log ...` → alert on exit code; `verify` weekly.
- **Logging levels**: routine no-ops (unchanged files) and individual
  rejected records at DEBUG; per-file actions and per-directory summaries
  at INFO; data problems at WARNING. Rejections are one WARNING per file
  giving the count **over the total attempted** — the ratio separates a
  few bad lines from an entirely wrong file. The INFO log of a
  steady-state cron run stays short.
- **Orchestration stays out of the core.** The pipeline is three linear
  nodes (fetch → ingest → verify) with its own incrementality; an
  orchestrator that can launch a backfill beside a scheduled run violates
  single-writer unless explicitly serialized. If one is ever adopted
  (multiple feeds, derived products), it lives in a **separate package**
  that imports the library API / shells the CLI and reads
  `ingest_runs`/`rejects` for metadata; the core never knows it exists.
- **Backups**: logical = `export` (re-ingestable anywhere, any dialect);
  physical backups of live servers belong to native tools
  (mariabackup/pg_basebackup); SQLite = copy the file while no writer
  runs.

## 12. Interchange and schema evolution

There is intentionally **no Alembic or other migration tooling.** Every
record reconstitutes losslessly from a full-fidelity `export`, so schema
changes ship as breaking releases migrated by `export` → `init-db --drop
--defer-indexes` → `ingest` → `init-db` (the finalize step builds the
deferred indexes in one sorted pass each). Restore performance on MariaDB
is index-bound, not parse-bound: modern MariaDB has no InnoDB change
buffer, so every secondary-index insert touches a real page — defer the
read-path index and size the buffer pool to hold the `tle` indexes. The
unique dedup index stays cheap during chronological restores because a
snapshot's epochs land in one hot region of the epoch-first index — a
reason its column order must not change. Purely additive changes may ship
as hand-written `ALTER` statements for all three dialects in the release
notes.

## 13. Testing strategy

- SQLite by default; `THISTLE_DB_TEST_MARIADB=1` /
  `THISTLE_DB_TEST_POSTGRES=1` (Docker) run MariaDB/PostgreSQL via
  testcontainers — CI runs all three. **Dialect-specific behavior (every
  upsert path, the merge UPDATE, JSON detail) must be tested against all
  three.**
- Fixtures in `tests/thistle_db/data/`: real TLE text and Space-Track OMM
  JSON, truncated to a few hundred records (realism from genuine provider
  output, not volume). `20250418.txt` keeps 6,001 records — above
  `CHUNK_SIZE` — so multi-chunk ingest stays exercised. `one.json` is the
  canonical TLE/OMM cross-check fixture. Add fixtures for: checksum
  mismatch (single flipped digit), future-epoch, line-less OMM-native
  records, a wholly corrupt file (fuse).
- Volume is generated, not committed (`scripts/bench_store.py` synthesis
  pattern); keep an ingest/rebuild bench to catch regressions in the flush
  engine and write pool.
- Invariant-level tests: double-ingest is a no-op (counts identical, zero
  new rows); kill-and-rerun converges (crash between flush steps 1/2/3);
  reject upsert bumps `times_seen` not row count; re-admission promotes a
  matured future-epoch reject exactly once; `verify` detects a truncated
  output file and a deleted quiet-object file.
- NaN-parsing caveat for tests: whether `nan` text parses to NaN is column-
  and process-dependent (numpy import changes the CRT parse of sgp4's
  exponent-format fields on Windows). Prefer a zero mean motion (infinite
  *derived* values) and unit-test `_clean_floats` directly.

## 14. Deleted by this plan

Relative to the shipping code — deleted, not deprecated:

- `generate` and all incremental machinery: `epoch_date_state`, watermark
  maintenance inside ingest, `lookback_days`,
  tail-guard-vs-mtime disambiguation, the ingest/generate serialization
  requirement.
- The dual-backend split: `[storage] backend`, `filestore.py` (TLE-line
  file store), the state-SQLite sidecar, the files-backend restrictions
  (OMM outputs refused, metadata discarded) and divergent tie rules.
- The filesystem reject quarantine (`reject_dir`, the quarantine tree) —
  replaced by the `rejects` table + `rejects --export`.
- `dump` as a distinct concept (subsumed by `export`).
- `line_hash` as a column name (now `record_hash`); nullable
  `norad_cat_id`; the `(created)` index.

## 15. Build order

Steps 1–3 of `PLAN-gp-store.md` are done (spec adopted; sgp4 type-4 spike
run; `record.py` implemented and tested). Remaining, re-planned for the
single-store decision:

**M4 — schema v2 + delta engine + outputs (the end-to-end core).**
`model.py`: `record_hash`, nullable lines, NOT NULL `norad_cat_id`,
`mean_element_theory`, metadata provenance + semantics-override columns,
`rejects`, `ingest_runs`, `ingest_files.last_run_id`; drop
`epoch_date_state` and `(created)`. `ingest.py` becomes the delta engine
(§6): checksum validation, DB reject sink with fuse, re-admission pass,
buffer/flush with the three-step order, run accounting, advisory lock.
Output application from the delta (§7) for `tle` and `omm-csv`, object
and date. Retire `generate` and watermarks.
*Acceptance*: fixture corpus double-ingest is a no-op; kill -9 between
flush steps then re-run converges byte-identically; checksum/future-epoch/
line-less fixtures land in `rejects` with correct reason codes; a cron-
shaped run logs ≤ a dozen INFO lines.

**M5 — operations surface.** `rebuild`, `verify` (counts, orphans,
resolution stamping), `export` (csv/tle/omm-xml/omm-kvn), `status`,
`rejects --export`; delete `filestore.py`, `generator.py`, `[storage]`,
`reject_dir`, legacy CLI paths; update README + `example.config.toml`;
sync CLAUDE.md to this plan and collapse its [NOT YET IMPLEMENTED]
markers.
*Acceptance*: `export` → `init-db --drop --defer-indexes` → `ingest` →
`init-db` round-trips the store losslessly on all three dialects; `verify`
catches a seeded mid-file truncation; an STK-loadable omm-csv sample is
spot-checked.

## 16. Invariants (the short list to never regress)

Idempotent ingest; tolerant scans (one bad record ≠ failed file, one bad
file ≠ failed scan); no stored non-finite float; validity = table
membership, never a flag; verbatim lines are the audit trail and TLE
outputs are pure passthrough; rejects are retained, deduped, fused per
file, and reject-write failures never fail ingest; flush order
store → outputs → file state, with every sink replay-idempotent;
O(delta) steady state independent of catalog size; minimal index budget;
single-writer ingest; DDL only in `init-db`; read paths read-only; no
credentials in config; short INFO logs for cron; progress UI on
interactive stderr only, command output on stdout; no orchestrator in the
core.

## 17. Open questions

None. Decisions new in this plan (table name kept as `tle`; derived
convenience columns retained but non-canonical; implicit-semantics
placement split between `tle.mean_element_theory` and nullable
`omm_metadata` overrides; 4096-byte reject payload cap; VARCHAR
vocabularies over ENUMs; advisory-lockfile concurrency; `(created)` index
dropped) are stated inline with rationale and take effect on adoption.
