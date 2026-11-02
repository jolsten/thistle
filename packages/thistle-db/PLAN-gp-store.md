# Plan: GP records, delta engine, pluggable store

Status: **adopted 2026-09-02** — folded into CLAUDE.md, which is now the
normative spec (it wins every conflict with this file). This document is
retained as the design rationale: the why behind the decisions, including
alternatives considered and rejected. Written **clean-slate: backwards
compatibility is explicitly out of scope** — no migration tooling, no
deprecation phases; an existing deployment adopts this by re-ingesting its
source files or a full-fidelity export.

Progress: build-order steps 1–3 are done (spec adopted; sgp4 spike run —
2.27 parses type-4 lines with `ephtype` preserved but has no XP propagation
model; `record.py` implemented with tests). Steps 4–6 remain.

Store-throughput spike (2026-09-15, `scripts/bench_store.py`, v1 TLE
filestore mechanics vs real SQLite `tle` schema, 2k objects × 240k records,
Windows/NTFS): files store ~320k rec/s ingest vs SQLite ~24–30k rec/s
(**~12× faster**), 34 MB on disk vs 120 MB. Reverse-chronological delivery
cost only ~15% at this scale — merge-rewrite amplification is
(per-object history)² / (2 × per-flush delta per object) records rewritten,
negligible until per-object history is large relative to the flush batch.
Two consequences folded into the design: ingest scans should sort source
files chronologically (date-named files make appends the common case), and
the GP-CSV store's merge is cheaper than v1's (RECORD_HASH and EPOCH are
columns — no per-line sgp4 re-parse). The files-store-vs-SQLite decision
gate is resolved: the files store earns its keep on ingest throughput alone,
before its scale advantage (SQLite index maintenance degrades past cache;
appends stay linear).

## Shape of the system

```
sources ──► readers ──► validation ──► delta buffer (GP records, by object)
(TLE/OMM)               (rejects,           │  flush at end-of-scan / flush_records
                         epoch guard,       ▼
                         quarantine)   1. store.add(delta)         dedup by RECORD_HASH
                                       2. output.apply(delta) ×N   merge-dedup / tail guard / date upsert
                                       3. commit ingest_files state
```

Three ideas, one per layer:

1. **One canonical record** — the GP record, an OMM-superset modeled on
   Space-Track's published GP class, carrying verbatim TLE lines as fields.
   Everything in the system produces, stores, or projects this record.
2. **One engine** — the single-pass delta flush. Ingest validates records,
   buffers them grouped by object, and at flush time hands the same delta to
   every sink. Source-file state commits only after all sinks succeed.
   **Every sink is idempotent against replay** (store dedups on
   `RECORD_HASH`; object outputs dedup on merge and guard appends by tail
   epoch; date outputs upsert on `epoch >=`), so that one ordering rule is
   the entire crash-safety story: a crash replays at most the files whose
   state was never recorded, and every sink absorbs the duplicates.
   Because outputs consume the *validated buffer, pre-dedup*, they never need
   to ask the store which rows were new — which is precisely what eliminates
   the watermark/lookback machinery for both store types.
3. **One store interface, two implementations** — filesystem or SQL, chosen
   per deployment. Outputs are lossy projections and identical in behavior
   regardless of store.

There is no steady-state `generate`. Ingest maintains the outputs. The cron
entry is `thistle-db ingest`.

## The canonical record

One flattened GP record per element set. Column vocabulary is CCSDS OMM
keywords; the column set is anchored to Space-Track's published GP class
(minus derived values, plus `RECORD_HASH`), so schema evolution has an
external trigger rather than being a thistle design activity.

| Class | Columns | Rule |
|---|---|---|
| Identity | `RECORD_HASH` (first column) | See Uniqueness |
| Parsed or delivered | `NORAD_CAT_ID`, `OBJECT_ID`, `EPOCH` (ISO-8601, µs), `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`, `MEAN_MOTION_DOT`, `MEAN_MOTION_DDOT`, `EPHEMERIS_TYPE`, `CLASSIFICATION_TYPE`, `ELEMENT_SET_NO`, `REV_AT_EPOCH` | From the OMM when delivered as OMM; parsed from the lines when TLE-only |
| Implicit semantics | `MEAN_ELEMENT_THEORY`, `REF_FRAME`, `TIME_SYSTEM`, `CENTER_NAME` | For TLE-only records, materialized from a **per-`EPHEMERIS_TYPE` table** (type 0 → SGP4/TEME/UTC/EARTH; type 4 → SGP4-XP with reinterpreted drag-term columns), never hardcoded — the first SGP4-XP TLE that arrives is stored truthfully |
| Metadata | `OBJECT_NAME`, `OBJECT_TYPE`, `RCS_SIZE`, `COUNTRY_CODE`, `LAUNCH_DATE`, `SITE`, `DECAY_DATE` | From OMM; null for TLE-only (except `OBJECT_NAME` from a 3LE name line). Never inferred |
| Provenance | `GP_ID`, `FILE`, `CREATION_DATE`, `ORIGINATOR` | From OMM; null for TLE-only. Never backfillable except by OMM re-delivery |
| Verbatim rendering | `TLE_LINE0`, `TLE_LINE1`, `TLE_LINE2` | Delivered bytes (TLE delivery) or the provider's embedded lines (OMM delivery); null for an OMM-native record with no lines |

Decisions with rationale:

- **Element columns are stored even when redundant with the lines**: uniform
  readers (no "parse the lines if columns are blank" second path),
  grep/spreadsheet usability, projection without running sgp4. ~600 B/record
  instead of ~350; accepted.
- **Derived values (`SEMIMAJOR_AXIS`, `PERIOD`, `APOAPSIS`, `PERIAPSIS`) are
  not canonical** — recomputed where needed. The one deliberate loss versus a
  delivered Space-Track OMM; they are derivable, and for today's type-0
  records the OMM element fields carry no more precision than the TLE
  rendering anyway (verified: `BSTAR "0.00168730000000"` is `16873-2`
  expanded).
- **TLE lines stay first-class and byte-verbatim.** TLE → OMM → TLE does not
  round-trip byte-exactly (space- vs zero-padded catalog numbers, sign
  rendering, `00000-0`/`00000+0` bstar forms). Storing delivered bytes
  preserves the audit trail and makes TLE output a passthrough.

### Uniqueness

`RECORD_HASH` (hex sha256), chosen on merits — verbatim text is the identity
of a delivery:

- **Lines present**: hash of `line1 + "\n" + line2`. Textually-distinct
  near-duplicates from different providers deliberately coexist.
- **Line-less** (OMM-native records unrepresentable as TLE): **Decided:
  semantic identity** — a re-issue with changed metadata or a new `GP_ID`
  enriches the existing record via the merge rule rather than coexisting,
  consistent with how line-present records behave. The frozen serialization,
  sha256 over the UTF-8 encoding of the following `"\n"`-joined fields:

  1. the literal `thistle-gp1` (domain-separates this hash space from the
     line-hash space and names the serialization version);
  2. `NORAD_CAT_ID` as unpadded decimal;
  3. `EPOCH` as `YYYY-MM-DDTHH:MM:SS.ffffff` — the stored naive-UTC
     microsecond value, always 6 fractional digits;
  4. `MEAN_ELEMENT_THEORY` verbatim, uppercase;
  5. then `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`,
     `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`, `MEAN_MOTION_DOT`,
     `MEAN_MOTION_DDOT`, each as Python `float.hex()` of the stored IEEE-754
     double with `-0.0` normalized to `0.0`, or the empty string for a null
     drag term.

  Rendered lowercase-hex like line hashes. Rationale: hex floats
  (C99 `%a` form, 13-hex-digit mantissa) are exact and reimplementable in
  any language without depending on shortest-round-trip formatting
  algorithms; hashing the *parsed doubles* means formatting variants of the
  same value (`.5` / `0.5` / `5.0E-1`) hash equal, while genuine last-ulp
  parse differences between providers coexist as separate records —
  mirroring how textual near-duplicates coexist for line-present records.
  Identity is over the *stored* record (µs-truncated epoch, cleaned floats),
  so hash-before-store and hash-after-read agree. Changing this
  serialization changes identity → major format version bump; the `gp1` tag
  makes any such bump self-announcing.

### Enrichment merge

"TLE first, identical-lined OMM later" updates the existing record's null
columns (in the files store this lands on the merge-rewrite path, which a
duplicate delivery forces anyway). Precedence:

1. Non-null beats null.
2. Both non-null and unequal (a re-issue: same lines, new `GP_ID`): newest
   `CREATION_DATE` wins field-wise; null `CREATION_DATE` loses to non-null;
   full tie (both dates equal or both absent) → **last-ingested wins**.
   Decided over storing an ingest-timestamp ordinal: matches the date-file
   tie rule and keeps thistle state out of the canonical record. Consequence
   accepted: a re-ingest replaying *genuinely conflicting* redeliveries in a
   different order can flip such fields (replay of identical data cannot).
3. `RECORD_HASH`, `EPOCH`, and the verbatim lines never change; element
   columns may be refined by rule 2.

Immutability convention: **identity and lines immutable; other fields
enrichable.** Epoch-tie ordering in files: existing records before new.

### Validation (unchanged in substance)

The record shape changes; what makes a record storable doesn't. Non-finite
mandatory elements reject; nullable columns null; future-epoch guard;
quarantine as a live view, bounded per file; a record with no NORAD ID has no
home and is quarantined. One bad record never aborts a file; one bad file
never aborts a scan.

## The store

One interface, two implementations, chosen in config:

```toml
[storage]
type = "files"                    # or "database"
state = "./thistle-state.sqlite"  # files store only: source-file tracking
flush_records = 1000000           # memory budget per flush
```

Interface: `add(delta)` (idempotent), linear scan (rebuild/export), per-object
read, date-candidate read, per-object and per-date counts (verify).

Outputs are **lossy projections** (TLE has no metadata; the STK CSV has no
provenance), so the store must be full-fidelity in either implementation —
the store is never reconstructible from the outputs. Moving between stores
goes through `export` + `ingest`.

### `type = "files"` — GP-CSV object tree

One epoch-ordered CSV file per object (naming via the existing
`object_id`/`zero_pad`/`filename` templating), one record per line, plus the
small state SQLite (a cache, created on first use, safe to delete).

- Append-or-merge per object: all buffered epochs strictly newer than the
  file's tail → append; anything else (late, tie, re-delivery, torn tail,
  enrichment) → merge-rewrite (tmp + atomic rename). Torn CSV tail = drop the
  unparseable last line, healed by the next merge.
- ~500–600 B/record; a 100k-elset worst-case object ≈ 60 MB. If a deployment
  proves size matters, per-object zstd with merge-time recompression is the
  escape hatch (never gzip-member appends).
- **Format versioning** (the price of normalization, paid deliberately):
  every file begins with a header row, authoritative for that file's columns;
  readers tolerate unknown columns and treat missing nullable columns as
  null, so additive change touches zero existing files. A tree-root
  `store.toml` records format version and store-defining rules. Only additive
  changes within a major version.
- `get-tle YYYYMMDD` reads date outputs when configured (candidates are each
  day's winners — a bounded approximation); exact nearest-to-noon needs the
  database store.

### `type = "database"` — SQL

The current schema shape survives on merits: `tle` (one row per record,
verbatim lines + parsed columns) + `omm_metadata` sidecar (sparse, unique FK
— now also carrying provenance: `GP_ID`, `FILE`, `CREATION_DATE`,
`ORIGINATOR`), `ingest_files` in the same database. Kept conventions: BIGINT
surrogate keys, naive-UTC `DATETIME(6)`, Float32 element columns with lines
as the precision authority, minimal index budget
(PK, `UNIQUE(epoch, record_hash)`, `(norad_cat_id, epoch)`), dialect-aware
`INSERT IGNORE`/`ON CONFLICT DO NOTHING` in 5000-row chunks, SQLite/MariaDB/
PostgreSQL. Enrichment (merge rule above) is an UPDATE of the metadata row —
the one sanctioned mutation.

Exact queries (`nearest_tles_for_date` over all elsets, ad-hoc SQL) are this
store's reason to exist. Its schema changes via `export` → `init-db --drop`
→ `ingest` (the deferred-index flow survives for large catalogs); no
migration framework, deliberately.

## Outputs

Declarative `[[output.files]]` entries, all maintained incrementally by the
flush engine, identical behavior on either store. Filename templating,
`extension`, `date_format`, `object_id` rendering as currently specced.

- **`format = "tle"`, `type = "object"` / `"date"`**: verbatim passthrough of
  stored lines. The compatibility surface for TLE-only consumers (GMAT,
  legacy pipelines). Records with null lines are **always skipped and
  counted** (one WARNING per run) — decided: TLE outputs are pure verbatim
  passthrough and thistle never synthesizes line bytes, keeping exporter
  fidelity out of the correctness surface; rendering can be added later
  behind a knob if OMM-native volume ever makes the skip count hurt. Object
  files: all records, epoch order. Date files: latest record per object per
  epoch date.
  An `object_id = "alpha5"` output meeting `NORAD_CAT_ID` > 339,999 (beyond
  alpha-5's range) **falls back to the plain integer stem for that object**
  with one WARNING per run — ingest never fails or drops valid data over an
  output naming choice; mixed naming in the directory is the accepted cost
  (`parse_object_name`/`object_glob` must accept both stems).
- **`format = "omm-csv"`, `type = "object"` / `"date"`**: header row + the
  CelesTrak/STK 17-column projection (`OBJECT_NAME,OBJECT_ID,EPOCH,
  MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,
  MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,
  ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT`) —
  verified against live CelesTrak output and STK's documented "OMM-based
  CSV" acceptance. All 17 fields are populated even for TLE-only deliveries,
  so **every** record is exportable — no subset caveat.
  `columns = "stk"` (default) | `"full"` (all store columns, for
  metadata-bearing outputs). Acceptance check when implemented: a generated
  file loads in STK's SGP4 propagator.

Filesystem work stays on the write pool (pure filesystem tasks only; all
queries on the calling thread; bounded chunks).

## Commands

- `init` — scaffold config + user secrets file.
- `init-db [--drop] [--yes] [--defer-indexes]` — database store: the only DDL
  path. Files store: create the state cache.
- `ingest [FILES...] [--force]` — everything: validate, store, outputs.
  Scans stay tolerant (exit 0); named files exit 1 on any failure.
- `rebuild` — linear store scan re-deriving all outputs (first run, disaster
  recovery, new output entry).
- `verify` — count reconciliation of every output against the store (the
  authority): per-object record counts, per-date winner counts. Periodic
  cron, not every run.
- `export OUTPUT [--format tle|csv|omm-xml|omm-kvn] [--force]` — full-fidelity
  CSV is the store-to-store migration and backup form; TLE concat for legacy;
  CCSDS OMM XML/KVN (core keywords + extras in `userDefinedParameters`) for
  standards-adherent interchange, emitting **OMM version 2.0 by default**
  (STK's documented acceptance; Space-Track emits 3.0 — offer a flag when
  someone needs it).
- `get-tle TARGET [--days N]` — as currently specced; exact on the database
  store, bounded approximation on files.

Privilege tiering: `init-db` = DDL; `ingest` = DML **and output writes** (the
ingest account owns the output tree); read commands use read-only
connections. **Single-writer ingest is universal** — output-file guards
assume serialized ingest on either store (`flock` / cron chaining).

## What does not exist in this design

Relative to the current codebase — deleted, not deprecated:

- `generate` and all its incremental machinery: `epoch_date_state`, mtime
  watermarks, `lookback_days`, tail-guard-vs-mtime disambiguation, the
  ingest/generate serialization requirement.
- The dual-backend split (`backend = "database" | "files"`) and its divergent
  tie rules.
- `dump` as a distinct concept (subsumed by `export`).
- The TLE-line store format and its metadata-discarding restrictions
  (omm outputs are legal everywhere; OMM ingest discards nothing).
- Any v1→v2 migration tooling. Adoption = re-ingest sources, or `export`
  from an old deployment and ingest that.

## Invariants carried forward

Idempotent ingest; tolerant scans; quarantine live-view, bounded, never
failing an ingest; future-epoch guard; atomic writes and replay-based crash
safety; O(delta) steady state independent of catalog size; minimal DB index
budget; no credentials in config; read-only reader connections; short INFO
logs for steady-state cron runs; progress bar on interactive stderr only,
command output on stdout.

## Build order

1. Freeze the spec: canonical column list. Fold this document into CLAUDE.md
   as the target.
2. Spikes: (a) python-sgp4 SGP4-XP / type-4 handling → the implicit-semantics
   table; (b) CSV store read/write throughput at bench scale
   (`scripts/bench_generate.py` shapes); (c) STK load test of a sample
   17-column file.
3. Record module: GP record type, TLE/OMM → record construction, per-type
   implicit semantics, `RECORD_HASH`, merge rule. Pure, heavily unit-tested.
4. Delta engine + files store + tle/omm-csv outputs (the serverless
   deployment, end to end).
5. Database store behind the same interface; exact `get-tle`.
6. `rebuild`, `verify`, `export`; retire the old code paths.

## Open questions

None. All prior open questions are resolved and folded into the body above
(line-less identity → semantic element hash with the frozen `thistle-gp1`
serialization; TLE outputs → always skip line-less records; merge tie without
`CREATION_DATE` → last-ingested wins; alpha-5 overflow → integer-stem
fallback with WARNING; exported OMM version → 2.0 default).
