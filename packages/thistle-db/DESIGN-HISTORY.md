# Design history

A condensed record of how thistle-db's design was reached — superseded
alternatives, the measurements that decided them, and options deliberately
left unbuilt. CLAUDE.md is the spec for the system as it is; this file
exists so no settled decision gets re-litigated without the evidence that
settled it. Full detail: git history (the original PLAN documents lived in
this directory until 2026-11).

## 2026-09 — the GP-record architecture

Replaced the v1 TLE-line pipeline with one canonical GP record
(OMM-superset, verbatim lines as fields, `RECORD_HASH` identity) feeding a
single-pass delta engine. Resolved then, still in force: line-less
identity = the frozen `thistle-gp1` semantic hash; TLE outputs always skip
line-less records; merge tie without `CREATION_DATE` → last-ingested wins;
alpha-5 overflow → integer-stem fallback; exported OMM version 2.0 by
default.

The original design also made the store pluggable: SQL **or** a GP-CSV
object tree ("files store") behind one interface. Supporting spike
(2026-09-15, a since-deleted `scripts/bench_store.py` on v1 mechanics, 2k objects × 240k
records, Windows/NTFS): files store ~320k rec/s ingest vs SQLite ~24–30k
(~12×), 34 MB vs 120 MB on disk; reverse-chronological delivery cost only
~15% — merge-rewrite amplification is (per-object history)² / (2 ×
per-flush delta per object), negligible until history is large relative
to the flush batch. Lasting consequence: ingest scans sort source files
chronologically so backfills probe the hot right edge of the epoch-first
dedup index.

## 2026-11-09 — single SQL store (0.16.0)

The dual-store interface was dropped before the files store was ever
built, despite its 12× bulk-ingest edge: SQLite already covers the
serverless case; every feature paid a lowest-common-denominator tax
(exact vs approximate `get-tle`, merge-as-UPDATE vs merge-rewrite, no
natural home for DB-resident state tables); and the throughput edge only
matters for rare full-catalog restores, which `init-db --defer-indexes`
serves. The same release moved rejects from a filesystem quarantine into
the `rejects` table (validity = table membership), added TLE checksum
validation and `ingest_runs` accounting, renamed `line_hash` →
`record_hash`, made `norad_cat_id` NOT NULL, and dropped the `(created)`
index together with `generate`'s incremental machinery
(`epoch_date_state`, mtime watermarks, `lookback_days`).

As shipped, ingest still maintained outputs inline: one flush handed the
same pre-dedup delta to every sink (store → outputs → file state), and
outputs were kept current incrementally — tail-epoch probe appends,
merge-rewrites, a date-file upsert, torn-tail healing.

## 2026-11-14 — resumable ingest, decoupled outputs

Ingest became store-only with a flush at every file boundary (batches
never span files; `FlushError` fails the run on infrastructure errors;
interrupts finalize `cancelled`), and output maintenance moved behind the
`output_cursor` table into `update-outputs`. The strategy choice for the
output phase, per steady-state run at design scale (~100k objects,
thousands dirty per run, ~3k elsets average history):

| Strategy | Rows read/run | Scales with |
|---|---|---|
| Full rebuild every run | ~whole store (10⁸) | total history |
| **Scoped re-render (chosen)** | dirty objects × history (~10⁷) | dirty count × history depth |
| Delta-apply (the incremental layer) | new rows (~10⁴–10⁵) | new rows |

Scoped re-render was chosen although it reads ~100–1000× more than
delta-apply, because it bought one write path (the entire incremental
apply layer was deleted, with its documented caveats), exactness by
construction (every touched file is byte-identical to a rebuild's —
touching is healing), and order-free idempotency. Accepted trend: cost
grows linearly with archive depth.

## Deliberately not built (escape hatches)

- **Delta-apply as a pure optimization** behind the same `output_cursor`,
  if scoped re-render ever gets too slow — the cursor bookkeeping is
  exactly what it would need.
- **Per-window cursor checkpointing** inside `update-outputs`, if large
  dirty sets outside `rebuild` territory ever appear.
- **A `last_changed` ordinal on `omm_metadata`** feeding the dirty scan,
  if the enrichment-only-redelivery gap (no new `tle.id` → no cursor
  motion) ever matters.
- **Rendering TLE lines for line-less records** behind a knob, if
  OMM-native volume ever makes the skip count hurt.
