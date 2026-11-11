# thistle-db

Orbital element database manager. Ingests TLE (Two-Line Element) and OMM
(Orbit Mean-Elements Message) files into a SQL database — the canonical,
full-fidelity store — and maintains organized output files (TLE by object
and by date; STK-compatible OMM CSV) in the same pass.

Nothing is silently lost: records that fail validation are retained in a
queryable `rejects` table with a reason code, and every run is recorded in
an `ingest_runs` table (`thistle-db status` is the dashboard).

## Installation

```bash
pip install thistle-db

# For MariaDB/MySQL support:
pip install thistle-db[mysql]
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add thistle-db
```

## Quick Start

### 1. Scaffold configuration

```bash
thistle-db init
```

This creates two files:

- `./config.toml` -- main configuration (database, ingest sources, outputs)
- `~/.config/thistle-db.toml` -- user-local database credentials

Use `-c` to specify a different config path:

```bash
thistle-db -c /etc/thistle-db/config.toml init
```

### 2. Configure

Edit `config.toml` to set your database, ingest sources, and outputs. The
generated file is fully commented — see `example.config.toml` for a
complete operational layout.

**SQLite (default; serverless):**

```toml
[database]
drivername = "sqlite"
name = "thistle-db.db"
```

**MariaDB/MySQL:**

```toml
[database]
drivername = "mysql+pymysql"
host = "localhost"
port = 3306
name = "thistle-db"
secrets_file = "/etc/thistle-db/secrets.toml"
```

Then add your credentials to `~/.config/thistle-db.toml`:

```toml
username = "myuser"
password = "mypassword"
```

Credentials never live in `config.toml`. Resolution order (highest
priority first): `THISTLE_DB_DATABASE__USERNAME`/`__PASSWORD` env vars,
the user secrets file, the system secrets file, config values.

### 3. Create the database schema

```bash
thistle-db init-db
```

Run once per database (idempotent — safe to re-run). Commands never create
schema implicitly, so on MariaDB/PostgreSQL you can run `init-db` with an
admin account and give the day-to-day account only read/write privileges.
`init-db --drop` destroys and recreates everything (asks for confirmation
unless `--yes`). For bulk restores, `init-db --drop --defer-indexes`
creates the schema without the read-path indexes; a later plain `init-db`
builds them in one fast sorted pass each.

### 4. Ingest

```bash
thistle-db ingest                      # scan configured source directories
thistle-db ingest 20260327.tle gp.json # or ingest specific files
```

One pass does everything: validate, store, and update every configured
output. There is no separate generate step — the steady-state cron entry
is just `thistle-db ingest`. Outputs are optional: with no
`[[output.files]]` entries, ingest is store-only (useful when the
database is queried directly, e.g. as `thistle`'s catalog fallback).

File format is auto-detected by extension:

| Extension              | Format               |
| ---------------------- | -------------------- |
| `.tle`, `.txt`, `.3le` | Two-Line Element     |
| `.json`                | Space-Track OMM JSON |
| `.csv`                 | OMM CSV              |
| `.xml`                 | OMM XML              |

Ingestion is **idempotent** (duplicates are silently absorbed — the dedup
key is the verbatim record text) and **incremental** (unchanged files are
skipped without being opened). Validation is tolerant: one bad record
never aborts a file, one bad file never aborts a scan.

Records that cannot be stored — unparseable text, non-finite mandatory
elements, TLE checksum mismatches, epochs beyond the configurable future
horizon, records with no NORAD ID — are **retained in the `rejects`
table** with a reason code and provenance. Future-epoch rejects are
re-admitted automatically once their epoch comes inside the horizon.

### 5. Outputs

Declared as `[[output.files]]` entries; `ingest` maintains all of them
incrementally, and `rebuild` re-derives everything from the store (first
run, disaster recovery, or after adding an entry):

- **Object files** (`type = "object"`) — all records for one satellite,
  epoch order. NORAD IDs render as integers (optionally zero-padded) or
  alpha-5; filenames are templatable (`filename = "tle_{id}{ext}"`).
- **Date files** (`type = "date"`) — the latest record per satellite per
  epoch date.
- **Formats**: `tle` (verbatim two-line passthrough — never a byte
  thistle synthesized) or `omm-csv` (the 17-column CelesTrak/STK
  projection, or `columns = "full"` for every store column).

## Operations

```bash
thistle-db status            # recent runs, reject trends, stuck runs
thistle-db verify            # reconcile outputs against the store (weekly cron)
thistle-db rejects           # summarize rejected input by reason
thistle-db rejects --export repair/   # write payloads out for repair + re-ingest
thistle-db get-tle 25544     # all TLEs for an object (alpha-5 IDs work too)
thistle-db get-tle 20260327  # nearest TLE per object to that date's noon
thistle-db export backup.csv # full-fidelity, re-ingestable backup
```

- **Runs are data**: every `ingest`/`rebuild` writes an `ingest_runs` row
  with per-run counters; `status` reads it back. Scans exit 0 even when
  individual files fail (they retry next scan); explicitly named files
  exit 1 on failure so cron wrappers can alert.
- **Single-writer**: `ingest` and `rebuild` take an advisory lock beside
  the config file; a second writer exits immediately.
- **Backup/migration**: `export` writes a full-fidelity CSV that
  re-ingests into any supported dialect (`init-db` + `ingest backup.csv`
  — also the schema-migration path; there is deliberately no migration
  framework). `--format tle|omm-xml|omm-kvn` cover legacy consumers and
  CCSDS OMM 2.0 interchange. Physical backups of a live server belong to
  the native tools.
- **Exit codes**: 0 success; 1 runtime failure / no match; 2 invalid
  config or usage; 3 database not initialized.

## Library use

```python
from thistle_db import get_tles

pairs = get_tles(25544)   # [(line1, line2), ...] from the configured DB
```

`thistle` (the propagation package) uses this as its catalog fallback.

## Supported databases

SQLite (default — dev, small and serverless deployments), MariaDB/MySQL
(operational), PostgreSQL. Dialect-specific behavior is tested against all
three in CI.

## License

MIT
