# thistle-db

Orbital element database manager. Ingests TLE (Two-Line Element) and OMM (Orbit Mean-Elements Message) files into a database and generates organized output files by date and satellite.

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

- `./config.toml` -- main configuration (database, ingest sources, output settings)
- `~/.config/thistle-db.toml` -- user-local database credentials

Use `-c` to specify a different config path:

```bash
thistle-db -c /etc/thistle-db/config.toml init
```

### 2. Configure

Edit `config.toml` to set your database and ingest sources. The generated file is fully commented -- see below for a summary.

**SQLite (default):**

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

### 3. Create the database schema

```bash
thistle-db init-db
```

Run once per database (idempotent — safe to re-run). Commands never create
schema implicitly, so on MariaDB/PostgreSQL you can run `init-db` with an
admin account and give the day-to-day account only read/write privileges.
`init-db --drop` destroys and recreates everything (asks for confirmation
unless `--yes`).

### 4. Ingest TLE/OMM files

Scan configured source directories:

```bash
thistle-db ingest
```

Or ingest specific files:

```bash
thistle-db ingest /path/to/20260327.tle /path/to/20260327.json
```

File format is auto-detected by extension:

| Extension          | Format              |
| ------------------ | ------------------- |
| `.tle`, `.txt`, `.3le` | Two-Line Element |
| `.json`            | Space-Track OMM JSON |
| `.csv`             | OMM CSV              |
| `.xml`             | OMM XML              |

Ingestion is idempotent -- duplicate records are silently skipped.

### 5. Generate output files

```bash
thistle-db generate
```

This produces files in the configured output directory:

- **Date files** (`YYYYMMDD.tle` / `YYYYMMDD.omm`) -- one TLE per satellite for each date (latest epoch that day)
- **Object files** (`25544.tle` / `25544.omm`) -- all TLEs for a single satellite, ordered by epoch

Generation is **incremental**: each run rewrites date files for a trailing
epoch window and appends newly ingested rows to object files, so cost
scales with new data rather than database size. Late-arriving TLEs are
placed correctly automatically. Run `generate --all` once after ingesting
pre-existing/historical data (or restoring a backup); routine cron runs
need no flags.

## Automating with Cron

thistle-db is designed to run via cron rather than as a long-running service. Both `ingest` and `generate` are idempotent and safe to re-run.

**Ingest and generate every 4 hours:**

```cron
0 */4 * * * thistle-db -c /etc/thistle-db/config.toml ingest && thistle-db -c /etc/thistle-db/config.toml generate
```

**Ingest hourly, generate once daily at 03:00 UTC:**

```cron
0 * * * * thistle-db -c /etc/thistle-db/config.toml ingest
0 3 * * * thistle-db -c /etc/thistle-db/config.toml generate
```

**With logging to a file:**

```cron
0 */4 * * * thistle-db -c /etc/thistle-db/config.toml ingest >> /var/log/thistle-db.log 2>&1 && thistle-db -c /etc/thistle-db/config.toml generate >> /var/log/thistle-db.log 2>&1
```

**Weekly integrity sweep** (reconciles every object file against the
database and repairs any damage the incremental runs can't see):

```cron
0 4 * * 0 thistle-db -c /etc/thistle-db/config.toml generate --verify
```

## MariaDB Deployment Tuning

The schema is designed so the hot working set (the dedup and per-object
indexes) stays small, but server configuration still decides whether inserts
run at memory speed or disk speed. In rough priority order:

### Must-have

- **`innodb_buffer_pool_size`** — the single setting that matters. The
  default (128 MB) is far too small for a growing catalog and is the classic
  cause of ingest performance "falling off a cliff" as the table grows.
  Size it to comfortably hold the `tle` indexes — roughly 1–2 GB of buffer
  pool per 10M rows — or simply give it 25–50% of the machine's RAM on a
  dedicated host. Check the current value with:

  ```sql
  SHOW VARIABLES LIKE 'innodb_buffer_pool_size';
  ```

### Nice-to-have

- **`innodb_flush_log_at_trx_commit = 2`** — one log flush per second
  instead of one per commit. Ingest commits every 5000-row chunk, so this
  meaningfully speeds bulk loads. The trade: a server crash (not a client
  crash) can lose up to ~1 second of committed rows — acceptable here
  because ingest is idempotent and re-running it restores anything lost.
  Keep the default (`1`) if the database also serves data you cannot
  re-derive.
- **`innodb_log_file_size`** (redo log; `innodb_redo_log_capacity` on
  newer MariaDB) — raise to 1–4 GB if large restores or backfills
  checkpoint-stall (visible as periodic throughput collapses during bulk
  ingest). Irrelevant for routine daily deliveries.
- **`wait_timeout` / `net_write_timeout`** — the defaults are fine for the
  normal cron cadence; only relevant if you script very long-running custom
  reads over the same connection.

## Credential Resolution

Database credentials are resolved in priority order:

1. **Environment variables** -- `THISTLE_DB_DATABASE__USERNAME` / `THISTLE_DB_DATABASE__PASSWORD`
2. **User secrets file** -- `~/.config/thistle-db.toml`
3. **System secrets file** -- path set via `secrets_file` in `config.toml`
4. **config.toml values** -- not recommended for credentials

For cron jobs, either use the user secrets file or export environment variables in the crontab:

```cron
THISTLE_DB_DATABASE__USERNAME=myuser
THISTLE_DB_DATABASE__PASSWORD=mypassword
0 */4 * * * thistle-db -c /etc/thistle-db/config.toml ingest && thistle-db -c /etc/thistle-db/config.toml generate
```

## CLI Reference

```
thistle-db [-c CONFIG] COMMAND

Commands:
  init       Scaffold config.toml and ~/.config/thistle-db.toml
  init-db    Create the database schema (idempotent; --drop recreates from
             scratch, destroying all data — asks unless --yes)
  ingest     Ingest TLE/OMM files into the database
  generate   Generate output TLE/OMM files from the database
  get-tle    Print TLEs from the database to stdout
  dump       Export the entire database as re-ingestable TLE/OMM files

Options:
  -c, --config PATH   Path to config.toml
                      (default: $THISTLE_DB_CONFIG if set, else ./config.toml)
```

### `generate`

Incremental by default (see *Generate output files* above). Flags:

```bash
thistle-db generate                    # routine incremental run
thistle-db generate --all              # full rebuild of every output file
thistle-db generate --verify           # incremental run + integrity sweep
thistle-db generate --window-days 90   # override output.window_days
thistle-db generate --lookback-days 14 # override output.lookback_days
```

Use `--all` for the first run over historical data, after restoring a
backup, or after `generate` hasn't run for longer than the lookback.
`--verify` reconciles every object file against the database and rewrites
any that disagree (it reads all output files — schedule it weekly rather
than every run).

### `get-tle`

Query the database directly and print TLEs to stdout. The positional argument
is either a NORAD ID (alpha-5 compatible, e.g. `25544`, `00022`, `E5693`) or
an 8-digit date (`YYYYMMDD`):

```bash
# All TLEs for one satellite, ordered by epoch
thistle-db get-tle 25544
thistle-db get-tle E5693   # alpha-5 IDs work too (= 145693)

# Nearest TLE per satellite to 12:00 UTC on a date, within +/- 7 days
thistle-db get-tle 20260717

# Widen (or narrow) the search window
thistle-db get-tle 20260717 --days 3
```

Exits with status 1 if no TLEs match.

### `dump` — backups and migration

Export the whole database as re-ingestable files (a *logical* backup):

```bash
thistle-db dump /backups/tles-20260722
# -> writes /backups/tles-20260722.tle
#    and    /backups/tles-20260722.json  (only if OMM metadata exists)
```

Restore into any empty database — including a different dialect (SQLite →
MariaDB, etc.):

```bash
thistle-db -c new-config.toml init-db
thistle-db -c new-config.toml ingest /backups/tles-20260722.tle /backups/tles-20260722.json
thistle-db -c new-config.toml generate --all
```

The export is lossless for element sets (rows store the TLE line text
verbatim, and dedup is by that exact text), and the JSON carries the OMM
metadata in Space-Track form so `ingest` reattaches it to the same rows.
The `ingest_files` change-detection state is deliberately not exported —
it is a cache; the next scan rebuilds it.

The final `generate --all` matters: a restore resets every row's `created`
timestamp, which is what incremental generation keys off, so output files
must be rebuilt once from scratch. This dump/restore cycle is also the
supported schema-migration path — thistle-db deliberately has no migration
framework; schema-changing releases document this procedure in their
release notes.

For *physical* backups of a live server, prefer the native tools: a file
copy or `VACUUM INTO` for SQLite, `mariadb-dump` / `pg_dump` for the
server dialects.

## Configuration Reference

### `[database]`

| Field          | Default      | Description                                |
| -------------- | ------------ | ------------------------------------------ |
| `drivername`   | `"sqlite"`   | SQLAlchemy driver (`sqlite`, `mysql+pymysql`) |
| `name`         | `":memory:"` | Database name or file path                 |
| `host`         |              | Database host                              |
| `port`         |              | Database port                              |
| `username`     |              | Database username (prefer secrets file)    |
| `password`     |              | Database password (prefer secrets file)    |
| `secrets_file` |              | Path to a TOML file with username/password |

### `[[ingest.sources]]`

| Field     | Default   | Description                        |
| --------- | --------- | ---------------------------------- |
| `path`    |           | Directory to scan for files        |
| `pattern` | `"*.tle"` | Glob pattern for matching files    |

### `[output]`

| Field           | Default      | Description                                      |
| --------------- | ------------ | ------------------------------------------------ |
| `dir`           | `"./output"` | Output directory                                 |
| `window_days`   | `60`         | Date files: trailing epoch window rewritten each run |
| `lookback_days` | `7`          | Object files: newly created rows considered each run (must exceed the ingest cron cadence) |

### `[output.formats]`

| Field | Default | Description                 |
| ----- | ------- | --------------------------- |
| `tle` | `true`  | Generate .tle output files  |
| `omm` | `true`  | Generate .omm (CSV) output  |

### `[output.types]`

| Field          | Default | Description                                        |
| -------------- | ------- | -------------------------------------------------- |
| `date_files`   | `true`  | YYYYMMDD files with latest TLE per satellite       |
| `object_files` | `true`  | Per-satellite files with all TLEs ordered by epoch  |

### `[logging]`

| Field   | Default  | Description                                    |
| ------- | -------- | ---------------------------------------------- |
| `level` | `"INFO"` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

## Development

### Running tests

Tests live at the workspace root under `tests/thistle_db/` and are
parametrized to run against SQLite, MariaDB, and PostgreSQL. SQLite runs
unconditionally; the MariaDB and PostgreSQL backends are opt-in and managed
automatically by [testcontainers](https://testcontainers-python.readthedocs.io/)
— one container per test session, one throwaway database per test. No manual
`docker run` needed, just a running Docker daemon.

SQLite only:

```bash
uv run pytest tests/thistle_db
```

All backends (requires Docker):

```bash
THISTLE_DB_TEST_MARIADB=1 THISTLE_DB_TEST_POSTGRES=1 uv run pytest tests/thistle_db
```

The images default to `mariadb:11` and `postgres:16`; override with
`THISTLE_DB_MARIADB_IMAGE` / `THISTLE_DB_POSTGRES_IMAGE`.
