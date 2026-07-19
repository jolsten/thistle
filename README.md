# thistle monorepo

A uv workspace containing two installable packages for working with satellite
orbital element data:

| Package | PyPI | Description |
|---|---|---|
| [thistle](packages/thistle/) | `pip install thistle` | Satellite orbit propagation and data generation with automatic TLE switching. Includes the `thistle` CLI (`pip install 'thistle[cli]'`). |
| [thistle-db](packages/thistle-db/) | `pip install thistle-db` | TLE/OMM database management: ingest element sets into SQLite/MariaDB/PostgreSQL and generate organized output files. Provides the `thistle-db` CLI. |

The packages compose: installing `thistle[db]` lets the `thistle` CLI resolve
NORAD catalog IDs directly from a thistle-db database when no matching file is
found in `THISTLE_TLE_DIR`.

## Development

```bash
uv sync --all-packages --all-extras --dev
uv run pytest                  # full suite (both packages + integration)
uv run pytest tests/thistle    # one package's tests
```

Tests live in `tests/{thistle,thistle_db,integration}/` at the workspace root.

## Releases

Each package is versioned independently from git tags via hatch-vcs:

- `thistle-vX.Y.Z` releases `thistle`
- `thistle-db-vX.Y.Z` releases `thistle-db`

Publishing to PyPI happens via the release workflow when a GitHub release is
created from one of those tags.

## History

`thistle-db` was previously developed as
[orbitable](https://github.com/jolsten/orbitable); it was renamed and merged
into this repository as a snapshot, and its pre-merge history remains in the
archived repo.
