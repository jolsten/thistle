"""The files storage backend: config validation, single-pass ingest into the
object-file store, derived date files, dedup/merge semantics, and rebuild.

The load-bearing properties under test mirror the spec:

- per-file dedup is globally equivalent to the database's unique index
  (the dedup key encodes the NORAD ID and epoch);
- ingest is idempotent and a re-ingest never rewrites an untouched file;
- late deliveries merge (sorted, deduplicated, atomic rewrite);
- the tie rule — equal epochs keep ingestion order in object files, and
  date files treat the last-ingested as "latest" — so incremental updates
  and `rebuild_derived` agree;
- source-file state commits only at flush, never before the records are on
  disk.
"""

import datetime
import pathlib

import pytest
from sqlalchemy import select

from .conftest import TLES
from thistle_db.config import Settings
from thistle_db.filestore import (
    FileStore,
    dump_store,
    nearest_tles_for_date,
    open_state_session,
    read_object_tles,
    rebuild_derived,
    store_entry,
)
from thistle_db.ingest import FileStatus, ingest_source_file, ingest_sources
from thistle_db.model import IngestFile
from thistle_db.reader import read_tle

# TLES holds three records: object 22 on 2025-04-17 and 2025-04-18, and
# object 81069 on 2025-04-18 (see conftest).
DATE_1, DATE_2 = datetime.date(2025, 4, 17), datetime.date(2025, 4, 18)

# TLES[0] with one digit of the mean anomaly changed: a textually distinct
# elset for the same object with the *same epoch* (line1 unchanged; sgp4
# does not validate the line-2 checksum). The tie-rule fixture.
TLE_TIE = (
    TLES[0][0],
    "2 00022  50.2761  24.4877 0093242 224.4518 134.8930 15.18021433735521",
)


def make_settings(tmp_path: pathlib.Path, **overrides) -> Settings:
    """A files-backend Settings over tmp_path (serial writes for determinism)."""
    kwargs = dict(
        storage={
            "backend": "files",
            "state": str(tmp_path / "state.sqlite"),
        },
        output={
            "files": [
                {"type": "object", "format": "tle", "dir": str(tmp_path / "objects")},
                {"type": "date", "format": "tle", "dir": str(tmp_path / "daily")},
            ],
            "write_workers": 1,
        },
        ingest={"sources": [{"path": str(tmp_path / "incoming"), "pattern": "*.tle"}]},
    )
    for key, value in overrides.items():
        kwargs[key] = {**kwargs[key], **value}
    return Settings(**kwargs)


def write_source(tmp_path: pathlib.Path, name: str, tles) -> pathlib.Path:
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    path = incoming / name
    path.write_text("".join(f"{l1}\n{l2}\n" for l1, l2 in tles))
    return path


def run_ingest(settings: Settings, **store_kwargs) -> tuple[int, dict]:
    """One full files-backend ingest run (scan + flush). Returns
    (total buffered, {norad or date -> pairs} of every output file)."""
    session, engine = open_state_session(settings)
    try:
        store = FileStore(session, settings.output, **store_kwargs)
        total = ingest_sources(session, settings.ingest.sources, store=store)
        store.close()
    finally:
        session.close()
        engine.dispose()
    return total, read_outputs(settings)


def read_outputs(settings: Settings) -> dict:
    out = {}
    for entry in settings.output.files:
        directory = pathlib.Path(entry.dir)
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix == ".tle":
                out[path.name] = list(read_tle(path))
    return out


class TestConfigValidation:
    def test_omm_output_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="format='omm'"):
            make_settings(
                tmp_path,
                output={
                    "files": [
                        {"type": "object", "format": "tle", "dir": str(tmp_path)},
                        {"type": "date", "format": "omm", "dir": str(tmp_path)},
                    ]
                },
            )

    def test_store_entry_required(self, tmp_path):
        with pytest.raises(ValueError, match="canonical store"):
            make_settings(
                tmp_path,
                output={
                    "files": [{"type": "date", "format": "tle", "dir": str(tmp_path)}]
                },
            )

    def test_database_backend_still_allows_omm(self, tmp_path):
        settings = make_settings(
            tmp_path,
            storage={"backend": "database"},
            output={
                "files": [{"type": "date", "format": "omm", "dir": str(tmp_path)}]
            },
        )
        assert settings.storage.backend == "database"

    def test_store_entry_is_first_object_tle(self, tmp_path):
        settings = make_settings(tmp_path)
        entry = store_entry(settings.output)
        assert (entry.type, entry.format) == ("object", "tle")


class TestIngest:
    def test_object_and_date_files(self, tmp_path):
        settings = make_settings(tmp_path)
        write_source(tmp_path, "feed.tle", TLES)
        total, outputs = run_ingest(settings)

        assert total == 3
        # Object files: full history, epoch order.
        assert outputs["22.tle"] == [TLES[0], TLES[1]]
        assert outputs["81069.tle"] == [TLES[2]]
        # Date files: latest per object that day, ordered by NORAD ID.
        assert outputs["20250417.tle"] == [TLES[0]]
        assert outputs["20250418.tle"] == [TLES[1], TLES[2]]

    def test_reingest_is_idempotent_and_writes_nothing(self, tmp_path):
        settings = make_settings(tmp_path)
        write_source(tmp_path, "feed.tle", TLES)
        run_ingest(settings)

        obj = tmp_path / "objects" / "22.tle"
        before = obj.stat().st_mtime_ns

        total, outputs = run_ingest(settings)
        assert total == 0  # unchanged source file skipped via state DB
        assert outputs["22.tle"] == [TLES[0], TLES[1]]
        assert obj.stat().st_mtime_ns == before

    def test_duplicate_records_in_renamed_file_are_dropped(self, tmp_path):
        # Same records via a different filename: state tracking can't skip
        # it, so on-disk dedup (the merge path's hash check) must.
        settings = make_settings(tmp_path)
        write_source(tmp_path, "feed.tle", TLES)
        run_ingest(settings)
        write_source(tmp_path, "redelivery.tle", TLES)
        _, outputs = run_ingest(settings)

        assert outputs["22.tle"] == [TLES[0], TLES[1]]
        assert outputs["81069.tle"] == [TLES[2]]

    def test_late_delivery_merges_sorted(self, tmp_path):
        settings = make_settings(tmp_path)
        write_source(tmp_path, "later.tle", [TLES[1]])
        run_ingest(settings)
        write_source(tmp_path, "earlier.tle", [TLES[0]])  # epoch before tail
        _, outputs = run_ingest(settings)

        assert outputs["22.tle"] == [TLES[0], TLES[1]]
        # The late earlier-epoch record becomes its own date's file, and
        # must not displace the existing later-epoch winner on DATE_2.
        assert outputs["20250417.tle"] == [TLES[0]]
        assert outputs["20250418.tle"] == [TLES[1]]

    def test_epoch_tie_keeps_ingestion_order(self, tmp_path):
        settings = make_settings(tmp_path)
        write_source(tmp_path, "a.tle", [TLES[0]])
        run_ingest(settings)
        write_source(tmp_path, "b.tle", [TLE_TIE])
        _, outputs = run_ingest(settings)

        # Object file: both records, ingestion order among the equal epochs.
        assert outputs["22.tle"] == [TLES[0], TLE_TIE]
        # Date file: newest-ingested wins the tie (the database's id DESC).
        assert outputs["20250417.tle"] == [TLE_TIE]

    def test_torn_tail_self_heals_on_merge(self, tmp_path):
        settings = make_settings(tmp_path)
        write_source(tmp_path, "a.tle", [TLES[0]])
        run_ingest(settings)

        obj = tmp_path / "objects" / "22.tle"
        with open(obj, "a") as f:
            f.write(TLES[1][0][:40])  # torn append: partial line, no newline

        write_source(tmp_path, "b.tle", [TLES[1]])
        _, outputs = run_ingest(settings)
        assert outputs["22.tle"] == [TLES[0], TLES[1]]
        assert obj.read_bytes().endswith(b"\n")

    def test_flush_budget_multiple_flushes(self, tmp_path):
        settings = make_settings(tmp_path)
        write_source(tmp_path, "feed.tle", TLES)
        total, outputs = run_ingest(settings, flush_records=1)

        assert total == 3
        assert outputs["22.tle"] == [TLES[0], TLES[1]]
        assert outputs["20250418.tle"] == [TLES[1], TLES[2]]

    def test_within_run_dedup(self, tmp_path):
        settings = make_settings(tmp_path)
        write_source(tmp_path, "feed.tle", [TLES[0], TLES[0], TLES[1]])
        total, outputs = run_ingest(settings)
        assert total == 2
        assert outputs["22.tle"] == [TLES[0], TLES[1]]

    def test_omm_input_stores_lines(self, tmp_path):
        settings = make_settings(
            tmp_path,
            ingest={
                "sources": [{"path": str(tmp_path / "incoming"), "pattern": "*.json"}]
            },
        )
        fixture = pathlib.Path(__file__).parent / "data" / "one.json"
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        (incoming / "one.json").write_bytes(fixture.read_bytes())

        total, outputs = run_ingest(settings)
        # The record's TLE lines are stored (metadata discarded with a
        # warning); every written file holds at least one full record.
        assert total >= 1
        assert outputs and all(pairs for pairs in outputs.values())

    def test_file_state_commits_only_at_flush(self, tmp_path):
        settings = make_settings(tmp_path)
        path = write_source(tmp_path, "feed.tle", TLES)

        session, engine = open_state_session(settings)
        try:
            store = FileStore(session, settings.output)
            status, _ = ingest_source_file(session, path, store=store)
            assert status == FileStatus.INGESTED
            # Records are only buffered: the file must not be marked
            # ingested yet, or a crash here would lose them silently.
            assert session.execute(select(IngestFile)).all() == []
            store.close()
            assert len(session.execute(select(IngestFile)).all()) == 1
        finally:
            session.close()
            engine.dispose()


class TestDerivedOutputs:
    def two_object_outputs(self, tmp_path) -> Settings:
        return make_settings(
            tmp_path,
            output={
                "files": [
                    {
                        "type": "object",
                        "format": "tle",
                        "dir": str(tmp_path / "objects"),
                    },
                    {
                        "type": "object",
                        "format": "tle",
                        "dir": str(tmp_path / "alpha5"),
                        "object_id": "alpha5",
                    },
                    {"type": "date", "format": "tle", "dir": str(tmp_path / "daily")},
                ],
                "write_workers": 1,
            },
        )

    def test_derived_object_output_written_in_same_pass(self, tmp_path):
        settings = self.two_object_outputs(tmp_path)
        write_source(tmp_path, "feed.tle", TLES)
        run_ingest(settings)
        assert list(read_tle(tmp_path / "alpha5" / "00022.tle")) == [
            TLES[0],
            TLES[1],
        ]

    def test_missing_derived_file_healed_from_store(self, tmp_path):
        settings = self.two_object_outputs(tmp_path)
        write_source(tmp_path, "a.tle", [TLES[0]])
        run_ingest(settings)
        (tmp_path / "alpha5" / "00022.tle").unlink()

        # The next delta for the object copies full history from the store.
        write_source(tmp_path, "b.tle", [TLES[1]])
        run_ingest(settings)
        assert list(read_tle(tmp_path / "alpha5" / "00022.tle")) == [
            TLES[0],
            TLES[1],
        ]

    def test_rebuild_derived_matches_incremental(self, tmp_path):
        settings = self.two_object_outputs(tmp_path)
        write_source(tmp_path, "feed.tle", TLES + [TLE_TIE])
        _, incremental = run_ingest(settings)

        # Wreck the derived outputs, keep the store.
        for name in ("alpha5", "daily"):
            for path in (tmp_path / name).iterdir():
                path.write_text("stale garbage\n")

        rebuild_derived(settings.output, flush_records=1)
        assert read_outputs(settings) == incremental

    def test_rebuild_with_store_only_is_noop(self, tmp_path):
        settings = make_settings(
            tmp_path,
            output={
                "files": [
                    {
                        "type": "object",
                        "format": "tle",
                        "dir": str(tmp_path / "objects"),
                    }
                ],
                "write_workers": 1,
            },
        )
        write_source(tmp_path, "feed.tle", TLES)
        run_ingest(settings)
        rebuild_derived(settings.output)  # nothing derived; must not raise


class TestReads:
    @pytest.fixture
    def populated(self, tmp_path) -> Settings:
        settings = make_settings(tmp_path)
        write_source(tmp_path, "feed.tle", TLES)
        run_ingest(settings)
        return settings

    def test_read_object_tles(self, populated):
        assert read_object_tles(populated.output, 22) == [TLES[0], TLES[1]]
        assert read_object_tles(populated.output, 99999) == []

    def test_nearest_tles_for_date(self, populated):
        pairs = nearest_tles_for_date(populated.output, DATE_2, 7.0)
        # Object 22: the DATE_2 record (epoch .23 of day 108) is nearer to
        # DATE_2 noon than the DATE_1 record; 81069 has only one record.
        assert pairs == [TLES[1], TLES[2]]

    def test_nearest_requires_date_output(self, tmp_path):
        settings = make_settings(
            tmp_path,
            output={
                "files": [
                    {"type": "object", "format": "tle", "dir": str(tmp_path / "o")}
                ],
                "write_workers": 1,
            },
        )
        with pytest.raises(ValueError, match="date"):
            nearest_tles_for_date(settings.output, DATE_2, 7.0)

    def test_dump_store(self, populated, tmp_path):
        target = tmp_path / "dump.tle"
        count = dump_store(populated.output, target)
        assert count == 3
        assert list(read_tle(target)) == [TLES[0], TLES[1], TLES[2]]

    def test_get_tles_api_uses_files_backend(self, populated):
        from thistle_db.api import get_tles

        assert get_tles(22, populated) == [TLES[0], TLES[1]]
