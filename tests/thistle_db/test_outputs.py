"""Output rendering: cursor-scoped updates, rebuild parity, naming."""

import json
import pathlib

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .conftest import TLES, engine_ingest, make_settings
from thistle_db.config import OutputFile
from thistle_db.model import TLE
from thistle_db.outputs import (
    MissingCursorError,
    load_cursors,
    output_lag,
    rebuild_outputs,
    update_outputs,
)
from thistle_db.record import epoch_from_lines

DATA = pathlib.Path(__file__).parent / "data"
ONE = json.loads((DATA / "one.json").read_text())

# TLES[0] and TLES[1] are the same object (22) at consecutive epochs on
# consecutive days (25107.9 / 25108.2); TLES[2] is another object (81069).
EARLY, LATE = TLES[0], TLES[1]


def _write_tle(path, *pairs):
    path.write_text("".join(f"{l1}\n{l2}\n" for l1, l2 in pairs))
    return path


def _pairs(path: pathlib.Path) -> list[tuple[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return list(zip(lines[::2], lines[1::2]))


def _bootstrap(db_session, settings) -> None:
    """Initialize the output cursors (rebuild on the possibly-empty store)."""
    rebuild_outputs(db_session, settings.output)


def _update(db_session, settings):
    return update_outputs(db_session, settings.output)


def _max_id(db_session) -> int:
    return db_session.execute(select(func.max(TLE.id))).scalar_one() or 0


# --- store-only ingest and the cursor lifecycle -------------------------------


def test_ingest_writes_no_output_files(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d.tle", EARLY)]
    )
    assert not (tmp_path / "objects").exists()
    assert not (tmp_path / "dates").exists()


def test_update_before_rebuild_errors(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d.tle", EARLY)]
    )
    with pytest.raises(MissingCursorError):
        update_outputs(db_session, settings.output)


def test_rebuild_sets_cursors_then_update_is_noop(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d.tle", EARLY, LATE)]
    )
    rebuild_outputs(db_session, settings.output)
    cursors = load_cursors(db_session, settings.output.files)
    assert set(cursors.values()) == {_max_id(db_session)}
    result = _update(db_session, settings)
    assert result.up_to_date
    assert result.written == {}


def test_cursor_advances_and_lag_reports(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    pending, missing = output_lag(db_session, settings.output.files)
    assert len(missing) == 2  # nothing initialized yet

    _bootstrap(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d.tle", EARLY)]
    )
    pending, missing = output_lag(db_session, settings.output.files)
    assert missing == [] and pending > 0

    result = _update(db_session, settings)
    assert not result.up_to_date
    pending, missing = output_lag(db_session, settings.output.files)
    assert (pending, missing) == (0, [])
    cursors = load_cursors(db_session, settings.output.files)
    assert set(cursors.values()) == {_max_id(db_session)}


# --- scoped rendering ----------------------------------------------------------


def test_update_renders_object_history_in_epoch_order(
    db_session: Session, tmp_path
):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d1.tle", EARLY)]
    )
    _update(db_session, settings)
    obj = tmp_path / "objects" / "22.tle"
    assert _pairs(obj) == [EARLY]

    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d2.tle", LATE)]
    )
    _update(db_session, settings)
    assert _pairs(obj) == [EARLY, LATE]


def test_late_delivery_rerenders_in_epoch_order(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d2.tle", LATE)]
    )
    _update(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d1.tle", EARLY)]
    )
    _update(db_session, settings)
    assert _pairs(tmp_path / "objects" / "22.tle") == [EARLY, LATE]


def test_redelivery_never_duplicates(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    path = _write_tle(tmp_path / "d.tle", EARLY, LATE)
    engine_ingest(db_session, settings, files=[path])
    _update(db_session, settings)
    engine_ingest(db_session, settings, files=[path])
    _update(db_session, settings)
    assert _pairs(tmp_path / "objects" / "22.tle") == [EARLY, LATE]


def test_touched_files_are_healed(db_session: Session, tmp_path):
    """Whole-file render: any damage to a file is gone the next time its
    object dirties — touching is healing."""
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d2.tle", LATE)]
    )
    _update(db_session, settings)
    obj = tmp_path / "objects" / "22.tle"
    with open(obj, "a") as f:
        f.write("1 00022U 59009A   25")  # torn garbage (crash artifact)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d1.tle", EARLY)]
    )
    _update(db_session, settings)
    assert _pairs(obj) == [EARLY, LATE]


def test_date_files_keep_latest_per_object(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    path = _write_tle(tmp_path / "d.tle", EARLY, LATE, TLES[2])
    engine_ingest(db_session, settings, files=[path])
    _update(db_session, settings)
    dates_dir = tmp_path / "dates"
    files = sorted(p.name for p in dates_dir.iterdir())
    assert files == ["20250417.tle", "20250418.tle"]
    # Each date file holds the day's winner per object, sorted by norad.
    assert _pairs(dates_dir / "20250417.tle") == [EARLY]
    assert _pairs(dates_dir / "20250418.tle") == [LATE, TLES[2]]


def test_stale_redelivery_cannot_regress_date_files(
    db_session: Session, tmp_path
):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "a.tle", LATE)]
    )
    _update(db_session, settings)
    engine_ingest(
        db_session,
        settings,
        files=[_write_tle(tmp_path / "b.tle", EARLY)],  # older, other date
    )
    _update(db_session, settings)
    assert _pairs(tmp_path / "dates" / "20250418.tle") == [LATE]
    assert _pairs(tmp_path / "dates" / "20250417.tle") == [EARLY]


def test_scoped_update_touches_only_dirty_files(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "a.tle", EARLY)]
    )
    first = _update(db_session, settings)
    assert sum(first.written.values()) == 2  # 22.tle + 20250417.tle
    obj22 = (tmp_path / "objects" / "22.tle").read_bytes()
    day17 = (tmp_path / "dates" / "20250417.tle").read_bytes()

    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "b.tle", TLES[2])]
    )
    second = _update(db_session, settings)
    assert second.dirty_objects == 1 and second.dirty_days == 1
    assert sum(second.written.values()) == 2  # 81069.tle + 20250418.tle
    # Clean files were not rewritten.
    assert (tmp_path / "objects" / "22.tle").read_bytes() == obj22
    assert (tmp_path / "dates" / "20250417.tle").read_bytes() == day17


def test_lineless_records_skip_tle_outputs(db_session: Session, tmp_path):
    lineless = {k: v for k, v in ONE.items() if not k.startswith("TLE_LINE")}
    path = tmp_path / "lineless.json"
    path.write_text(json.dumps([lineless]))
    settings = make_settings(
        tmp_path,
        outputs=[
            OutputFile(type="object", format="tle", dir=str(tmp_path / "objects")),
            OutputFile(
                type="object", format="omm-csv", dir=str(tmp_path / "omm")
            ),
        ],
    )
    _bootstrap(db_session, settings)
    report = engine_ingest(db_session, settings, files=[path])
    assert report.records_inserted == 1
    result = _update(db_session, settings)
    assert result.lineless_skipped == 1
    assert not list((tmp_path / "objects").iterdir())  # no tle file at all
    omm_files = list((tmp_path / "omm").iterdir())
    assert len(omm_files) == 1  # but the CSV projection has it
    assert "VANGUARD 2" in omm_files[0].read_text()


def test_omm_csv_object_output_full_columns(db_session: Session, tmp_path):
    settings = make_settings(
        tmp_path,
        outputs=[
            OutputFile(
                type="object",
                format="omm-csv",
                columns="full",
                dir=str(tmp_path / "full"),
            ),
        ],
    )
    _bootstrap(db_session, settings)
    engine_ingest(db_session, settings, files=[DATA / "one.json"])
    _update(db_session, settings)
    text = (tmp_path / "full" / "11.csv").read_text()
    header = text.splitlines()[0]
    assert header.startswith("RECORD_HASH,")
    assert "TLE_LINE1" in header
    assert ONE["TLE_LINE1"] in text  # verbatim lines survive the projection


# --- resumability ----------------------------------------------------------------


def test_crash_mid_update_leaves_cursors_and_resumes(
    db_session: Session, tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    before = load_cursors(db_session, settings.output.files)
    engine_ingest(
        db_session,
        settings,
        files=[_write_tle(tmp_path / "d.tle", EARLY, LATE, TLES[2])],
    )

    def boom(entry, norad, records):
        raise RuntimeError("simulated crash mid-render")

    monkeypatch.setattr("thistle_db.outputs._render_object_file", boom)
    with pytest.raises(RuntimeError):
        update_outputs(db_session, settings.output)
    assert load_cursors(db_session, settings.output.files) == before
    monkeypatch.undo()

    result = _update(db_session, settings)
    assert not result.up_to_date
    assert _pairs(tmp_path / "objects" / "22.tle") == [EARLY, LATE]
    assert _pairs(tmp_path / "dates" / "20250418.tle") == [LATE, TLES[2]]


def test_stray_tmp_files_are_swept(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    _bootstrap(db_session, settings)
    stray = tmp_path / "objects" / "22.tle.tmp"
    stray.write_text("leftover of a killed render")
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d.tle", EARLY)]
    )
    _update(db_session, settings)
    assert not stray.exists()


# --- the core equivalence property ----------------------------------------------


def test_update_outputs_matches_rebuild(db_session: Session, tmp_path):
    """Every file update-outputs produces is byte-identical to a rebuild's."""
    settings = make_settings(
        tmp_path,
        outputs=[
            OutputFile(type="object", format="tle", dir=str(tmp_path / "objects")),
            OutputFile(type="date", format="tle", dir=str(tmp_path / "dates")),
            OutputFile(
                type="object", format="omm-csv", dir=str(tmp_path / "omm_obj")
            ),
            OutputFile(
                type="date", format="omm-csv", dir=str(tmp_path / "omm_date")
            ),
        ],
    )
    _bootstrap(db_session, settings)
    # Out-of-order delivery across runs exercises re-render of clean files.
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "b.tle", LATE, TLES[2])]
    )
    _update(db_session, settings)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "a.tle", EARLY)]
    )
    _update(db_session, settings)
    engine_ingest(db_session, settings, files=[DATA / "one.json"])
    _update(db_session, settings)

    def snapshot() -> dict[str, bytes]:
        out = {}
        for sub in ("objects", "dates", "omm_obj", "omm_date"):
            for p in sorted((tmp_path / sub).iterdir()):
                out[f"{sub}/{p.name}"] = p.read_bytes()
        return out

    incremental = snapshot()
    for sub in ("objects", "dates", "omm_obj", "omm_date"):
        for p in (tmp_path / sub).iterdir():
            p.unlink()

    rebuild_outputs(db_session, settings.output)
    assert snapshot() == incremental


# --- naming (pure config, no store) ----------------------------------------------


def test_object_file_naming_round_trip():
    entry = OutputFile(
        type="object",
        format="tle",
        object_id="alpha5",
        filename="tle_{id}{ext}",
        extension=".txt",
    )
    assert entry.object_path(145693).name == "tle_E5693.txt"
    assert entry.parse_object_name("tle_E5693.txt") == 145693
    assert entry.parse_object_name("tle_25544.txt") == 25544
    # Beyond alpha-5 range: integer-stem fallback, still parseable.
    assert entry.object_path(400000).name == "tle_400000.txt"
    assert entry.parse_object_name("tle_400000.txt") == 400000
    # 8-digit date stems never look like object files.
    assert entry.parse_object_name("tle_20260327.txt") is None


def test_zero_pad_and_glob():
    entry = OutputFile(type="object", format="tle", zero_pad=True)
    assert entry.object_path(900).name == "00900.tle"
    assert entry.parse_object_name("00900.tle") == 900
    assert entry.object_glob() == "*.tle"


def test_epochs_are_exact():
    # Renders compare store epochs against file-derived expectations in
    # verify and in tests; both must come from the same sgp4 parse.
    from thistle_db.record import from_tle

    for line1, line2 in TLES:
        assert epoch_from_lines(line1, line2) == from_tle(line1, line2).epoch
