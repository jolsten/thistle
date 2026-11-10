"""Output maintenance: append/merge, date upsert, naming, rebuild parity."""

import json
import pathlib

from sqlalchemy.orm import Session

from .conftest import TLES, engine_ingest, make_settings
from thistle_db.config import OutputFile
from thistle_db.outputs import rebuild_outputs
from thistle_db.record import epoch_from_lines

DATA = pathlib.Path(__file__).parent / "data"
ONE = json.loads((DATA / "one.json").read_text())

# TLES[0] and TLES[1] are the same object (22) at consecutive epochs on
# consecutive days (25107.9 / 25108.2); TLES[2] is another object.
EARLY, LATE = TLES[0], TLES[1]


def _write_tle(path, *pairs):
    path.write_text("".join(f"{l1}\n{l2}\n" for l1, l2 in pairs))
    return path


def _pairs(path: pathlib.Path) -> list[tuple[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return list(zip(lines[::2], lines[1::2]))


def test_object_file_appends_in_epoch_order(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d1.tle", EARLY)]
    )
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d2.tle", LATE)]
    )
    obj = tmp_path / "objects" / "22.tle"
    assert _pairs(obj) == [EARLY, LATE]


def test_late_delivery_merges_into_epoch_order(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d2.tle", LATE)]
    )
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d1.tle", EARLY)]
    )
    obj = tmp_path / "objects" / "22.tle"
    assert _pairs(obj) == [EARLY, LATE]


def test_redelivery_never_duplicates(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    path = _write_tle(tmp_path / "d.tle", EARLY, LATE)
    engine_ingest(db_session, settings, files=[path])
    engine_ingest(db_session, settings, files=[path])
    assert _pairs(tmp_path / "objects" / "22.tle") == [EARLY, LATE]


def test_torn_tail_heals_on_merge(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d2.tle", LATE)]
    )
    obj = tmp_path / "objects" / "22.tle"
    with open(obj, "a") as f:
        f.write("1 00022U 59009A   25")  # torn mid-record (crash artifact)
    # An older record forces the merge path, which heals the tear.
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "d1.tle", EARLY)]
    )
    assert _pairs(obj) == [EARLY, LATE]


def test_date_files_keep_latest_per_object(db_session: Session, tmp_path):
    settings = make_settings(tmp_path)
    path = _write_tle(tmp_path / "d.tle", EARLY, LATE, TLES[2])
    engine_ingest(db_session, settings, files=[path])
    dates_dir = tmp_path / "dates"
    files = sorted(p.name for p in dates_dir.iterdir())
    assert files == ["20250417.tle", "20250418.tle"]
    # Each date file holds the day's winner per object, sorted by norad.
    day1 = _pairs(dates_dir / "20250417.tle")
    assert day1 == [EARLY]
    day2 = _pairs(dates_dir / "20250418.tle")
    assert day2 == [LATE, TLES[2]]


def test_date_upsert_ignores_stale_redelivery(db_session: Session, tmp_path):
    # EARLY and LATE are different dates; craft two same-day records by
    # using LATE and TLES[2] (both 25108) for distinct objects, then a
    # stale extra delivery of LATE's day-mate EARLY cannot regress day2.
    settings = make_settings(tmp_path)
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "a.tle", LATE)]
    )
    engine_ingest(
        db_session,
        settings,
        files=[_write_tle(tmp_path / "b.tle", EARLY)],  # older, other date
    )
    day2 = _pairs(tmp_path / "dates" / "20250418.tle")
    assert day2 == [LATE]


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
    report = engine_ingest(db_session, settings, files=[path])
    assert report.records_inserted == 1
    assert report.lineless_skipped == 1
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
    engine_ingest(db_session, settings, files=[DATA / "one.json"])
    text = (tmp_path / "full" / "11.csv").read_text()
    header = text.splitlines()[0]
    assert header.startswith("RECORD_HASH,")
    assert "TLE_LINE1" in header
    assert ONE["TLE_LINE1"] in text  # verbatim lines survive the projection


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


def test_rebuild_matches_incremental(db_session: Session, tmp_path):
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
    # Out-of-order delivery exercises both append and merge paths.
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "b.tle", LATE, TLES[2])]
    )
    engine_ingest(
        db_session, settings, files=[_write_tle(tmp_path / "a.tle", EARLY)]
    )
    engine_ingest(db_session, settings, files=[DATA / "one.json"])

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
    rebuilt = snapshot()
    assert rebuilt == incremental


def test_epochs_are_exact():
    # The tail guard and date upsert compare file-parsed epochs against
    # GPRecord epochs; both must come from the same sgp4 parse.
    from thistle_db.record import from_tle

    for line1, line2 in TLES:
        assert epoch_from_lines(line1, line2) == from_tle(line1, line2).epoch
