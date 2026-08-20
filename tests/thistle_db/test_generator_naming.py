"""Filename configuration: object ID rendering, extensions, date formats.

Synthetic TLEs (as in the incremental tests): sgp4 does not validate
checksums, so edited satnum fields are distinct valid records. E5693 is the
alpha-5 encoding of 145693.
"""

from sqlalchemy.orm import Session

from thistle_db.config import OutputConfig, OutputFile
from thistle_db.generator import generate
from thistle_db.ingest import ingest_tles

TLE_900 = (
    "1 00900U 59009A   25100.10000000  .00013443  00000-0  59558-3 0  9999",
    "2 00900  50.2761  24.4877 0093242 224.4518 134.8929 15.18021433735521",
)
TLE_ALPHA = (
    "1 E5693U 59009A   25100.10000000  .00013443  00000-0  59558-3 0  9999",
    "2 E5693  50.2761  24.4877 0093242 224.4518 134.8929 15.18021433735521",
)


def _object_config(outdir, **options) -> OutputConfig:
    return OutputConfig(
        files=[OutputFile(type="object", format="tle", dir=str(outdir), **options)]
    )


def test_object_id_int_unpadded(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900])
    generate(db_session, _object_config(tmp_path))
    assert (tmp_path / "900.tle").exists()


def test_object_id_int_zero_pad(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900])
    generate(db_session, _object_config(tmp_path, zero_pad=True))
    assert (tmp_path / "00900.tle").exists()
    assert not (tmp_path / "900.tle").exists()


def test_object_id_alpha5(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900, TLE_ALPHA])
    generate(db_session, _object_config(tmp_path, object_id="alpha5"))
    # Below 100000 alpha-5 is the zero-padded integer; above, letter + 4 digits.
    assert (tmp_path / "00900.tle").exists()
    assert (tmp_path / "E5693.tle").exists()
    assert not (tmp_path / "145693.tle").exists()


def test_custom_extension(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900])
    generate(db_session, _object_config(tmp_path, extension="txt"))
    assert (tmp_path / "900.txt").exists()
    assert not (tmp_path / "900.tle").exists()


def test_date_format(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900])
    config = OutputConfig(
        files=[
            OutputFile(
                type="date", format="tle", dir=str(tmp_path),
                date_format="%Y-%m-%d",
            )
        ]
    )
    generate(db_session, config)
    # 2025 day 100 = 2025-04-10
    assert (tmp_path / "2025-04-10.tle").exists()


def test_incremental_append_resolves_same_alpha5_file(
    db_session: Session, tmp_path
):
    """A second run must find the alpha-5-named file and leave it unchanged."""
    config = _object_config(tmp_path, object_id="alpha5")
    ingest_tles(db_session, [TLE_ALPHA])
    generate(db_session, config)
    before = (tmp_path / "E5693.tle").read_bytes()

    generate(db_session, config)
    assert (tmp_path / "E5693.tle").read_bytes() == before
    assert list(tmp_path.iterdir()) == [tmp_path / "E5693.tle"]


def test_verify_restores_zero_padded_file(db_session: Session, tmp_path):
    config = _object_config(tmp_path, zero_pad=True)
    ingest_tles(db_session, [TLE_900])
    generate(db_session, config)
    (tmp_path / "00900.tle").unlink()

    generate(db_session, config, lookback_days=0, verify=True)
    assert (tmp_path / "00900.tle").read_text().splitlines() == [*TLE_900]


def test_parse_object_stem():
    out_int = OutputFile(type="object", format="tle")
    assert out_int.parse_object_stem("900") == 900
    assert out_int.parse_object_stem("00900") == 900
    assert out_int.parse_object_stem("20250410") is None  # date stem
    assert out_int.parse_object_stem("E5693") is None

    out_a5 = OutputFile(type="object", format="tle", object_id="alpha5")
    assert out_a5.parse_object_stem("E5693") == 145693
    assert out_a5.parse_object_stem("00900") == 900
    assert out_a5.parse_object_stem("20250410") is None
    assert out_a5.parse_object_stem("I1234") is None  # I is invalid in alpha-5
