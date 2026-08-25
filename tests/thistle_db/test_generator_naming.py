"""Filename configuration: object ID rendering, extensions, date formats.

Synthetic TLEs (as in the incremental tests): sgp4 does not validate
checksums, so edited satnum fields are distinct valid records. E5693 is the
alpha-5 encoding of 145693.
"""

import re

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from thistle_db.config import OutputConfig, OutputFile
from thistle_db.generator import generate, verify_object_files
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


def test_parse_object_name():
    out_int = OutputFile(type="object", format="tle")
    assert out_int.parse_object_name("900.tle") == 900
    assert out_int.parse_object_name("00900.tle") == 900
    assert out_int.parse_object_name("20250410.tle") is None  # date file
    assert out_int.parse_object_name("E5693.tle") is None
    assert out_int.parse_object_name("900.omm") is None  # another output's file

    out_a5 = OutputFile(type="object", format="tle", object_id="alpha5")
    assert out_a5.parse_object_name("E5693.tle") == 145693
    assert out_a5.parse_object_name("00900.tle") == 900
    assert out_a5.parse_object_name("20250410.tle") is None
    assert out_a5.parse_object_name("I1234.tle") is None  # I invalid in alpha-5


# ---------------------------------------------------------------------------
# Filename templates
# ---------------------------------------------------------------------------


def test_object_filename_template(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900])
    generate(db_session, _object_config(tmp_path, filename="tle_{id}.txt"))
    assert (tmp_path / "tle_900.txt").exists()


def test_template_composes_with_id_rendering(db_session: Session, tmp_path):
    """The template arranges; object_id/zero_pad/extension still render."""
    ingest_tles(db_session, [TLE_900])
    config = _object_config(
        tmp_path, filename="{id}-{format}{ext}", zero_pad=True, extension="dat"
    )
    generate(db_session, config)
    assert (tmp_path / "00900-tle.dat").exists()


def test_date_filename_template(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900])
    config = OutputConfig(
        files=[
            OutputFile(
                type="date",
                format="tle",
                dir=str(tmp_path),
                filename="{date}_gp{ext}",
                date_format="%Y-%m-%d",
            )
        ]
    )
    generate(db_session, config)
    assert (tmp_path / "2025-04-10_gp.tle").exists()


def test_templated_output_round_trips_through_parse_and_glob():
    """The orphan scan's inverse must track the forward rendering."""
    out = OutputFile(
        type="object", format="tle", filename="tle_{id}.txt", object_id="alpha5"
    )
    name = out.object_path(145693).name
    assert name == "tle_E5693.txt"
    assert out.parse_object_name(name) == 145693
    assert out.object_glob() == "tle_*.txt"
    # A date file sharing the directory must not look like an object file.
    assert out.parse_object_name("tle_20250410.txt") is None


def test_templated_orphans_are_detected(db_session: Session, tmp_path):
    ingest_tles(db_session, [TLE_900])
    config = _object_config(tmp_path, filename="tle_{id}.txt")
    generate(db_session, config)

    orphan = tmp_path / "tle_54321.txt"
    orphan.write_text("stale\n")
    verify_object_files(db_session, config.files)

    # Reported, not deleted — verify never removes files it did not write.
    assert orphan.exists()
    assert (tmp_path / "tle_900.txt").exists()


@pytest.mark.parametrize(
    "kind,template,message",
    [
        ("object", "fixed.tle", "must contain {id}"),
        ("date", "fixed.tle", "must contain {date}"),
        ("object", "{date}{ext}", "unknown placeholder {date}"),
        ("date", "{nope}{date}", "unknown placeholder {nope}"),
        ("object", "sub/{id}.tle", "must name a file"),
        ("object", "..{id}.tle", "must name a file"),
        ("object", "{id:>8}.tle", "may not carry a format spec"),
        ("object", "   ", "empty"),
    ],
)
def test_bad_templates_fail_at_config_load(kind, template, message):
    """Loudly, before generate has written a partial output tree."""
    with pytest.raises(ValidationError, match=re.escape(message)):
        OutputFile(type=kind, format="tle", filename=template)
