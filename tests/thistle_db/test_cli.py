"""CLI surface: command wiring, exit codes, stdout discipline."""

import json
import pathlib

import pytest
from typer.testing import CliRunner

from .conftest import TLES
from thistle_db.cli import app

runner = CliRunner()

L1, L2 = TLES[0]


def write_config(tmp_path: pathlib.Path, extra: str = "") -> pathlib.Path:
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[database]
drivername = "sqlite"
name = "{(tmp_path / 'cli.db').as_posix()}"

[[ingest.sources]]
path = "{incoming.as_posix()}"
pattern = "*.tle"

[[output.files]]
type = "object"
format = "tle"
dir = "{(tmp_path / 'objects').as_posix()}"

[[output.files]]
type = "date"
format = "tle"
dir = "{(tmp_path / 'dates').as_posix()}"
{extra}
"""
    )
    return config


def invoke(config, *args):
    return runner.invoke(app, ["-c", str(config), "--no-progress", *args])


@pytest.fixture
def config(tmp_path) -> pathlib.Path:
    config = write_config(tmp_path)
    result = invoke(config, "init-db")
    assert result.exit_code == 0, result.output
    return config


def test_init_scaffolds_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    config = tmp_path / "config.toml"
    result = runner.invoke(app, ["-c", str(config), "init"])
    assert result.exit_code == 0, result.output
    assert config.exists()
    assert (home / ".config" / "thistle-db.toml").exists()


def test_init_db_idempotent_and_drop_guard(tmp_path):
    config = write_config(tmp_path)
    assert invoke(config, "init-db").exit_code == 0
    result = invoke(config, "init-db")
    assert result.exit_code == 0
    assert "left untouched" in result.output
    # --drop without --yes fails closed non-interactively.
    assert invoke(config, "init-db", "--drop").exit_code == 2
    # --defer-indexes without --drop is a usage error.
    assert invoke(config, "init-db", "--defer-indexes").exit_code == 2


def test_uninitialized_database_exits_3(tmp_path):
    config = write_config(tmp_path)
    result = invoke(config, "status")
    assert result.exit_code == 3


def test_invalid_config_exits_2(tmp_path):
    config = write_config(tmp_path, extra='[ingest]\nreject_dir = "./rejects"\n')
    result = invoke(config, "status")
    assert result.exit_code == 2


def test_ingest_scan_status_and_get_tle(config, tmp_path):
    (tmp_path / "incoming" / "a.tle").write_text(
        "".join(f"{a}\n{b}\n" for a, b in TLES)
    )
    result = invoke(config, "ingest")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "objects" / "22.tle").exists()

    result = invoke(config, "get-tle", "22")
    assert result.exit_code == 0
    assert L1 in result.output and L2 in result.output

    result = invoke(config, "get-tle", "20250418")
    assert result.exit_code == 0
    assert TLES[1][0] in result.output

    result = invoke(config, "get-tle", "99999")
    assert result.exit_code == 1

    result = invoke(config, "status")
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "succeeded" in result.output
    assert "No unresolved rejects" in result.output


def test_ingest_named_missing_file_exits_1(config, tmp_path):
    result = invoke(config, "ingest", str(tmp_path / "nope.tle"))
    assert result.exit_code == 1


def test_ingest_requires_outputs(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[database]
drivername = "sqlite"
name = "{(tmp_path / 'cli.db').as_posix()}"

[[ingest.sources]]
path = "{incoming.as_posix()}"
"""
    )
    result = invoke(config, "ingest")
    assert result.exit_code == 2


def test_rejects_summary_and_export(config, tmp_path):
    bad_l1 = L1[:30] + ("9" if L1[30] != "9" else "8") + L1[31:]
    bad = tmp_path / "bad.tle"
    bad.write_text(f"{bad_l1}\n{L2}\n")
    assert invoke(config, "ingest", str(bad)).exit_code == 0

    result = invoke(config, "rejects")
    assert result.exit_code == 0
    assert "checksum" in result.output

    result = invoke(config, "rejects", "--export", str(tmp_path / "repair"))
    assert result.exit_code == 0
    assert (tmp_path / "repair" / "rejects.tle").exists()

    result = invoke(config, "status")
    assert "checksum" in result.output


def test_export_and_force_guard(config, tmp_path):
    (tmp_path / "incoming" / "a.tle").write_text(f"{L1}\n{L2}\n")
    invoke(config, "ingest")
    out = tmp_path / "backup.csv"
    result = invoke(config, "export", str(out))
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert invoke(config, "export", str(out)).exit_code == 1
    assert invoke(config, "export", str(out), "--force").exit_code == 0
    assert invoke(config, "export", str(out), "-f", "bogus", "--force").exit_code == 2


def test_verify_and_rebuild(config, tmp_path):
    (tmp_path / "incoming" / "a.tle").write_text(
        "".join(f"{a}\n{b}\n" for a, b in TLES)
    )
    invoke(config, "ingest")
    assert invoke(config, "verify").exit_code == 0

    (tmp_path / "objects" / "22.tle").unlink()
    assert invoke(config, "verify").exit_code == 1

    assert invoke(config, "rebuild").exit_code == 0
    assert (tmp_path / "objects" / "22.tle").exists()
    assert invoke(config, "verify").exit_code == 0


def test_omm_ingest_via_cli(config, tmp_path):
    data = pathlib.Path(__file__).parent / "data" / "one.json"
    result = invoke(config, "ingest", str(data))
    assert result.exit_code == 0, result.output
    result = invoke(config, "get-tle", "11")
    assert result.exit_code == 0
    one = json.loads(data.read_text())
    assert one["TLE_LINE1"] in result.output
