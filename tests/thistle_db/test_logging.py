"""Logging behavior: skipped-at-DEBUG, per-source summaries, --log file
sink, and progress-related CLI flags."""

import pathlib
import shutil

from loguru import logger
from typer.testing import CliRunner

from thistle_db.cli import app
from thistle_db.config import IngestSource
from thistle_db.ingest import ingest_sources

DATA = pathlib.Path(__file__).parent / "data"

runner = CliRunner()


class _Capture:
    """Collect loguru records (level name + message text)."""

    def __init__(self):
        self.records = []

    def __call__(self, message):
        record = message.record
        self.records.append((record["level"].name, record["message"]))

    def messages(self, level: str) -> list[str]:
        return [msg for lvl, msg in self.records if lvl == level]


def _scan_twice(db_session, tmp_path) -> _Capture:
    shutil.copy(DATA / "25544.txt", tmp_path / "25544.txt")
    sources = [IngestSource(path=str(tmp_path), pattern="*")]
    capture = _Capture()
    sink_id = logger.add(capture, level="DEBUG")
    try:
        ingest_sources(db_session, sources)
        ingest_sources(db_session, sources)
    finally:
        logger.remove(sink_id)
    return capture


def test_skipped_files_log_at_debug(db_session, tmp_path):
    capture = _scan_twice(db_session, tmp_path)

    # First scan ingests at INFO; second scan skips at DEBUG only.
    assert any(": ingested" in m for m in capture.messages("INFO"))
    assert any(": skipped" in m for m in capture.messages("DEBUG"))
    assert not any(": skipped" in m for m in capture.messages("INFO"))


def test_per_source_summary_logged_at_info(db_session, tmp_path):
    capture = _scan_twice(db_session, tmp_path)

    summaries = [m for m in capture.messages("INFO") if "new records (" in m]
    assert len(summaries) == 2
    assert "1 ingested, 0 skipped" in summaries[0]
    assert "0 ingested, 1 skipped" in summaries[1]


def _sqlite_config(tmp_path) -> pathlib.Path:
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "log-test.db"
    config_path.write_text(
        f'[database]\ndrivername = "sqlite"\nname = "{db_path.as_posix()}"\n'
    )
    assert runner.invoke(app, ["-c", str(config_path), "init-db"]).exit_code == 0
    return config_path


def test_log_flag_writes_rotating_file(tmp_path):
    config_path = _sqlite_config(tmp_path)
    log_path = tmp_path / "logs" / "thistle-db.log"
    log_path.parent.mkdir()

    result = runner.invoke(
        app,
        [
            "-c",
            str(config_path),
            "--log",
            str(log_path),
            "ingest",
            str(DATA / "25544.txt"),
        ],
    )
    assert result.exit_code == 0
    content = log_path.read_text()
    assert "Ingesting" in content
    assert "new records" in content


def test_no_progress_flag_accepted(tmp_path):
    config_path = _sqlite_config(tmp_path)
    result = runner.invoke(
        app,
        ["-c", str(config_path), "--no-progress", "ingest", str(DATA / "25544.txt")],
    )
    assert result.exit_code == 0
