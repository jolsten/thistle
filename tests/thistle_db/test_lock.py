"""Single-writer advisory lock."""

import pytest

from thistle_db.lock import LockHeldError, SingleWriterLock, lock_path_for


def test_lock_excludes_second_writer(tmp_path):
    path = lock_path_for(tmp_path / "config.toml")
    with SingleWriterLock(path):
        with pytest.raises(LockHeldError):
            with SingleWriterLock(path):
                pass


def test_lock_releases_on_exit(tmp_path):
    path = lock_path_for(tmp_path / "config.toml")
    with SingleWriterLock(path):
        pass
    with SingleWriterLock(path):
        pass  # re-acquirable immediately


def test_lock_path_is_beside_config(tmp_path):
    config = tmp_path / "deploy" / "config.toml"
    assert lock_path_for(config).parent == config.parent
