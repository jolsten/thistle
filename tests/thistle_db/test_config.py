"""Config loading: layered credential resolution and config-path discovery.

The documented credential resolution order (highest priority first):

1. Env vars ``THISTLE_DB_DATABASE__USERNAME`` / ``__PASSWORD``
2. User secrets file ``~/.config/thistle-db.toml``
3. System secrets file (``database.secrets_file`` in config.toml)
4. Values in config.toml

With no explicit path, the config file is ``$THISTLE_DB_CONFIG`` when set,
else ``./config.toml`` (the CLI default, shared with thistle's db fallback).
"""

import pathlib

import pytest

from thistle_db.config import load_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip THISTLE_DB_* vars so a developer's environment can't leak in."""
    for var in (
        "THISTLE_DB_CONFIG",
        "THISTLE_DB_DATABASE__USERNAME",
        "THISTLE_DB_DATABASE__PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def fake_home(tmp_path, monkeypatch):
    """Isolate ~/.config/thistle-db.toml from the developer's real one."""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    return home


def _write_user_secrets(fake_home: pathlib.Path, body: str) -> pathlib.Path:
    path = fake_home / ".config" / "thistle-db.toml"
    path.write_text(body)
    return path


class TestCredentialPrecedence:
    @pytest.fixture
    def layers(self, tmp_path, fake_home) -> pathlib.Path:
        """All file layers present, each with distinct credentials."""
        system = tmp_path / "system-secrets.toml"
        system.write_text('username = "system_user"\npassword = "system_pw"\n')
        _write_user_secrets(
            fake_home, 'username = "user_user"\npassword = "user_pw"\n'
        )
        config = tmp_path / "config.toml"
        config.write_text(
            "[database]\n"
            'drivername = "sqlite"\n'
            'name = "x.db"\n'
            'username = "toml_user"\n'
            'password = "toml_pw"\n'
            f'secrets_file = "{system.as_posix()}"\n'
        )
        return config

    def test_env_vars_beat_everything(self, layers, monkeypatch):
        monkeypatch.setenv("THISTLE_DB_DATABASE__USERNAME", "env_user")
        monkeypatch.setenv("THISTLE_DB_DATABASE__PASSWORD", "env_pw")
        db = load_config(layers).database
        assert (db.username, db.password) == ("env_user", "env_pw")

    def test_env_var_overrides_single_key(self, layers, monkeypatch):
        # Only the password comes from the environment; the username still
        # resolves through the file layers (user secrets win there).
        monkeypatch.setenv("THISTLE_DB_DATABASE__PASSWORD", "env_pw")
        db = load_config(layers).database
        assert (db.username, db.password) == ("user_user", "env_pw")

    def test_user_secrets_beat_system_secrets(self, layers):
        db = load_config(layers).database
        assert (db.username, db.password) == ("user_user", "user_pw")

    def test_system_secrets_beat_config_values(self, layers, fake_home):
        (fake_home / ".config" / "thistle-db.toml").unlink()
        db = load_config(layers).database
        assert (db.username, db.password) == ("system_user", "system_pw")

    def test_config_values_are_last_resort(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text(
            "[database]\n"
            'drivername = "sqlite"\n'
            'name = "x.db"\n'
            'username = "toml_user"\n'
            'password = "toml_pw"\n'
        )
        db = load_config(config).database
        assert (db.username, db.password) == ("toml_user", "toml_pw")

    def test_scaffolded_empty_user_secrets_do_not_mask(self, layers, fake_home):
        # `thistle-db init` scaffolds the user file with empty strings; an
        # unfilled scaffold must not mask real credentials below it.
        _write_user_secrets(fake_home, 'username = ""\npassword = ""\n')
        db = load_config(layers).database
        assert (db.username, db.password) == ("system_user", "system_pw")

    def test_partial_user_secrets_resolve_per_key(self, layers, fake_home):
        # username from the user file; password falls through to system.
        _write_user_secrets(fake_home, 'username = "user_user"\n')
        db = load_config(layers).database
        assert (db.username, db.password) == ("user_user", "system_pw")

    def test_missing_system_secrets_file_is_ignored(self, tmp_path, fake_home):
        config = tmp_path / "config.toml"
        config.write_text(
            "[database]\n"
            'drivername = "sqlite"\n'
            'name = "x.db"\n'
            f'secrets_file = "{(tmp_path / "nope.toml").as_posix()}"\n'
            'username = "toml_user"\n'
        )
        db = load_config(config).database
        assert db.username == "toml_user"
        assert db.password is None


class TestConfigPathDiscovery:
    def test_env_var_points_to_config(self, tmp_path, monkeypatch):
        config = tmp_path / "elsewhere" / "special.toml"
        config.parent.mkdir()
        config.write_text('[database]\ndrivername = "sqlite"\nname = "from-env.db"\n')
        monkeypatch.setenv("THISTLE_DB_CONFIG", str(config))
        assert load_config(None).database.name == "from-env.db"

    def test_env_var_config_still_layers_credentials(
        self, tmp_path, fake_home, monkeypatch
    ):
        # The env-var-discovered config participates in the same credential
        # layering as an explicit path (user secrets beat its inline values).
        config = tmp_path / "env-config.toml"
        config.write_text(
            '[database]\ndrivername = "sqlite"\nname = "x.db"\n'
            'username = "toml_user"\n'
        )
        _write_user_secrets(fake_home, 'username = "user_user"\n')
        monkeypatch.setenv("THISTLE_DB_CONFIG", str(config))
        assert load_config(None).database.username == "user_user"

    def test_env_var_relative_path_resolves_against_cwd(
        self, tmp_path, monkeypatch
    ):
        # THISTLE_DB_CONFIG=configs/custom.toml (a relative value, e.g. set
        # in a crontab that cd's first) resolves against the working dir.
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "custom.toml").write_text(
            '[database]\ndrivername = "sqlite"\nname = "relative.db"\n'
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("THISTLE_DB_CONFIG", "configs/custom.toml")
        assert load_config(None).database.name == "relative.db"

    def test_explicit_path_beats_env_var(self, tmp_path, monkeypatch):
        env_config = tmp_path / "env.toml"
        env_config.write_text('[database]\ndrivername = "sqlite"\nname = "env.db"\n')
        explicit = tmp_path / "explicit.toml"
        explicit.write_text(
            '[database]\ndrivername = "sqlite"\nname = "explicit.db"\n'
        )
        monkeypatch.setenv("THISTLE_DB_CONFIG", str(env_config))
        assert load_config(explicit).database.name == "explicit.db"

    def test_defaults_to_cwd_config_toml(self, tmp_path, monkeypatch):
        (tmp_path / "config.toml").write_text(
            '[database]\ndrivername = "sqlite"\nname = "cwd.db"\n'
        )
        monkeypatch.chdir(tmp_path)
        assert load_config(None).database.name == "cwd.db"

    def test_missing_config_yields_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no config.toml here
        settings = load_config(None)
        assert settings.database.drivername == "sqlite"
        assert settings.database.name == ":memory:"
