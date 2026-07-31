import os
import tomllib
from functools import cached_property
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL

CONFIG_PATH_ENV = "THISTLE_DB_CONFIG"

_READONLY_STMTS = {
    "sqlite": "PRAGMA query_only=ON",
    "postgresql": "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
    "mysql": "SET SESSION TRANSACTION READ ONLY",
    "mariadb": "SET SESSION TRANSACTION READ ONLY",
}


def _install_readonly_guard(engine: Engine) -> None:
    """Reject writes at the connection level (defense in depth for the read
    tier — grants are the real enforcement, this catches bugs)."""
    stmt = _READONLY_STMTS.get(engine.dialect.name)
    if stmt is None:
        return  # unknown dialect: rely on grants

    @event.listens_for(engine, "connect")
    def _set_read_only(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute(stmt)
        finally:
            cursor.close()
        dbapi_conn.commit()


class Database(BaseModel):
    drivername: str = "sqlite"
    username: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    name: Optional[str] = ":memory:"
    secrets_file: Optional[str] = None

    @property
    def url(self) -> URL:
        return URL.create(
            drivername=self.drivername,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.name,
        )

    @cached_property
    def engine(self) -> Engine:
        # Cached: repeated access must not build a new pool each time.
        # pool_pre_ping revalidates connections that a long generate run
        # (or MariaDB's wait_timeout) may have silently dropped.
        return create_engine(self.url, pool_pre_ping=True)

    @cached_property
    def readonly_engine(self) -> Engine:
        # A separate engine: the readonly guard is a connect-event listener,
        # and attaching it to the shared write engine would poison later
        # write use (both engines are cached for the process lifetime).
        engine = create_engine(self.url, pool_pre_ping=True)
        _install_readonly_guard(engine)
        return engine


class IngestSource(BaseModel):
    path: str
    pattern: str = "*.tle"


class IngestConfig(BaseModel):
    sources: list[IngestSource] = []


class OutputFormats(BaseModel):
    tle: bool = True
    omm: bool = True


class OutputTypes(BaseModel):
    date_files: bool = True
    object_files: bool = True


class OutputConfig(BaseModel):
    dir: str = "./output"
    formats: OutputFormats = OutputFormats()
    types: OutputTypes = OutputTypes()
    window_days: int = 60
    """Date files: trailing epoch window (days) rewritten every run."""
    lookback_days: int = 7
    """Object files: rows created within this many days are (re)considered.

    Must comfortably exceed the ingest/generate cron cadence; after an outage
    longer than this, run `generate --all`."""


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="THISTLE_DB_",
        env_nested_delimiter="__",
    )

    database: Database = Database()
    ingest: IngestConfig = IngestConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # load_config passes the merged TOML layers as init kwargs, and
        # pydantic-settings ranks init kwargs above env vars by default —
        # which would let file values shadow THISTLE_DB_* env vars. Env
        # vars are the documented highest-priority source, so rank them
        # above init here.
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


def _overlay_secrets(db_data: dict, secrets_path: Path) -> None:
    """Overlay username/password from a secrets TOML onto `db_data`.

    Non-empty values overwrite whatever a lower-priority layer supplied.
    Empty values are ignored: `thistle-db init` scaffolds the user secrets
    file with empty strings, and an unfilled scaffold must not mask real
    credentials from a lower layer.
    """
    if not secrets_path.exists():
        return
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    for key in ("username", "password"):
        if secrets.get(key):
            db_data[key] = secrets[key]


def load_config(path: Path | None = None) -> Settings:
    """Load settings from TOML file with layered credential resolution.

    Resolution order (highest priority first):
    1. Environment variables (THISTLE_DB_DATABASE__USERNAME, etc.)
    2. User secrets (~/.config/thistle-db.toml)
    3. System secrets file (database.secrets_file in config)
    4. Values in config.toml

    With ``path=None`` the config file is ``$THISTLE_DB_CONFIG`` when set,
    else ``./config.toml`` — the same default the CLI uses, shared with
    thistle's db fallback.
    """
    if path is None:
        env_path = os.environ.get(CONFIG_PATH_ENV)
        path = Path(env_path) if env_path else Path("config.toml")

    toml_data: dict = {}
    if path.exists():
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)

    # Credential layers, applied lowest priority first so each overlays the
    # one below: config.toml values, then the system secrets file, then the
    # user secrets file. Env vars beat all of these (Settings ranks env
    # above init kwargs).
    db_data = toml_data.get("database", {})
    if db_data.get("secrets_file"):
        _overlay_secrets(db_data, Path(db_data["secrets_file"]))
    _overlay_secrets(db_data, Path.home() / ".config" / "thistle-db.toml")
    if db_data:
        toml_data["database"] = db_data

    return Settings(**toml_data)
