import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL


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

    @property
    def engine(self) -> Engine:
        return create_engine(self.url)


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


def load_config(path: Path | None = None) -> Settings:
    """Load settings from TOML file with layered credential resolution.

    Resolution order (highest priority first):
    1. Environment variables (THISTLE_DB_DATABASE__USERNAME, etc.)
    2. User secrets (~/.config/thistle-db.toml)
    3. System secrets file (database.secrets_file in config)
    4. Values in config.toml
    """
    toml_data: dict = {}

    if path is not None and path.exists():
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)

    # Load system secrets file if specified
    db_data = toml_data.get("database", {})
    secrets_file = db_data.get("secrets_file")
    if secrets_file:
        secrets_path = Path(secrets_file)
        if secrets_path.exists():
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
            # Merge secrets into database config (secrets_file has lower priority
            # than values already in db_data, which have lower priority than env vars)
            for key in ("username", "password"):
                if key in secrets and key not in db_data:
                    db_data[key] = secrets[key]
            toml_data["database"] = db_data

    # Load user-local secrets (~/.config/thistle-db.toml)
    user_secrets_path = Path.home() / ".config" / "thistle-db.toml"
    if user_secrets_path.exists():
        with open(user_secrets_path, "rb") as f:
            user_secrets = tomllib.load(f)
        for key in ("username", "password"):
            if key in user_secrets and key not in db_data:
                db_data[key] = user_secrets[key]
        toml_data["database"] = db_data

    # pydantic-settings handles env var overrides automatically
    return Settings(**toml_data)
