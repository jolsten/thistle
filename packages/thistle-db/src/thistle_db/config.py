import datetime
import glob
import os
import re
import tomllib
from functools import cached_property
from pathlib import Path
from string import Formatter
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from sgp4.alpha5 import from_alpha5, to_alpha5
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL

CONFIG_PATH_ENV = "THISTLE_DB_CONFIG"

# Default for [ingest] max_epoch_ahead_days — how far into the future an
# elset's epoch may run before ingest rejects it. Real feeds deliver elsets
# at most hours to a day ahead of now, so 30 days is generous by orders of
# magnitude while still catching corrupted epoch fields, which parse
# cleanly into 2000-2056 (TLE years are two digits, 00-56 -> 20xx).
DEFAULT_MAX_EPOCH_AHEAD_DAYS = 30

# Default for [ingest] flush_records — buffered records that trigger an
# early delta flush. Purely a memory budget (~600 B/record), never a
# correctness knob: flushes are idempotent, so an early flush only means
# touching hot output files more than once per run.
DEFAULT_FLUSH_RECORDS = 1_000_000

# Alpha-5 tops out at Z9999, so no real catalog number exceeds this. Lets
# object-name parsing reject 8-digit YYYYMMDD date stems numerically.
MAX_SATNUM = 339_999

# Filename template placeholders. The per-file one ({id}/{date}) is required
# in its output type — without it every file of that output would render to
# the same name and overwrite the last.
_REQUIRED_PLACEHOLDER = {"object": "id", "date": "date"}
_ALLOWED_PLACEHOLDERS = {
    "object": {"id", "ext", "format", "type"},
    "date": {"date", "ext", "format", "type"},
}

# Default file extension per output format ({ext} placeholder).
_DEFAULT_EXTENSION = {"tle": ".tle", "omm-csv": ".csv"}

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
        # pool_pre_ping revalidates connections that a long run (or
        # MariaDB's wait_timeout) may have silently dropped.
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
    # extra="forbid": retired keys (reject_dir) fail loudly with a pointer
    # at the key instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    sources: list[IngestSource] = []
    max_epoch_ahead_days: int = Field(default=DEFAULT_MAX_EPOCH_AHEAD_DAYS, ge=0)
    """Reject elsets whose epoch is more than this many days in the future.

    A corrupted epoch field parses cleanly into 2000-2056 (two-digit TLE
    years), and once stored such a row permanently poisons its object
    file's tail guard — every later legitimate elset triggers a full
    rewrite instead of an append — and plants a junk date file nothing
    cleans up. Real feeds run at most hours to a day ahead of now, so the
    default (30) is generous; raise it for predicted-elset workflows, or
    set 0 to disable the guard entirely. One-sided: epochs in the past are
    never bounded, so historical archives ingest untouched. Rejected
    records are retained in the `rejects` table and re-admitted
    automatically once their epoch comes inside the horizon."""
    verify_checksums: bool = True
    """Reject TLE lines whose mod-10 checksum (column 69) does not match.

    sgp4 never verifies checksums, and a single flipped digit changes an
    element value *and* breaks the checksum — for a record whose elements
    still parse finite, the checksum is the only corruption signal
    validation has. Lines shorter than 69 characters skip the check (some
    tools emit trimmed lines; historical archives contain them)."""
    flush_records: int = Field(default=DEFAULT_FLUSH_RECORDS, ge=1)
    """Buffered records that trigger an early delta flush (memory budget)."""


class OutputFile(BaseModel):
    """One maintained output: a file type + format + destination + naming.

    Declared as ``[[output.files]]`` entries in config.toml. Entries may
    share a directory (extensions keep them apart) or spread across
    directories; each is maintained independently by the flush engine.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["date", "object"]
    """"date": one file per epoch date (latest elset per object that day);
    "object": one file per satellite (full history, epoch order)."""
    format: Literal["tle", "omm-csv"]
    """"tle": verbatim two-line text; "omm-csv": STK-compatible OMM CSV."""
    dir: str = "./output"
    columns: Literal["stk", "full"] = "stk"
    """omm-csv outputs: "stk" = the 17-column CelesTrak/STK projection;
    "full" = every store column (metadata-bearing outputs)."""
    object_id: Literal["int", "alpha5"] = "int"
    """Object files: render the NORAD ID as a plain integer or alpha-5
    (always 5 characters, e.g. 00900, E5693)."""
    zero_pad: bool = False
    """Object files with object_id="int": left-pad the ID to 5 digits."""
    date_format: str = "%Y%m%d"
    """Date files: strftime pattern for the `{date}` placeholder."""
    extension: Optional[str] = None
    """The `{ext}` placeholder; defaults to ".tle"/".csv" per format."""
    filename: Optional[str] = None
    """Filename template, e.g. "tle_{id}.txt" or "{date}_gp{ext}".

    Defaults to "{id}{ext}" for object outputs and "{date}{ext}" for date
    outputs. The template only arranges the pieces — `object_id`/`zero_pad`,
    `date_format` and `extension` still decide how each renders.

    Placeholders: `{id}` (object outputs, required there), `{date}` (date
    outputs, required there), `{ext}`, `{format}`, `{type}`."""

    @field_validator("extension")
    @classmethod
    def _dot_extension(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith("."):
            return f".{v}"
        return v

    @model_validator(mode="after")
    def _check_columns(self) -> "OutputFile":
        if self.format == "tle" and self.columns != "stk":
            raise ValueError(
                "columns applies to omm-csv outputs only — tle outputs are "
                "verbatim two-line text"
            )
        return self

    @model_validator(mode="after")
    def _check_filename(self) -> "OutputFile":
        """Reject a bad template at config load, not mid-run.

        A KeyError raised while naming the 40,000th file would leave a
        half-written output tree behind, so unknown placeholders, a missing
        required one, and anything that would escape `dir` fail here.
        """
        if self.filename is None:
            return self
        if not self.filename.strip():
            raise ValueError("filename template is empty")

        required = _REQUIRED_PLACEHOLDER[self.type]
        allowed = _ALLOWED_PLACEHOLDERS[self.type]
        seen: set[str] = set()
        for literal, field, spec, conversion in Formatter().parse(self.filename):
            if any(sep in literal for sep in ("/", "\\")) or ".." in literal:
                raise ValueError(
                    f"filename template {self.filename!r} must name a file, "
                    "not a path — use `dir` for the destination directory"
                )
            if field is None:
                continue
            if field not in allowed:
                raise ValueError(
                    f"unknown placeholder {{{field}}} in filename template "
                    f"{self.filename!r} for {self.type} outputs; available: "
                    + ", ".join(f"{{{name}}}" for name in sorted(allowed))
                )
            if spec or conversion:
                raise ValueError(
                    f"placeholder {{{field}}} in filename template "
                    f"{self.filename!r} may not carry a format spec"
                )
            seen.add(field)
        if required not in seen:
            raise ValueError(
                f"filename template {self.filename!r} for {self.type} outputs must "
                f"contain {{{required}}}, or files would overwrite each other"
            )
        return self

    @property
    def ext(self) -> str:
        if self.extension is not None:
            return self.extension
        return _DEFAULT_EXTENSION[self.format]

    @property
    def template(self) -> str:
        """The filename template in force, explicit or default."""
        if self.filename is not None:
            return self.filename
        return "{id}{ext}" if self.type == "object" else "{date}{ext}"

    def _fixed(self, field: str) -> str:
        """Value of a placeholder that doesn't vary per file."""
        return {"ext": self.ext, "format": self.format, "type": self.type}[field]

    def object_stem(self, norad_id: int) -> str:
        """The `{id}` placeholder's value for `norad_id`.

        An alpha5 output meeting an ID beyond alpha-5's range (> MAX_SATNUM)
        falls back to the plain integer stem — ingest never fails or drops
        valid data over an output naming choice; the caller warns once per
        run.
        """
        if self.object_id == "alpha5" and norad_id <= MAX_SATNUM:
            return to_alpha5(norad_id)
        if self.object_id == "int" and self.zero_pad:
            return f"{norad_id:05d}"
        return str(norad_id)

    def object_path(self, norad_id: int) -> Path:
        name = self.template.format(
            id=self.object_stem(norad_id),
            ext=self.ext,
            format=self.format,
            type=self.type,
        )
        return Path(self.dir) / name

    def date_path(self, date_val: datetime.date) -> Path:
        name = self.template.format(
            date=date_val.strftime(self.date_format),
            ext=self.ext,
            format=self.format,
            type=self.type,
        )
        return Path(self.dir) / name

    @cached_property
    def _object_name_re(self) -> re.Pattern:
        """Regex matching filenames this output's template produces.

        Built from the template so the inverse always tracks the forward
        rendering: literals escaped, `{id}` captured, the fixed placeholders
        substituted. Alpha-5 outputs also accept a bare integer stem — the
        overflow fallback names files that way.
        """
        # zero_pad only pads; a 6-digit ID stays 6 digits, and an unpadded
        # leftover file should still be recognizable. So both int modes
        # accept a bare run of digits, bounded below by MAX_SATNUM.
        id_pattern = (
            # 5 chars, leading digit or alpha-5 letter (I and O are
            # invalid), OR the integer fallback stem.
            r"(?:[0-9A-HJ-NP-Z][0-9]{4}|[0-9]+)"
            if self.object_id == "alpha5"
            else r"[0-9]+"
        )
        parts = []
        for literal, field, _, _ in Formatter().parse(self.template):
            parts.append(re.escape(literal))
            if field is None:
                continue
            parts.append(
                f"({id_pattern})" if field == "id" else re.escape(self._fixed(field))
            )
        return re.compile("".join(parts))

    def object_glob(self) -> str:
        """Glob matching this output's object files, for the orphan scan."""
        parts = []
        for literal, field, _, _ in Formatter().parse(self.template):
            parts.append(glob.escape(literal))
            if field is None:
                continue
            parts.append("*" if field == "id" else glob.escape(self._fixed(field)))
        return "".join(parts)

    def parse_object_name(self, name: str) -> Optional[int]:
        """The NORAD ID this output's naming assigns to filename `name`, or
        None if `name` cannot be one of its object files (used for orphan
        detection — date files sharing the directory must not look like
        orphans)."""
        match = self._object_name_re.fullmatch(name)
        if match is None:
            return None
        raw = match.group(1)
        if self.object_id == "alpha5":
            if not raw.isdigit():
                return from_alpha5(raw)
            # All-digit stem: either a numeric alpha-5 (≤ MAX_SATNUM) or the
            # integer overflow-fallback stem. 8-digit stems are excluded —
            # they are YYYYMMDD date files sharing the directory.
            return int(raw) if len(raw) != 8 else None
        norad_id = int(raw)
        return norad_id if norad_id <= MAX_SATNUM else None


class OutputConfig(BaseModel):
    # extra="forbid": retired keys (lookback_days) fail loudly with a
    # pointer at the key instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    files: list[OutputFile] = []
    """Outputs to maintain. Empty is valid config for query-only use, but
    `ingest` and `rebuild` refuse to run with nothing to produce."""
    write_workers: int = 0
    """Threads used for output file writes. 0 = auto (CPU count x 4, capped).

    Output maintenance is dominated by per-file syscall latency, which
    releases the GIL, so a pool recovers most of it. Set to 1 to write
    serially — for debugging, or a network filesystem that dislikes
    concurrent writes."""


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
