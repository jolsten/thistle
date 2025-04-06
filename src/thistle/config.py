import pathlib
import string
from typing import Union

from platformdirs import user_config_path
from pydantic import DirectoryPath
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

user_config_file = user_config_path() / "thistle.toml"

_config_template = string.Template("""
archive = "$archive"
daily   = "$daily"
object  = "$object"
suffix  = "$suffix"
""")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="THISTLE_",
        env_nested_delimiter="__",
        toml_file=user_config_file,
    )

    archive: DirectoryPath
    daily: Union[DirectoryPath, str] = "day"
    object: Union[DirectoryPath, str] = "obj"
    suffix: str = ".txt"

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self.archive = self.archive.absolute()

        self.daily = pathlib.Path(self.daily)
        if not self.daily.is_absolute():
            self.daily = self.archive / self.daily

        self.object = pathlib.Path(self.object)
        if not self.object.is_absolute():
            self.object = self.archive / self.object

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )
