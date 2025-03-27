from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="THISTLE_",
        env_nested_delimiter="__",
    )

    archive: DirectoryPath
    daily: DirectoryPath = "day"
    object: DirectoryPath = "obj"
    suffix: str = ".tce"

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self.archive = self.archive.absolute()

        if not self.daily.is_absolute():
            self.daily = self.archive / self.daily

        if not self.object.is_absolute():
            self.object = self.archive / self.object


# settings = Settings()
