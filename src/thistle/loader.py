import datetime
from typing import Union

from thistle.alpha5 import ensure_alpha5
from thistle.config import Settings
from thistle.io import read_tle


class Loader:
    settings: Settings

    def __init__(self, config: Settings) -> None:
        self.settings = config

        if not self.settings.archive.exists():
            raise FileNotFoundError(self.settings.archive)

        if not self.settings.object.exists():
            raise FileNotFoundError(self.settings.object)

        if not self.settings.daily.exists():
            raise FileNotFoundError(self.settings.daily)

    def load_object(self, satnum: Union[str, int]) -> None:
        satnum = ensure_alpha5(satnum)
        file = self.settings.object / f"{satnum}{self.settings.suffix}"
        return read_tle(file)

    def load_day(self, date: str) -> None:
        date = datetime.datetime.strptime(date, "%Y%m%d")
        file = self.settings.daily / f"{date.strftime('%Y%m%d')}{self.settings.suffix}"
        return read_tle(file)
