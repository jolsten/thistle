# from typing import Union

# from sgp4.api import Satrec

# from thistle.config import Settings
# from thistle.io import read_tle


# class Loader:
#     satrecs: dict[int, list[Satrec]]
#     settings: Settings

#     def __init__(self, config: Settings) -> None:
#         self.settings = config
#         self.satrecs = {}

#     def load_object(self, satnum: Union[str, int]) -> None:
#         file = self.settings.daily / f"{satnum}.tce"
#         self.satrecs[satnum] = read_tle(file)
