import datetime
import os
import pathlib
from typing import Union

import numpy as np

PathLike = Union[str, bytes, os.PathLike, pathlib.Path]
TLETuple = tuple[str, str]
Satnum = Union[str, int]
DateTime = Union[datetime.datetime, np.datetime64]
