"""Common type aliases used throughout the thistle package."""

import datetime
import os
import pathlib
from typing import Union

import numpy as np

PathLike = Union[str, bytes, os.PathLike, pathlib.Path]
"""A file system path: str, bytes, os.PathLike, or pathlib.Path."""

TLETuple = tuple[str, str]
"""A two-line element set as a ``(line1, line2)`` tuple of strings."""

Satnum = Union[str, int]
"""A satellite catalog number, either as an integer or Alpha-5 string."""

DateTime = Union[datetime.datetime, np.datetime64]
"""A point in time as a standard datetime or numpy datetime64."""
