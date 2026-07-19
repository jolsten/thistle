"""Environment-based user configuration for the thistle CLI.

Settings (env vars only, no config file):

- ``THISTLE_TLE_DIR``: directory of per-object TLE files, enabling catalog-ID
  file arguments (``thistle plot 25544`` instead of a path).
- ``THISTLE_TLE_EXT``: filename extension for those files (default ``.tle``).
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass
from typing import Optional

TLE_DIR_ENV = "THISTLE_TLE_DIR"
TLE_EXT_ENV = "THISTLE_TLE_EXT"
DEFAULT_EXT = ".tle"

# All digits, or Alpha-5: satnums >= 100000 encode the leading digits as a
# letter (A=10 ... Z=33, I and O excluded).
_CATALOG_ID_RE = re.compile(r"^(\d{1,9}|[A-HJ-NP-Za-hj-np-z]\d{4})$")


@dataclass
class CliConfig:
    tle_dir: Optional[pathlib.Path]
    tle_ext: str


def load_config() -> CliConfig:
    """Read CLI configuration from the environment."""
    dir_val = os.environ.get(TLE_DIR_ENV)
    ext = os.environ.get(TLE_EXT_ENV, DEFAULT_EXT)
    if ext and not ext.startswith("."):
        ext = "." + ext
    return CliConfig(
        tle_dir=pathlib.Path(dir_val) if dir_val else None,
        tle_ext=ext,
    )


def is_catalog_id(name: str) -> bool:
    """True if ``name`` is a NORAD catalog ID (digits or Alpha-5)."""
    return _CATALOG_ID_RE.match(name) is not None


def candidate_names(name: str) -> list[str]:
    """Filenames (sans extension) to try for a catalog ID.

    Alpha-5 is defined uppercase, so a lowercase-typed ID also tries the
    uppercased form (matters on case-sensitive filesystems).
    """
    names = [name]
    if name.upper() != name:
        names.append(name.upper())
    return names
