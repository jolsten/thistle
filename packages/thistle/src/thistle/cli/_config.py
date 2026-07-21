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


def to_satnum(name: str) -> Optional[int]:
    """Convert a catalog ID (digits or Alpha-5) to its integer satnum."""
    if name.isdigit():
        return int(name)
    try:
        from sgp4.alpha5 import from_alpha5

        return from_alpha5(name.upper())
    except ValueError:
        return None


def candidate_names(name: str) -> list[str]:
    """Filenames (sans extension) to try for a catalog ID, in order.

    Tries the ID as typed (plus uppercased — Alpha-5 is defined uppercase,
    which matters on case-sensitive filesystems), then the canonical integer
    form (leading zeros stripped, Alpha-5 decoded) and the canonical
    5-character catalog form (zero-padded below 100000, Alpha-5 above). This
    finds the file whatever naming convention the directory uses, however
    the ID was typed.
    """
    names = [name]
    if name.upper() != name:
        names.append(name.upper())
    satnum = to_satnum(name)
    if satnum is not None:
        names.append(str(satnum))
        try:
            from sgp4.alpha5 import to_alpha5

            names.append(to_alpha5(satnum))
        except ValueError:
            pass  # satnum too large for the Alpha-5 range
    return list(dict.fromkeys(names))


def find_candidate_file(
    name: str, cfg: CliConfig
) -> "tuple[Optional[pathlib.Path], list[str]]":
    """Look up a catalog ID in the configured TLE directory.

    Returns ``(path, tried)``: the first existing candidate file (or None),
    and every path tried without success. Both are empty-handed when no
    ``tle_dir`` is configured.
    """
    tried: list[str] = []
    if cfg.tle_dir is None:
        return None, tried
    for cand in candidate_names(name):
        candidate = cfg.tle_dir / f"{cand}{cfg.tle_ext}"
        if candidate.exists():
            return candidate, tried
        tried.append(str(candidate))
    return None, tried
