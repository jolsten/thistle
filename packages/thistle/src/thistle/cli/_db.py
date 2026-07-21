"""Optional thistle-db database fallback for catalog-ID resolution.

Active only when the thistle-db package is installed (``pip install
'thistle[db]'``) and a database is actually configured, via ``THISTLE_DB_*``
environment variables or a config.toml named by ``THISTLE_DB_CONFIG``. All
thistle_db imports are lazy so core thistle stays importable without it.
"""

from __future__ import annotations

import atexit
import importlib.util
import os
import pathlib
import sys
import tempfile
from typing import List, Optional, Tuple

from thistle.cli._config import to_satnum

DB_CONFIG_ENV = "THISTLE_DB_CONFIG"
_DB_DATABASE_ENV_PREFIX = "THISTLE_DB_DATABASE__"


def db_installed() -> bool:
    return importlib.util.find_spec("thistle_db") is not None


def db_configured() -> bool:
    """True if the user has pointed thistle-db at an actual database.

    Guards against silently querying thistle-db's unconfigured default, an
    in-memory SQLite database that is always empty.
    """
    if os.environ.get(DB_CONFIG_ENV):
        return True
    return any(key.startswith(_DB_DATABASE_ENV_PREFIX) for key in os.environ)


def _load_settings():
    from thistle_db.config import load_config

    path = os.environ.get(DB_CONFIG_ENV)
    return load_config(pathlib.Path(path) if path else None)


def lookup_tles(catalog_id: str) -> Optional[List[Tuple[str, str]]]:
    """Query the thistle-db database for a catalog ID.

    Returns None when the fallback was not attempted (not installed, not
    configured, or the query failed) and an empty list when the database was
    queried but holds no TLEs for this object.
    """
    if not (db_installed() and db_configured()):
        return None
    satnum = to_satnum(catalog_id)
    if satnum is None:
        return None
    try:
        from thistle_db import get_tles

        return get_tles(satnum, _load_settings())
    except Exception as err:
        print(f"Warning: thistle-db lookup failed: {err}", file=sys.stderr)
        return None


def _unlink_quiet(path: pathlib.Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def tles_to_tempfile(tles: List[Tuple[str, str]], suffix: str) -> pathlib.Path:
    """Write (line1, line2) pairs to a temp file removed at interpreter exit.

    The file is closed before returning so callers can reopen it by path
    (required on Windows).
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        prefix="thistle-db-",
        delete=False,
        encoding="utf-8",
    )
    try:
        for line1, line2 in tles:
            tmp.write(f"{line1}\n{line2}\n")
    finally:
        tmp.close()
    path = pathlib.Path(tmp.name)
    atexit.register(_unlink_quiet, path)
    return path


def db_status() -> dict:
    """Fallback availability info for ``thistle config``."""
    info: dict = {
        "db_installed": db_installed(),
        "db_configured": db_configured(),
        "db_version": None,
        "db_url": None,
    }
    if not info["db_installed"]:
        return info
    try:
        from importlib.metadata import version

        info["db_version"] = version("thistle-db")
    except Exception:
        pass
    if info["db_configured"]:
        try:
            settings = _load_settings()
            info["db_url"] = settings.database.url.render_as_string(
                hide_password=True
            )
        except Exception as err:
            info["db_url"] = f"<error: {err}>"
    return info
