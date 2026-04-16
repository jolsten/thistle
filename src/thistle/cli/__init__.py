"""Thistle command-line interface.

The CLI requires the ``cli`` extra: ``pip install 'thistle[cli]'``.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``thistle`` CLI.

    Fails gracefully with a helpful message if CLI dependencies aren't installed.
    """
    try:
        from thistle.cli._app import app
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        print(
            f"thistle: CLI requires additional packages (missing: {missing}).\n"
            f"Install with: pip install 'thistle[cli]'",
            file=sys.stderr,
        )
        sys.exit(1)
    app()
