try:
    from thistle._version import __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("thistle")

from thistle.io import read_tle, read_tles, write_tle, write_tles
from thistle.propagator import (
    EpochSwitchStrategy,
    MidpointSwitchStrategy,
    Propagator,
    TCASwitchStrategy,
)

__all__ = [
    "Propagator",
    "read_tle",
    "read_tles",
    "write_tle",
    "write_tles",
    "EpochSwitchStrategy",
    "MidpointSwitchStrategy",
    "TCASwitchStrategy",
]
