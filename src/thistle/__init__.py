try:
    from thistle._version import __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("thistle")

from thistle.propagator import (
    EpochSwitchStrategy,
    MidpointSwitchStrategy,
    Propagator,
    TCASwitchStrategy,
)
from thistle.utils import read_tle

__all__ = [
    "Propagator",
    "read_tle",
    "EpochSwitchStrategy",
    "MidpointSwitchStrategy",
    "TCASwitchStrategy",
]
