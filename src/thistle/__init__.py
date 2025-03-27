from importlib.metadata import version

__version__ = version("thistle")

from thistle.io import Loader, read_tle, read_tles, write_tle, write_tles
from thistle.propagator import Propagator

__all__ = ["Propagator", "Loader", "read_tle", "read_tles", "write_tle", "write_tles"]
