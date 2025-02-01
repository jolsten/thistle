from importlib.metadata import version

__version__ = version("thistle")

from thistle.propagator import Propagator
from thistle.reader import TLEReader

__all__ = ["Propagator", "TLEReader"]
