from importlib.metadata import version

__version__ = version("thistle")

from thistle.reader import TLEReader
from thistle.propagator import Propagator
