try:
    from thistle._version import __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("thistle")

from thistle.events import (
    find_ascending_periods,
    find_descending_periods,
    find_eclipse_periods,
    find_node_crossings,
    find_passes,
    find_sunlit_periods,
)
from thistle.ground_sites import doppler_shift, generate_range, visibility_circle
from thistle._core import Site, Sites
from thistle.orbit_data import generate
from thistle.propagator import (
    EpochSwitchStrategy,
    MidpointSwitchStrategy,
    Propagator,
    SwitchingStrategy,
    TCASwitchStrategy,
)
from thistle.utils import read_tle

__all__ = [
    "Propagator",
    "read_tle",
    "SwitchingStrategy",
    "EpochSwitchStrategy",
    "MidpointSwitchStrategy",
    "TCASwitchStrategy",
    "visibility_circle",
    "generate_range",
    "doppler_shift",
    "generate",
    "Site",
    "Sites",
    "find_passes",
    "find_node_crossings",
    "find_sunlit_periods",
    "find_eclipse_periods",
    "find_ascending_periods",
    "find_descending_periods",
]
