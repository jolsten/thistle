try:
    from thistle._version import __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("thistle")

from thistle.events import (
    AscendingPeriod,
    DescendingPeriod,
    EclipsePeriod,
    Event,
    NodeCrossing,
    SatellitePass,
    SunlitPeriod,
    find_ascending_periods,
    find_descending_periods,
    find_eclipse_periods,
    find_node_crossings,
    find_passes,
    find_sunlit_periods,
)
from thistle.ground_sites import doppler_shift, generate_range, visibility_circle
from thistle.orbit_data import Site, Sites, generate
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
    "visibility_circle",
    "generate_range",
    "doppler_shift",
    "generate",
    "Site",
    "Sites",
    "Event",
    "SatellitePass",
    "NodeCrossing",
    "SunlitPeriod",
    "EclipsePeriod",
    "AscendingPeriod",
    "DescendingPeriod",
    "find_passes",
    "find_node_crossings",
    "find_sunlit_periods",
    "find_eclipse_periods",
    "find_ascending_periods",
    "find_descending_periods",
]
