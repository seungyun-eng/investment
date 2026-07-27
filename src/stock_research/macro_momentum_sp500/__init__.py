"""Point-in-time macro and momentum research for an adjusted SPY proxy."""

from .config import ResearchConfig, load_research_config
from .data import load_research_data
from .features import build_features
from .targets import build_targets

__all__ = [
    "ResearchConfig",
    "build_features",
    "build_targets",
    "load_research_config",
    "load_research_data",
]
