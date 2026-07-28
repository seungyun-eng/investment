"""Shared multi-equity integrated-signal research workflow."""

from .config import EquitySpec, load_equity_specs
from .research import run_equity_research

__all__ = ["EquitySpec", "load_equity_specs", "run_equity_research"]
