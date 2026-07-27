"""Independent macro timing workflow for an investable S&P 500 proxy."""

from .config import MacroSp500Params, MacroSp500Settings
from .config_v2 import MacroSp500V2Params, MacroSp500V2Settings
from .features import add_macro_features
from .features_v2 import add_v2_features
from .portfolio import PortfolioResult, run_target_weight_backtest
from .strategy import generate_target_weights
from .strategy_v2 import generate_v2_target_weights

__all__ = [
    "MacroSp500Params",
    "MacroSp500Settings",
    "MacroSp500V2Params",
    "MacroSp500V2Settings",
    "PortfolioResult",
    "add_macro_features",
    "add_v2_features",
    "generate_target_weights",
    "generate_v2_target_weights",
    "run_target_weight_backtest",
]
