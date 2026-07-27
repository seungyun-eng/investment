"""Contrarian S&P 500 macro fear-buy research workflow."""

from .config import FearBuyParams, FearBuySettings
from .features import build_fear_features
from .portfolio import PortfolioResult, run_fear_buy_backtest
from .strategy import generate_fear_buy_signals

__all__ = [
    "FearBuyParams",
    "FearBuySettings",
    "PortfolioResult",
    "build_fear_features",
    "generate_fear_buy_signals",
    "run_fear_buy_backtest",
]
