from __future__ import annotations

from pathlib import Path

from stock_research.multi_equity.config import load_equity_specs
from stock_research.multi_equity.research import _ticker_seed

EXPECTED_TICKERS = [
    "NVDA",
    "AAPL",
    "GOOG",
    "MSFT",
    "TSM",
    "AVGO",
    "META",
    "TSLA",
    "COST",
    "UNH",
    "CVX",
    "PLTR",
    "NVO",
    "APP",
    "VRT",
]


def test_requested_universe_is_exactly_configured() -> None:
    specs = load_equity_specs(
        Path("config/multi_equity/research.json")
    )
    assert [spec.ticker for spec in specs] == EXPECTED_TICKERS


def test_ticker_seed_is_stable_and_ticker_specific() -> None:
    assert _ticker_seed(100, "NVDA") == _ticker_seed(100, "NVDA")
    assert _ticker_seed(100, "NVDA") != _ticker_seed(100, "AAPL")
