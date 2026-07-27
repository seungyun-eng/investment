from __future__ import annotations

import pandas as pd

from stock_research.macro_fear_buy_sp500.validation import (
    flow_adjusted_block_bootstrap,
    yearly_flow_adjusted_comparison,
)


def test_yearly_comparison_uses_flow_adjusted_returns() -> None:
    dates = pd.to_datetime(
        ["2023-12-29", "2024-01-02", "2024-01-03"]
    )
    strategy = pd.DataFrame(
        {
            "Date": dates,
            "FlowAdjustedReturn": [0.0, 0.10, -0.05],
        }
    )
    benchmark = pd.DataFrame(
        {
            "Date": dates,
            "FlowAdjustedReturn": [0.0, 0.02, 0.01],
        }
    )
    result = yearly_flow_adjusted_comparison(strategy, benchmark)
    row = result[result["Year"] == 2024].iloc[0]
    assert round(float(row["StrategyReturn(%)"]), 3) == 4.5
    assert round(float(row["BuyHoldReturn(%)"]), 3) == 3.02
    assert bool(row["StrategyWon"])


def test_bootstrap_is_reproducible() -> None:
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    strategy = pd.DataFrame(
        {"Date": dates, "FlowAdjustedReturn": [0.001] * 50}
    )
    benchmark = pd.DataFrame(
        {"Date": dates, "FlowAdjustedReturn": [0.0005] * 50}
    )
    first = flow_adjusted_block_bootstrap(
        strategy,
        benchmark,
        block_size=5,
        samples=20,
        seed=7,
    )
    second = flow_adjusted_block_bootstrap(
        strategy,
        benchmark,
        block_size=5,
        samples=20,
        seed=7,
    )
    pd.testing.assert_frame_equal(first, second)
