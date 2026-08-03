from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.big_tech_10 import (
    build_equal_weight_buy_hold_equity,
    calendar_return_rows,
    summarize_equity_curve,
)


def test_equal_weight_buy_hold_does_not_rebalance_winners() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-12-31"])
    panel = pd.DataFrame(
        {
            "Date": [dates[0], dates[1], dates[0], dates[1]],
            "Ticker": ["A", "A", "B", "B"],
            "Close": [100.0, 200.0, 100.0, 50.0],
        }
    )
    equity = build_equal_weight_buy_hold_equity(
        panel,
        start="2020-01-02",
        end="2020-12-31",
        initial_capital=100_000.0,
    )
    assert equity["Equity"].tolist() == pytest.approx(
        [100_000.0, 125_000.0]
    )


def test_equity_summary_and_calendar_returns_use_net_growth() -> None:
    curve = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-02", "2020-12-31", "2021-12-31"]
            ),
            "Equity": [100_000.0, 120_000.0, 108_000.0],
        }
    )
    summary = summarize_equity_curve(
        curve,
        start="2020-01-02",
        end="2021-12-31",
    )
    annual = calendar_return_rows(curve, series="TEST")
    assert summary["ROI"] == pytest.approx(8.0)
    assert summary["MaxDrawdown"] == pytest.approx(-10.0)
    assert [row["Return"] for row in annual] == pytest.approx([20.0, -10.0])
