from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.cross_sectional.portfolio import (
    run_portfolio_backtest,
)
from stock_research.cross_sectional.winner_attribution import (
    _add_baseline_deltas,
    summarize_ticker_contributions,
)


def _attributed_result():
    dates = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    )
    prices = {
        "A": {
            "Open": [10.0, 10.0, 12.0, 13.0],
            "Close": [10.0, 11.0, 13.0, 13.0],
        },
        "B": {
            "Open": [20.0, 20.0, 18.0, 21.0],
            "Close": [20.0, 20.0, 20.0, 22.0],
        },
    }
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "Open": values["Open"],
                    "Close": values["Close"],
                }
            )
            for ticker, values in prices.items()
        ],
        ignore_index=True,
    )
    targets = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "Ticker": ["A", "B"],
            "TargetWeight": [1.0, 1.0],
        }
    )
    return run_portfolio_backtest(
        panel,
        targets,
        start="2025-01-02",
        end="2025-01-07",
        initial_capital=100.0,
        transaction_cost_bps=10.0,
        record_attribution=True,
    )


def test_ticker_attribution_reconciles_daily_equity_and_costs() -> None:
    result = _attributed_result()
    assert result.attribution is not None
    attributed_by_day = (
        result.attribution.groupby("Date")["NetPnL"].sum()
        .reindex(result.daily["Date"], fill_value=0.0)
        .to_numpy()
    )
    expected = np.diff(
        np.r_[result.summary.initial_capital, result.daily["Equity"]]
    )
    assert attributed_by_day == pytest.approx(expected)
    assert result.attribution["NetPnL"].sum() == pytest.approx(
        result.summary.final_value - result.summary.initial_capital
    )
    assert result.attribution["TransactionCost"].sum() == pytest.approx(
        result.executions["TransactionCost"].sum()
    )


def test_period_contributions_rank_and_reconcile() -> None:
    result = _attributed_result()
    contributions = summarize_ticker_contributions(
        result,
        {
            "all": ("2025-01-02", "2025-01-07"),
            "after_rotation": ("2025-01-06", "2025-01-07"),
        },
    )
    for period, group in contributions.groupby("Period"):
        assert group["NetPnL"].sum() == pytest.approx(
            group["PortfolioNetPnL"].iloc[0]
        ), period
        assert group["ReconciliationError"].iloc[0] == pytest.approx(0.0)
    top = contributions.loc[
        contributions["Period"].eq("all")
        & contributions["ContributionRank"].eq(1),
        "Ticker",
    ].item()
    assert top == "B"


def test_leave_winner_summary_reports_excess_retention() -> None:
    summary = pd.DataFrame(
        {
            "Scenario": ["BASELINE", "LEAVE_TOP_1_OUT"],
            "Period": ["2025", "2025"],
            "StrategyROI": [100.0, 20.0],
            "ExcessROI": [80.0, 0.0],
        }
    )
    result = _add_baseline_deltas(summary)
    leave_one = result.loc[
        result["Scenario"].eq("LEAVE_TOP_1_OUT")
    ].iloc[0]
    assert leave_one["StrategyROILossVsBaseline"] == pytest.approx(80.0)
    assert leave_one["ExcessReturnRetentionPct"] == pytest.approx(0.0)
    assert not leave_one["BeatSameUniverseBenchmark"]
