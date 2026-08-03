from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.config import StrategyParams
from stock_research.cross_sectional.portfolio import run_portfolio_backtest
from stock_research.cross_sectional.v7_trade_report import (
    add_factor_contributions,
    build_execution_ledger,
    explain_trade,
)


def _params() -> StrategyParams:
    return StrategyParams(
        momentum_weight=0.04,
        trend_weight=0.36,
        growth_weight=0.44,
        quality_weight=0.155,
        risk_control_weight=0.005,
        top_k=5,
        exit_rank=9,
        trend_floor=0.05,
        momentum_floor=0.10,
        loss_aware_exit_enabled=True,
        minimum_exit_gain=0.01,
        conviction_exit_rank=20,
        conviction_trend_floor=-0.10,
        conviction_momentum_floor=-0.20,
        hard_stop_return=-0.35,
        minimum_hold_rebalances=4,
        profit_rotation_exit_rank=9,
        replacement_score_advantage=0.05,
    )


def test_execution_ledger_reconciles_exact_portfolio() -> None:
    dates = pd.to_datetime(
        ["2025-01-03", "2025-01-06", "2025-01-10", "2025-01-13"]
    )
    panel = pd.DataFrame(
        [
            {
                "Date": date,
                "Ticker": ticker,
                "Open": price,
                "Close": price + 1,
            }
            for date, a, b in zip(
                dates,
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                strict=True,
            )
            for ticker, price in (("A", a), ("B", b))
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "Date": dates[0],
                "Ticker": "A",
                "TargetWeight": 1.0,
                "TradeAction": "BUY",
            },
            {
                "Date": dates[0],
                "Ticker": "B",
                "TargetWeight": 0.0,
                "TradeAction": "NONE",
            },
            {
                "Date": dates[2],
                "Ticker": "A",
                "TargetWeight": 0.0,
                "TradeAction": "SELL",
                "ExitReason": "PROFITABLE_ROTATION",
            },
            {
                "Date": dates[2],
                "Ticker": "B",
                "TargetWeight": 1.0,
                "TradeAction": "BUY",
            },
        ]
    )
    portfolio = run_portfolio_backtest(
        panel,
        targets,
        start="2025-01-03",
        end="2025-01-13",
        initial_capital=100_000,
        transaction_cost_bps=10,
    )
    ledger, checks = build_execution_ledger(
        panel,
        targets,
        portfolio,
        start="2025-01-03",
        end="2025-01-13",
        initial_capital=100_000,
        transaction_cost_bps=10,
    )
    assert set(ledger["ExecutionType"]) == {"OPEN", "CLOSE"}
    assert len(ledger) == 3
    assert (
        checks[["EquityError", "TurnoverError", "CostError"]]
        .abs()
        .max()
        .max()
        < 1e-8
    )
    first = ledger.iloc[0]
    assert first["NotionalAfter"] == pytest.approx(99_900)
    assert first["SharesAfter"] == pytest.approx(99_900 / 11)


def test_factor_contributions_sum_to_alpha() -> None:
    params = _params()
    frame = pd.DataFrame(
        {
            "MomentumFactor": [0.2],
            "MAFactor": [0.3],
            "MACDFactor": [0.1],
            "OBVFactor": [0.2],
            "GrowthFactor": [0.4],
            "QualityFactor": [0.1],
            "RiskControlFactor": [-0.2],
        }
    )
    enriched = add_factor_contributions(frame, params)
    contributions = enriched[
        [
            "MomentumContribution",
            "MAContribution",
            "MACDContribution",
            "OBVContribution",
            "GrowthContribution",
            "QualityContribution",
            "RiskContribution",
        ]
    ].iloc[0]
    expected = (
        0.2 * 0.04
        + ((0.3 + 0.1 + 0.2) / 3) * 0.36
        + 0.4 * 0.44
        + 0.1 * 0.155
        - 0.2 * 0.005
    )
    assert contributions.sum() == pytest.approx(expected)
    assert "재무 성장" in enriched.iloc[0]["TopFactorContributions"]


def test_explain_profitable_rotation_contains_thresholds() -> None:
    reason = explain_trade(
        pd.Series(
            {
                "TradeAction": "SELL",
                "ExitReason": "PROFITABLE_ROTATION",
                "SignalReferenceReturn": 0.12,
                "Rank": 11,
                "ReplacementScoreAdvantage": 0.08,
            }
        ),
        _params(),
    )
    assert "수익 +12.0%" in reason
    assert "회전 기준 9위 밖" in reason
    assert "필요 0.050" in reason
