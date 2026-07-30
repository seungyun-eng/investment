from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.config import StrategyParams
from stock_research.cross_sectional.v7_slot_sweep import (
    balanced_slot_ranking,
    slot_sweep_params,
    spy_buy_and_hold,
)


def _params() -> StrategyParams:
    return StrategyParams(
        momentum_weight=0.10,
        trend_weight=0.20,
        growth_weight=0.30,
        quality_weight=0.30,
        risk_control_weight=0.10,
        top_k=3,
        exit_rank=9,
        trend_floor=0.05,
        momentum_floor=0.10,
        loss_aware_exit_enabled=True,
        profit_rotation_exit_rank=9,
        replacement_score_advantage=0.05,
    )


def test_slot_sweep_preserves_four_rank_exit_buffer() -> None:
    base = _params()
    updated = slot_sweep_params(base, 10, exit_buffer=4)
    assert updated.top_k == 10
    assert updated.exit_rank == 14
    assert updated.profit_rotation_exit_rank == 14
    for key, value in base.as_dict().items():
        if key not in {
            "top_k",
            "exit_rank",
            "profit_rotation_exit_rank",
        }:
            assert updated.as_dict()[key] == value


def test_spy_buy_and_hold_uses_adjusted_open_and_entry_cost() -> None:
    spy = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "Open": [100.0, 109.0],
            "Close": [105.0, 110.0],
        }
    )
    summary, equity = spy_buy_and_hold(
        spy,
        start="2025-01-01",
        end="2025-12-31",
        initial_capital=100_000.0,
        transaction_cost_bps=10.0,
    )
    assert summary["FinalValue"] == pytest.approx(109_890.0)
    assert summary["ROI"] == pytest.approx(9.89)
    assert equity.iloc[-1]["Equity"] == pytest.approx(109_890.0)


def test_balanced_ranking_rewards_consistent_candidate() -> None:
    rows = []
    for top_k, cagr, sharpe, drawdown, excess in (
        (1, [100, -20, 80], [2.0, -0.5, 1.8], [-50, -60, -40], [50, -30, 40]),
        (5, [30, 25, 35], [1.0, 0.9, 1.1], [-20, -18, -15], [10, 8, 12]),
    ):
        for index in range(3):
            rows.append(
                {
                    "TopK": top_k,
                    "CAGR": cagr[index],
                    "Sharpe": sharpe[index],
                    "MaxDrawdown": drawdown[index],
                    "SPYExcessROI": excess[index],
                }
            )
    result = balanced_slot_ranking(pd.DataFrame(rows))
    assert result.iloc[0]["TopK"] == 5
    assert result.iloc[0]["BalancedRank"] == 1
