from __future__ import annotations

import pandas as pd

from stock_research.macro_sp500.config import (
    MacroSp500Params,
    MacroSp500Settings,
)
from stock_research.macro_sp500.optimization import (
    _candidate_score,
    optimize_walk_forward,
)
from stock_research.macro_sp500.portfolio import PerformanceSummary


def _summary(*, cagr: float, calmar: float) -> PerformanceSummary:
    return PerformanceSummary(
        initial_capital=100_000.0,
        total_injected=100_000.0,
        final_value=100_000.0,
        roi_percent=0.0,
        cagr_percent=cagr,
        max_drawdown_percent=-10.0,
        calmar_ratio=calmar,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        average_exposure_percent=70.0,
        turnover_multiple=1.0,
        rebalance_count=1,
    )


def test_candidate_below_cagr_floor_is_not_eligible(
    macro_settings: MacroSp500Settings,
) -> None:
    strategy = _summary(cagr=7.0, calmar=5.0)
    benchmark = _summary(cagr=10.0, calmar=1.0)
    settings = MacroSp500Settings(
        **{
            **macro_settings.__dict__,
            "minimum_cagr_fraction_of_buy_hold": 0.8,
        }
    )

    score = _candidate_score(strategy, benchmark, settings)

    assert score[0] == float("-inf")


def test_walk_forward_selects_only_on_prior_years_and_chains_oos_capital(
    synthetic_macro_data: pd.DataFrame,
    macro_settings: MacroSp500Settings,
    macro_params: MacroSp500Params,
) -> None:
    result = optimize_walk_forward(
        synthetic_macro_data,
        [macro_params],
        macro_settings,
    )

    assert result.folds["TestYear"].tolist() == [2020, 2021]
    assert (
        pd.to_datetime(result.folds["TrainEnd"])
        < pd.to_datetime(result.folds["TestStart"])
    ).all()
    assert result.oos_daily["Date"].is_monotonic_increasing
    assert result.oos_daily["TotalInjected"].nunique() == 1
    assert result.oos_daily["TotalInjected"].iloc[0] == 100_000.0
    expected_roi = (
        result.oos_summary.final_value / result.oos_summary.total_injected - 1.0
    ) * 100.0
    assert result.oos_summary.roi_percent == expected_roi
