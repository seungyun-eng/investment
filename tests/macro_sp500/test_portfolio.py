from __future__ import annotations

import pandas as pd

from stock_research.macro_sp500.config import (
    MacroSp500Params,
    MacroSp500Settings,
)
from stock_research.macro_sp500.portfolio import run_target_weight_backtest


def test_signal_is_executed_at_next_open_and_roi_uses_fixed_injected_capital(
    macro_settings: MacroSp500Settings,
) -> None:
    params = MacroSp500Params(
        vix_lookback_years=3,
        core_weight=0.50,
        warning_score_min=2,
        warning_addition=0.50,
        vix_entry_quantile=0.90,
        exit_vix_quantile=0.50,
        minimum_hold_days=1,
    )
    dates = pd.bdate_range("2024-01-02", periods=3)
    features = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 100.0, 100.0],
            "Close": [100.0, 100.0, 100.0],
            "VIX": [15.0, 15.0, 15.0],
            "VixPercentile": [0.50, 0.50, 0.50],
            "Drawdown": [0.0, 0.0, 0.0],
            "WarningScore": [3, 3, 3],
            "FeaturesReady": [True, True, True],
        }
    )

    result = run_target_weight_backtest(features, params, macro_settings)
    warning_trade = result.trades.iloc[1]

    assert warning_trade["Date"] == dates[1]
    assert warning_trade["SignalDate"] == dates[0]
    assert warning_trade["Reason"] == "WARNING_SCORE_3"
    assert result.summary.total_injected == 100_000.0
    assert result.summary.final_value == 100_000.0
    assert result.summary.roi_percent == 0.0
