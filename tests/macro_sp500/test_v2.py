from __future__ import annotations

import pandas as pd
import pytest

from stock_research.macro_sp500.config_v2 import (
    MacroSp500V2Params,
    MacroSp500V2Settings,
)
from stock_research.macro_sp500.portfolio_v2 import (
    run_static_weight_v2,
    run_v2_portfolio_from_signals,
)
from stock_research.macro_sp500.strategy_v2 import generate_v2_target_weights


def _params() -> MacroSp500V2Params:
    return MacroSp500V2Params(
        core_weight=0.60,
        vix_entry_quantile=0.90,
        drawdown_profile="10_20_30",
        target_profile="70_85_100",
        rebound_threshold=0.05,
        rebalance_band=0.05,
        minimum_hold_days=1,
    )


def test_v2_waits_for_reversal_before_final_tranche(
    v2_settings: MacroSp500V2Settings,
) -> None:
    features = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=5),
            "Close": [90.0, 75.0, 65.0, 70.0, 71.0],
            "VixPercentile": [0.91, 0.95, 0.97, 0.40, 0.40],
            "Drawdown": [-0.10, -0.25, -0.35, -0.30, -0.25],
            "SMA20": [92.0, 80.0, 70.0, 68.0, 69.0],
            "SMA200": [100.0, 95.0, 90.0, 69.0, 69.0],
            "Rebound20": [0.00, 0.00, 0.00, 0.08, 0.09],
            "VixOffPeak20": [0.00, -0.05, -0.10, -0.25, -0.30],
            "FeaturesReady": True,
        }
    )

    signals, _ = generate_v2_target_weights(features, _params(), v2_settings)

    assert signals["TargetWeight"].tolist() == pytest.approx(
        [0.70, 0.85, 0.85, 1.00, 0.60]
    )
    assert signals.loc[2, "ReversalConfirmed"] == 0
    assert signals.loc[3, "ReversalConfirmed"] == 1
    assert signals.loc[4, "Reason"].startswith("RECOVERY_EXIT")


def test_v2_rebalance_band_avoids_daily_micro_trades(
    v2_settings: MacroSp500V2Settings,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    features = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "VIX": 15.0,
            "VixPercentile": 0.50,
            "Drawdown": 0.0,
            "CashRate": 0.0,
        }
    )
    signals = pd.DataFrame(
        {
            "Date": dates,
            "State": "NORMAL",
            "TargetWeight": 0.70,
            "CoreWeight": 0.70,
            "RebalanceBand": 0.05,
            "Reason": "NORMAL_CORE",
        }
    )

    result = run_v2_portfolio_from_signals(features, signals, v2_settings)

    assert result.summary.rebalance_count == 1
    assert result.trades.iloc[0]["Action"] == "BUY"


def test_v2_cash_receives_historical_rate(
    v2_settings: MacroSp500V2Settings,
) -> None:
    features = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2025-01-01"]),
            "Open": [100.0, 100.0],
            "Close": [100.0, 100.0],
            "VIX": [15.0, 15.0],
            "VixPercentile": [0.50, 0.50],
            "Drawdown": [0.0, 0.0],
            "CashRate": [10.0, 10.0],
        }
    )

    result = run_static_weight_v2(features, v2_settings, weight=0.0)

    assert result.summary.final_value == pytest.approx(110_000.0, rel=1e-3)
