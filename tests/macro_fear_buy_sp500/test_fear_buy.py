from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.macro_fear_buy_sp500.config import (
    FearBuyParams,
    FearBuySettings,
)
from stock_research.macro_fear_buy_sp500.features import build_fear_features
from stock_research.macro_fear_buy_sp500.portfolio import run_signal_backtest
from stock_research.macro_fear_buy_sp500.strategy import (
    generate_fear_buy_signals,
)


def _predictions(periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = 100.0 + np.arange(periods) * 0.05
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "Close": close,
            "CashRate": 0.0,
            "VIX": np.linspace(15.0, 30.0, periods),
            "Drawdown252": 0.0,
            "MacroConfirmationScore": 0.50,
            "RiskProbability_63": np.linspace(0.10, 0.60, periods),
            "RiskProbability_126": np.linspace(0.20, 0.70, periods),
        }
    )


def test_feature_history_is_point_in_time() -> None:
    params = FearBuyParams(percentile_min_history_days=20, vix_lookback_days=50)
    original = _predictions(100)
    extended = pd.concat([original, _predictions(120).iloc[100:]], ignore_index=True)

    original_features = build_fear_features(original, params)
    extended_features = build_fear_features(extended, params).iloc[: len(original)]

    pd.testing.assert_series_equal(
        original_features["FearScore"],
        extended_features["FearScore"],
        check_names=False,
    )


def test_deeper_fear_adds_weekly_tranches() -> None:
    params = FearBuyParams(
        core_weight=0.70,
        tranche_weight=0.10,
        minimum_hold_sessions=100,
    )
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    features = pd.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 95.0, 90.0, 85.0],
            "VixPercentile": [0.50, 0.85, 0.93, 0.99],
            "FearScore": [0.40, 0.60, 0.70, 0.80],
            "EuphoriaScore": [0.20, 0.10, 0.10, 0.05],
            "Drawdown252": [0.0, -0.09, -0.16, -0.26],
        }
    )

    signals = generate_fear_buy_signals(features, params)

    assert np.allclose(
        signals["TargetWeight"],
        [0.70, 0.80, 0.90, 1.00],
    )
    assert signals["TransitionReason"].iloc[-1] == "PANIC_BUY_TRANCHE"


def test_tactical_trim_requires_hold_profit_and_euphoria() -> None:
    params = FearBuyParams(
        core_weight=0.70,
        tranche_weight=0.10,
        minimum_hold_sessions=2,
        trim_profit_buffer=0.02,
    )
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    features = pd.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 99.0, 101.0, 103.0],
            "VixPercentile": [0.90, 0.30, 0.30, 0.30],
            "FearScore": [0.70, 0.30, 0.30, 0.30],
            "EuphoriaScore": [0.10, 0.70, 0.70, 0.70],
            "Drawdown252": [-0.16, -0.01, -0.01, -0.01],
        }
    )

    signals = generate_fear_buy_signals(features, params)

    assert np.allclose(
        signals["TargetWeight"],
        [0.80, 0.80, 0.80, 0.70],
    )
    assert signals["TransitionReason"].iloc[-1] == "EUPHORIA_PROFIT_TRIM"


def test_backtest_executes_prior_signal_and_never_sells_core() -> None:
    params = FearBuyParams(core_weight=0.70, tranche_weight=0.10)
    settings = FearBuySettings(
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    dates = pd.bdate_range("2024-01-01", periods=5)
    signals = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 90.0, 80.0, 100.0, 110.0],
            "Close": [100.0, 90.0, 80.0, 100.0, 110.0],
            "CashRate": 0.0,
            "TargetWeight": [0.70, 0.80, 0.90, 0.80, 0.70],
            "SignalState": ["CORE", "MILD", "FEAR", "TRIM", "CORE"],
            "TransitionReason": ["", "BUY", "BUY", "SELL", "SELL"],
        }
    )

    result = run_signal_backtest(signals, params, settings, name="test")

    assert result.trades.iloc[0]["Sleeve"] == "CORE"
    assert result.trades.iloc[1]["Date"] == dates[2]
    assert result.trades.iloc[1]["SignalDate"] == dates[1]
    assert result.trades.query("Action == 'SELL'")["Sleeve"].eq("TACTICAL").all()
    assert result.daily["CoreShares"].nunique() == 1
    assert np.isclose(
        result.summary.roi_percent,
        (result.summary.final_value / result.summary.total_injected - 1.0) * 100.0,
    )
