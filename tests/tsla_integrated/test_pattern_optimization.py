from __future__ import annotations

import pandas as pd

from stock_research.tsla_integrated.pattern_optimization import (
    ReentryFilterParams,
    add_obv_credit_features,
    generate_reentry_signal,
)


def test_obv_and_credit_proxy_are_causally_constructed() -> None:
    dates = pd.date_range("2024-01-01", periods=140, freq="B")
    patterns = pd.DataFrame(
        {
            "Date": dates,
            "AdjClose": [100 + index for index in range(140)],
            "Treasury10Y": [4.0] * 140,
            "VIX_Level": [20.0] * 140,
        }
    )
    volume = pd.DataFrame({"Date": dates, "Volume": [1_000.0] * 140})
    credit = pd.DataFrame({"Date": dates, "HYYield": [10.0] * 140})
    result = add_obv_credit_features(patterns, volume, credit)
    assert result.loc[139, "OBVFlow21"] == 1.0
    assert result.loc[139, "HYExcessYieldProxy"] == 6.0
    assert result.loc[139, "HYExcessYieldPercentile"] == 1.0


def test_reentry_signal_applies_optional_obv_and_macro_filters() -> None:
    frame = pd.DataFrame(
        {
            "Return60D": [-0.20, -0.20],
            "RSI14": [30.0, 30.0],
            "DistanceFromSMA200": [-0.10, -0.10],
            "OBVFlow21": [-0.2, 0.2],
            "OBVFlow63": [-0.1, 0.1],
            "OBV": [90.0, 110.0],
            "OBVSignal9": [100.0, 100.0],
            "VIX_Level": [30.0, 30.0],
            "VIXChange5": [-2.0, -2.0],
            "HYExcessYieldPercentile": [0.8, 0.8],
            "HYExcessYieldChange5": [-0.2, -0.2],
        }
    )
    params = ReentryFilterParams(
        obv_mode="FLOW21_POSITIVE",
        vix_mode="VIX20_FALLING",
        credit_mode="STRESS60_FALLING",
    )
    assert generate_reentry_signal(frame, params).tolist() == [False, True]
