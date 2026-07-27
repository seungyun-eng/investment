from __future__ import annotations

import pandas as pd

from .config_v2 import MacroSp500V2Settings
from .features import add_macro_features


def add_v2_features(
    data: pd.DataFrame,
    settings: MacroSp500V2Settings,
) -> pd.DataFrame:
    frame = add_macro_features(
        data,
        vix_lookback_years=settings.vix_lookback_years,
        warning_lookback_days=settings.warning_lookback_days,
        drawdown_lookback_days=settings.drawdown_lookback_days,
        minimum_vix_observations=settings.minimum_vix_observations,
    )
    close = frame["Close"]
    vix = frame["VIX"]
    reversal_days = settings.reversal_lookback_days
    frame["SMA20"] = close.rolling(reversal_days).mean()
    frame["SMA200"] = close.rolling(settings.recovery_sma_days).mean()
    frame["Low20"] = close.rolling(reversal_days).min()
    frame["Rebound20"] = close / frame["Low20"] - 1.0
    frame["VixPeak20"] = vix.rolling(reversal_days).max()
    frame["VixOffPeak20"] = vix / frame["VixPeak20"] - 1.0
    required = [
        "VixPercentile",
        "Drawdown",
        "SMA20",
        "SMA200",
        "Rebound20",
        "VixOffPeak20",
    ]
    frame["FeaturesReady"] = frame[required].notna().all(axis=1)
    return frame
