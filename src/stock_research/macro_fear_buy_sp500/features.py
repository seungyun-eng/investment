from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FearBuyParams

REQUIRED_COLUMNS = {
    "Date",
    "Open",
    "Close",
    "CashRate",
    "VIX",
    "Drawdown252",
    "MacroConfirmationScore",
    "RiskProbability_63",
    "RiskProbability_126",
}


def _rolling_percentile(
    values: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Percentile of today's observation using only history available today."""

    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rolling(window, min_periods=min_periods).rank(pct=True)


def build_fear_features(
    predictions: pd.DataFrame,
    params: FearBuyParams,
) -> pd.DataFrame:
    """Build point-in-time fear and euphoria scores from strict-OOS inputs."""

    missing = REQUIRED_COLUMNS - set(predictions)
    if missing:
        raise ValueError(f"Fear-buy inputs are missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = (
        frame.dropna(subset=["Date", "Open", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    frame["VixPercentile"] = _rolling_percentile(
        frame["VIX"],
        window=params.vix_lookback_days,
        min_periods=params.percentile_min_history_days,
    )
    frame["ModelRiskRaw"] = (
        0.40 * pd.to_numeric(frame["RiskProbability_63"], errors="coerce")
        + 0.60 * pd.to_numeric(frame["RiskProbability_126"], errors="coerce")
    )
    frame["ModelRiskPercentile"] = _rolling_percentile(
        frame["ModelRiskRaw"],
        window=params.vix_lookback_days,
        min_periods=params.percentile_min_history_days,
    )
    frame["Momentum63"] = frame["Close"].pct_change(63)
    frame["Momentum126"] = frame["Close"].pct_change(126)
    frame["SMA200"] = frame["Close"].rolling(200, min_periods=200).mean()
    frame["SMA200Extension"] = frame["Close"] / frame["SMA200"] - 1.0
    frame["DrawdownIntensity"] = (
        -pd.to_numeric(frame["Drawdown252"], errors="coerce") / 0.25
    ).clip(0.0, 1.0)
    frame["DownsideMomentumIntensity"] = (-frame["Momentum63"] / 0.20).clip(
        0.0,
        1.0,
    )
    macro = pd.to_numeric(
        frame["MacroConfirmationScore"],
        errors="coerce",
    ).clip(0.0, 1.0)
    frame["FearScore"] = (
        params.vix_weight * frame["VixPercentile"]
        + params.macro_weight * macro
        + params.model_risk_weight * frame["ModelRiskPercentile"]
        + params.drawdown_weight * frame["DrawdownIntensity"]
        + params.downside_momentum_weight * frame["DownsideMomentumIntensity"]
    ).clip(0.0, 1.0)
    trend_extension = ((frame["SMA200Extension"] - 0.05) / 0.15).clip(0.0, 1.0)
    positive_momentum = ((frame["Momentum126"] - 0.05) / 0.25).clip(0.0, 1.0)
    frame["EuphoriaScore"] = (
        0.30 * (1.0 - frame["VixPercentile"])
        + 0.20 * (1.0 - macro)
        + 0.30 * trend_extension
        + 0.20 * positive_momentum
    ).clip(0.0, 1.0)
    frame["FearScoreAvailable"] = np.isfinite(frame["FearScore"])
    return frame
