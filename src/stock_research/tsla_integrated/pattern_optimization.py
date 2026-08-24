from __future__ import annotations

"""Feature and signal helpers for the 50%-rise/50%-sell TSLA strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReentryFilterParams:
    return60_max: float = -0.10
    rsi_max: float = 42.0
    distance_sma200_max: float = 0.0
    obv_mode: str = "NONE"
    vix_mode: str = "NONE"
    credit_mode: str = "NONE"

    def validate(self) -> None:
        if self.obv_mode not in {"NONE", "FLOW21_POSITIVE", "FLOW21_IMPROVING", "ABOVE_SIGNAL9"}:
            raise ValueError("Unknown obv_mode.")
        if self.vix_mode not in {"NONE", "VIX20", "VIX25", "VIX_FALLING", "VIX20_FALLING"}:
            raise ValueError("Unknown vix_mode.")
        if self.credit_mode not in {"NONE", "STRESS60", "FALLING", "STRESS60_FALLING"}:
            raise ValueError("Unknown credit_mode.")


def _rolling_percentile(series: pd.Series, window: int = 756) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    def percentile(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return np.nan
        return float(np.count_nonzero(finite <= finite[-1]) / len(finite))

    return numeric.rolling(window, min_periods=126).apply(percentile, raw=True)


def add_obv_credit_features(
    patterns: pd.DataFrame,
    volume_history: pd.DataFrame,
    high_yield_effective_yield: pd.DataFrame,
) -> pd.DataFrame:
    """Add causal OBV and truthfully labelled credit-yield proxy features."""

    required_pattern = {"Date", "AdjClose", "Treasury10Y", "VIX_Level"}
    if missing := required_pattern.difference(patterns.columns):
        raise ValueError(f"Missing pattern columns: {sorted(missing)}")
    if missing := {"Date", "Volume"}.difference(volume_history.columns):
        raise ValueError(f"Missing volume columns: {sorted(missing)}")
    if missing := {"Date", "HYYield"}.difference(high_yield_effective_yield.columns):
        raise ValueError(f"Missing high-yield columns: {sorted(missing)}")
    frame = patterns.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    volume = volume_history[["Date", "Volume"]].copy()
    volume["Date"] = pd.to_datetime(volume["Date"], errors="coerce")
    volume["Volume"] = pd.to_numeric(volume["Volume"], errors="coerce")
    frame = frame.merge(volume, how="left", on="Date", validate="one_to_one")
    close = pd.to_numeric(frame["AdjClose"], errors="coerce")
    signed_volume = np.sign(close.diff()).fillna(0.0) * frame["Volume"]
    absolute_volume = frame["Volume"].abs()
    frame["OBV"] = signed_volume.fillna(0.0).cumsum()
    frame["OBVSignal9"] = frame["OBV"].rolling(9, min_periods=5).mean()
    frame["OBVFlow21"] = (
        signed_volume.rolling(21, min_periods=10).sum()
        / absolute_volume.rolling(21, min_periods=10).sum()
    )
    frame["OBVFlow63"] = (
        signed_volume.rolling(63, min_periods=21).sum()
        / absolute_volume.rolling(63, min_periods=21).sum()
    )

    credit = high_yield_effective_yield[["Date", "HYYield"]].copy()
    credit["Date"] = pd.to_datetime(credit["Date"], errors="coerce")
    credit["HYYield"] = pd.to_numeric(credit["HYYield"], errors="coerce")
    credit = credit.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    frame = pd.merge_asof(
        frame.sort_values("Date"),
        credit,
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    frame["HYExcessYieldProxy"] = frame["HYYield"] - pd.to_numeric(
        frame["Treasury10Y"], errors="coerce"
    )
    frame["HYExcessYieldPercentile"] = _rolling_percentile(frame["HYExcessYieldProxy"])
    frame["HYExcessYieldChange5"] = frame["HYExcessYieldProxy"].diff(5)
    frame["VIXChange5"] = pd.to_numeric(frame["VIX_Level"], errors="coerce").diff(5)
    frame["VIXPercentile"] = _rolling_percentile(frame["VIX_Level"])
    return frame


def generate_reentry_signal(
    features: pd.DataFrame,
    params: ReentryFilterParams,
) -> pd.Series:
    """Generate the exact causal re-entry mask shared by search and simulation."""

    params.validate()
    required = {"Return60D", "RSI14", "DistanceFromSMA200"}
    if missing := required.difference(features.columns):
        raise ValueError(f"Missing re-entry columns: {sorted(missing)}")
    signal = (
        pd.to_numeric(features["Return60D"], errors="coerce").le(params.return60_max)
        & pd.to_numeric(features["RSI14"], errors="coerce").le(params.rsi_max)
        & pd.to_numeric(features["DistanceFromSMA200"], errors="coerce").le(
            params.distance_sma200_max
        )
    )
    if params.obv_mode == "FLOW21_POSITIVE":
        signal &= pd.to_numeric(features["OBVFlow21"], errors="coerce").gt(0.0)
    elif params.obv_mode == "FLOW21_IMPROVING":
        signal &= pd.to_numeric(features["OBVFlow21"], errors="coerce").gt(
            pd.to_numeric(features["OBVFlow63"], errors="coerce")
        )
    elif params.obv_mode == "ABOVE_SIGNAL9":
        signal &= pd.to_numeric(features["OBV"], errors="coerce").gt(
            pd.to_numeric(features["OBVSignal9"], errors="coerce")
        )

    vix = pd.to_numeric(features.get("VIX_Level"), errors="coerce")
    vix_change = pd.to_numeric(features.get("VIXChange5"), errors="coerce")
    if params.vix_mode == "VIX20":
        signal &= vix.ge(20.0)
    elif params.vix_mode == "VIX25":
        signal &= vix.ge(25.0)
    elif params.vix_mode == "VIX_FALLING":
        signal &= vix_change.lt(0.0)
    elif params.vix_mode == "VIX20_FALLING":
        signal &= vix.ge(20.0) & vix_change.lt(0.0)

    credit_percentile = pd.to_numeric(
        features.get("HYExcessYieldPercentile"), errors="coerce"
    )
    credit_change = pd.to_numeric(features.get("HYExcessYieldChange5"), errors="coerce")
    if params.credit_mode == "STRESS60":
        signal &= credit_percentile.ge(0.60)
    elif params.credit_mode == "FALLING":
        signal &= credit_change.lt(0.0)
    elif params.credit_mode == "STRESS60_FALLING":
        signal &= credit_percentile.ge(0.60) & credit_change.lt(0.0)
    return signal.fillna(False).astype(bool)
