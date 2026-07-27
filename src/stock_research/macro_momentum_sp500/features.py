from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig

IDENTIFIER_COLUMNS = {
    "Date",
    "Open",
    "Close",
    "Volume",
    "CashRate",
}


def _trailing_percentile(values: pd.Series, window: int, minimum: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan)
    for index, current in enumerate(numeric):
        if not np.isfinite(current):
            continue
        history = numeric[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if len(history) >= minimum:
            output[index] = float(np.mean(history <= current))
    return pd.Series(output, index=values.index, dtype=float)


def _trailing_zscore(values: pd.Series, window: int, minimum: int) -> pd.Series:
    history = pd.to_numeric(values, errors="coerce").shift(1).rolling(window, min_periods=minimum)
    mean = history.mean()
    std = history.std(ddof=1).replace(0, np.nan)
    return (pd.to_numeric(values, errors="coerce") - mean) / std


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    relative = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative)


def _add_changes(frame: pd.DataFrame, name: str) -> None:
    values = pd.to_numeric(frame[name], errors="coerce")
    change_21 = values.diff(21)
    frame[f"{name}_Change5"] = values.diff(5)
    frame[f"{name}_Change21"] = change_21
    frame[f"{name}_PctChange21"] = values.pct_change(21, fill_method=None)
    frame[f"{name}_Z252"] = _trailing_zscore(values, 252, 126)
    frame[f"{name}_Change21_Z252"] = _trailing_zscore(change_21, 252, 126)


def _stress_probability(values: pd.Series, direction: float = 1.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").clip(-6, 6) * direction
    return 1.0 / (1.0 + np.exp(-numeric))


def _channel_score(
    frame: pd.DataFrame,
    specifications: tuple[tuple[str, float], ...],
) -> pd.Series:
    components = [
        _stress_probability(frame[name], direction)
        for name, direction in specifications
        if name in frame
    ]
    if not components:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(components, axis=1).mean(axis=1)


def build_features(data: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    required = {"Date", "Open", "Close", "Volume", "VIX", "CashRate"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Research data is missing columns: {sorted(missing)}")
    frame = data.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for name in frame.columns:
        if name != "Date" and not name.endswith("ObservationDate"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["Date", "Open", "Close", "Volume"]).reset_index(drop=True)

    close = frame["Close"]
    log_return = np.log(close).diff()
    frame["Return1"] = close.pct_change(fill_method=None)
    for window in config.momentum_windows:
        frame[f"Momentum_{window}"] = close.pct_change(window, fill_method=None)

    for window in config.sma_windows:
        sma = close.rolling(window, min_periods=window).mean()
        frame[f"SMA{window}_Ratio"] = close / sma - 1
        slope_window = max(5, min(21, window // 4))
        frame[f"SMA{window}_Slope"] = sma.pct_change(slope_window, fill_method=None)

    for window in config.volatility_windows:
        frame[f"RealizedVol_{window}"] = (
            log_return.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
        )
        downside = log_return.clip(upper=0)
        frame[f"DownsideVol_{window}"] = (
            downside.pow(2).rolling(window, min_periods=window).mean().pow(0.5)
            * np.sqrt(252)
        )

    rolling_peak = close.rolling(252, min_periods=63).max()
    rolling_low = close.rolling(63, min_periods=21).min()
    frame["Drawdown252"] = close / rolling_peak - 1
    frame["Rebound63"] = close / rolling_low - 1
    frame["RSI14"] = _rsi(close)
    frame["VolumeZ63"] = (
        (np.log1p(frame["Volume"]) - np.log1p(frame["Volume"]).shift(1).rolling(63).mean())
        / np.log1p(frame["Volume"]).shift(1).rolling(63).std(ddof=1)
    )

    for window in config.distribution_windows:
        minimum = max(126, min(window // 2, 500))
        frame[f"VIX_Percentile_{window}"] = _trailing_percentile(
            frame["VIX"], window, minimum
        )
        frame[f"VIX_Z_{window}"] = _trailing_zscore(frame["VIX"], window, minimum)
    _add_changes(frame, "VIX")

    macro_names = (
        "VIX3M",
        "T10Y2Y",
        "T10Y3M",
        "NFCI",
        "GS10",
        "TB3MS",
        "BAA10Y",
        "HYOAS",
        "HYYield",
        "CPI",
        "FedFundsKnown",
        "Unemployment",
        "WTI",
        "YieldCurveLegacy",
    )
    for name in macro_names:
        if name in frame:
            _add_changes(frame, name)

    if {"VIX", "VIX3M"} <= set(frame):
        frame["VIX_TermStructure"] = frame["VIX"] / frame["VIX3M"] - 1
        _add_changes(frame, "VIX_TermStructure")
    if {"HYYield", "GS10"} <= set(frame):
        # Explicitly a proxy, not the official option-adjusted spread.
        frame["HYExcessYieldProxy"] = frame["HYYield"] - frame["GS10"]
    if {"Momentum_63", "VIX_Z_252"} <= set(frame):
        frame["Interaction_Momentum63_VIXZ"] = frame["Momentum_63"] * frame["VIX_Z_252"]
    if {"Drawdown252", "HYYield_Z252"} <= set(frame):
        frame["Interaction_Drawdown_HY"] = frame["Drawdown252"] * frame["HYYield_Z252"]
    if {"SMA200_Ratio", "YieldCurveLegacy"} <= set(frame):
        frame["Interaction_Trend_Curve"] = (
            frame["SMA200_Ratio"] * frame["YieldCurveLegacy"]
        )

    frame = frame.copy()
    level_channels = {
        "Volatility": (
            ("VIX_Z_1260", 1.0),
            ("VIX_TermStructure_Z252", 1.0),
        ),
        "Credit": (
            ("HYYield_Z252", 1.0),
            ("BAA10Y_Z252", 1.0),
        ),
        "FinancialConditions": (("NFCI_Z252", 1.0),),
        "Labor": (("Unemployment_Z252", 1.0),),
        "YieldCurve": (
            ("T10Y3M_Z252", -1.0),
            ("T10Y2Y_Z252", -1.0),
        ),
    }
    trend_channels = {
        "Volatility": (
            ("VIX_Change21_Z252", 1.0),
            ("VIX_TermStructure_Change21_Z252", 1.0),
        ),
        "Credit": (
            ("HYYield_Change21_Z252", 1.0),
            ("BAA10Y_Change21_Z252", 1.0),
        ),
        "FinancialConditions": (("NFCI_Change21_Z252", 1.0),),
        "Labor": (("Unemployment_Change21_Z252", 1.0),),
        "YieldCurve": (
            ("T10Y3M_Change21_Z252", -1.0),
            ("T10Y2Y_Change21_Z252", -1.0),
        ),
    }
    level_names: list[str] = []
    trend_names: list[str] = []
    for channel, specifications in level_channels.items():
        name = f"MacroLevel_{channel}"
        frame[name] = _channel_score(frame, specifications)
        level_names.append(name)
    for channel, specifications in trend_channels.items():
        name = f"MacroTrend_{channel}"
        frame[name] = _channel_score(frame, specifications)
        trend_names.append(name)
    frame["MacroStressLevel"] = frame[level_names].mean(axis=1)
    frame["MacroStressTrend"] = frame[trend_names].mean(axis=1)
    level = frame["MacroStressLevel"]
    trend = frame["MacroStressTrend"]
    available_weight = level.notna().astype(float) * 0.65 + trend.notna().astype(float) * 0.35
    frame["MacroConfirmationScore"] = (
        level.fillna(0) * 0.65 + trend.fillna(0) * 0.35
    ) / available_weight.replace(0, np.nan)
    frame["MacroStressBreadth"] = frame[level_names].gt(0.50).where(
        frame[level_names].notna()
    ).mean(axis=1)

    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    frame.attrs.update(data.attrs)
    frame.attrs["feature_information_timing"] = (
        "close-of-day features; eligible for next-trading-day execution"
    )
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_suffixes = ("ObservationDate", "TargetEndDate")
    excluded_prefixes = ("Forward", "ExcessReturn_", "DrawdownHit_", "FutureMinReturn_")
    result = []
    for name in frame.columns:
        if name in IDENTIFIER_COLUMNS or name.endswith(excluded_suffixes):
            continue
        if name.startswith(excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[name]):
            result.append(name)
    return result


def select_feature_group(
    columns: list[str],
    group: str,
    config: ResearchConfig,
) -> list[str]:
    patterns = config.feature_groups[group]
    if patterns == ("*",):
        return columns.copy()
    return [name for name in columns if any(name.startswith(prefix) for prefix in patterns)]
