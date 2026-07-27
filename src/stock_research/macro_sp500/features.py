from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_vix_percentile(
    dates: pd.Series,
    values: pd.Series,
    *,
    lookback_years: int,
    minimum_observations: int,
) -> pd.Series:
    parsed_dates = pd.to_datetime(dates, errors="coerce").reset_index(drop=True)
    numeric_values = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    result = np.full(len(numeric_values), np.nan, dtype=float)
    for index in range(len(numeric_values)):
        current_date = parsed_dates.iloc[index]
        current_value = numeric_values.iloc[index]
        if pd.isna(current_date) or pd.isna(current_value):
            continue
        left_date = current_date - pd.DateOffset(years=lookback_years)
        left = int(parsed_dates.searchsorted(left_date, side="left"))
        history = numeric_values.iloc[left:index].dropna().to_numpy(dtype=float)
        if len(history) < minimum_observations:
            continue
        result[index] = float(np.mean(history <= float(current_value)))
    return pd.Series(result, index=dates.index, name="VixPercentile")


def add_macro_features(
    data: pd.DataFrame,
    *,
    vix_lookback_years: int,
    warning_lookback_days: int = 10,
    drawdown_lookback_days: int = 60,
    minimum_vix_observations: int = 500,
) -> pd.DataFrame:
    required = {"Date", "Close", "Open", "Volume", "VIX"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Macro feature data is missing columns: {sorted(missing)}")
    frame = data.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ("Close", "Open", "Volume", "VIX"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).reset_index(drop=True)

    close = frame["Close"]
    delta = close.diff()
    average_gain = delta.clip(lower=0).rolling(14).mean()
    average_loss = (-delta.clip(upper=0)).rolling(14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    frame["RSI14"] = 100 - 100 / (1 + relative_strength)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["MACD"] = ema12 - ema26
    frame["MACDSignal"] = frame["MACD"].ewm(span=9, adjust=False).mean()
    frame["MACDGap"] = frame["MACD"] - frame["MACDSignal"]

    direction = np.sign(close.diff()).fillna(0)
    frame["OBV"] = (direction * frame["Volume"]).cumsum()
    frame["OBVSignal9"] = frame["OBV"].rolling(9).mean()

    frame["ReturnWarning"] = close.pct_change(warning_lookback_days)
    frame["RSIChangeWarning"] = frame["RSI14"].diff(warning_lookback_days)
    frame["MACDGapChangeWarning"] = frame["MACDGap"].diff(warning_lookback_days)
    frame["Drawdown"] = (
        close / close.rolling(drawdown_lookback_days).max() - 1.0
    )
    frame["VixPercentile"] = trailing_vix_percentile(
        frame["Date"],
        frame["VIX"],
        lookback_years=vix_lookback_years,
        minimum_observations=minimum_vix_observations,
    )
    frame["WarningScore"] = (
        (frame["ReturnWarning"] < 0).astype(int)
        + (frame["MACDGapChangeWarning"] < 0).astype(int)
        + (frame["OBV"] < frame["OBVSignal9"]).astype(int)
    )
    frame["FeaturesReady"] = frame[
        ["VixPercentile", "Drawdown", "MACDGapChangeWarning", "OBVSignal9"]
    ].notna().all(axis=1)
    return frame
