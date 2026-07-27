from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig


def _future_sum(values: pd.Series, horizon: int) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(array), np.nan)
    for index in range(len(array) - horizon):
        future = array[index + 1 : index + horizon + 1]
        if np.isfinite(future).all():
            output[index] = float(future.sum())
    return pd.Series(output, index=values.index)


def _future_min_return(close: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    for index in range(len(values) - horizon):
        future = values[index + 1 : index + horizon + 1]
        if np.isfinite(values[index]) and np.isfinite(future).all():
            output[index] = float(future.min() / values[index] - 1)
    return pd.Series(output, index=close.index)


def build_targets(data: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    required = {"Date", "Close", "CashRate"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Target data is missing columns: {sorted(missing)}")
    result = pd.DataFrame({"Date": pd.to_datetime(data["Date"], errors="coerce")})
    close = pd.to_numeric(data["Close"], errors="coerce")
    cash_rate = pd.to_numeric(data["CashRate"], errors="coerce").ffill().fillna(0)
    cash_daily_log = np.log1p(cash_rate.clip(lower=-99.0) / 100.0) / 252.0

    all_horizons = sorted(set(config.return_horizons) | set(config.risk_horizons))
    for horizon in all_horizons:
        result[f"TargetEndDate_{horizon}"] = result["Date"].shift(-horizon)

    for horizon in config.return_horizons:
        market_return = close.shift(-horizon) / close - 1
        cash_return = np.expm1(_future_sum(cash_daily_log, horizon))
        result[f"ForwardReturn_{horizon}"] = market_return
        result[f"ForwardCashReturn_{horizon}"] = cash_return
        result[f"ExcessReturn_{horizon}"] = market_return - cash_return

    for horizon in config.risk_horizons:
        future_min = _future_min_return(close, horizon)
        result[f"FutureMinReturn_{horizon}"] = future_min
        for threshold in config.drawdown_thresholds:
            label = round(abs(threshold) * 100)
            target = (future_min <= threshold + 1e-12).astype(float)
            target[future_min.isna()] = np.nan
            result[f"DrawdownHit_{horizon}_{label}"] = target

    return result


def primary_target_names(config: ResearchConfig) -> tuple[str, str]:
    drawdown_label = round(abs(config.primary_drawdown_threshold) * 100)
    return (
        f"DrawdownHit_{config.primary_risk_horizon}_{drawdown_label}",
        f"ExcessReturn_{config.primary_return_horizon}",
    )
