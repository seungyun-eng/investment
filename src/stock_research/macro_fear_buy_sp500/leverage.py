from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contributions import _xirr


@dataclass(frozen=True)
class OverlaySummary:
    final_value: float
    net_profit: float
    roi_percent: float
    time_weighted_cagr_percent: float
    money_weighted_return_percent: float
    max_drawdown_percent: float
    sharpe_ratio: float
    average_overlay_weight_percent: float


def daily_reset_instrument(
    signals: pd.DataFrame,
    *,
    multiple: float,
    annual_expense_ratio: float = 0.0095,
    annual_financing_spread: float = 0.01,
) -> pd.DataFrame:
    """Replace prices with a synthetic daily-reset leveraged instrument.

    Signals and features stay unchanged. Only the traded instrument's open and
    close series are replaced. Financing uses the point-in-time cash rate plus
    a fixed spread and an ETF-like expense ratio.
    """

    if multiple == 0:
        raise ValueError("multiple must be non-zero.")
    required = {"Date", "Open", "Close", "CashRate"}
    missing = required - set(signals)
    if missing:
        raise ValueError(f"Leveraged inputs are missing: {sorted(missing)}")
    frame = signals.copy().sort_values("Date").reset_index(drop=True)
    synthetic_open: list[float] = []
    synthetic_close: list[float] = []
    prior_underlying_close: float | None = None
    prior_synthetic_close = 100.0
    financing_multiplier = max(0.0, abs(multiple) - 1.0)

    for row in frame.itertuples(index=False):
        open_price = float(row.Open)
        close_price = float(row.Close)
        cash_rate = max(0.0, float(row.CashRate) / 100.0)
        annual_cost = (
            annual_expense_ratio
            + financing_multiplier
            * (cash_rate + annual_financing_spread)
        )
        half_day_cost = annual_cost / 504.0
        if prior_underlying_close is None:
            overnight_return = 0.0
        else:
            overnight_return = open_price / prior_underlying_close - 1.0
        intraday_return = close_price / open_price - 1.0
        open_growth = max(
            0.001,
            1.0 + multiple * overnight_return - half_day_cost,
        )
        synthetic_open_value = prior_synthetic_close * open_growth
        close_growth = max(
            0.001,
            1.0 + multiple * intraday_return - half_day_cost,
        )
        synthetic_close_value = synthetic_open_value * close_growth
        synthetic_open.append(synthetic_open_value)
        synthetic_close.append(synthetic_close_value)
        prior_underlying_close = close_price
        prior_synthetic_close = synthetic_close_value

    frame["UnderlyingOpen"] = frame["Open"]
    frame["UnderlyingClose"] = frame["Close"]
    frame["Open"] = synthetic_open
    frame["Close"] = synthetic_close
    frame["LeverageMultiple"] = multiple
    return frame


def _cash_returns(daily: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(daily["Date"]).reset_index(drop=True)
    rates = (
        pd.to_numeric(daily["CashRate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    result = np.zeros(len(daily), dtype=float)
    for index in range(1, len(daily)):
        days = max(0, int((dates.iloc[index] - dates.iloc[index - 1]).days))
        annual_rate = max(-0.99, rates[index - 1] / 100.0)
        result[index] = (1.0 + annual_rate) ** (days / 365.25) - 1.0
    return result


def _summarize_overlay(
    base_daily: pd.DataFrame,
    adjusted_returns: np.ndarray,
    overlay_weights: np.ndarray,
) -> OverlaySummary:
    contributions = pd.to_numeric(
        base_daily["Contribution"],
        errors="coerce",
    ).fillna(0.0)
    initial = float(base_daily["TotalInjected"].iloc[0])
    total_injected = float(base_daily["TotalInjected"].iloc[-1])
    values = np.zeros(len(base_daily), dtype=float)
    flow_index = np.ones(len(base_daily), dtype=float)
    value = initial
    index_value = 1.0
    for index, adjusted_return in enumerate(adjusted_returns):
        if index > 0:
            value += float(contributions.iloc[index])
        adjusted_return = max(-0.999, float(adjusted_return))
        value *= 1.0 + adjusted_return
        index_value *= 1.0 + adjusted_return
        values[index] = value
        flow_index[index] = index_value
    dates = pd.to_datetime(base_daily["Date"]).reset_index(drop=True)
    elapsed_days = max(1, int((dates.iloc[-1] - dates.iloc[0]).days))
    years = elapsed_days / 365.25
    cagr = (flow_index[-1] ** (1.0 / years) - 1.0) * 100.0
    drawdown = flow_index / np.maximum.accumulate(flow_index) - 1.0
    volatility = float(np.std(adjusted_returns, ddof=1))
    sharpe = (
        float(np.mean(adjusted_returns) / volatility * np.sqrt(252))
        if volatility > 0
        else float("nan")
    )
    flow_dates = [pd.Timestamp(dates.iloc[0])]
    flow_values = [-initial]
    for index in np.flatnonzero(contributions.to_numpy() > 0):
        flow_dates.append(pd.Timestamp(dates.iloc[index]))
        flow_values.append(-float(contributions.iloc[index]))
    flow_dates.append(pd.Timestamp(dates.iloc[-1]))
    flow_values.append(float(values[-1]))
    return OverlaySummary(
        final_value=float(values[-1]),
        net_profit=float(values[-1] - total_injected),
        roi_percent=float((values[-1] / total_injected - 1.0) * 100.0),
        time_weighted_cagr_percent=float(cagr),
        money_weighted_return_percent=float(
            _xirr(flow_dates, flow_values) * 100.0
        ),
        max_drawdown_percent=float(drawdown.min() * 100.0),
        sharpe_ratio=sharpe,
        average_overlay_weight_percent=float(
            np.mean(overlay_weights) * 100.0
        ),
    )


def tactical_two_x_overlay(
    base_daily: pd.DataFrame,
    *,
    annual_expense_ratio: float = 0.0095,
    annual_financing_spread: float = 0.01,
) -> OverlaySummary:
    """Approximate one extra unit of SPY exposure on the tactical sleeve."""

    required = {
        "Close",
        "CashRate",
        "TacticalShares",
        "TotalValue",
        "FlowAdjustedReturn",
    }
    missing = required - set(base_daily)
    if missing:
        raise ValueError(f"Tactical leverage inputs are missing: {sorted(missing)}")
    closes = pd.to_numeric(base_daily["Close"], errors="coerce")
    returns = closes.pct_change().fillna(0.0).to_numpy(dtype=float)
    tactical_weight = (
        pd.to_numeric(base_daily["TacticalShares"], errors="coerce")
        * closes
        / pd.to_numeric(base_daily["TotalValue"], errors="coerce")
    ).fillna(0.0)
    overlay_weight = tactical_weight.shift(1).fillna(0.0).to_numpy(dtype=float)
    cash_rate = (
        pd.to_numeric(base_daily["CashRate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        / 100.0
    )
    daily_cost = (
        annual_expense_ratio + np.maximum(cash_rate, 0.0)
        + annual_financing_spread
    ) / 252.0
    base_returns = pd.to_numeric(
        base_daily["FlowAdjustedReturn"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    adjusted = base_returns + overlay_weight * (returns - daily_cost)
    return _summarize_overlay(base_daily, adjusted, overlay_weight)


def conditional_two_x_short_hedge(
    base_daily: pd.DataFrame,
    *,
    maximum_capital_fraction: float,
    euphoria_threshold: float,
    max_fear_score: float,
    max_vix_percentile: float,
    annual_expense_ratio: float = 0.0095,
    annual_financing_spread: float = 0.01,
    transaction_cost_bps: float = 10.0,
) -> OverlaySummary:
    """Approximate a -2x ETF held only with otherwise-unused cash.

    Eligibility is based on the prior close, so the overlay does not use future
    information. Capital at risk is capped by both the requested fraction and
    the strategy's unused cash weight.
    """

    if not 0.0 <= maximum_capital_fraction <= 1.0:
        raise ValueError("maximum_capital_fraction must be in [0, 1].")
    required = {
        "Close",
        "CashRate",
        "ActualWeight",
        "FearScore",
        "EuphoriaScore",
        "VixPercentile",
        "FlowAdjustedReturn",
    }
    missing = required - set(base_daily)
    if missing:
        raise ValueError(f"Short-hedge inputs are missing: {sorted(missing)}")
    closes = pd.to_numeric(base_daily["Close"], errors="coerce")
    underlying_returns = closes.pct_change().fillna(0.0).to_numpy(dtype=float)
    euphoria = pd.to_numeric(
        base_daily["EuphoriaScore"],
        errors="coerce",
    )
    fear = pd.to_numeric(base_daily["FearScore"], errors="coerce")
    vix = pd.to_numeric(base_daily["VixPercentile"], errors="coerce")
    eligible = (
        (euphoria >= euphoria_threshold)
        & (fear <= max_fear_score)
        & (vix <= max_vix_percentile)
    )
    cash_weight = (
        1.0
        - pd.to_numeric(
            base_daily["ActualWeight"],
            errors="coerce",
        ).fillna(0.0)
    ).clip(0.0, 1.0)
    overlay_weight = np.minimum(
        cash_weight.shift(1).fillna(0.0).to_numpy(dtype=float),
        maximum_capital_fraction,
    )
    prior_eligible = eligible.shift(1, fill_value=False).astype(bool)
    overlay_weight *= prior_eligible.to_numpy(dtype=float)
    cash_returns = _cash_returns(base_daily)
    cash_rate = (
        pd.to_numeric(base_daily["CashRate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        / 100.0
    )
    daily_cost = (
        annual_expense_ratio + np.maximum(cash_rate, 0.0)
        + annual_financing_spread
    ) / 252.0
    inverse_returns = -2.0 * underlying_returns - daily_cost
    turnover_cost = (
        np.abs(np.diff(overlay_weight, prepend=0.0))
        * transaction_cost_bps
        / 10_000.0
    )
    base_returns = pd.to_numeric(
        base_daily["FlowAdjustedReturn"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    adjusted = base_returns + overlay_weight * (
        inverse_returns - cash_returns
    ) - turnover_cost
    return _summarize_overlay(base_daily, adjusted, overlay_weight)


def conditional_two_x_risk_hedge(
    base_daily: pd.DataFrame,
    *,
    maximum_capital_fraction: float,
    model_risk_threshold: float,
    macro_threshold: float,
    maximum_vix_percentile: float = 0.90,
    minimum_drawdown: float = -0.15,
    annual_expense_ratio: float = 0.0095,
    annual_financing_spread: float = 0.01,
    transaction_cost_bps: float = 10.0,
) -> OverlaySummary:
    """Approximate a pre-fear -2x hedge using only prior-close risk data."""

    if not 0.0 <= maximum_capital_fraction <= 1.0:
        raise ValueError("maximum_capital_fraction must be in [0, 1].")
    required = {
        "Close",
        "CashRate",
        "ActualWeight",
        "ModelRiskPercentile",
        "MacroConfirmationScore",
        "VixPercentile",
        "Drawdown252",
        "FlowAdjustedReturn",
    }
    missing = required - set(base_daily)
    if missing:
        raise ValueError(f"Risk-hedge inputs are missing: {sorted(missing)}")
    closes = pd.to_numeric(base_daily["Close"], errors="coerce")
    underlying_returns = closes.pct_change().fillna(0.0).to_numpy(dtype=float)
    risk = pd.to_numeric(
        base_daily["ModelRiskPercentile"],
        errors="coerce",
    )
    macro = pd.to_numeric(
        base_daily["MacroConfirmationScore"],
        errors="coerce",
    )
    vix = pd.to_numeric(base_daily["VixPercentile"], errors="coerce")
    drawdown = pd.to_numeric(base_daily["Drawdown252"], errors="coerce")
    eligible = (
        (risk >= model_risk_threshold)
        & (macro >= macro_threshold)
        & (vix <= maximum_vix_percentile)
        & (drawdown >= minimum_drawdown)
    )
    cash_weight = (
        1.0
        - pd.to_numeric(
            base_daily["ActualWeight"],
            errors="coerce",
        ).fillna(0.0)
    ).clip(0.0, 1.0)
    overlay_weight = np.minimum(
        cash_weight.shift(1).fillna(0.0).to_numpy(dtype=float),
        maximum_capital_fraction,
    )
    prior_eligible = eligible.shift(1, fill_value=False).astype(bool)
    overlay_weight *= prior_eligible.to_numpy(dtype=float)
    cash_returns = _cash_returns(base_daily)
    cash_rate = (
        pd.to_numeric(base_daily["CashRate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        / 100.0
    )
    daily_cost = (
        annual_expense_ratio + np.maximum(cash_rate, 0.0)
        + annual_financing_spread
    ) / 252.0
    inverse_returns = -2.0 * underlying_returns - daily_cost
    turnover_cost = (
        np.abs(np.diff(overlay_weight, prepend=0.0))
        * transaction_cost_bps
        / 10_000.0
    )
    base_returns = pd.to_numeric(
        base_daily["FlowAdjustedReturn"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    adjusted = base_returns + overlay_weight * (
        inverse_returns - cash_returns
    ) - turnover_cost
    return _summarize_overlay(base_daily, adjusted, overlay_weight)
