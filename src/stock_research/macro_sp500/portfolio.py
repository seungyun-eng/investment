from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MacroSp500Params, MacroSp500Settings
from .strategy import generate_target_weights


@dataclass(frozen=True)
class PerformanceSummary:
    initial_capital: float
    total_injected: float
    final_value: float
    roi_percent: float
    cagr_percent: float
    max_drawdown_percent: float
    calmar_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    average_exposure_percent: float
    turnover_multiple: float
    rebalance_count: int


@dataclass
class PortfolioResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: PerformanceSummary


def _performance_summary(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
) -> PerformanceSummary:
    if daily.empty:
        raise ValueError("Cannot summarize an empty portfolio.")
    final_value = float(daily["TotalValue"].iloc[-1])
    roi = (final_value / initial_capital - 1.0) * 100.0
    elapsed_days = max(
        1,
        int((daily["Date"].iloc[-1] - daily["Date"].iloc[0]).days),
    )
    years = elapsed_days / 365.25
    cagr = ((final_value / initial_capital) ** (1.0 / years) - 1.0) * 100.0
    running_peak = daily["TotalValue"].cummax()
    drawdown = daily["TotalValue"] / running_peak - 1.0
    maximum_drawdown = float(drawdown.min() * 100.0)
    calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else float("inf")

    returns = daily["TotalValue"].pct_change().dropna()
    standard_deviation = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / standard_deviation * np.sqrt(252))
        if standard_deviation > 0
        else float("nan")
    )
    downside = returns[returns < 0]
    downside_deviation = float(downside.std(ddof=1))
    sortino = (
        float(returns.mean() / downside_deviation * np.sqrt(252))
        if downside_deviation > 0
        else float("nan")
    )
    turnover = (
        float(trades["Notional"].abs().sum() / initial_capital)
        if not trades.empty
        else 0.0
    )
    return PerformanceSummary(
        initial_capital=initial_capital,
        total_injected=initial_capital,
        final_value=final_value,
        roi_percent=roi,
        cagr_percent=cagr,
        max_drawdown_percent=maximum_drawdown,
        calmar_ratio=calmar,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        average_exposure_percent=float(daily["ActualWeight"].mean() * 100.0),
        turnover_multiple=turnover,
        rebalance_count=len(trades),
    )


def run_target_weight_backtest(
    features: pd.DataFrame,
    params: MacroSp500Params,
    settings: MacroSp500Settings,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> PortfolioResult:
    frame = features.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if start is not None:
        frame = frame[frame["Date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["Date"] <= pd.Timestamp(end)]
    frame = frame.dropna(subset=["Date", "Open", "Close"]).sort_values("Date")
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("Backtest date range contains no valid rows.")

    signals = generate_target_weights(frame, params, settings)
    cash = float(settings.initial_capital)
    shares = 0.0
    fee_rate = settings.transaction_cost_bps / 10_000.0
    slippage_rate = settings.slippage_bps / 10_000.0
    daily_rate = (1.0 + settings.cash_annual_rate) ** (1.0 / 365.25) - 1.0
    previous_date: pd.Timestamp | None = None
    trade_rows: list[dict[str, object]] = []
    dates = pd.to_datetime(frame["Date"]).to_numpy()
    open_values = frame["Open"].to_numpy(dtype=float)
    close_values = frame["Close"].to_numpy(dtype=float)
    vix_values = frame["VIX"].to_numpy(dtype=float)
    percentile_values = frame["VixPercentile"].to_numpy(dtype=float)
    drawdown_values = frame["Drawdown"].to_numpy(dtype=float)
    warning_values = frame["WarningScore"].to_numpy(dtype=int)
    signal_targets = signals["TargetWeight"].to_numpy(dtype=float)
    signal_reasons = signals["Reason"].to_numpy(dtype=str)
    signal_states = signals["State"].to_numpy(dtype=str)
    actual_weights: list[float] = []
    cash_values: list[float] = []
    share_values: list[float] = []
    total_values: list[float] = []
    roi_values: list[float] = []

    for index in range(len(frame)):
        date = pd.Timestamp(dates[index])
        open_price = open_values[index]
        close_price = close_values[index]
        if previous_date is not None and cash > 0 and daily_rate:
            cash *= (1.0 + daily_rate) ** max(0, (date - previous_date).days)

        if index == 0:
            target_weight = params.core_weight
            signal_date = date
            reason = "INITIAL_CORE"
            signal_state = "NORMAL"
        else:
            target_weight = signal_targets[index - 1]
            signal_date = pd.Timestamp(dates[index - 1])
            reason = signal_reasons[index - 1]
            signal_state = signal_states[index - 1]

        value_at_open = cash + shares * open_price
        desired_stock_value = target_weight * value_at_open
        current_stock_value = shares * open_price
        requested_notional = desired_stock_value - current_stock_value
        action = ""
        executed_notional = 0.0
        fee = 0.0
        execution_price = open_price

        if requested_notional > max(1e-8, value_at_open * 1e-8):
            action = "BUY"
            execution_price = open_price * (1.0 + slippage_rate)
            affordable_notional = cash / (1.0 + fee_rate)
            executed_notional = min(requested_notional, affordable_notional)
            quantity = executed_notional / execution_price
            fee = executed_notional * fee_rate
            shares += quantity
            cash -= executed_notional + fee
        elif requested_notional < -max(1e-8, value_at_open * 1e-8):
            action = "SELL"
            execution_price = open_price * (1.0 - slippage_rate)
            requested_sale = min(-requested_notional, shares * open_price)
            quantity = requested_sale / open_price
            executed_notional = quantity * execution_price
            fee = executed_notional * fee_rate
            shares -= quantity
            cash += executed_notional - fee

        value_at_close = cash + shares * close_price
        actual_weight = shares * close_price / value_at_close if value_at_close else 0.0
        roi = (value_at_close / settings.initial_capital - 1.0) * 100.0
        if action:
            trade_rows.append(
                {
                    "Date": date,
                    "SignalDate": signal_date,
                    "Action": action,
                    "ExecutionPrice": execution_price,
                    "Notional": executed_notional,
                    "Fee": fee,
                    "TargetWeight": target_weight,
                    "State": signal_state,
                    "Reason": reason,
                    "VIX": vix_values[index],
                    "VixPercentile": percentile_values[index],
                    "Drawdown": drawdown_values[index],
                    "WarningScore": warning_values[index],
                    "ROI": roi,
                }
            )
        actual_weights.append(actual_weight)
        cash_values.append(cash)
        share_values.append(shares)
        total_values.append(value_at_close)
        roi_values.append(roi)
        previous_date = date

    daily = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_values,
            "Close": close_values,
            "VIX": vix_values,
            "VixPercentile": percentile_values,
            "Drawdown": drawdown_values,
            "WarningScore": warning_values,
            "State": signal_states,
            "NextTargetWeight": signal_targets,
            "ActualWeight": actual_weights,
            "Cash": cash_values,
            "Shares": share_values,
            "TotalValue": total_values,
            "TotalInjected": settings.initial_capital,
            "ROI": roi_values,
        }
    )
    trades = pd.DataFrame(trade_rows)
    summary = _performance_summary(
        daily,
        trades,
        initial_capital=settings.initial_capital,
    )
    return PortfolioResult(daily=daily, trades=trades, summary=summary)


def run_buy_and_hold(
    features: pd.DataFrame,
    settings: MacroSp500Settings,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> PortfolioResult:
    params = MacroSp500Params(
        vix_lookback_years=3,
        core_weight=1.0,
        warning_score_min=3,
        warning_addition=0.0,
        vix_entry_quantile=0.90,
        exit_vix_quantile=0.50,
        minimum_hold_days=0,
    )
    return run_target_weight_backtest(
        features,
        params,
        settings,
        start=start,
        end=end,
    )
