from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import IntegratedParams


@dataclass(frozen=True)
class IntegratedSummary:
    initial_capital: float
    total_injected: float
    final_value: float
    roi_percent: float
    max_drawdown_percent: float
    completed_trades: int


@dataclass(frozen=True)
class IntegratedResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: IntegratedSummary


def run_integrated_backtest(
    signals: pd.DataFrame,
    params: IntegratedParams,
    *,
    initial_capital: float = 40_000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    annual_short_borrow_bps: float = 300.0,
    initial_long: bool = False,
) -> IntegratedResult:
    """Execute prior-close signals at the next open in long/cash/short states."""

    frame = signals.sort_values("Date").reset_index(drop=True).copy()
    cash = float(initial_capital)
    shares = 0.0
    short_units = 0.0
    entry_price: float | None = None
    long_peak: float | None = None
    short_trough: float | None = None
    held = 0
    trades: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    buy_cost = 1 + (transaction_cost_bps + slippage_bps) / 10_000
    sell_cost = 1 - (transaction_cost_bps + slippage_bps) / 10_000
    completed = 0
    initial_position_opened = False

    dates = pd.to_datetime(frame["Date"]).to_numpy()
    opens = pd.to_numeric(frame["Open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(
        frame["CompositeScore"], errors="coerce"
    ).to_numpy(dtype=float)
    cash_rates = pd.to_numeric(
        frame.get("CashRate", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    downside_column = (
        "DownsideProbability21"
        if "DownsideProbability21" in frame
        else "TslaDownsideProbability21"
    )
    downside_probabilities = pd.to_numeric(
        frame.get(downside_column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    buys = frame["BuySignal"].fillna(False).to_numpy(dtype=bool)
    sells = frame["SellSignal"].fillna(False).to_numpy(dtype=bool)
    shorts = frame.get(
        "ShortSignal", pd.Series(False, index=frame.index)
    ).fillna(False).to_numpy(dtype=bool)
    covers = frame.get(
        "CoverSignal", pd.Series(True, index=frame.index)
    ).fillna(True).to_numpy(dtype=bool)
    daily_borrow_rate = annual_short_borrow_bps / 10_000 / 252

    for index in range(len(frame)):
        open_price = opens[index]
        close_price = closes[index]
        action = "HOLD"
        if cash > 0 and short_units == 0:
            cash *= 1 + max(cash_rates[index], 0.0) / 100 / 252
        if short_units > 0:
            cash -= short_units * close_price * daily_borrow_rate
        if index and shares == 0 and short_units == 0:
            if shorts[index - 1]:
                execution = open_price * sell_cost
                short_units = cash / execution
                cash += short_units * execution
                entry_price = execution
                short_trough = execution
                long_peak = None
                held = 0
                initial_position_opened = True
                action = "SHORT"
            elif buys[index - 1] or (
                initial_long and not initial_position_opened
            ):
                execution = open_price * buy_cost
                shares = cash / execution
                cash = 0.0
                entry_price = execution
                long_peak = execution
                short_trough = None
                held = 0
                initial_position_opened = True
                action = "BUY"
        elif shares > 0:
            held += 1
            stopped = (
                entry_price is not None
                and open_price <= entry_price * (1 - params.stop_loss)
            )
            trailing_stopped = (
                long_peak is not None
                and open_price <= long_peak * (1 - params.trailing_stop)
            )
            if stopped or trailing_stopped or (
                held >= params.minimum_hold_sessions and sells[index - 1]
            ):
                execution = open_price * sell_cost
                cash = shares * execution
                shares = 0.0
                entry_price = None
                long_peak = None
                completed += 1
                action = "SELL"
        elif short_units > 0:
            held += 1
            stopped = (
                entry_price is not None
                and open_price >= entry_price * (1 + params.short_stop_loss)
            )
            trailing_stopped = (
                short_trough is not None
                and open_price
                >= short_trough * (1 + params.short_trailing_stop)
            )
            if stopped or trailing_stopped or (
                held >= params.minimum_hold_sessions and covers[index - 1]
            ):
                execution = open_price * buy_cost
                cash -= short_units * execution
                short_units = 0.0
                entry_price = None
                short_trough = None
                completed += 1
                action = "COVER"
        equity = cash + shares * close_price - short_units * close_price
        if shares > 0:
            long_peak = max(
                long_peak if long_peak is not None else close_price,
                close_price,
            )
        if short_units > 0:
            short_trough = min(
                (
                    short_trough
                    if short_trough is not None
                    else close_price
                ),
                close_price,
            )
        state = "LONG" if shares > 0 else "SHORT" if short_units > 0 else "CASH"
        daily.append(
            {
                "Date": dates[index],
                "Open": open_price,
                "Close": close_price,
                "Action": action,
                "State": state,
                "Cash": cash,
                "Shares": shares,
                "ShortUnits": short_units,
                "Equity": equity,
                "CompositeScore": scores[index],
                "DownsideProbability21": downside_probabilities[index],
            }
        )
        if action != "HOLD":
            trades.append(daily[-1].copy())

    daily_frame = pd.DataFrame(daily)
    final_value = float(daily_frame["Equity"].iloc[-1])
    drawdown = daily_frame["Equity"] / np.maximum.accumulate(
        daily_frame["Equity"].to_numpy(dtype=float)
    ) - 1
    return IntegratedResult(
        daily=daily_frame,
        trades=pd.DataFrame(trades),
        summary=IntegratedSummary(
            initial_capital=initial_capital,
            total_injected=initial_capital,
            final_value=final_value,
            roi_percent=(final_value / initial_capital - 1) * 100,
            max_drawdown_percent=float(drawdown.min() * 100),
            completed_trades=completed,
        ),
    )
