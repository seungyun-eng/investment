from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd

P = TypeVar("P")
Signal = Callable[[pd.Series, P], bool]
LogDetails = Callable[[pd.Series, P, str], dict[str, object]]


@dataclass(frozen=True)
class BacktestSummary:
    initial_cash: float
    total_injected: float
    final_value: float
    roi_percent: float
    buys: int
    sells: int
    liquidations: int
    completed_trades: int


@dataclass
class BacktestResult(Generic[P]):
    trades: pd.DataFrame
    summary: BacktestSummary


def run_long_only(
    data: pd.DataFrame,
    params: P,
    buy_signal: Signal[P],
    sell_signal: Signal[P],
    *,
    initial_cash: float = 10_000.0,
    extra_on_buy: bool = False,
    extra_contribution: float = 10_000.0,
    cooldown_days: int = 0,
    liquidate_at_end: bool = True,
    log_details: LogDetails[P] | None = None,
) -> BacktestResult:
    if data.empty:
        raise ValueError("Backtest data is empty.")
    required = {"날짜", "종가"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Backtest missing columns: {sorted(missing)}")
    frame = data.copy().sort_values("날짜").reset_index(drop=True)
    frame["날짜"] = pd.to_datetime(frame["날짜"], errors="coerce")
    frame["종가"] = pd.to_numeric(frame["종가"], errors="coerce")
    frame = frame.dropna(subset=["날짜", "종가"])
    if frame.empty:
        raise ValueError("No valid backtest rows after date/price conversion.")

    cash, shares = float(initial_cash), 0.0
    total_injected = float(initial_cash)
    last_buy_date = None
    buys = sells = liquidations = 0
    logs: list[dict[str, object]] = []

    def add_log(row: pd.Series, action: str, price: float) -> None:
        total_value = cash + shares * price
        entry: dict[str, object] = {
            "Date": row["날짜"], "Action": action, "StockPrice": price,
            "Cash": cash, "Shares": shares, "TotalValue": total_value,
            "TotalInjected": total_injected,
            "ROI": (total_value / total_injected - 1.0) * 100.0,
        }
        if log_details:
            entry.update(log_details(row, params, action))
        logs.append(entry)

    for _, row in frame.iterrows():
        date, price = row["날짜"], float(row["종가"])
        cooldown_ok = last_buy_date is None or (date - last_buy_date).days >= cooldown_days
        if shares == 0 and cooldown_ok and buy_signal(row, params):
            if extra_on_buy:
                cash += extra_contribution
                total_injected += extra_contribution
            shares, cash = cash / price, 0.0
            last_buy_date, buys = date, buys + 1
            add_log(row, "BUY", price)
        elif shares > 0 and sell_signal(row, params):
            cash, shares = shares * price, 0.0
            last_buy_date, sells = None, sells + 1
            add_log(row, "SELL", price)

    if shares > 0 and liquidate_at_end:
        row = frame.iloc[-1]
        price = float(row["종가"])
        cash, shares = shares * price, 0.0
        liquidations += 1
        add_log(row, "LIQUIDATE", price)

    final_value = cash + shares * float(frame.iloc[-1]["종가"])
    roi = (final_value / total_injected - 1.0) * 100.0
    return BacktestResult(
        trades=pd.DataFrame(logs),
        summary=BacktestSummary(
            initial_cash, total_injected, final_value, roi, buys, sells,
            liquidations, sells,
        ),
    )
