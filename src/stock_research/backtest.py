from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import pandas as pd

P = TypeVar("P")
Signal = Callable[[pd.Series, P], bool]


@dataclass(frozen=True)
class BacktestSummary:
    initial_cash: float
    total_injected: float
    final_value: float
    roi_percent: float
    buys: int
    sells: int


@dataclass
class BacktestResult:
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

    cash = float(initial_cash)
    shares = 0.0
    total_injected = float(initial_cash)
    last_buy_date = None
    buys = sells = 0
    logs: list[dict] = []

    for _, row in frame.iterrows():
        date = row["날짜"]
        price = float(row["종가"])
        cooldown_ok = (
            last_buy_date is None
            or (date - last_buy_date).days >= cooldown_days
        )

        # Keep optimization and simulation consistent: only open a new position when flat.
        if shares == 0 and cooldown_ok and buy_signal(row, params):
            if extra_on_buy:
                cash += extra_contribution
                total_injected += extra_contribution
            shares = cash / price
            cash = 0.0
            last_buy_date = date
            buys += 1
            logs.append({
                "날짜": date, "액션": "BUY", "가격": price,
                "보유주": shares, "현금": cash,
                "총자산": cash + shares * price, "투입금액": total_injected,
            })
        elif shares > 0 and sell_signal(row, params):
            cash = shares * price
            shares = 0.0
            last_buy_date = None
            sells += 1
            logs.append({
                "날짜": date, "액션": "SELL", "가격": price,
                "보유주": shares, "현금": cash,
                "총자산": cash, "투입금액": total_injected,
            })

    if shares > 0 and liquidate_at_end:
        row = frame.iloc[-1]
        price = float(row["종가"])
        cash = shares * price
        shares = 0.0
        sells += 1
        logs.append({
            "날짜": row["날짜"], "액션": "LIQUIDATE", "가격": price,
            "보유주": 0.0, "현금": cash,
            "총자산": cash, "투입금액": total_injected,
        })

    final_value = cash + shares * float(frame.iloc[-1]["종가"])
    roi = (final_value / total_injected - 1.0) * 100.0
    trades = pd.DataFrame(logs)
    if not trades.empty:
        trades["ROI(%)"] = (
            trades["총자산"] / trades["투입금액"] - 1.0
        ) * 100.0

    return BacktestResult(
        trades=trades,
        summary=BacktestSummary(
            initial_cash=initial_cash,
            total_injected=total_injected,
            final_value=final_value,
            roi_percent=roi,
            buys=buys,
            sells=sells,
        ),
    )
