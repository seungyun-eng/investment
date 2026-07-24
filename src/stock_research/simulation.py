from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_long_only
from .io_utils import atomic_to_csv
from .strategies.technical import (
    TechnicalParams,
    technical_buy_signal,
    technical_sell_signal,
)
from .strategies.vix import (
    VixParams,
    load_vix_rule_config,
    vix_buy_signal,
    vix_sell_signal,
    vix_trade_log_details,
)


def run_strategy(
    data: pd.DataFrame,
    strategy: str,
    params: dict,
    *,
    extra_on_buy: bool = False,
) -> BacktestResult:
    strategy = strategy.lower()
    if strategy == "vix":
        rules = load_vix_rule_config()
        typed = VixParams(
            rsi_buy_th=float(params["rsi_buy_th"]),
            rsi_sell_th=float(params["rsi_sell_th"]),
            boll_buffer=float(params["boll_buffer"]),
            vix_buy_level=rules.vix_buy_level,
            vix_sell_level=rules.vix_sell_level,
        )
        return run_long_only(
            data, typed, vix_buy_signal, vix_sell_signal,
            extra_on_buy=extra_on_buy,
            log_details=vix_trade_log_details,
        )
    if strategy == "technical":
        typed = TechnicalParams(
            **{k: float(params[k]) for k in TechnicalParams.__annotations__}
        )
        return run_long_only(
            data, typed, technical_buy_signal, technical_sell_signal,
            extra_on_buy=extra_on_buy,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def buy_and_hold(data: pd.DataFrame, initial_cash: float = 10_000.0) -> pd.DataFrame:
    frame = data.sort_values("날짜").dropna(subset=["종가"])
    if frame.empty:
        return pd.DataFrame()
    first, last = frame.iloc[0], frame.iloc[-1]
    shares = initial_cash / float(first["종가"])
    final = shares * float(last["종가"])
    roi = (final / initial_cash - 1) * 100
    return pd.DataFrame([
        {
            "날짜": first["날짜"], "액션": "BUY_AND_HOLD_BUY",
            "가격": first["종가"], "보유주": shares,
            "총자산": initial_cash, "투입금액": initial_cash, "ROI(%)": 0.0,
        },
        {
            "날짜": last["날짜"], "액션": "BUY_AND_HOLD_END",
            "가격": last["종가"], "보유주": shares,
            "총자산": final, "투입금액": initial_cash, "ROI(%)": roi,
        },
    ])


def perfect_foresight(data: pd.DataFrame, initial_cash: float = 10_000.0) -> pd.DataFrame:
    """Diagnostic only. This is look-ahead biased and must not be treated as a benchmark."""
    frame = data.sort_values("날짜").dropna(subset=["종가"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    best_roi = float("-inf")
    best_pair = None
    for buy_idx in range(len(frame) - 1):
        future = frame.iloc[buy_idx + 1:]
        sell_idx = future["종가"].idxmax()
        roi = float(frame.loc[sell_idx, "종가"] / frame.loc[buy_idx, "종가"] - 1)
        if roi > best_roi:
            best_roi = roi
            best_pair = (buy_idx, sell_idx)
    if best_pair is None:
        return pd.DataFrame()
    buy_idx, sell_idx = best_pair
    buy, sell = frame.loc[buy_idx], frame.loc[sell_idx]
    shares = initial_cash / float(buy["종가"])
    final = shares * float(sell["종가"])
    return pd.DataFrame([
        {
            "날짜": buy["날짜"], "액션": "PERFECT_FORESIGHT_BUY",
            "가격": buy["종가"], "보유주": shares,
            "총자산": initial_cash, "투입금액": initial_cash, "ROI(%)": 0.0,
        },
        {
            "날짜": sell["날짜"], "액션": "PERFECT_FORESIGHT_SELL",
            "가격": sell["종가"], "보유주": 0.0,
            "총자산": final, "투입금액": initial_cash,
            "ROI(%)": (final / initial_cash - 1) * 100,
        },
    ])


def daily_dca(data: pd.DataFrame, daily_amount: float = 10_000.0) -> pd.DataFrame:
    frame = data.sort_values("날짜").dropna(subset=["종가"])
    shares = 0.0
    invested = 0.0
    rows = []
    for _, row in frame.iterrows():
        shares += daily_amount / float(row["종가"])
        invested += daily_amount
        value = shares * float(row["종가"])
        rows.append({
            "날짜": row["날짜"], "액션": "DCA_BUY", "가격": row["종가"],
            "보유주": shares, "총자산": value, "투입금액": invested,
            "ROI(%)": (value / invested - 1) * 100,
        })
    return pd.DataFrame(rows)


def save_simulation_outputs(
    result_root: Path,
    company: str,
    strategy: str,
    parameter_index: int,
    start: str,
    end: str,
    outputs: dict[str, pd.DataFrame],
) -> list[Path]:
    folder = result_root / company / strategy
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for label, frame in outputs.items():
        if frame.empty:
            continue
        roi_column = "ROI" if "ROI" in frame else "ROI(%)"
        roi = float(frame[roi_column].iloc[-1]) if roi_column in frame else float("nan")
        filename = (
            f"{parameter_index}_{company}_{strategy}_{label}_"
            f"{start}_{end}_ROI_{roi:.2f}.csv"
        )
        saved.append(atomic_to_csv(frame, folder / filename, index=False))
    return saved
