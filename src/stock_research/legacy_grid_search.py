from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_to_csv


@dataclass(frozen=True)
class LegacyGridParams:
    rsi_buy_threshold: float
    bollinger_buffer: float
    two_week_max_pct: float
    rsi_sell_threshold: float = 70.0


def _buy(row: pd.Series, params: LegacyGridParams) -> bool:
    return bool(
        row.get("RSI (14일)", 100) < params.rsi_buy_threshold
        and row["종가"] < row.get("볼린저밴드 하단", row["종가"])
        * (1 + params.bollinger_buffer)
        and row.get("MACD", 0) > row.get("MACD 시그널", 0)
        and row.get("SMA 5일", 0) > row.get("SMA 10일", 0)
        and row.get("SMA 5일", 0) < row.get("SMA 60일", 0)
        and row.get("가격 상승률 (2주)", 100) < params.two_week_max_pct
    )


def _sell(row: pd.Series, params: LegacyGridParams) -> bool:
    return bool(row.get("RSI (14일)", 0) > params.rsi_sell_threshold)


def run_legacy_period(
    frame: pd.DataFrame,
    params: LegacyGridParams,
    start: str,
    end: str,
    *,
    allow_sell: bool = True,
    initial_cash: float = 10_000.0,
    tranche_fraction: float = 0.10,
    cooldown_rows: int = 5,
) -> dict:
    data = frame[
        (frame["날짜"] >= pd.Timestamp(start))
        & (frame["날짜"] < pd.Timestamp(end))
    ].copy()
    if data.empty:
        raise ValueError(f"No rows for {start} to {end}")

    cash = initial_cash
    shares = 0.0
    cooldown = 0
    buys, sells = [], []

    for _, row in data.iterrows():
        if cooldown > 0:
            cooldown -= 1
            continue
        price = float(row["종가"])
        if _buy(row, params) and cash > 0:
            amount = cash * tranche_fraction
            shares += amount / price
            cash -= amount
            buys.append((row["날짜"], price))
            cooldown = cooldown_rows
        elif allow_sell and shares > 0 and _sell(row, params):
            cash += shares * price
            shares = 0.0
            sells.append((row["날짜"], price))
            cooldown = cooldown_rows

    final_value = cash + shares * float(data.iloc[-1]["종가"])
    return {
        "시작일": pd.Timestamp(start).date(),
        "종료일": pd.Timestamp(end).date(),
        "최종 자산": final_value,
        "수익률 (%)": (final_value / initial_cash - 1) * 100,
        "매수 횟수": len(buys),
        "매도 횟수": len(sells),
        "매수 가격": [f"{d:%Y-%m-%d}: {p:.2f}" for d, p in buys],
        "매도 가격": [f"{d:%Y-%m-%d}: {p:.2f}" for d, p in sells],
    }


def run_legacy_grid_search(
    frame: pd.DataFrame,
    output: Path,
    *,
    periods: list[tuple[str, str]] | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    data = frame.copy()
    data["날짜"] = pd.to_datetime(data["날짜"], errors="coerce")
    periods = periods or [
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
    ]
    results = []
    for rsi, buffer, two_week in itertools.product(
        range(30, 60, 5),
        [0.01, 0.015, 0.02, 0.025],
        [1, 2, 3, 4, 5],
    ):
        params = LegacyGridParams(rsi, buffer, two_week)
        period_results = [
            run_legacy_period(data, params, start, end)
            for start, end in periods
        ]
        results.append({
            "RSI 조건": rsi,
            "볼린저 하단 버퍼": buffer,
            "2주 상승률 조건": two_week,
            "평균 수익률 (%)": np.mean(
                [row["수익률 (%)"] for row in period_results]
            ),
            "기간별 결과": period_results,
        })
    ranked = (
        pd.DataFrame(results)
        .sort_values("평균 수익률 (%)", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    rows = []
    for _, strategy in ranked.iterrows():
        for period_result in strategy["기간별 결과"]:
            rows.append({
                **period_result,
                "RSI 조건": strategy["RSI 조건"],
                "볼린저 하단 버퍼": strategy["볼린저 하단 버퍼"],
                "2주 상승률 조건": strategy["2주 상승률 조건"],
            })
    detailed = pd.DataFrame(rows)
    atomic_to_csv(detailed, output, index=False)
    return detailed
