from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class VixParams:
    vix_buy_th: float
    vix_sell_th: float
    rsi_buy_th: float
    rsi_sell_th: float
    boll_buffer: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


VIX_REQUIRED_COLUMNS = {
    "VIX", "RSI (14일)", "종가", "볼린저밴드 하단",
    "볼린저밴드 상단", "MACD", "MACD 시그널",
}


def vix_buy_signal(row: pd.Series, params: VixParams) -> bool:
    return bool(
        pd.notna(row["VIX"])
        and row["VIX"] >= params.vix_buy_th
        and row["RSI (14일)"] < params.rsi_buy_th
        and row["종가"] < row["볼린저밴드 하단"] * (1 + params.boll_buffer)
        and row["MACD"] > row["MACD 시그널"]
    )


def vix_sell_signal(row: pd.Series, params: VixParams) -> bool:
    return bool(
        pd.notna(row["VIX"])
        and row["VIX"] <= params.vix_sell_th
        and row["RSI (14일)"] > params.rsi_sell_th
        and row["종가"] > row["볼린저밴드 상단"] * (1 + params.boll_buffer)
        and row["MACD"] < row["MACD 시그널"]
    )
