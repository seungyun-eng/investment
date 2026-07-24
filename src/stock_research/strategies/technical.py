from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TechnicalParams:
    rsi_sell_th: float
    boll_buffer: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


TECHNICAL_REQUIRED_COLUMNS = {
    "OBV", "OBV_SIG9", "RSI_14", "RSI_SIG9", "종가",
    "BB_LOWER", "BB_UPPER", "MACD", "MACD_SIG",
}


def technical_buy_signal(row: pd.Series, params: TechnicalParams) -> bool:
    return bool(
        row["OBV"] > row["OBV_SIG9"]
        and row["RSI_14"] < row["RSI_SIG9"]
        and row["종가"] < row["BB_LOWER"] * (1 + params.boll_buffer)
        and row["MACD"] > row["MACD_SIG"]
    )


def technical_sell_signal(row: pd.Series, params: TechnicalParams) -> bool:
    condition_group = (
        row["RSI_14"] > params.rsi_sell_th
        and row["OBV"] < row["OBV_SIG9"]
        and row["종가"] > row["BB_UPPER"] * (1 + params.boll_buffer)
    )
    macd_exit = row["MACD"] < row["MACD_SIG"]
    return bool(condition_group or macd_exit)
