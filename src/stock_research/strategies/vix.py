from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class VixParams:
    rsi_buy_th: float
    rsi_sell_th: float
    boll_buffer: float
    vix_buy_level: float
    vix_sell_level: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class VixRuleConfig:
    vix_buy_level: float
    vix_sell_level: float
    source: str
    rule_type: str


VIX_REQUIRED_COLUMNS = {
    "VIX", "RSI (14일)", "종가", "볼린저밴드 하단",
    "볼린저밴드 상단", "MACD", "MACD 시그널",
}


def load_vix_rule_config(path: str | Path | None = None) -> VixRuleConfig:
    config_path = Path(path) if path else (
        Path(__file__).resolve().parents[3] / "config" / "vix_strategy.json"
    )
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    config = VixRuleConfig(
        vix_buy_level=float(raw["vix_buy_level"]),
        vix_sell_level=float(raw["vix_sell_level"]),
        source=str(raw["source"]),
        rule_type=str(raw["rule_type"]),
    )
    if config.vix_buy_level <= 0 or config.vix_sell_level <= 0:
        raise ValueError("Configured VIX levels must be greater than zero.")
    if config.source != "actual_daily_vix" or config.rule_type != "fixed_levels":
        raise ValueError("VIX strategy requires actual_daily_vix with fixed_levels.")
    return config


def vix_buy_signal(row: pd.Series, params: VixParams) -> bool:
    bollinger_buy = (
        row["종가"] < row["볼린저밴드 하단"] * (1 + params.boll_buffer)
    )
    macd_buy = row["MACD"] > row["MACD 시그널"]
    return bool(
        pd.notna(row["VIX"])
        and row["VIX"] >= params.vix_buy_level
        and row["RSI (14일)"] < params.rsi_buy_th
        and (bollinger_buy or macd_buy)
    )


def vix_sell_signal(row: pd.Series, params: VixParams) -> bool:
    bollinger_sell = (
        row["종가"] > row["볼린저밴드 상단"] * (1 + params.boll_buffer)
    )
    macd_sell = row["MACD"] < row["MACD 시그널"]
    return bool(
        pd.notna(row["VIX"])
        and row["VIX"] <= params.vix_sell_level
        and row["RSI (14일)"] > params.rsi_sell_th
        and (bollinger_sell or macd_sell)
    )


def vix_trade_log_details(
    row: pd.Series, params: VixParams, action: str
) -> dict[str, object]:
    actual_vix = float(row["VIX"])
    rsi = float(row["RSI (14일)"])
    lower = float(row["볼린저밴드 하단"])
    upper = float(row["볼린저밴드 상단"])
    macd = float(row["MACD"])
    macd_signal = float(row["MACD 시그널"])
    price = float(row["종가"])
    bollinger_buy = price < lower * (1 + params.boll_buffer)
    bollinger_sell = price > upper * (1 + params.boll_buffer)
    macd_buy = macd > macd_signal
    macd_sell = macd < macd_signal
    if action == "BUY":
        reason = (
            f"BUY: ActualVIX {actual_vix:.4g} >= fixed buy level {params.vix_buy_level:.4g}; "
            f"RSI {rsi:.4g} < {params.rsi_buy_th:.4g}; technical OR passed "
            f"(Bollinger={bollinger_buy}: price {price:.4g} < lower {lower:.4g} * "
            f"(1 + {params.boll_buffer:.4g}), MACD={macd_buy}: {macd:.4g} > "
            f"signal {macd_signal:.4g})."
        )
    elif action == "SELL":
        reason = (
            f"SELL: ActualVIX {actual_vix:.4g} <= fixed sell level {params.vix_sell_level:.4g}; "
            f"RSI {rsi:.4g} > {params.rsi_sell_th:.4g}; technical OR passed "
            f"(Bollinger={bollinger_sell}: price {price:.4g} > upper {upper:.4g} * "
            f"(1 + {params.boll_buffer:.4g}), MACD={macd_sell}: {macd:.4g} < "
            f"signal {macd_signal:.4g})."
        )
    else:
        reason = (
            "LIQUIDATE: end-of-period portfolio closure; not a VIX/technical SELL "
            f"signal. ActualVIX was {actual_vix:.4g}; fixed levels were buy "
            f"{params.vix_buy_level:.4g} and sell {params.vix_sell_level:.4g}."
        )
    return {
        "ActualVIX": actual_vix, "VixBuyLevel": params.vix_buy_level,
        "VixSellLevel": params.vix_sell_level, "RSI": rsi,
        "BollingerLower": lower, "BollingerUpper": upper, "MACD": macd,
        "MACDSignal": macd_signal, "Reason": reason,
    }
