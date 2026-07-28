from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegratedParams:
    technical_weight: float = 0.35
    financial_weight: float = 0.35
    macro_weight: float = 0.30
    buy_threshold: float = 0.62
    sell_threshold: float = 0.42
    short_threshold: float = 0.28
    cover_threshold: float = 0.48
    short_macro_score_max: float = 0.45
    buy_macro_score_min: float = 0.45
    short_downside_probability_min: float = 0.65
    sell_downside_probability_min: float = 0.55
    reentry_downside_probability_max: float = 0.55
    cover_downside_probability_max: float = 0.45
    buy_downside_probability_max: float = 0.40
    reentry_macro_score_min: float = 0.35
    reentry_return21_min: float = 0.00
    trend_entry_window: int = 50
    trend_exit_window: int = 50
    trend_entry_threshold: float = 0.00
    trend_exit_threshold: float = -0.05
    rsi_oversold: float = 38.0
    rsi_overbought: float = 72.0
    stop_loss: float = 0.22
    trailing_stop: float = 0.18
    short_stop_loss: float = 0.15
    short_trailing_stop: float = 0.15
    minimum_hold_sessions: int = 10

    def __post_init__(self) -> None:
        weights = (
            self.technical_weight,
            self.financial_weight,
            self.macro_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Signal weights must be non-negative.")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Signal weights must sum to 1.")
        if not 0 <= self.short_threshold < self.sell_threshold:
            raise ValueError("Require 0 <= short < sell.")
        if not self.short_threshold < self.cover_threshold <= 1:
            raise ValueError("Require short < cover <= 1.")
        if not 0 <= self.short_macro_score_max <= 1:
            raise ValueError("short_macro_score_max must be in [0, 1].")
        if not 0 <= self.buy_macro_score_min <= 1:
            raise ValueError("buy_macro_score_min must be in [0, 1].")
        if not 0 <= self.cover_downside_probability_max:
            raise ValueError("cover downside probability must be non-negative.")
        if not (
            self.cover_downside_probability_max
            < self.short_downside_probability_min
            <= 1
        ):
            raise ValueError("Require cover downside probability < short.")
        if not (
            0 <= self.buy_downside_probability_max
            < self.short_downside_probability_min
        ):
            raise ValueError("Require buy downside probability < short.")
        if not 0 <= self.sell_downside_probability_min <= 1:
            raise ValueError("sell downside probability must be in [0, 1].")
        if not (
            0 <= self.reentry_downside_probability_max
            < self.short_downside_probability_min
        ):
            raise ValueError("Require reentry downside probability < short.")
        if not 0 <= self.reentry_macro_score_min <= 1:
            raise ValueError("reentry macro score must be in [0, 1].")
        if not -1 <= self.reentry_return21_min <= 1:
            raise ValueError("reentry 21-session return must be in [-1, 1].")
        if self.trend_entry_window < 2 or self.trend_exit_window < 2:
            raise ValueError("Trend windows must be at least two sessions.")
        if not -1 <= self.trend_entry_threshold <= 1:
            raise ValueError("trend entry threshold must be in [-1, 1].")
        if not -1 <= self.trend_exit_threshold <= 1:
            raise ValueError("trend exit threshold must be in [-1, 1].")
        if not self.sell_threshold < self.buy_threshold <= 1:
            raise ValueError("Require sell < buy <= 1.")
        if not 0 <= self.stop_loss < 1:
            raise ValueError("stop_loss must be in [0, 1).")
        if not 0 <= self.trailing_stop < 1:
            raise ValueError("trailing_stop must be in [0, 1).")
        if not 0 <= self.short_stop_loss < 1:
            raise ValueError("short_stop_loss must be in [0, 1).")
        if not 0 <= self.short_trailing_stop < 1:
            raise ValueError("short_trailing_stop must be in [0, 1).")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class IntegratedSettings:
    initial_capital: float = 40_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    annual_short_borrow_bps: float = 300.0
    financial_release_lag_days: int = 45
    development_start: str = "2019-01-01"
    development_end: str = "2025-12-31"
    holdout_start: str = "2026-01-01"
    maximum_drawdown_percent: float = -40.0
    optimization_candidates: int = 2_000
    consensus_members: int = 50
    entry_consensus: float = 0.70
    exit_consensus: float = 0.50
    initial_long: bool = False
    seed: int = 20260727


def load_config(path: Path) -> tuple[IntegratedParams, IntegratedSettings]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        IntegratedParams(**payload.get("strategy", {})),
        IntegratedSettings(**payload.get("research", {})),
    )
