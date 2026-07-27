from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FearBuyParams:
    """Trading parameters used by both optimization and the final simulation."""

    core_weight: float = 0.70
    tranche_weight: float = 0.10
    vix_lookback_days: int = 1260
    percentile_min_history_days: int = 252
    mild_vix_percentile: float = 0.80
    fear_vix_percentile: float = 0.90
    panic_vix_percentile: float = 0.97
    mild_fear_score: float = 0.55
    fear_fear_score: float = 0.65
    panic_fear_score: float = 0.75
    mild_drawdown: float = -0.08
    fear_drawdown: float = -0.15
    panic_drawdown: float = -0.25
    trim_euphoria_score: float = 0.55
    trim_max_fear_score: float = 0.45
    trim_max_vix_percentile: float = 0.60
    minimum_hold_sessions: int = 126
    trim_profit_buffer: float = 0.02
    rebalance_band: float = 0.025
    decision_frequency: str = "W-FRI"
    vix_weight: float = 0.30
    macro_weight: float = 0.20
    model_risk_weight: float = 0.20
    drawdown_weight: float = 0.20
    downside_momentum_weight: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.core_weight <= 1:
            raise ValueError("core_weight must be in (0, 1].")
        if not 0 < self.tranche_weight <= 1 - self.core_weight + 1e-12:
            raise ValueError("tranche_weight must fit inside the tactical reserve.")
        percentiles = (
            self.mild_vix_percentile,
            self.fear_vix_percentile,
            self.panic_vix_percentile,
        )
        if not 0 < percentiles[0] < percentiles[1] < percentiles[2] <= 1:
            raise ValueError("VIX percentiles must increase from mild to panic.")
        fear_scores = (
            self.mild_fear_score,
            self.fear_fear_score,
            self.panic_fear_score,
        )
        if not 0 <= fear_scores[0] < fear_scores[1] < fear_scores[2] <= 1:
            raise ValueError("Fear thresholds must increase from mild to panic.")
        drawdowns = (
            self.mild_drawdown,
            self.fear_drawdown,
            self.panic_drawdown,
        )
        if not 0 > drawdowns[0] > drawdowns[1] > drawdowns[2] >= -1:
            raise ValueError("Drawdown thresholds must deepen from mild to panic.")
        if self.minimum_hold_sessions < 0:
            raise ValueError("minimum_hold_sessions must be non-negative.")
        if self.vix_lookback_days < self.percentile_min_history_days:
            raise ValueError("VIX lookback must cover the minimum history.")
        if not 0 <= self.trim_profit_buffer < 1:
            raise ValueError("trim_profit_buffer must be in [0, 1).")
        if not 0 < self.rebalance_band <= 1:
            raise ValueError("rebalance_band must be in (0, 1].")
        score_weights = (
            self.vix_weight,
            self.macro_weight,
            self.model_risk_weight,
            self.drawdown_weight,
            self.downside_momentum_weight,
        )
        if any(weight < 0 for weight in score_weights):
            raise ValueError("Fear-score weights must be non-negative.")
        if abs(sum(score_weights) - 1.0) > 1e-9:
            raise ValueError("Fear-score weights must sum to one.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FearBuySettings:
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    development_end: str = "2016-12-30"
    holdout_start: str = "2017-01-03"
    optimization_core_weights: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80)
    optimization_mild_scores: tuple[float, ...] = (0.50, 0.55, 0.60)
    optimization_fear_scores: tuple[float, ...] = (0.60, 0.65, 0.70)
    optimization_euphoria_scores: tuple[float, ...] = (0.50, 0.55, 0.60)
    optimization_hold_sessions: tuple[int, ...] = (63, 126, 189)
    optimization_profit_buffers: tuple[float, ...] = (0.02, 0.05)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if min(self.transaction_cost_bps, self.slippage_bps) < 0:
            raise ValueError("Trading costs must be non-negative.")
        if not self.optimization_core_weights:
            raise ValueError("At least one optimization core weight is required.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_fear_buy_config(
    path: str | Path,
) -> tuple[FearBuyParams, FearBuySettings]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    params = FearBuyParams(**payload.get("strategy", {}))
    settings_payload = payload.get("research", {})
    tuple_fields = (
        "optimization_core_weights",
        "optimization_mild_scores",
        "optimization_fear_scores",
        "optimization_euphoria_scores",
        "optimization_hold_sessions",
        "optimization_profit_buffers",
    )
    for name in tuple_fields:
        if name in settings_payload:
            settings_payload[name] = tuple(settings_payload[name])
    settings = FearBuySettings(**settings_payload)
    return params, settings
