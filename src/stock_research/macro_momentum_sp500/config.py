from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    """Configuration shared by features, models, evaluation, and portfolio."""

    training_years: int = 10
    expanding_training: bool = False
    training_stride: int = 1
    first_test_year: int = 2007
    inner_validation_years: int = 3
    random_seed: int = 42
    monthly_release_lag_days: int = 45
    weekly_release_lag_days: int = 7
    momentum_windows: tuple[int, ...] = (5, 10, 21, 42, 63, 126, 189, 252)
    sma_windows: tuple[int, ...] = (20, 50, 100, 200)
    volatility_windows: tuple[int, ...] = (10, 20, 60, 126)
    distribution_windows: tuple[int, ...] = (252, 1260)
    return_horizons: tuple[int, ...] = (21, 63, 126, 252)
    risk_horizons: tuple[int, ...] = (21, 63, 126)
    drawdown_thresholds: tuple[float, ...] = (-0.05, -0.10, -0.15, -0.20)
    primary_return_horizon: int = 126
    primary_risk_horizon: int = 126
    primary_drawdown_threshold: float = -0.10
    minimum_train_rows: int = 1000
    search_budget_per_task: int = 24
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_capital: float = 100_000.0
    defensive_weight: float = 0.25
    caution_weight: float = 0.70
    risk_probability_threshold: float = 0.60
    caution_probability_threshold: float = 0.50
    negative_return_threshold: float = 0.0
    rebalance_band: float = 0.05
    state_risk_horizons: tuple[int, ...] = (63, 126)
    state_risk_weights: tuple[float, ...] = (0.40, 0.60)
    state_risk_smoothing_days: int = 10
    state_macro_smoothing_days: int = 20
    state_caution_entry_risk: float = 0.55
    state_defensive_entry_risk: float = 0.70
    state_exit_risk: float = 0.45
    state_recovery_risk: float = 0.50
    state_caution_macro: float = 0.55
    state_defensive_macro: float = 0.60
    state_macro_only_caution: float = 0.60
    state_macro_only_defensive: float = 0.70
    state_macro_only_emergency: float = 0.85
    state_momentum_recovery_macro: float = 0.60
    state_momentum_days: int = 63
    state_trend_sma_days: int = 126
    state_exit_macro: float = 0.50
    state_recovery_macro: float = 0.52
    state_emergency_risk: float = 0.80
    state_emergency_macro: float = 0.65
    state_loss_exit_risk: float = 0.65
    state_loss_exit_macro: float = 0.65
    state_loss_tolerance: float = 0.0025
    state_entry_confirmations: int = 2
    state_exit_confirmations: int = 2
    state_min_hold_days: int = 20
    state_recovery_days: int = 10
    state_normal_cooldown_days: int = 10
    state_rebalance_band: float = 0.10
    feature_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "price": ("Momentum_", "SMA", "RealizedVol_", "DownsideVol_", "Drawdown"),
            "price_vix": (
                "Momentum_",
                "SMA",
                "RealizedVol_",
                "DownsideVol_",
                "Drawdown",
                "VIX",
            ),
            "all": ("*",),
        }
    )

    def __post_init__(self) -> None:
        if self.training_years < 3:
            raise ValueError("training_years must be at least 3.")
        if self.training_stride < 1:
            raise ValueError("training_stride must be positive.")
        if self.inner_validation_years < 2:
            raise ValueError("inner_validation_years must be at least 2.")
        if self.search_budget_per_task < 1:
            raise ValueError("search_budget_per_task must be positive.")
        if not 0 <= self.defensive_weight <= self.caution_weight <= 1:
            raise ValueError("Portfolio weights must satisfy 0 <= defensive <= caution <= 1.")
        if self.primary_return_horizon not in self.return_horizons:
            raise ValueError("primary_return_horizon must be in return_horizons.")
        if self.primary_risk_horizon not in self.risk_horizons:
            raise ValueError("primary_risk_horizon must be in risk_horizons.")
        if self.primary_drawdown_threshold not in self.drawdown_thresholds:
            raise ValueError("primary_drawdown_threshold must be in drawdown_thresholds.")
        if not self.state_risk_horizons:
            raise ValueError("state_risk_horizons must not be empty.")
        if len(self.state_risk_horizons) != len(self.state_risk_weights):
            raise ValueError("state risk horizons and weights must have equal length.")
        if any(weight < 0 for weight in self.state_risk_weights):
            raise ValueError("state risk weights must be non-negative.")
        if sum(self.state_risk_weights) <= 0:
            raise ValueError("state risk weights must have positive total weight.")
        probability_fields = (
            self.state_caution_entry_risk,
            self.state_defensive_entry_risk,
            self.state_exit_risk,
            self.state_recovery_risk,
            self.state_caution_macro,
            self.state_defensive_macro,
            self.state_macro_only_caution,
            self.state_macro_only_defensive,
            self.state_macro_only_emergency,
            self.state_momentum_recovery_macro,
            self.state_exit_macro,
            self.state_recovery_macro,
            self.state_emergency_risk,
            self.state_emergency_macro,
            self.state_loss_exit_risk,
            self.state_loss_exit_macro,
        )
        if any(not 0 <= value <= 1 for value in probability_fields):
            raise ValueError("State transition risk and macro thresholds must be in [0, 1].")
        if not self.state_exit_risk < self.state_caution_entry_risk:
            raise ValueError("State exit risk must be below caution entry risk.")
        if not self.state_exit_macro < self.state_caution_macro:
            raise ValueError("State exit macro must be below caution entry macro.")
        if not (
            self.state_caution_macro
            < self.state_macro_only_caution
            < self.state_macro_only_defensive
            < self.state_macro_only_emergency
        ):
            raise ValueError(
                "Macro-only thresholds must increase from caution to emergency."
            )
        if min(
            self.state_risk_smoothing_days,
            self.state_macro_smoothing_days,
            self.state_entry_confirmations,
            self.state_exit_confirmations,
            self.state_min_hold_days,
            self.state_recovery_days,
            self.state_normal_cooldown_days,
            self.state_momentum_days,
            self.state_trend_sma_days,
        ) < 1:
            raise ValueError("State transition windows and confirmations must be positive.")
        if not 0 <= self.state_loss_tolerance < 1:
            raise ValueError("state_loss_tolerance must be in [0, 1).")
        if not 0 < self.state_rebalance_band <= 1:
            raise ValueError("state_rebalance_band must be in (0, 1].")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _tuple_fields() -> tuple[str, ...]:
    return (
        "momentum_windows",
        "sma_windows",
        "volatility_windows",
        "distribution_windows",
        "return_horizons",
        "risk_horizons",
        "drawdown_thresholds",
        "state_risk_horizons",
        "state_risk_weights",
    )


def load_research_config(path: str | Path) -> ResearchConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in _tuple_fields():
        if name in payload:
            payload[name] = tuple(payload[name])
    if "feature_groups" in payload:
        payload["feature_groups"] = {
            name: tuple(patterns) for name, patterns in payload["feature_groups"].items()
        }
    return ResearchConfig(**payload)
