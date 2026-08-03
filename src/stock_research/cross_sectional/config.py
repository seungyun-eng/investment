from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchSettings:
    train_start: str
    train_end: str
    validation_periods: dict[str, tuple[str, str]]
    research_label: str = "baseline"
    validation_is_fresh: bool = True
    selection_validation_label: str | None = None
    candidate_count: int = 400
    seed: int = 20260727
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 10.0
    financial_release_lag_days: int = 45
    max_financial_age_days: int = 180
    minimum_price_history_sessions: int = 200
    minimum_cross_section_size: int = 8
    minimum_financial_weight: float = 0.0
    financial_feature_mode: str = "legacy"
    force_universe_exit: bool = False
    loss_aware_exit_enabled: bool = False
    minimum_exit_gain: float = 0.01
    conviction_exit_rank: int = 20
    conviction_trend_floor: float = -0.10
    conviction_momentum_floor: float = -0.20
    hard_stop_return: float = -0.35
    minimum_hold_rebalances: int = 4
    profit_rotation_exit_rank: int | None = None
    profit_rotation_confirmation_rebalances: int = 1
    replacement_score_advantage: float = 0.0
    overheated_entry_enabled: bool = False
    overheated_return126: float = 1.0
    overheated_trend200: float = 0.50
    overheated_drawdown126_floor: float = -0.03
    trailing_stop_enabled: bool = False
    trailing_stop_activation_gain: float = 0.20
    trailing_stop_drawdown: float = -0.15
    dcf_wacc: float = 0.10
    dcf_cost_of_equity: float = 0.10
    dcf_short_growth: float = 0.05
    dcf_projection_years: int = 5
    dcf_terminal_growth: float = 0.025
    rebalance_weekday: int = 4
    training_folds: tuple[tuple[str, str], ...] = (
        ("2020-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2024-12-31"),
    )

    def __post_init__(self) -> None:
        if self.train_start > self.train_end:
            raise ValueError("train_start must not be after train_end")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0 <= self.minimum_financial_weight <= 1:
            raise ValueError("minimum_financial_weight must be between zero and one")
        if self.financial_feature_mode not in {
            "legacy",
            "ttm_value_momentum",
        }:
            raise ValueError(
                "financial_feature_mode must be legacy or ttm_value_momentum"
            )
        if self.minimum_exit_gain < 0:
            raise ValueError("minimum_exit_gain must be non-negative")
        if self.conviction_exit_rank < 1:
            raise ValueError("conviction_exit_rank must be positive")
        if self.conviction_trend_floor >= 0:
            raise ValueError("conviction_trend_floor must be negative")
        if self.conviction_momentum_floor >= 0:
            raise ValueError("conviction_momentum_floor must be negative")
        if not -1 < self.hard_stop_return < 0:
            raise ValueError("hard_stop_return must be between -1 and zero")
        if self.minimum_hold_rebalances < 1:
            raise ValueError("minimum_hold_rebalances must be positive")
        if (
            self.profit_rotation_exit_rank is not None
            and self.profit_rotation_exit_rank < 1
        ):
            raise ValueError("profit_rotation_exit_rank must be positive")
        if self.profit_rotation_confirmation_rebalances < 1:
            raise ValueError(
                "profit_rotation_confirmation_rebalances must be positive"
            )
        if self.replacement_score_advantage < 0:
            raise ValueError(
                "replacement_score_advantage must be non-negative"
            )
        if self.overheated_return126 <= 0:
            raise ValueError("overheated_return126 must be positive")
        if self.overheated_trend200 <= 0:
            raise ValueError("overheated_trend200 must be positive")
        if not -1 < self.overheated_drawdown126_floor <= 0:
            raise ValueError(
                "overheated_drawdown126_floor must be in (-1, 0]"
            )
        if self.trailing_stop_activation_gain <= 0:
            raise ValueError(
                "trailing_stop_activation_gain must be positive"
            )
        if not -1 < self.trailing_stop_drawdown < 0:
            raise ValueError(
                "trailing_stop_drawdown must be between -1 and zero"
            )
        if self.dcf_wacc <= self.dcf_terminal_growth:
            raise ValueError("dcf_wacc must exceed dcf_terminal_growth")
        if self.dcf_cost_of_equity <= self.dcf_terminal_growth:
            raise ValueError(
                "dcf_cost_of_equity must exceed dcf_terminal_growth"
            )
        if self.dcf_projection_years < 1:
            raise ValueError("dcf_projection_years must be positive")
        if not 0 <= self.rebalance_weekday <= 4:
            raise ValueError("rebalance_weekday must be a weekday from 0 to 4")
        for name, (start, end) in self.validation_periods.items():
            if start > end:
                raise ValueError(f"Validation period {name} starts after it ends")
            if start <= self.train_end:
                raise ValueError(
                    f"Validation period {name} overlaps the training period"
                )
        if (
            self.selection_validation_label is not None
            and self.selection_validation_label not in self.validation_periods
        ):
            raise ValueError(
                "selection_validation_label must name a validation period"
            )


WEIGHTING_SCHEMES = {"equal", "score_proportional", "inverse_volatility"}


@dataclass(frozen=True)
class StrategyParams:
    momentum_weight: float
    trend_weight: float
    growth_weight: float
    quality_weight: float
    risk_control_weight: float
    top_k: int
    exit_rank: int
    trend_floor: float
    momentum_floor: float
    loss_aware_exit_enabled: bool = False
    minimum_exit_gain: float = 0.01
    conviction_exit_rank: int = 20
    conviction_trend_floor: float = -0.10
    conviction_momentum_floor: float = -0.20
    hard_stop_return: float = -0.35
    minimum_hold_rebalances: int = 4
    profit_rotation_exit_rank: int | None = None
    profit_rotation_confirmation_rebalances: int = 1
    replacement_score_advantage: float = 0.0
    overheated_entry_enabled: bool = False
    overheated_return126: float = 1.0
    overheated_trend200: float = 0.50
    overheated_drawdown126_floor: float = -0.03
    trailing_stop_enabled: bool = False
    trailing_stop_activation_gain: float = 0.20
    trailing_stop_drawdown: float = -0.15
    weighting_scheme: str = "equal"

    def __post_init__(self) -> None:
        weights = self.factor_weights
        if any(value < 0 for value in weights.values()):
            raise ValueError("Factor weights must be non-negative")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Factor weights must sum to one")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.exit_rank < self.top_k:
            raise ValueError("exit_rank must be at least top_k")
        if self.conviction_exit_rank < self.exit_rank:
            raise ValueError(
                "conviction_exit_rank must be at least exit_rank"
            )
        if self.minimum_exit_gain < 0:
            raise ValueError("minimum_exit_gain must be non-negative")
        if self.conviction_trend_floor >= 0:
            raise ValueError("conviction_trend_floor must be negative")
        if self.conviction_momentum_floor >= 0:
            raise ValueError("conviction_momentum_floor must be negative")
        if not -1 < self.hard_stop_return < 0:
            raise ValueError("hard_stop_return must be between -1 and zero")
        if self.minimum_hold_rebalances < 1:
            raise ValueError("minimum_hold_rebalances must be positive")
        if (
            self.profit_rotation_exit_rank is not None
            and self.profit_rotation_exit_rank < self.exit_rank
        ):
            raise ValueError(
                "profit_rotation_exit_rank must be at least exit_rank"
            )
        if self.profit_rotation_confirmation_rebalances < 1:
            raise ValueError(
                "profit_rotation_confirmation_rebalances must be positive"
            )
        if self.replacement_score_advantage < 0:
            raise ValueError(
                "replacement_score_advantage must be non-negative"
            )
        if self.overheated_return126 <= 0:
            raise ValueError("overheated_return126 must be positive")
        if self.overheated_trend200 <= 0:
            raise ValueError("overheated_trend200 must be positive")
        if not -1 < self.overheated_drawdown126_floor <= 0:
            raise ValueError(
                "overheated_drawdown126_floor must be in (-1, 0]"
            )
        if self.trailing_stop_activation_gain <= 0:
            raise ValueError(
                "trailing_stop_activation_gain must be positive"
            )
        if not -1 < self.trailing_stop_drawdown < 0:
            raise ValueError(
                "trailing_stop_drawdown must be between -1 and zero"
            )
        if self.weighting_scheme not in WEIGHTING_SCHEMES:
            raise ValueError(
                f"weighting_scheme must be one of {sorted(WEIGHTING_SCHEMES)}"
            )

    @property
    def factor_weights(self) -> dict[str, float]:
        return {
            "MomentumFactor": self.momentum_weight,
            "TrendFactor": self.trend_weight,
            "GrowthFactor": self.growth_weight,
            "QualityFactor": self.quality_weight,
            "RiskControlFactor": self.risk_control_weight,
        }

    @property
    def financial_weight(self) -> float:
        return self.growth_weight + self.quality_weight

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> StrategyParams:
        return cls(
            momentum_weight=float(values["momentum_weight"]),
            trend_weight=float(values["trend_weight"]),
            growth_weight=float(values["growth_weight"]),
            quality_weight=float(values["quality_weight"]),
            risk_control_weight=float(values["risk_control_weight"]),
            top_k=int(values["top_k"]),
            exit_rank=int(values["exit_rank"]),
            trend_floor=float(values["trend_floor"]),
            momentum_floor=float(values["momentum_floor"]),
            loss_aware_exit_enabled=bool(
                values.get("loss_aware_exit_enabled", False)
            ),
            minimum_exit_gain=float(values.get("minimum_exit_gain", 0.01)),
            conviction_exit_rank=int(
                values.get("conviction_exit_rank", 20)
            ),
            conviction_trend_floor=float(
                values.get("conviction_trend_floor", -0.10)
            ),
            conviction_momentum_floor=float(
                values.get("conviction_momentum_floor", -0.20)
            ),
            hard_stop_return=float(values.get("hard_stop_return", -0.35)),
            minimum_hold_rebalances=int(
                values.get("minimum_hold_rebalances", 4)
            ),
            profit_rotation_exit_rank=(
                int(values["profit_rotation_exit_rank"])
                if values.get("profit_rotation_exit_rank") is not None
                else None
            ),
            profit_rotation_confirmation_rebalances=int(
                values.get(
                    "profit_rotation_confirmation_rebalances",
                    1,
                )
            ),
            replacement_score_advantage=float(
                values.get("replacement_score_advantage", 0.0)
            ),
            overheated_entry_enabled=bool(
                values.get("overheated_entry_enabled", False)
            ),
            overheated_return126=float(
                values.get("overheated_return126", 1.0)
            ),
            overheated_trend200=float(
                values.get("overheated_trend200", 0.50)
            ),
            overheated_drawdown126_floor=float(
                values.get("overheated_drawdown126_floor", -0.03)
            ),
            trailing_stop_enabled=bool(
                values.get("trailing_stop_enabled", False)
            ),
            trailing_stop_activation_gain=float(
                values.get("trailing_stop_activation_gain", 0.20)
            ),
            trailing_stop_drawdown=float(
                values.get("trailing_stop_drawdown", -0.15)
            ),
            weighting_scheme=str(
                values.get("weighting_scheme", "equal")
            ),
        )


def load_settings(path: str | Path) -> ResearchSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return settings_from_dict(raw)


def settings_from_dict(raw: dict[str, object]) -> ResearchSettings:
    validation = {
        name: (str(period[0]), str(period[1]))
        for name, period in dict(raw["validation_periods"]).items()
    }
    folds = tuple(
        (str(start), str(end))
        for start, end in raw.get(
            "training_folds",
            (
                ("2020-01-01", "2021-12-31"),
                ("2022-01-01", "2022-12-31"),
                ("2023-01-01", "2024-12-31"),
            ),
        )
    )
    return ResearchSettings(
        train_start=str(raw["train_start"]),
        train_end=str(raw["train_end"]),
        validation_periods=validation,
        research_label=str(raw.get("research_label", "baseline")),
        validation_is_fresh=bool(raw.get("validation_is_fresh", True)),
        selection_validation_label=(
            str(raw["selection_validation_label"])
            if raw.get("selection_validation_label") is not None
            else None
        ),
        candidate_count=int(raw.get("candidate_count", 400)),
        seed=int(raw.get("seed", 20260727)),
        initial_capital=float(raw.get("initial_capital", 100_000.0)),
        transaction_cost_bps=float(raw.get("transaction_cost_bps", 10.0)),
        financial_release_lag_days=int(
            raw.get("financial_release_lag_days", 45)
        ),
        max_financial_age_days=int(raw.get("max_financial_age_days", 180)),
        minimum_price_history_sessions=int(
            raw.get("minimum_price_history_sessions", 200)
        ),
        minimum_cross_section_size=int(
            raw.get("minimum_cross_section_size", 8)
        ),
        minimum_financial_weight=float(
            raw.get("minimum_financial_weight", 0.0)
        ),
        financial_feature_mode=str(
            raw.get("financial_feature_mode", "legacy")
        ),
        force_universe_exit=bool(raw.get("force_universe_exit", False)),
        loss_aware_exit_enabled=bool(
            raw.get("loss_aware_exit_enabled", False)
        ),
        minimum_exit_gain=float(raw.get("minimum_exit_gain", 0.01)),
        conviction_exit_rank=int(raw.get("conviction_exit_rank", 20)),
        conviction_trend_floor=float(
            raw.get("conviction_trend_floor", -0.10)
        ),
        conviction_momentum_floor=float(
            raw.get("conviction_momentum_floor", -0.20)
        ),
        hard_stop_return=float(raw.get("hard_stop_return", -0.35)),
        minimum_hold_rebalances=int(
            raw.get("minimum_hold_rebalances", 4)
        ),
        profit_rotation_exit_rank=(
            int(raw["profit_rotation_exit_rank"])
            if raw.get("profit_rotation_exit_rank") is not None
            else None
        ),
        profit_rotation_confirmation_rebalances=int(
            raw.get("profit_rotation_confirmation_rebalances", 1)
        ),
        replacement_score_advantage=float(
            raw.get("replacement_score_advantage", 0.0)
        ),
        overheated_entry_enabled=bool(
            raw.get("overheated_entry_enabled", False)
        ),
        overheated_return126=float(
            raw.get("overheated_return126", 1.0)
        ),
        overheated_trend200=float(
            raw.get("overheated_trend200", 0.50)
        ),
        overheated_drawdown126_floor=float(
            raw.get("overheated_drawdown126_floor", -0.03)
        ),
        trailing_stop_enabled=bool(
            raw.get("trailing_stop_enabled", False)
        ),
        trailing_stop_activation_gain=float(
            raw.get("trailing_stop_activation_gain", 0.20)
        ),
        trailing_stop_drawdown=float(
            raw.get("trailing_stop_drawdown", -0.15)
        ),
        dcf_wacc=float(raw.get("dcf_wacc", 0.10)),
        dcf_cost_of_equity=float(raw.get("dcf_cost_of_equity", 0.10)),
        dcf_short_growth=float(raw.get("dcf_short_growth", 0.05)),
        dcf_projection_years=int(raw.get("dcf_projection_years", 5)),
        dcf_terminal_growth=float(raw.get("dcf_terminal_growth", 0.025)),
        rebalance_weekday=int(raw.get("rebalance_weekday", 4)),
        training_folds=folds,
    )
