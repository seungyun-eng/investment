from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MacroSp500Settings:
    initial_capital: float
    warning_lookback_days: int
    drawdown_lookback_days: int
    panic_confirmation_window: int
    exit_confirmation_days: int
    stage_2_quantile_step: float
    stage_3_quantile: float
    drawdown_levels: tuple[float, float, float]
    stage_target_weights: tuple[float, float, float]
    transaction_cost_bps: float
    slippage_bps: float
    cash_annual_rate: float
    training_years: int
    first_test_year: int
    minimum_vix_observations: int
    minimum_cagr_fraction_of_buy_hold: float

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if len(self.drawdown_levels) != 3 or len(self.stage_target_weights) != 3:
            raise ValueError("Exactly three drawdown levels and target weights are required.")
        if tuple(sorted(self.drawdown_levels, reverse=True)) != self.drawdown_levels:
            raise ValueError("drawdown_levels must progress from mild to severe.")
        if tuple(sorted(self.stage_target_weights)) != self.stage_target_weights:
            raise ValueError("stage_target_weights must be nondecreasing.")
        if self.stage_target_weights[-1] > 1.0:
            raise ValueError("Leverage is not supported in the first optimization.")


@dataclass(frozen=True)
class MacroSp500Params:
    vix_lookback_years: int
    core_weight: float
    warning_score_min: int
    warning_addition: float
    vix_entry_quantile: float
    exit_vix_quantile: float
    minimum_hold_days: int

    def __post_init__(self) -> None:
        if self.vix_lookback_years <= 0:
            raise ValueError("vix_lookback_years must be positive.")
        if not 0 <= self.core_weight <= 1:
            raise ValueError("core_weight must be between zero and one.")
        if not 0 <= self.warning_addition <= 1 - self.core_weight:
            raise ValueError("warning_addition is incompatible with core_weight.")
        if self.warning_score_min not in (1, 2, 3):
            raise ValueError("warning_score_min must be between one and three.")
        if not 0 < self.exit_vix_quantile < self.vix_entry_quantile < 1:
            raise ValueError("VIX exit quantile must be below the entry quantile.")
        if self.minimum_hold_days < 0:
            raise ValueError("minimum_hold_days cannot be negative.")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _default_config_folder() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "macro_sp500"


def load_settings(path: str | Path | None = None) -> MacroSp500Settings:
    config_path = Path(path) if path else _default_config_folder() / "strategy.json"
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["drawdown_levels"] = tuple(float(value) for value in raw["drawdown_levels"])
    raw["stage_target_weights"] = tuple(
        float(value) for value in raw["stage_target_weights"]
    )
    return MacroSp500Settings(**raw)


def load_search_space(path: str | Path | None = None) -> dict[str, list[object]]:
    config_path = Path(path) if path else _default_config_folder() / "search_space.json"
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    required = set(MacroSp500Params.__annotations__)
    if set(raw) != required:
        missing = required - set(raw)
        extra = set(raw) - required
        raise ValueError(f"Invalid search space. Missing={sorted(missing)}, extra={sorted(extra)}")
    return raw


def iter_parameter_candidates(
    search_space: dict[str, list[object]],
) -> list[MacroSp500Params]:
    names = list(MacroSp500Params.__annotations__)
    candidates: list[MacroSp500Params] = []
    for values in itertools.product(*(search_space[name] for name in names)):
        candidates.append(MacroSp500Params(**dict(zip(names, values, strict=True))))
    return candidates
