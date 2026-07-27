from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DRAWDOWN_PROFILES = {
    "10_20_30": (-0.10, -0.20, -0.30),
    "10_20_35": (-0.10, -0.20, -0.35),
    "15_25_35": (-0.15, -0.25, -0.35),
}
TARGET_PROFILES = {
    "70_80_95": (0.70, 0.80, 0.95),
    "70_85_100": (0.70, 0.85, 1.00),
    "80_90_100": (0.80, 0.90, 1.00),
}


@dataclass(frozen=True)
class MacroSp500V2Settings:
    initial_capital: float
    vix_lookback_years: int
    warning_lookback_days: int
    drawdown_lookback_days: int
    reversal_lookback_days: int
    recovery_sma_days: int
    exit_vix_quantile: float
    exit_confirmation_days: int
    vix_decline_from_peak: float
    minimum_trade_fraction: float
    transaction_cost_bps: float
    slippage_bps: float
    fallback_cash_annual_rate: float
    training_years: int
    first_test_year: int
    minimum_vix_observations: int
    minimum_cagr_fraction_of_buy_hold: float
    maximum_mdd_fraction_of_buy_hold: float
    static_benchmark_weight: float

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0 < self.minimum_trade_fraction < 1:
            raise ValueError("minimum_trade_fraction must be between zero and one.")
        if not 0 < self.minimum_cagr_fraction_of_buy_hold <= 1:
            raise ValueError("Invalid CAGR constraint.")
        if not 0 < self.maximum_mdd_fraction_of_buy_hold <= 1:
            raise ValueError("Invalid MDD constraint.")


@dataclass(frozen=True)
class MacroSp500V2Params:
    core_weight: float
    vix_entry_quantile: float
    drawdown_profile: str
    target_profile: str
    rebound_threshold: float
    rebalance_band: float
    minimum_hold_days: int

    def __post_init__(self) -> None:
        if not 0 < self.core_weight < 1:
            raise ValueError("core_weight must be between zero and one.")
        if not 0 < self.vix_entry_quantile < 1:
            raise ValueError("vix_entry_quantile must be between zero and one.")
        if self.drawdown_profile not in DRAWDOWN_PROFILES:
            raise ValueError(f"Unknown drawdown profile: {self.drawdown_profile}")
        if self.target_profile not in TARGET_PROFILES:
            raise ValueError(f"Unknown target profile: {self.target_profile}")
        if not 0 < self.rebound_threshold < 1:
            raise ValueError("rebound_threshold must be between zero and one.")
        if not 0 < self.rebalance_band < 1:
            raise ValueError("rebalance_band must be between zero and one.")
        if self.minimum_hold_days < 0:
            raise ValueError("minimum_hold_days cannot be negative.")

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _config_folder() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "macro_sp500"


def load_v2_settings(path: str | Path | None = None) -> MacroSp500V2Settings:
    config_path = Path(path) if path else _config_folder() / "strategy_v2.json"
    with config_path.open(encoding="utf-8") as handle:
        return MacroSp500V2Settings(**json.load(handle))


def load_v2_search_space(path: str | Path | None = None) -> dict[str, list[object]]:
    config_path = Path(path) if path else _config_folder() / "search_space_v2.json"
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    required = set(MacroSp500V2Params.__annotations__)
    if set(raw) != required:
        missing = required - set(raw)
        extra = set(raw) - required
        raise ValueError(f"Invalid V2 search space. Missing={missing}, extra={extra}")
    return raw


def iter_v2_candidates(
    search_space: dict[str, list[object]],
) -> list[MacroSp500V2Params]:
    names = list(MacroSp500V2Params.__annotations__)
    return [
        MacroSp500V2Params(**dict(zip(names, values, strict=True)))
        for values in itertools.product(*(search_space[name] for name in names))
    ]
