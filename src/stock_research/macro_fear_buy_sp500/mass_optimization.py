from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FearBuyParams
from .contributions import (
    ContributionConfig,
    ContributionDeploymentPolicy,
    run_contribution_backtest,
)
from .features import build_fear_features
from .strategy import generate_fear_buy_signals

_WORKER_FEATURES: dict[str, pd.DataFrame] = {}
_WORKER_CONFIG: ContributionConfig | None = None
_WORKER_BENCHMARKS: dict[str, dict[str, float]] = {}


def sample_candidates(
    count: int,
    *,
    seed: int,
    baseline_params: FearBuyParams,
    baseline_policy: ContributionDeploymentPolicy,
) -> list[dict[str, float | int]]:
    """Draw reproducible, ordered strategy candidates.

    The first row is always the current baseline. Remaining rows are random
    proposals whose ordered thresholds satisfy ``FearBuyParams`` validation.
    """

    if count < 1:
        raise ValueError("count must be at least one.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    baseline = {
        **asdict(baseline_params),
        "mild_fraction": baseline_policy.mild_fraction,
        "fear_fraction": baseline_policy.fear_fraction,
        "panic_fraction": baseline_policy.panic_fraction,
        "cooldown_sessions": baseline_policy.cooldown_sessions,
    }
    rows.append(baseline)
    fraction_grid = np.asarray((0.0, 0.1, 0.25, 0.5, 0.75, 1.0))
    hold_grid = np.asarray(
        (21, 42, 63, 84, 126, 168, 189, 252, 315, 378),
        dtype=int,
    )
    cooldown_grid = np.asarray((0, 5, 10, 21, 42, 63, 84, 126), dtype=int)

    for _ in range(count - 1):
        core = float(rng.uniform(0.55, 0.90))
        tranche = float(rng.uniform(0.025, min(0.20, 1.0 - core)))

        mild_vix = float(rng.uniform(0.65, 0.90))
        fear_vix = float(rng.uniform(max(0.75, mild_vix + 0.03), 0.97))
        panic_vix = float(rng.uniform(max(0.90, fear_vix + 0.01), 0.995))

        mild_fear = float(rng.uniform(0.35, 0.65))
        fear_fear = float(rng.uniform(mild_fear + 0.05, 0.85))
        panic_fear = float(rng.uniform(fear_fear + 0.03, 0.95))

        mild_magnitude = float(rng.uniform(0.04, 0.14))
        fear_magnitude = float(
            rng.uniform(max(0.10, mild_magnitude + 0.03), 0.25)
        )
        panic_magnitude = float(
            rng.uniform(max(0.18, fear_magnitude + 0.05), 0.40)
        )

        weights = rng.dirichlet(np.asarray((2.0, 1.5, 1.5, 1.5, 1.0)))
        weights = weights / weights.sum()
        fraction_indices = np.sort(rng.integers(0, len(fraction_grid), size=3))
        fractions = fraction_grid[fraction_indices]
        rebalance_ceiling = min(0.04, tranche * 0.80)

        rows.append(
            {
                "core_weight": core,
                "tranche_weight": tranche,
                "vix_lookback_days": baseline_params.vix_lookback_days,
                "percentile_min_history_days": (
                    baseline_params.percentile_min_history_days
                ),
                "mild_vix_percentile": mild_vix,
                "fear_vix_percentile": fear_vix,
                "panic_vix_percentile": panic_vix,
                "mild_fear_score": mild_fear,
                "fear_fear_score": fear_fear,
                "panic_fear_score": panic_fear,
                "mild_drawdown": -mild_magnitude,
                "fear_drawdown": -fear_magnitude,
                "panic_drawdown": -panic_magnitude,
                "trim_euphoria_score": float(rng.uniform(0.40, 0.80)),
                "trim_max_fear_score": float(rng.uniform(0.25, 0.60)),
                "trim_max_vix_percentile": float(rng.uniform(0.30, 0.75)),
                "minimum_hold_sessions": int(rng.choice(hold_grid)),
                "trim_profit_buffer": float(rng.uniform(0.0, 0.20)),
                "rebalance_band": float(
                    rng.uniform(0.005, rebalance_ceiling)
                ),
                "decision_frequency": baseline_params.decision_frequency,
                "vix_weight": float(weights[0]),
                "macro_weight": float(weights[1]),
                "model_risk_weight": float(weights[2]),
                "drawdown_weight": float(weights[3]),
                "downside_momentum_weight": float(weights[4]),
                "mild_fraction": float(fractions[0]),
                "fear_fraction": float(fractions[1]),
                "panic_fraction": float(fractions[2]),
                "cooldown_sessions": int(rng.choice(cooldown_grid)),
            }
        )
    return rows


def candidate_to_params(
    candidate: dict[str, float | int],
) -> FearBuyParams:
    fields = FearBuyParams.__dataclass_fields__
    return FearBuyParams(
        **{name: candidate[name] for name in fields},
    )


def candidate_to_policy(
    candidate: dict[str, float | int],
) -> ContributionDeploymentPolicy:
    return ContributionDeploymentPolicy(
        mild_fraction=float(candidate["mild_fraction"]),
        fear_fraction=float(candidate["fear_fraction"]),
        panic_fraction=float(candidate["panic_fraction"]),
        cooldown_sessions=int(candidate["cooldown_sessions"]),
    )


def candidate_features(
    base: pd.DataFrame,
    params: FearBuyParams,
) -> pd.DataFrame:
    frame = base.copy()
    macro = pd.to_numeric(
        frame["MacroConfirmationScore"],
        errors="coerce",
    ).clip(0.0, 1.0)
    frame["FearScore"] = (
        params.vix_weight * frame["VixPercentile"]
        + params.macro_weight * macro
        + params.model_risk_weight * frame["ModelRiskPercentile"]
        + params.drawdown_weight * frame["DrawdownIntensity"]
        + params.downside_momentum_weight
        * frame["DownsideMomentumIntensity"]
    ).clip(0.0, 1.0)
    return frame


def initialize_worker(
    prediction_path: str,
    baseline_params_payload: dict[str, object],
    config_payload: dict[str, float],
    development_end: str,
    include_development_folds: bool = True,
) -> None:
    """Load point-in-time inputs once in each spawned worker."""

    global _WORKER_CONFIG
    predictions = pd.read_csv(Path(prediction_path), parse_dates=["Date"])
    baseline_params = FearBuyParams(**baseline_params_payload)
    features = build_fear_features(predictions, baseline_params)
    development = features[
        features["Date"] <= pd.Timestamp(development_end)
    ].reset_index(drop=True)
    periods = {"Development": development}
    if include_development_folds:
        periods.update(
            {
                "DevelopmentCrisis": development[
                    development["Date"] <= pd.Timestamp("2011-12-30")
                ].reset_index(drop=True),
                "DevelopmentBull": development[
                    development["Date"] >= pd.Timestamp("2012-01-03")
                ].reset_index(drop=True),
            }
        )
    _WORKER_FEATURES.clear()
    _WORKER_FEATURES.update(periods)
    _WORKER_CONFIG = ContributionConfig(**config_payload)


def _evaluate_period(
    features: pd.DataFrame,
    params: FearBuyParams,
    policy: ContributionDeploymentPolicy,
) -> dict[str, float]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Mass-optimization worker is not initialized.")
    features_with_score = candidate_features(features, params)
    signals = generate_fear_buy_signals(features_with_score, params)
    result = run_contribution_backtest(
        signals,
        params,
        _WORKER_CONFIG,
        name="MassOptimizationCandidate",
        deployment_policy=policy,
    )
    summary = result.summary
    return {
        "FinalValue": summary.final_value,
        "NetProfit": summary.net_profit,
        "ROI(%)": summary.roi_percent,
        "XIRR(%)": summary.money_weighted_return_percent,
        "MDD(%)": summary.max_drawdown_percent,
        "Sharpe": summary.sharpe_ratio,
        "AverageExposure(%)": summary.average_exposure_percent,
        "Trades": float(summary.trade_count),
    }


def evaluate_candidate_batch(
    indexed_candidates: Iterable[
        tuple[int, dict[str, float | int]]
    ],
) -> list[dict[str, float | int]]:
    """Evaluate a batch with canonical feature, signal, and portfolio functions."""

    rows: list[dict[str, float | int]] = []
    for candidate_id, candidate in indexed_candidates:
        params = candidate_to_params(candidate)
        policy = candidate_to_policy(candidate)
        row: dict[str, float | int] = {
            "CandidateId": candidate_id,
            **candidate,
        }
        for period, features in _WORKER_FEATURES.items():
            metrics = _evaluate_period(features, params, policy)
            row.update(
                {
                    f"{period}{name}": value
                    for name, value in metrics.items()
                }
            )
        rows.append(row)
    return rows


def constant_signals(signals: pd.DataFrame) -> pd.DataFrame:
    benchmark = signals[["Date", "Open", "Close", "CashRate"]].copy()
    benchmark["TargetWeight"] = 1.0
    benchmark["SignalState"] = "BUY_HOLD"
    benchmark["TransitionReason"] = ""
    return benchmark


def evaluate_frozen_candidate(
    features: pd.DataFrame,
    candidate: dict[str, float | int],
    config: ContributionConfig,
    *,
    name: str,
) -> tuple[FearBuyParams, ContributionDeploymentPolicy, object]:
    params = candidate_to_params(candidate)
    policy = candidate_to_policy(candidate)
    features_with_score = candidate_features(features, params)
    signals = generate_fear_buy_signals(features_with_score, params)
    result = run_contribution_backtest(
        signals,
        params,
        config,
        name=name,
        deployment_policy=policy,
    )
    return params, policy, result


def add_selection_scores(
    candidates: pd.DataFrame,
    benchmark_metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Add transparent development-only return, balance, and safety scores."""

    scored = candidates.copy()
    for period in ("Development", "DevelopmentCrisis", "DevelopmentBull"):
        benchmark_profit = benchmark_metrics[period]["NetProfit"]
        scored[f"{period}ProfitRatio"] = (
            scored[f"{period}NetProfit"] / benchmark_profit
        )
        benchmark_mdd = benchmark_metrics[period]["MDD(%)"]
        scored[f"{period}MDDImprovement"] = (
            scored[f"{period}MDD(%)"] - benchmark_mdd
        )
    scored["WorstDevelopmentFoldProfitRatio"] = scored[
        ["DevelopmentCrisisProfitRatio", "DevelopmentBullProfitRatio"]
    ].min(axis=1)
    positive_drawdown = -scored["DevelopmentMDD(%)"].clip(upper=-1e-6)
    scored["DevelopmentCalmar"] = (
        scored["DevelopmentXIRR(%)"] / positive_drawdown
    )
    scored["ReturnScore"] = (
        0.75 * scored["DevelopmentProfitRatio"]
        + 0.25 * scored["WorstDevelopmentFoldProfitRatio"]
    )
    scored["BalancedScore"] = (
        0.40 * scored["DevelopmentProfitRatio"]
        + 0.25 * scored["WorstDevelopmentFoldProfitRatio"]
        + 0.20 * scored["DevelopmentSharpe"]
        + 0.15 * scored["DevelopmentCalmar"]
    )
    eligible = (
        (scored["DevelopmentProfitRatio"] >= 0.85)
        & (scored["WorstDevelopmentFoldProfitRatio"] >= 0.60)
    )
    scored["SafetyScore"] = np.where(
        eligible,
        0.60 * scored["DevelopmentMDDImprovement"]
        + 0.25 * scored["DevelopmentSharpe"]
        + 0.15 * scored["DevelopmentProfitRatio"],
        -np.inf,
    )
    return scored


def select_category_winners(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select three distinct candidates without consulting holdout data."""

    selections: list[pd.Series] = []
    used: set[int] = set()
    specifications = (
        ("Return", "ReturnScore"),
        ("Balanced", "BalancedScore"),
        ("Safety", "SafetyScore"),
    )
    for label, column in specifications:
        ordered = candidates.sort_values(
            [column, "WorstDevelopmentFoldProfitRatio"],
            ascending=False,
        )
        if label == "Safety" and np.isneginf(
            pd.to_numeric(ordered[column], errors="coerce")
        ).all():
            ordered = candidates.sort_values(
                [
                    "DevelopmentMDD(%)",
                    "DevelopmentProfitRatio",
                ],
                ascending=False,
            )
        winner = next(
            row
            for _, row in ordered.iterrows()
            if int(row["CandidateId"]) not in used
        )
        winner = winner.copy()
        winner["SelectionCategory"] = label
        used.add(int(winner["CandidateId"]))
        selections.append(winner)
    return pd.DataFrame(selections).reset_index(drop=True)
