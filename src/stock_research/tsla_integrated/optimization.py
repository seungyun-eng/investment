from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import IntegratedParams
from .portfolio import IntegratedResult, run_integrated_backtest
from .strategy import generate_integrated_signals


@dataclass(frozen=True)
class OptimizationResult:
    selected_params: IntegratedParams
    candidates: pd.DataFrame


DEFAULT_FOLDS = (
    ("Fold_2019_2021", "2019-01-01", "2021-12-31"),
    ("Fold_2022_2023", "2022-01-01", "2023-12-31"),
    ("Fold_2024_2025", "2024-01-01", "2025-12-31"),
)

INTEGER_PARAM_NAMES = {
    "minimum_hold_sessions",
    "trend_entry_window",
    "trend_exit_window",
}


def buy_and_hold_params() -> IntegratedParams:
    return IntegratedParams(
        stop_loss=0.999999,
        trailing_stop=0.999999,
        short_stop_loss=0.999999,
        short_trailing_stop=0.999999,
        minimum_hold_sessions=1_000_000,
    )


def params_from_candidate_row(row: pd.Series) -> IntegratedParams:
    values: dict[str, float | int] = {}
    for name in IntegratedParams.__dataclass_fields__:
        value = row[name]
        values[name] = (
            int(value) if name in INTEGER_PARAM_NAMES else float(value)
        )
    return IntegratedParams(**values)


def consensus_execution_params(
    members: Sequence[IntegratedParams],
) -> IntegratedParams:
    if not members:
        raise ValueError("Consensus execution requires at least one member.")
    return IntegratedParams(
        stop_loss=float(np.median([member.stop_loss for member in members])),
        trailing_stop=float(
            np.median([member.trailing_stop for member in members])
        ),
        short_stop_loss=float(
            np.median([member.short_stop_loss for member in members])
        ),
        short_trailing_stop=float(
            np.median(
                [member.short_trailing_stop for member in members]
            )
        ),
        short_leverage=float(
            np.median([member.short_leverage for member in members])
        ),
        minimum_hold_sessions=int(
            np.median([member.minimum_hold_sessions for member in members])
        ),
    )


def sample_params(rng: np.random.Generator) -> IntegratedParams:
    weights = rng.dirichlet([2.0, 2.0, 2.0])
    sell = float(rng.uniform(0.35, 0.58))
    short = float(rng.uniform(0.22, min(0.50, sell - 0.01)))
    short_downside = float(rng.uniform(0.55, 0.85))
    entry_window = int(
        rng.choice([10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200])
    )
    exit_window = (
        entry_window
        if rng.random() < 0.5
        else int(
            rng.choice(
                [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
            )
        )
    )
    stop_loss = (
        0.999999
        if rng.random() < 0.25
        else float(rng.uniform(0.12, 0.35))
    )
    trailing_stop = (
        0.999999
        if rng.random() < 0.25
        else float(rng.uniform(0.08, 0.35))
    )
    return IntegratedParams(
        technical_weight=float(weights[0]),
        financial_weight=float(weights[1]),
        macro_weight=float(weights[2]),
        buy_threshold=float(rng.uniform(max(0.52, sell + 0.05), 0.82)),
        sell_threshold=sell,
        short_threshold=short,
        cover_threshold=float(rng.uniform(max(0.40, short + 0.04), 0.70)),
        short_macro_score_max=float(rng.uniform(0.35, 0.70)),
        buy_macro_score_min=float(rng.uniform(0.35, 0.70)),
        short_downside_probability_min=short_downside,
        sell_downside_probability_min=float(rng.uniform(0.45, 0.75)),
        reentry_downside_probability_max=float(
            rng.uniform(0.35, min(0.95, short_downside - 0.01))
        ),
        cover_downside_probability_max=float(rng.uniform(0.25, 0.50)),
        buy_downside_probability_max=float(rng.uniform(0.20, 0.50)),
        reentry_macro_score_min=float(rng.uniform(0.0, 0.60)),
        reentry_return21_min=float(rng.uniform(-0.25, 0.12)),
        trend_entry_window=entry_window,
        trend_exit_window=exit_window,
        trend_entry_threshold=float(rng.uniform(-0.08, 0.12)),
        trend_exit_threshold=float(rng.uniform(-0.25, 0.04)),
        rsi_oversold=float(rng.uniform(28, 45)),
        rsi_overbought=float(rng.uniform(65, 82)),
        stop_loss=stop_loss,
        trailing_stop=trailing_stop,
        short_stop_loss=float(rng.uniform(0.08, 0.25)),
        short_trailing_stop=float(rng.uniform(0.08, 0.25)),
        short_leverage=float(rng.uniform(1.0, 2.5)),
        minimum_hold_sessions=int(rng.choice([5, 10, 21, 42, 63])),
    )


def evaluate(
    features: pd.DataFrame,
    params: IntegratedParams,
    **backtest_kwargs: float,
) -> IntegratedResult:
    return run_integrated_backtest(
        generate_integrated_signals(features, params),
        params,
        **backtest_kwargs,
    )


def _alpha_robustness_tiers(
    candidates: pd.DataFrame,
    folds: tuple[tuple[str, str, str], ...],
) -> tuple[pd.Series, pd.Series]:
    fold_alpha_positive = [
        candidates[f"{name}ExcessROI(%)"] > 0
        for name, _, _ in folds
    ]
    positive_count = sum(
        condition.astype(int)
        for condition in fold_alpha_positive
    )
    recent_alpha = fold_alpha_positive[-1]
    last_two_alpha = (
        fold_alpha_positive[-2] & recent_alpha
        if len(fold_alpha_positive) >= 2
        else recent_alpha
    )
    every_fold_alpha = positive_count == len(folds)
    tiers = pd.Series(
        np.select(
            [every_fold_alpha, last_two_alpha, recent_alpha],
            [3, 2, 1],
            default=0,
        ),
        index=candidates.index,
    )
    return positive_count, tiers


def optimize_on_development(
    features: pd.DataFrame,
    *,
    candidate_count: int,
    seed: int,
    minimum_trades: int = 2,
    minimum_folds_with_trades: int = 2,
    maximum_drawdown_percent: float = -40.0,
    require_buy_hold_outperformance: bool = False,
    allow_no_eligible: bool = False,
    folds: tuple[tuple[str, str, str], ...] = DEFAULT_FOLDS,
    **backtest_kwargs: float,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    benchmark_by_period: dict[str, IntegratedResult] = {}
    for name, start, end in (("Development", None, None), *folds):
        period = features
        if start:
            period = period[period["Date"] >= start]
        if end:
            period = period[period["Date"] <= end]
        benchmark_signals = period.copy()
        benchmark_signals["CompositeScore"] = 1.0
        benchmark_signals["BuySignal"] = True
        benchmark_signals["SellSignal"] = False
        benchmark_by_period[name] = run_integrated_backtest(
            benchmark_signals,
            buy_and_hold_params(),
            **backtest_kwargs,
        )
    rows: list[dict[str, float | int]] = []
    params_by_id: dict[int, IntegratedParams] = {}
    for candidate_id in range(1, candidate_count + 1):
        params = sample_params(rng)
        signals = generate_integrated_signals(features, params)
        result = run_integrated_backtest(
            signals, params, **backtest_kwargs
        )
        params_by_id[candidate_id] = params
        row: dict[str, float | int] = {
            "CandidateId": candidate_id,
            **params.as_dict(),
            "DevelopmentROI(%)": result.summary.roi_percent,
            "DevelopmentMDD(%)": result.summary.max_drawdown_percent,
            "DevelopmentTrades": result.summary.completed_trades,
        }
        development_benchmark = benchmark_by_period["Development"].summary
        development_log_return = float(
            np.log1p(max(result.summary.roi_percent, -99.999999) / 100.0)
        )
        development_benchmark_log_return = float(
            np.log1p(
                max(
                    development_benchmark.roi_percent,
                    -99.999999,
                )
                / 100.0
            )
        )
        row["DevelopmentLogReturn"] = development_log_return
        row["DevelopmentAlphaLogReturn"] = (
            development_log_return - development_benchmark_log_return
        )
        row["DevelopmentExcessROI(%)"] = (
            result.summary.roi_percent - development_benchmark.roi_percent
        )
        fold_excess: list[float] = []
        fold_rois: list[float] = []
        fold_mdd_improvement: list[float] = []
        fold_mdds: list[float] = []
        fold_alpha_log_returns: list[float] = []
        folds_with_trades = 0
        for name, start, end in folds:
            period_signals = signals[
                (signals["Date"] >= start) & (signals["Date"] <= end)
            ].reset_index(drop=True)
            fold_result = run_integrated_backtest(
                period_signals, params, **backtest_kwargs
            )
            benchmark_summary = benchmark_by_period[name].summary
            excess = (
                fold_result.summary.roi_percent
                - benchmark_summary.roi_percent
            )
            mdd_improvement = (
                fold_result.summary.max_drawdown_percent
                - benchmark_summary.max_drawdown_percent
            )
            row[f"{name}ExcessROI(%)"] = excess
            row[f"{name}ROI(%)"] = fold_result.summary.roi_percent
            row[f"{name}MDDImprovement(%)"] = mdd_improvement
            row[f"{name}MDD(%)"] = fold_result.summary.max_drawdown_percent
            row[f"{name}Trades"] = fold_result.summary.completed_trades
            fold_excess.append(excess)
            fold_rois.append(fold_result.summary.roi_percent)
            fold_mdd_improvement.append(mdd_improvement)
            fold_mdds.append(fold_result.summary.max_drawdown_percent)
            fold_alpha_log_returns.append(
                float(
                    np.log1p(
                        max(
                            fold_result.summary.roi_percent,
                            -99.999999,
                        )
                        / 100.0
                    )
                    - np.log1p(
                        max(
                            benchmark_summary.roi_percent,
                            -99.999999,
                        )
                        / 100.0
                    )
                )
            )
            folds_with_trades += int(fold_result.summary.completed_trades > 0)
        row["WorstFoldExcessROI(%)"] = min(fold_excess)
        row["WorstFoldROI(%)"] = min(fold_rois)
        fold_log_returns = np.log1p(
            np.maximum(np.asarray(fold_rois), -99.999999) / 100.0
        )
        row["WorstFoldLogReturn"] = float(np.min(fold_log_returns))
        row["MeanFoldLogReturn"] = float(np.mean(fold_log_returns))
        row["WorstFoldAlphaLogReturn"] = min(fold_alpha_log_returns)
        row["MeanFoldAlphaLogReturn"] = float(
            np.mean(fold_alpha_log_returns)
        )
        row["MeanFoldExcessROI(%)"] = float(np.mean(fold_excess))
        row["MeanFoldMDDImprovement(%)"] = float(
            np.mean(fold_mdd_improvement)
        )
        row["MeanFoldMDD(%)"] = float(np.mean(fold_mdds))
        row["WorstFoldMDD(%)"] = min(fold_mdds)
        row["FoldsWithTrades"] = folds_with_trades
        rows.append(row)
    candidates = pd.DataFrame(rows)
    candidates["WithinDrawdownLimit"] = (
        (candidates["DevelopmentMDD(%)"] >= maximum_drawdown_percent)
        & (candidates["WorstFoldMDD(%)"] >= maximum_drawdown_percent)
    )
    candidates["Eligible"] = (
        (candidates["DevelopmentTrades"] >= minimum_trades)
        & (candidates["FoldsWithTrades"] >= minimum_folds_with_trades)
    )
    if require_buy_hold_outperformance:
        candidates["Eligible"] &= (
            candidates["DevelopmentExcessROI(%)"] > 0
        )
        (
            candidates["AlphaPositiveFoldCount"],
            candidates["AlphaRobustnessTier"],
        ) = _alpha_robustness_tiers(
            candidates,
            folds,
        )
        candidates["Score"] = (
            0.45 * candidates["DevelopmentAlphaLogReturn"]
            + 0.30 * candidates["WorstFoldAlphaLogReturn"]
            + 0.20 * candidates["MeanFoldAlphaLogReturn"]
            + 0.05 * candidates["MeanFoldMDDImprovement(%)"] / 100.0
        )
    else:
        candidates["AlphaPositiveFoldCount"] = 0
        candidates["AlphaRobustnessTier"] = 0
        candidates["Score"] = (
            0.55 * candidates["DevelopmentLogReturn"]
            + 0.25 * candidates["MeanFoldLogReturn"]
            + 0.15 * candidates["WorstFoldLogReturn"]
            + 0.05 * candidates["MeanFoldMDD(%)"] / 100.0
        )
    eligible = candidates[candidates["Eligible"]].copy()
    if eligible.empty:
        if not allow_no_eligible:
            raise RuntimeError(
                "No candidate completed all eligibility conditions."
            )
        ranked = candidates.sort_values(
            [
                "AlphaRobustnessTier",
                "Score",
                "WorstFoldROI(%)",
                "DevelopmentROI(%)",
            ],
            ascending=False,
        ).reset_index(drop=True)
        winner_id = int(ranked.iloc[0]["CandidateId"])
        return OptimizationResult(params_by_id[winner_id], ranked)
    eligible = eligible.sort_values(
        [
            "AlphaRobustnessTier",
            "Score",
            "WorstFoldROI(%)",
            "DevelopmentROI(%)",
        ],
        ascending=False,
    ).reset_index(drop=True)
    winner_id = int(eligible.iloc[0]["CandidateId"])
    rejected = candidates[~candidates["Eligible"]].sort_values(
        [
            "AlphaRobustnessTier",
            "Score",
            "WorstFoldROI(%)",
            "DevelopmentROI(%)",
        ],
        ascending=False,
    )
    ranked = pd.concat([eligible, rejected], ignore_index=True)
    return OptimizationResult(params_by_id[winner_id], ranked)
