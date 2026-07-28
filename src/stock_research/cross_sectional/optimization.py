from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import ResearchSettings, StrategyParams
from .portfolio import (
    PortfolioResult,
    PreparedMarket,
    prepare_market,
    run_portfolio_backtest,
)
from .signals import (
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)


@dataclass(frozen=True)
class OptimizationResult:
    params: StrategyParams
    candidates: pd.DataFrame
    benchmark: PortfolioResult
    selection_mode: str


@dataclass(frozen=True)
class SelectionPeriod:
    label: str
    start: str
    end: str
    signal_days: pd.DataFrame
    market: PreparedMarket
    benchmark: PortfolioResult


_WORKER_CONTEXT: tuple[
    pd.DataFrame,
    PreparedMarket,
    PortfolioResult,
    ResearchSettings,
    SelectionPeriod | None,
] | None = None


def _initialize_candidate_worker(
    train_signal_days: pd.DataFrame,
    prepared_market: PreparedMarket,
    benchmark: PortfolioResult,
    settings: ResearchSettings,
    selection_period: SelectionPeriod | None,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (
        train_signal_days,
        prepared_market,
        benchmark,
        settings,
        selection_period,
    )


def _evaluate_candidate_worker(
    item: tuple[int, StrategyParams],
) -> tuple[int, StrategyParams, dict[str, object]]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Candidate worker was not initialized.")
    index, params = item
    return _evaluate_candidate(
        index,
        params,
        *_WORKER_CONTEXT,
    )


def optimize_strategy(
    panel: pd.DataFrame,
    settings: ResearchSettings,
) -> OptimizationResult:
    """Optimize on training data, then optionally select on one later period."""

    train_signal_days = signal_day_panel(
        panel,
        settings.train_start,
        settings.train_end,
        settings.rebalance_weekday,
    ).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    prepared_market = prepare_market(
        panel,
        start=settings.train_start,
        end=settings.train_end,
    )
    benchmark_targets = generate_equal_weight_targets(train_signal_days)
    benchmark = run_portfolio_backtest(
        panel,
        benchmark_targets,
        start=settings.train_start,
        end=settings.train_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        prepared_market=prepared_market,
    )
    selection_period = _prepare_selection_period(panel, settings)
    rows: list[dict[str, object]] = []
    params_by_index: dict[int, StrategyParams] = {}
    indexed_params = list(
        enumerate(_candidate_parameters(settings), start=1)
    )
    worker_count = min(
        8,
        settings.candidate_count,
        max((os.cpu_count() or 1) - 1, 1),
    )
    print(f"Optimization workers: {worker_count}", flush=True)
    if settings.candidate_count >= 100 and worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_candidate_worker,
            initargs=(
                train_signal_days,
                prepared_market,
                benchmark,
                settings,
                selection_period,
            ),
        ) as executor:
            evaluated = executor.map(
                _evaluate_candidate_worker,
                indexed_params,
                chunksize=1,
            )
            _collect_candidate_rows(
                evaluated,
                rows,
                params_by_index,
                settings.candidate_count,
            )
    else:
        evaluated = (
            _evaluate_candidate(
                index,
                params,
                train_signal_days,
                prepared_market,
                benchmark,
                settings,
                selection_period,
            )
            for index, params in indexed_params
        )
        _collect_candidate_rows(
            evaluated,
            rows,
            params_by_index,
            settings.candidate_count,
        )

    candidates = pd.DataFrame(rows)
    if selection_period is not None:
        candidates = candidates.sort_values(
            [
                "PassTrainConstraints",
                "PassSelectionConstraints",
                "SelectionObjective",
                "Objective",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        passing = candidates.loc[
            candidates["PassTrainConstraints"]
            & candidates["PassSelectionConstraints"]
        ]
        if passing.empty:
            chosen = candidates.iloc[0]
            selection_mode = (
                f"BEST_AVAILABLE_NO_{selection_period.label}_PASS"
            )
        else:
            chosen = passing.iloc[0]
            selection_mode = (
                f"TRAIN_AND_{selection_period.label}_SELECTION_PASS"
            )
    else:
        candidates = candidates.sort_values(
            ["PassTrainConstraints", "Objective"],
            ascending=[False, False],
        ).reset_index(drop=True)
        passing = candidates.loc[candidates["PassTrainConstraints"]]
        if passing.empty:
            chosen = candidates.iloc[0]
            selection_mode = "BEST_AVAILABLE_NO_STRICT_TRAIN_PASS"
        else:
            chosen = passing.iloc[0]
            selection_mode = "STRICT_TRAIN_PASS"
    return OptimizationResult(
        params=params_by_index[int(chosen["Candidate"])],
        candidates=candidates,
        benchmark=benchmark,
        selection_mode=selection_mode,
    )


def _collect_candidate_rows(
    evaluated: Iterable[
        tuple[int, StrategyParams, dict[str, object]]
    ],
    rows: list[dict[str, object]],
    params_by_index: dict[int, StrategyParams],
    candidate_count: int,
) -> None:
    for completed, (index, params, row) in enumerate(evaluated, start=1):
        rows.append(row)
        params_by_index[index] = params
        if completed % 50 == 0 or completed == candidate_count:
            print(
                f"Optimization progress: {completed}/{candidate_count}",
                flush=True,
            )


def _prepare_selection_period(
    panel: pd.DataFrame,
    settings: ResearchSettings,
) -> SelectionPeriod | None:
    label = settings.selection_validation_label
    if label is None:
        return None
    start, end = settings.validation_periods[label]
    signal_days = signal_day_panel(
        panel,
        start,
        end,
        settings.rebalance_weekday,
    ).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    market = prepare_market(panel, start=start, end=end)
    benchmark_targets = generate_equal_weight_targets(signal_days)
    benchmark = run_portfolio_backtest(
        panel,
        benchmark_targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        prepared_market=market,
    )
    return SelectionPeriod(
        label=label,
        start=start,
        end=end,
        signal_days=signal_days,
        market=market,
        benchmark=benchmark,
    )


def _evaluate_candidate(
    index: int,
    params: StrategyParams,
    train_signal_days: pd.DataFrame,
    prepared_market: PreparedMarket,
    benchmark: PortfolioResult,
    settings: ResearchSettings,
    selection_period: SelectionPeriod | None,
) -> tuple[int, StrategyParams, dict[str, object]]:
    scored = score_panel(
        train_signal_days,
        params,
        compact=True,
        presorted=True,
    )
    targets = generate_rebalance_targets(scored, params, compact=True)
    result = run_portfolio_backtest(
        pd.DataFrame(),
        targets,
        start=settings.train_start,
        end=settings.train_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        prepared_market=prepared_market,
    )
    fold_excess: list[float] = []
    fold_strategy: list[float] = []
    row: dict[str, object] = {
        "Candidate": index,
        **params.as_dict(),
        "FinancialWeight": params.financial_weight,
        "TrainROI": result.summary.roi_percent,
        "TrainCAGR": result.summary.cagr_percent,
        "TrainMaxDrawdown": result.summary.max_drawdown_percent,
        "TrainSharpe": result.summary.sharpe_ratio,
        "TrainAnnualizedTurnover": result.summary.annualized_turnover,
        "BenchmarkROI": benchmark.summary.roi_percent,
        "BenchmarkCAGR": benchmark.summary.cagr_percent,
        "ExcessROI": (
            result.summary.roi_percent - benchmark.summary.roi_percent
        ),
        "ExcessCAGR": (
            result.summary.cagr_percent - benchmark.summary.cagr_percent
        ),
    }
    for fold_index, (start, end) in enumerate(
        settings.training_folds,
        start=1,
    ):
        strategy_fold = _period_metrics(result.daily, start, end)
        benchmark_fold = _period_metrics(benchmark.daily, start, end)
        excess_cagr = strategy_fold["CAGR"] - benchmark_fold["CAGR"]
        fold_strategy.append(strategy_fold["CAGR"])
        fold_excess.append(excess_cagr)
        row[f"Fold{fold_index}CAGR"] = strategy_fold["CAGR"]
        row[f"Fold{fold_index}ExcessCAGR"] = excess_cagr
    positive_excess_folds = sum(value > 0 for value in fold_excess)
    row["PositiveExcessFolds"] = positive_excess_folds
    row["MedianFoldExcessCAGR"] = float(np.median(fold_excess))
    row["WorstFoldExcessCAGR"] = float(min(fold_excess))
    row["PositiveStrategyFolds"] = sum(value > 0 for value in fold_strategy)
    row["PassTrainConstraints"] = bool(
        result.summary.roi_percent > 0
        and result.summary.roi_percent > benchmark.summary.roi_percent
        and positive_excess_folds >= 2
        and result.summary.max_drawdown_percent >= -65
        and params.financial_weight >= settings.minimum_financial_weight
    )
    row["Objective"] = _objective(row)
    if selection_period is not None:
        _add_selection_metrics(
            row,
            params,
            settings,
            selection_period,
        )
    return index, params, row


def _add_selection_metrics(
    row: dict[str, object],
    params: StrategyParams,
    settings: ResearchSettings,
    selection_period: SelectionPeriod,
) -> None:
    scored = score_panel(
        selection_period.signal_days,
        params,
        compact=True,
        presorted=True,
    )
    targets = generate_rebalance_targets(scored, params, compact=True)
    result = run_portfolio_backtest(
        pd.DataFrame(),
        targets,
        start=selection_period.start,
        end=selection_period.end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        prepared_market=selection_period.market,
    )
    summary = result.summary
    benchmark = selection_period.benchmark.summary
    row.update(
        {
            "SelectionLabel": selection_period.label,
            "SelectionROI": summary.roi_percent,
            "SelectionCAGR": summary.cagr_percent,
            "SelectionMaxDrawdown": summary.max_drawdown_percent,
            "SelectionSharpe": summary.sharpe_ratio,
            "SelectionAnnualizedTurnover": summary.annualized_turnover,
            "SelectionBenchmarkROI": benchmark.roi_percent,
            "SelectionBenchmarkCAGR": benchmark.cagr_percent,
            "SelectionExcessROI": (
                summary.roi_percent - benchmark.roi_percent
            ),
            "SelectionExcessCAGR": (
                summary.cagr_percent - benchmark.cagr_percent
            ),
        }
    )
    row["PassSelectionConstraints"] = bool(
        row["PassTrainConstraints"]
        and summary.roi_percent > 0
        and summary.roi_percent > benchmark.roi_percent
        and summary.max_drawdown_percent >= -65
    )
    row["SelectionObjective"] = _selection_objective(row)


def _candidate_parameters(
    settings: ResearchSettings,
) -> list[StrategyParams]:
    if settings.minimum_financial_weight > 0:
        anchors = [
            StrategyParams(0.15, 0.20, 0.20, 0.20, 0.25, 3, 6, -0.10, -0.15),
            StrategyParams(0.25, 0.20, 0.18, 0.17, 0.20, 3, 6, 0.00, -0.10),
            StrategyParams(0.15, 0.10, 0.30, 0.25, 0.20, 5, 9, -0.10, -0.15),
            StrategyParams(0.30, 0.25, 0.16, 0.15, 0.14, 5, 9, 0.00, 0.00),
        ]
    else:
        anchors = [
            StrategyParams(0.20, 0.20, 0.20, 0.20, 0.20, 3, 6, -0.10, -0.15),
            StrategyParams(0.35, 0.30, 0.10, 0.10, 0.15, 3, 6, 0.00, -0.10),
            StrategyParams(0.20, 0.15, 0.25, 0.25, 0.15, 5, 9, -0.10, -0.15),
            StrategyParams(0.45, 0.30, 0.05, 0.05, 0.15, 5, 9, 0.00, 0.00),
        ]
    anchors = [_apply_exit_policy(params, settings) for params in anchors]
    rng = np.random.default_rng(settings.seed)
    candidates = anchors[: settings.candidate_count]
    choices = (
        (3, (2, 4, 6)),
        (5, (2, 4, 7)),
        (8, (3, 5, 8)),
    )
    while len(candidates) < settings.candidate_count:
        if settings.minimum_financial_weight > 0:
            financial_weight = float(
                rng.uniform(settings.minimum_financial_weight, 0.60)
            )
            growth_share = float(rng.beta(2.0, 2.0))
            non_financial = rng.dirichlet(np.array([1.5, 1.5, 1.2]))
            remaining = 1 - financial_weight
            weights = np.array(
                [
                    remaining * non_financial[0],
                    remaining * non_financial[1],
                    financial_weight * growth_share,
                    financial_weight * (1 - growth_share),
                    remaining * non_financial[2],
                ]
            )
        else:
            weights = rng.dirichlet(
                np.array([1.5, 1.5, 1.0, 1.0, 1.2])
            )
        top_k, buffers = choices[int(rng.integers(0, len(choices)))]
        buffer = buffers[int(rng.integers(0, len(buffers)))]
        candidates.append(
            _apply_exit_policy(
                StrategyParams(
                    momentum_weight=float(weights[0]),
                    trend_weight=float(weights[1]),
                    growth_weight=float(weights[2]),
                    quality_weight=float(weights[3]),
                    risk_control_weight=float(weights[4]),
                    top_k=top_k,
                    exit_rank=top_k + buffer,
                    trend_floor=float(
                        rng.choice(np.array([-0.20, -0.10, 0.00, 0.05]))
                    ),
                    momentum_floor=float(
                        rng.choice(np.array([-0.35, -0.15, 0.00, 0.10]))
                    ),
                ),
                settings,
            )
        )
    return candidates


def _apply_exit_policy(
    params: StrategyParams,
    settings: ResearchSettings,
) -> StrategyParams:
    return replace(
        params,
        loss_aware_exit_enabled=settings.loss_aware_exit_enabled,
        minimum_exit_gain=settings.minimum_exit_gain,
        conviction_exit_rank=max(
            settings.conviction_exit_rank,
            params.exit_rank,
        ),
        conviction_trend_floor=settings.conviction_trend_floor,
        conviction_momentum_floor=settings.conviction_momentum_floor,
        hard_stop_return=settings.hard_stop_return,
        minimum_hold_rebalances=settings.minimum_hold_rebalances,
    )


def _period_metrics(
    daily: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, float]:
    period = daily.loc[daily["Date"].between(start, end)]
    if len(period) < 2:
        return {"ROI": 0.0, "CAGR": 0.0}
    first = float(period["Equity"].iloc[0])
    last = float(period["Equity"].iloc[-1])
    days = max(
        (pd.Timestamp(period["Date"].iloc[-1]) - pd.Timestamp(period["Date"].iloc[0])).days,
        1,
    )
    years = max(days / 365.25, 1 / 252)
    return {
        "ROI": (last / first - 1) * 100,
        "CAGR": ((last / first) ** (1 / years) - 1) * 100,
    }


def _objective(row: dict[str, object]) -> float:
    excess_cagr = float(row["ExcessCAGR"])
    train_cagr = float(row["TrainCAGR"])
    median_fold = float(row["MedianFoldExcessCAGR"])
    worst_fold = float(row["WorstFoldExcessCAGR"])
    drawdown = abs(float(row["TrainMaxDrawdown"]))
    sharpe = float(row["TrainSharpe"])
    turnover = float(row["TrainAnnualizedTurnover"])
    return (
        0.40 * excess_cagr
        + 0.20 * train_cagr
        + 0.20 * median_fold
        + 0.10 * worst_fold
        + 1.5 * sharpe
        - 0.08 * drawdown
        - 0.10 * turnover
    )


def _selection_objective(row: dict[str, object]) -> float:
    """Rank training-pass candidates on the configured selection period.

    The later period rewards absolute and benchmark-relative CAGR while
    retaining a smaller contribution from the multi-fold training objective.
    The final holdout period is never referenced here.
    """

    excess_cagr = float(row["SelectionExcessCAGR"])
    selection_cagr = float(row["SelectionCAGR"])
    drawdown = abs(float(row["SelectionMaxDrawdown"]))
    sharpe = float(row["SelectionSharpe"])
    turnover = float(row["SelectionAnnualizedTurnover"])
    training_objective = float(row["Objective"])
    return (
        0.45 * excess_cagr
        + 0.25 * selection_cagr
        + 1.5 * sharpe
        - 0.08 * drawdown
        - 0.05 * turnover
        + 0.10 * training_objective
    )
