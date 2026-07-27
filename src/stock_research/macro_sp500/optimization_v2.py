from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config_v2 import MacroSp500V2Params, MacroSp500V2Settings
from .features_v2 import add_v2_features
from .portfolio import PerformanceSummary, PortfolioResult
from .portfolio_v2 import (
    run_static_weight_v2,
    run_v2_backtest,
    run_v2_portfolio_from_signals,
    run_v2_with_memory,
)
from .strategy_v2 import CrisisMemory


@dataclass
class WalkForwardV2Result:
    folds: pd.DataFrame
    oos_strategy: PortfolioResult
    buy_hold: PortfolioResult
    static_70_30: PortfolioResult
    latest_params: MacroSp500V2Params
    candidate_count: int


def _is_eligible(
    strategy: PerformanceSummary,
    benchmark: PerformanceSummary,
    settings: MacroSp500V2Settings,
) -> bool:
    if benchmark.cagr_percent >= 0:
        minimum_cagr = (
            benchmark.cagr_percent
            * settings.minimum_cagr_fraction_of_buy_hold
        )
    else:
        minimum_cagr = benchmark.cagr_percent
    maximum_mdd = (
        abs(benchmark.max_drawdown_percent)
        * settings.maximum_mdd_fraction_of_buy_hold
    )
    return (
        strategy.cagr_percent >= minimum_cagr
        and abs(strategy.max_drawdown_percent) <= maximum_mdd
    )


def _constraint_violations(
    strategy: PerformanceSummary,
    benchmark: PerformanceSummary,
    settings: MacroSp500V2Settings,
) -> tuple[float, float]:
    minimum_cagr = (
        benchmark.cagr_percent
        * settings.minimum_cagr_fraction_of_buy_hold
        if benchmark.cagr_percent >= 0
        else benchmark.cagr_percent
    )
    maximum_mdd = max(
        1e-9,
        abs(benchmark.max_drawdown_percent)
        * settings.maximum_mdd_fraction_of_buy_hold,
    )
    cagr_scale = max(1.0, abs(minimum_cagr))
    cagr_violation = max(
        0.0,
        (minimum_cagr - strategy.cagr_percent) / cagr_scale,
    )
    mdd_violation = max(
        0.0,
        (abs(strategy.max_drawdown_percent) - maximum_mdd) / maximum_mdd,
    )
    return cagr_violation, mdd_violation


def optimize_v2_walk_forward(
    data: pd.DataFrame,
    candidates: list[MacroSp500V2Params],
    settings: MacroSp500V2Settings,
) -> WalkForwardV2Result:
    if not candidates:
        raise ValueError("At least one V2 candidate is required.")
    features = add_v2_features(data, settings)
    years = pd.to_datetime(features["Date"]).dt.year
    first_data_year = int(years.min())
    last_data_year = int(years.max())
    first_test_year = max(
        settings.first_test_year,
        first_data_year + settings.training_years,
    )
    folds: list[dict[str, object]] = []
    selected: list[tuple[int, MacroSp500V2Params]] = []

    for test_year in range(first_test_year, last_data_year + 1):
        test_start = pd.Timestamp(test_year, 1, 1)
        test_end = pd.Timestamp(test_year, 12, 31)
        test_mask = (features["Date"] >= test_start) & (features["Date"] <= test_end)
        if not test_mask.any():
            continue
        train_start = pd.Timestamp(test_year - settings.training_years, 1, 1)
        train_end = test_start - pd.Timedelta(days=1)
        train = features[
            (features["Date"] >= train_start) & (features["Date"] <= train_end)
        ].reset_index(drop=True)
        benchmark = run_static_weight_v2(train, settings, weight=1.0)

        best_params: MacroSp500V2Params | None = None
        best_result: PortfolioResult | None = None
        best_score = (float("-inf"), float("-inf"))
        fallback_params: MacroSp500V2Params | None = None
        fallback_result: PortfolioResult | None = None
        fallback_score = (
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )
        eligible_count = 0

        for candidate in candidates:
            result = run_v2_backtest(train, candidate, settings)
            if _is_eligible(result.summary, benchmark.summary, settings):
                eligible_count += 1
                score = (
                    result.summary.calmar_ratio,
                    result.summary.cagr_percent,
                )
                if score > best_score:
                    best_score = score
                    best_params = candidate
                    best_result = result
            violations = _constraint_violations(
                result.summary,
                benchmark.summary,
                settings,
            )
            score = (
                -sum(violations),
                result.summary.calmar_ratio,
                result.summary.cagr_percent,
            )
            if score > fallback_score:
                fallback_score = score
                fallback_params = candidate
                fallback_result = result

        used_fallback = best_params is None
        if used_fallback:
            best_params = fallback_params
            best_result = fallback_result
        if best_params is None or best_result is None:
            raise RuntimeError(f"No V2 candidate selected for {test_year}.")

        test = features[test_mask].reset_index(drop=True)
        test_result = run_v2_backtest(test, best_params, settings)
        test_benchmark = run_static_weight_v2(test, settings, weight=1.0)
        folds.append(
            {
                "TestYear": test_year,
                "TrainStart": train_start,
                "TrainEnd": train_end,
                "TestStart": test["Date"].min(),
                "TestEnd": test["Date"].max(),
                "EligibleCandidateCount": eligible_count,
                "FallbackUsed": used_fallback,
                "TrainCAGRViolation": _constraint_violations(
                    best_result.summary,
                    benchmark.summary,
                    settings,
                )[0],
                "TrainMDDViolation": _constraint_violations(
                    best_result.summary,
                    benchmark.summary,
                    settings,
                )[1],
                **best_params.as_dict(),
                "TrainROI(%)": best_result.summary.roi_percent,
                "TrainCAGR(%)": best_result.summary.cagr_percent,
                "TrainMDD(%)": best_result.summary.max_drawdown_percent,
                "TrainCalmar": best_result.summary.calmar_ratio,
                "TestROI(%)": test_result.summary.roi_percent,
                "TestMDD(%)": test_result.summary.max_drawdown_percent,
                "BenchmarkTestROI(%)": test_benchmark.summary.roi_percent,
                "BenchmarkTestMDD(%)": (
                    test_benchmark.summary.max_drawdown_percent
                ),
            }
        )
        selected.append((test_year, best_params))

    if not selected:
        raise RuntimeError("V2 walk-forward optimization produced no folds.")

    memory: CrisisMemory | None = None
    oos_features: list[pd.DataFrame] = []
    oos_signals: list[pd.DataFrame] = []
    for test_year, params in selected:
        year_features = features[years == test_year].reset_index(drop=True)
        signals, memory = run_v2_with_memory(
            year_features,
            params,
            settings,
            memory,
        )
        oos_features.append(year_features)
        oos_signals.append(signals)

    continuous_features = pd.concat(oos_features, ignore_index=True)
    continuous_signals = pd.concat(oos_signals, ignore_index=True)
    strategy = run_v2_portfolio_from_signals(
        continuous_features,
        continuous_signals,
        settings,
    )
    buy_hold = run_static_weight_v2(
        continuous_features,
        settings,
        weight=1.0,
    )
    static_70_30 = run_static_weight_v2(
        continuous_features,
        settings,
        weight=settings.static_benchmark_weight,
    )
    return WalkForwardV2Result(
        folds=pd.DataFrame(folds),
        oos_strategy=strategy,
        buy_hold=buy_hold,
        static_70_30=static_70_30,
        latest_params=selected[-1][1],
        candidate_count=len(candidates),
    )
