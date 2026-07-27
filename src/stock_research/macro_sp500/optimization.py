from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import MacroSp500Params, MacroSp500Settings
from .features import add_macro_features
from .portfolio import (
    PerformanceSummary,
    PortfolioResult,
    _performance_summary,
    run_buy_and_hold,
    run_target_weight_backtest,
)


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos_daily: pd.DataFrame
    oos_trades: pd.DataFrame
    benchmark_daily: pd.DataFrame
    oos_summary: PerformanceSummary
    benchmark_summary: PerformanceSummary
    latest_params: MacroSp500Params
    candidate_count: int


def _candidate_score(
    strategy: PerformanceSummary,
    benchmark: PerformanceSummary,
    settings: MacroSp500Settings,
) -> tuple[float, float]:
    minimum_cagr = (
        benchmark.cagr_percent * settings.minimum_cagr_fraction_of_buy_hold
    )
    if strategy.cagr_percent < minimum_cagr:
        return (float("-inf"), strategy.cagr_percent)
    return (strategy.calmar_ratio, strategy.cagr_percent)


def _scaled_fold(
    result: PortfolioResult,
    *,
    starting_value: float,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    scale = starting_value / result.summary.initial_capital
    daily = result.daily.copy()
    for column in ("Cash", "TotalValue"):
        daily[column] = daily[column] * scale
    daily["TotalInjected"] = result.summary.initial_capital
    daily["Fold"] = label
    trades = result.trades.copy()
    if not trades.empty:
        for column in ("Notional", "Fee"):
            trades[column] = trades[column] * scale
        trades["Fold"] = label
    return daily, trades, float(daily["TotalValue"].iloc[-1])


def optimize_walk_forward(
    data: pd.DataFrame,
    candidates: list[MacroSp500Params],
    settings: MacroSp500Settings,
) -> WalkForwardResult:
    if not candidates:
        raise ValueError("At least one parameter candidate is required.")
    features_by_lookback = {
        years: add_macro_features(
            data,
            vix_lookback_years=years,
            warning_lookback_days=settings.warning_lookback_days,
            drawdown_lookback_days=settings.drawdown_lookback_days,
            minimum_vix_observations=settings.minimum_vix_observations,
        )
        for years in sorted({candidate.vix_lookback_years for candidate in candidates})
    }
    first_data_year = int(pd.to_datetime(data["Date"]).dt.year.min())
    last_data_year = int(pd.to_datetime(data["Date"]).dt.year.max())
    first_test_year = max(
        settings.first_test_year,
        first_data_year + settings.training_years,
    )
    fold_rows: list[dict[str, object]] = []
    oos_daily_parts: list[pd.DataFrame] = []
    oos_trade_parts: list[pd.DataFrame] = []
    benchmark_daily_parts: list[pd.DataFrame] = []
    benchmark_trade_parts: list[pd.DataFrame] = []
    strategy_capital = settings.initial_capital
    benchmark_capital = settings.initial_capital
    latest_params: MacroSp500Params | None = None

    for test_year in range(first_test_year, last_data_year + 1):
        test_start = pd.Timestamp(test_year, 1, 1)
        test_end = pd.Timestamp(test_year, 12, 31)
        if not (
            (pd.to_datetime(data["Date"]) >= test_start)
            & (pd.to_datetime(data["Date"]) <= test_end)
        ).any():
            continue
        train_start = pd.Timestamp(test_year - settings.training_years, 1, 1)
        train_end = test_start - pd.Timedelta(days=1)
        benchmark_features = features_by_lookback[min(features_by_lookback)]
        train_benchmark = run_buy_and_hold(
            benchmark_features,
            settings,
            start=train_start,
            end=train_end,
        )

        best_params: MacroSp500Params | None = None
        best_result: PortfolioResult | None = None
        best_score = (float("-inf"), float("-inf"))
        fallback_score = (float("-inf"), float("-inf"))
        fallback_params: MacroSp500Params | None = None
        fallback_result: PortfolioResult | None = None
        for candidate in candidates:
            train_result = run_target_weight_backtest(
                features_by_lookback[candidate.vix_lookback_years],
                candidate,
                settings,
                start=train_start,
                end=train_end,
            )
            score = _candidate_score(train_result.summary, train_benchmark.summary, settings)
            if score[0] != float("-inf") and score > best_score:
                best_score = score
                best_params = candidate
                best_result = train_result
            raw_score = (
                train_result.summary.calmar_ratio,
                train_result.summary.cagr_percent,
            )
            if raw_score > fallback_score:
                fallback_score = raw_score
                fallback_params = candidate
                fallback_result = train_result

        if best_params is None or best_result is None:
            best_params = fallback_params
            best_result = fallback_result
        if best_params is None or best_result is None:
            raise RuntimeError(f"No valid parameter candidate for test year {test_year}.")

        test_features = features_by_lookback[best_params.vix_lookback_years]
        test_result = run_target_weight_backtest(
            test_features,
            best_params,
            settings,
            start=test_start,
            end=test_end,
        )
        test_benchmark = run_buy_and_hold(
            benchmark_features,
            settings,
            start=test_start,
            end=test_end,
        )
        label = str(test_year)
        scaled_daily, scaled_trades, strategy_capital = _scaled_fold(
            test_result,
            starting_value=strategy_capital,
            label=label,
        )
        scaled_benchmark, scaled_benchmark_trades, benchmark_capital = _scaled_fold(
            test_benchmark,
            starting_value=benchmark_capital,
            label=label,
        )
        oos_daily_parts.append(scaled_daily)
        oos_trade_parts.append(scaled_trades)
        benchmark_daily_parts.append(scaled_benchmark)
        benchmark_trade_parts.append(scaled_benchmark_trades)
        fold_rows.append(
            {
                "TestYear": test_year,
                "TrainStart": train_start,
                "TrainEnd": train_end,
                "TestStart": test_start,
                "TestEnd": min(
                    test_end,
                    pd.to_datetime(test_result.daily["Date"]).max(),
                ),
                **best_params.as_dict(),
                "TrainROI(%)": best_result.summary.roi_percent,
                "TrainCAGR(%)": best_result.summary.cagr_percent,
                "TrainMDD(%)": best_result.summary.max_drawdown_percent,
                "TrainCalmar": best_result.summary.calmar_ratio,
                "TestROI(%)": test_result.summary.roi_percent,
                "TestCAGR(%)": test_result.summary.cagr_percent,
                "TestMDD(%)": test_result.summary.max_drawdown_percent,
                "TestCalmar": test_result.summary.calmar_ratio,
                "BenchmarkTestROI(%)": test_benchmark.summary.roi_percent,
                "BenchmarkTestMDD(%)": test_benchmark.summary.max_drawdown_percent,
            }
        )
        latest_params = best_params

    if not oos_daily_parts or latest_params is None:
        raise RuntimeError("Walk-forward optimization produced no test folds.")
    oos_daily = pd.concat(oos_daily_parts, ignore_index=True)
    benchmark_daily = pd.concat(benchmark_daily_parts, ignore_index=True)
    oos_trades = (
        pd.concat(
            [frame for frame in oos_trade_parts if not frame.empty],
            ignore_index=True,
        )
        if any(not frame.empty for frame in oos_trade_parts)
        else pd.DataFrame()
    )
    benchmark_trades = (
        pd.concat(
            [frame for frame in benchmark_trade_parts if not frame.empty],
            ignore_index=True,
        )
        if any(not frame.empty for frame in benchmark_trade_parts)
        else pd.DataFrame()
    )
    oos_summary = _performance_summary(
        oos_daily,
        oos_trades,
        initial_capital=settings.initial_capital,
    )
    benchmark_summary = _performance_summary(
        benchmark_daily,
        benchmark_trades,
        initial_capital=settings.initial_capital,
    )
    return WalkForwardResult(
        folds=pd.DataFrame(fold_rows),
        oos_daily=oos_daily,
        oos_trades=oos_trades,
        benchmark_daily=benchmark_daily,
        oos_summary=oos_summary,
        benchmark_summary=benchmark_summary,
        latest_params=latest_params,
        candidate_count=len(candidates),
    )
