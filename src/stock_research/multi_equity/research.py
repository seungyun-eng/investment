from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths
from stock_research.tsla_integrated.config import (
    IntegratedParams,
    IntegratedSettings,
)
from stock_research.tsla_integrated.data import (
    load_equity_financials,
    load_equity_prices,
)
from stock_research.tsla_integrated.downside import (
    add_strict_oos_downside_probability,
)
from stock_research.tsla_integrated.features import build_integrated_features
from stock_research.tsla_integrated.optimization import (
    DEFAULT_FOLDS,
    buy_and_hold_params,
    consensus_execution_params,
    optimize_on_development,
    params_from_candidate_row,
)
from stock_research.tsla_integrated.portfolio import (
    IntegratedResult,
    run_integrated_backtest,
)
from stock_research.tsla_integrated.strategy import (
    generate_consensus_signals,
)

from .config import EquitySpec


@dataclass(frozen=True)
class EquityResearchRun:
    summary: dict[str, object]
    manifest: dict[str, object]


@dataclass(frozen=True)
class AlphaConsensusSelection:
    member_count: int
    entry_consensus: float
    execution_params: IntegratedParams
    development: IntegratedResult
    development_benchmark: IntegratedResult
    folds: dict[str, IntegratedResult]
    fold_benchmarks: dict[str, IntegratedResult]
    trials: pd.DataFrame


def _ticker_seed(base_seed: int, ticker: str) -> int:
    return base_seed + sum(
        (index + 1) * ord(character)
        for index, character in enumerate(ticker)
    )


def _buy_and_hold(
    features: pd.DataFrame,
    *,
    backtest_kwargs: dict[str, float | bool],
) -> IntegratedResult:
    signals = features.copy()
    signals["CompositeScore"] = 1.0
    signals["BuySignal"] = True
    signals["SellSignal"] = False
    signals["ShortSignal"] = False
    signals["CoverSignal"] = True
    return run_integrated_backtest(
        signals,
        buy_and_hold_params(),
        **backtest_kwargs,
    )


def _select_alpha_consensus(
    development: pd.DataFrame,
    eligible_candidates: pd.DataFrame,
    *,
    maximum_members: int,
    minimum_entry_consensus: float,
    exit_consensus: float,
    backtest_kwargs: dict[str, float | bool],
) -> AlphaConsensusSelection:
    if eligible_candidates.empty:
        raise RuntimeError("No eligible candidates are available for consensus.")
    maximum_members = min(maximum_members, len(eligible_candidates))
    member_counts = sorted(
        {
            count
            for count in (1, 3, 5, 10, 25, maximum_members)
            if count <= maximum_members
        }
    )
    step_count = round((1.0 - minimum_entry_consensus) / 0.05)
    thresholds = [
        round(minimum_entry_consensus + 0.05 * index, 2)
        for index in range(step_count + 1)
    ]
    if thresholds[-1] < 1.0:
        thresholds.append(1.0)
    development_benchmark = _buy_and_hold(
        development,
        backtest_kwargs=backtest_kwargs,
    )
    fold_benchmarks = {
        name: _buy_and_hold(
            development[
                (development["Date"] >= start)
                & (development["Date"] <= end)
            ].reset_index(drop=True),
            backtest_kwargs=backtest_kwargs,
        )
        for name, start, end in DEFAULT_FOLDS
    }
    trials: list[dict[str, float | int | bool]] = []
    successful: list[
        tuple[
            float,
            int,
            float,
            IntegratedParams,
            IntegratedResult,
            dict[str, IntegratedResult],
        ]
    ] = []
    for member_count in member_counts:
        member_params = [
            params_from_candidate_row(row)
            for _, row in eligible_candidates.head(member_count).iterrows()
        ]
        execution_params = consensus_execution_params(member_params)
        candidate_thresholds = (
            [minimum_entry_consensus] if member_count == 1 else thresholds
        )
        for threshold in candidate_thresholds:
            development_signals = generate_consensus_signals(
                development,
                member_params,
                entry_consensus=threshold,
                exit_consensus=exit_consensus,
            )
            development_result = run_integrated_backtest(
                development_signals,
                execution_params,
                **backtest_kwargs,
            )
            fold_results = {}
            fold_alpha_log_returns = []
            for name, start, end in DEFAULT_FOLDS:
                fold_signals = development_signals[
                    (development_signals["Date"] >= start)
                    & (development_signals["Date"] <= end)
                ].reset_index(drop=True)
                fold_result = run_integrated_backtest(
                    fold_signals,
                    execution_params,
                    **backtest_kwargs,
                )
                fold_results[name] = fold_result
                fold_alpha_log_returns.append(
                    np.log1p(fold_result.summary.roi_percent / 100)
                    - np.log1p(
                        fold_benchmarks[name].summary.roi_percent / 100
                    )
                )
            development_roi = development_result.summary.roi_percent
            benchmark_roi = development_benchmark.summary.roi_percent
            development_alpha_log_return = (
                np.log1p(development_roi / 100)
                - np.log1p(benchmark_roi / 100)
            )
            score = float(
                0.70 * development_alpha_log_return
                + 0.20 * np.mean(fold_alpha_log_returns)
                + 0.10 * np.min(fold_alpha_log_returns)
            )
            eligible = bool(
                development_roi > max(0.0, benchmark_roi)
                and development_result.summary.completed_trades >= 1
            )
            trials.append(
                {
                    "MemberCount": member_count,
                    "EntryConsensus": threshold,
                    "DevelopmentROI(%)": development_roi,
                    "BuyHoldROI(%)": benchmark_roi,
                    "ExcessROI(%)": development_roi - benchmark_roi,
                    "DevelopmentMDD(%)": (
                        development_result.summary.max_drawdown_percent
                    ),
                    "DevelopmentTrades": (
                        development_result.summary.completed_trades
                    ),
                    "WorstFoldROI(%)": min(
                        result.summary.roi_percent
                        for result in fold_results.values()
                    ),
                    "Score": score,
                    "Eligible": eligible,
                }
            )
            if eligible:
                successful.append(
                    (
                        score,
                        member_count,
                        threshold,
                        execution_params,
                        development_result,
                        fold_results,
                    )
                )
    if not successful:
        raise RuntimeError(
            "No consensus configuration beat Buy & Hold with positive "
            "development ROI and at least one completed trade."
        )
    (
        _,
        selected_member_count,
        selected_threshold,
        selected_execution_params,
        selected_development,
        selected_folds,
    ) = max(successful, key=lambda item: item[0])
    return AlphaConsensusSelection(
        member_count=selected_member_count,
        entry_consensus=selected_threshold,
        execution_params=selected_execution_params,
        development=selected_development,
        development_benchmark=development_benchmark,
        folds=selected_folds,
        fold_benchmarks=fold_benchmarks,
        trials=pd.DataFrame(trials).sort_values(
            ["Eligible", "Score"],
            ascending=False,
        ),
    )


def run_equity_research(
    spec: EquitySpec,
    *,
    paths: ProjectPaths,
    macro: pd.DataFrame,
    settings: IntegratedSettings,
    candidate_count: int,
    timestamp: str,
) -> EquityResearchRun:
    price_path = paths.processed / spec.price_file
    financial_path = paths.financial_raw / f"{spec.ticker}_financials_Q.xlsx"
    prices = load_equity_prices(price_path)
    financials = load_equity_financials(financial_path)
    features = build_integrated_features(
        prices,
        financials,
        macro,
        financial_release_lag_days=settings.financial_release_lag_days,
    )
    features = add_strict_oos_downside_probability(features)
    development = features[
        (features["Date"] >= settings.development_start)
        & (features["Date"] <= settings.development_end)
    ].reset_index(drop=True)
    holdout = features[
        features["Date"] >= settings.holdout_start
    ].reset_index(drop=True)
    if development.empty:
        raise ValueError(
            f"{spec.ticker} has no prices in the development period."
        )
    if len(holdout) < 2:
        raise ValueError(
            f"{spec.ticker} has fewer than two holdout sessions."
        )

    backtest_kwargs: dict[str, float | bool] = {
        "initial_capital": settings.initial_capital,
        "transaction_cost_bps": settings.transaction_cost_bps,
        "slippage_bps": settings.slippage_bps,
        "annual_short_borrow_bps": settings.annual_short_borrow_bps,
        "initial_long": settings.initial_long,
    }
    optimized = optimize_on_development(
        development,
        candidate_count=candidate_count,
        seed=_ticker_seed(settings.seed, spec.ticker),
        minimum_trades=1,
        minimum_folds_with_trades=1,
        maximum_drawdown_percent=settings.maximum_drawdown_percent,
        require_buy_hold_outperformance=True,
        **backtest_kwargs,
    )
    eligible_candidates = optimized.candidates[
        optimized.candidates["Eligible"]
    ]
    selection = _select_alpha_consensus(
        development,
        eligible_candidates,
        maximum_members=settings.consensus_members,
        minimum_entry_consensus=settings.entry_consensus,
        exit_consensus=settings.exit_consensus,
        backtest_kwargs=backtest_kwargs,
    )
    selected_members = eligible_candidates.head(selection.member_count)
    member_params = [
        params_from_candidate_row(row)
        for _, row in selected_members.iterrows()
    ]
    holdout_signals = generate_consensus_signals(
        holdout,
        member_params,
        entry_consensus=selection.entry_consensus,
        exit_consensus=settings.exit_consensus,
    )
    holdout_result = run_integrated_backtest(
        holdout_signals,
        selection.execution_params,
        **backtest_kwargs,
    )
    holdout_benchmark = _buy_and_hold(
        holdout,
        backtest_kwargs=backtest_kwargs,
    )

    output = paths.results / "Multi_Equity" / "integrated_signal" / spec.ticker
    output_files = {
        "candidates": atomic_to_csv(
            optimized.candidates,
            output / f"development_candidates_{timestamp}.csv",
            index=False,
        ),
        "consensus_members": atomic_to_csv(
            selected_members,
            output / f"consensus_members_{timestamp}.csv",
            index=False,
        ),
        "consensus_trials": atomic_to_csv(
            selection.trials,
            output / f"consensus_trials_{timestamp}.csv",
            index=False,
        ),
        "development_daily": atomic_to_csv(
            selection.development.daily,
            output / f"development_daily_{timestamp}.csv",
            index=False,
        ),
        "development_trades": atomic_to_csv(
            selection.development.trades,
            output / f"development_trades_{timestamp}.csv",
            index=False,
        ),
        "holdout_daily": atomic_to_csv(
            holdout_result.daily,
            output / f"holdout_daily_{timestamp}.csv",
            index=False,
        ),
        "holdout_trades": atomic_to_csv(
            holdout_result.trades,
            output / f"holdout_trades_{timestamp}.csv",
            index=False,
        ),
    }
    last_financial_period = pd.to_datetime(
        financials["Date"], errors="coerce"
    ).max()
    summary = {
        "Ticker": spec.ticker,
        "Status": "COMPLETE",
        "LeadCandidateId": int(selected_members.iloc[0]["CandidateId"]),
        "CandidateCount": candidate_count,
        "ConsensusMembers": len(selected_members),
        "EntryConsensus": selection.entry_consensus,
        "DevelopmentStart": development["Date"].min(),
        "DevelopmentEnd": development["Date"].max(),
        "FinancialDataThrough": last_financial_period,
        "DevelopmentROI(%)": selection.development.summary.roi_percent,
        "DevelopmentBuyHoldROI(%)": (
            selection.development_benchmark.summary.roi_percent
        ),
        "DevelopmentExcessROI(%)": (
            selection.development.summary.roi_percent
            - selection.development_benchmark.summary.roi_percent
        ),
        "DevelopmentMDD(%)": (
            selection.development.summary.max_drawdown_percent
        ),
        "DevelopmentTrades": (
            selection.development.summary.completed_trades
        ),
        "Fold_2019_2021ROI(%)": (
            selection.folds["Fold_2019_2021"].summary.roi_percent
        ),
        "Fold_2022_2023ROI(%)": (
            selection.folds["Fold_2022_2023"].summary.roi_percent
        ),
        "Fold_2024_2025ROI(%)": (
            selection.folds["Fold_2024_2025"].summary.roi_percent
        ),
        "HoldoutStart": holdout["Date"].min(),
        "HoldoutEnd": holdout["Date"].max(),
        "HoldoutROI(%)": holdout_result.summary.roi_percent,
        "HoldoutBuyHoldROI(%)": holdout_benchmark.summary.roi_percent,
        "HoldoutExcessROI(%)": (
            holdout_result.summary.roi_percent
            - holdout_benchmark.summary.roi_percent
        ),
        "HoldoutMDD(%)": holdout_result.summary.max_drawdown_percent,
        "HoldoutTrades": holdout_result.summary.completed_trades,
        "HoldoutPositive": holdout_result.summary.roi_percent > 0,
        "HoldoutBeatsBuyHoldAndPositive": (
            holdout_result.summary.roi_percent
            > max(0.0, holdout_benchmark.summary.roi_percent)
        ),
        "HoldoutFinalState": holdout_result.daily.iloc[-1]["State"],
    }
    manifest = {
        "ticker": spec.ticker,
        "selection_objective": (
            "require 2019-2025 net ROI above both zero and same-period "
            "Buy & Hold, with at least one completed trade; rank qualifying "
            "configurations by full-period and subperiod alpha"
        ),
        "selected_on": (
            f"{settings.development_start} <= Date <= "
            f"{settings.development_end}"
        ),
        "evaluation_start": settings.holdout_start,
        "evaluation_status": (
            "diagnostic recheck; 2026 was observed by the earlier single-member run"
        ),
        "financial_release_lag_days": settings.financial_release_lag_days,
        "inputs": {
            "prices": str(price_path.resolve()),
            "financials": str(financial_path.resolve()),
        },
        "consensus": {
            "member_candidate_ids": (
                selected_members["CandidateId"].astype(int).tolist()
            ),
            "member_count": selection.member_count,
            "minimum_entry_consensus": settings.entry_consensus,
            "selected_entry_consensus": selection.entry_consensus,
            "exit_consensus": settings.exit_consensus,
            "execution_stop_loss": selection.execution_params.stop_loss,
            "execution_trailing_stop": (
                selection.execution_params.trailing_stop
            ),
            "execution_short_stop_loss": (
                selection.execution_params.short_stop_loss
            ),
            "execution_minimum_hold_sessions": (
                selection.execution_params.minimum_hold_sessions
            ),
        },
        "development": asdict(selection.development.summary),
        "development_buy_and_hold": asdict(
            selection.development_benchmark.summary
        ),
        "development_folds": {
            name: asdict(result.summary)
            for name, result in selection.folds.items()
        },
        "development_fold_buy_and_hold": {
            name: asdict(result.summary)
            for name, result in selection.fold_benchmarks.items()
        },
        "holdout": asdict(holdout_result.summary),
        "holdout_buy_and_hold": asdict(holdout_benchmark.summary),
        "outputs": {
            key: str(value)
            for key, value in output_files.items()
        },
    }
    return EquityResearchRun(summary=summary, manifest=manifest)
