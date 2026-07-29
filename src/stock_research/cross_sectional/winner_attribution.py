from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths

from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .features import add_cross_sectional_factors, build_panel
from .portfolio import PortfolioResult, run_portfolio_backtest
from .signals import (
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)


@dataclass(frozen=True)
class WinnerDependenceResult:
    contributions: pd.DataFrame
    excluded_winners: pd.DataFrame
    scenario_summary: pd.DataFrame
    scenario_equity: pd.DataFrame
    baseline_daily_attribution: pd.DataFrame
    manifest: pd.DataFrame


@dataclass(frozen=True)
class WinnerDependenceArtifacts:
    output_dir: Path
    contributions_csv: Path
    excluded_winners_csv: Path
    scenario_summary_csv: Path
    scenario_equity_csv: Path
    daily_attribution_csv: Path
    manifest_csv: Path


def analyze_winner_dependence(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
) -> WinnerDependenceResult:
    """Measure V6-B ticker PnL and rerun after ex-post winner exclusions.

    Winner exclusions are a diagnostic stress test, not an investable
    out-of-sample rule. The frozen strategy parameters, signal functions,
    weekly schedule, next-open execution, and transaction-cost model remain
    unchanged.
    """

    latest_end = str(pd.Timestamp(panel["Date"].max()).date())
    live_start = min(
        start for start, _ in settings.validation_periods.values()
    )
    continuous_label = "2025-2026"
    contribution_periods = {
        label: (start, min(end, latest_end))
        for label, (start, end) in settings.validation_periods.items()
        if start <= latest_end
    }
    contribution_periods[continuous_label] = (live_start, latest_end)
    attributed_period_results: dict[
        str,
        tuple[PortfolioResult, PortfolioResult],
    ] = {}
    contribution_frames: list[pd.DataFrame] = []
    for label, (start, end) in contribution_periods.items():
        period_results = _run_period(
            panel,
            params,
            settings,
            start=start,
            end=end,
            record_attribution=True,
        )
        attributed_period_results[label] = period_results
        contribution_frames.append(
            summarize_ticker_contributions(
                period_results[0],
                {label: (start, end)},
            )
        )
    contributions = pd.concat(contribution_frames, ignore_index=True)
    if contributions.empty:
        raise ValueError("No ticker contributions were produced")
    winner_rows = (
        contributions.sort_values(
            ["Period", "NetPnL", "Ticker"],
            ascending=[True, False, True],
        )
        .groupby("Period", as_index=False, group_keys=False)
        .head(2)
        .copy()
    )
    winner_rows["ExclusionOrder"] = (
        winner_rows.groupby("Period").cumcount() + 1
    )
    excluded_winners = winner_rows.loc[
        :,
        [
            "Period",
            "ExclusionOrder",
            "Ticker",
            "NetPnL",
            "ContributionToPeriodReturnPct",
            "ShareOfPortfolioPnLPct",
        ],
    ].sort_values(
        ["Period", "ExclusionOrder"],
    ).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    evaluation_periods = contribution_periods
    for label, (start, end) in evaluation_periods.items():
        winner_tickers = excluded_winners.loc[
            excluded_winners["Period"].eq(label),
            "Ticker",
        ].tolist()
        scenarios: list[tuple[str, list[str]]] = [("BASELINE", [])]
        if winner_tickers:
            scenarios.append(("LEAVE_TOP_1_OUT", winner_tickers[:1]))
        if len(winner_tickers) >= 2:
            scenarios.append(("LEAVE_TOP_2_OUT", winner_tickers[:2]))
        for scenario, excluded in scenarios:
            if scenario == "BASELINE":
                strategy, benchmark = attributed_period_results[label]
            else:
                scenario_panel = _panel_without_tickers(
                    panel,
                    excluded,
                    settings,
                )
                strategy, benchmark = _run_period(
                    scenario_panel,
                    params,
                    settings,
                    start=start,
                    end=end,
                )
            summary_rows.append(
                _scenario_summary_row(
                    scenario,
                    excluded,
                    label,
                    strategy,
                    benchmark,
                )
            )
            if label == continuous_label:
                equity = strategy.daily.copy()
                equity.insert(0, "Scenario", scenario)
                equity.insert(1, "ExcludedTickers", ",".join(excluded))
                equity_frames.append(equity)

    scenario_summary = _add_baseline_deltas(pd.DataFrame(summary_rows))
    scenario_equity = pd.concat(equity_frames, ignore_index=True)
    attribution = attributed_period_results[continuous_label][0].attribution
    if attribution is None:
        raise RuntimeError("Baseline attribution was not recorded")
    manifest = pd.DataFrame(
        [
            {
                "GeneratedAt": datetime.now(UTC).isoformat(),
                "ValidationIsFresh": settings.validation_is_fresh,
                "ModelStatus": "POST_HOC_DIAGNOSTIC_PASS",
                "UniverseCount": int(panel["Ticker"].nunique()),
                "LatestPriceDate": latest_end,
                "WinnerSelectionPeriod": (
                    "each validation period plus 2025-current continuous"
                ),
                "WinnerSelectionMethod": (
                    "ex-post rank by arithmetic net dollar PnL"
                ),
                "AttributionMethod": (
                    "overnight PnL + intraday PnL - allocated transaction cost"
                ),
                "SignalFunctions": (
                    "score_panel -> generate_rebalance_targets"
                ),
                "ExecutionFunction": "run_portfolio_backtest",
                "SignalTiming": (
                    "last trading session of W-FRI week close"
                ),
                "ExecutionTiming": "next available session open",
                "TransactionCostBps": settings.transaction_cost_bps,
                "SurvivorshipCaveat": (
                    "current 200-stock snapshot applied historically"
                ),
                "WinnerSelectionCaveat": (
                    "leave-winner-out exclusions use full-period outcomes and "
                    "are diagnostic, not tradable"
                ),
            }
        ]
    )
    return WinnerDependenceResult(
        contributions=contributions,
        excluded_winners=excluded_winners,
        scenario_summary=scenario_summary,
        scenario_equity=scenario_equity,
        baseline_daily_attribution=attribution,
        manifest=manifest,
    )


def summarize_ticker_contributions(
    result: PortfolioResult,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Aggregate exactly reconciling arithmetic ticker PnL by period."""

    if result.attribution is None:
        raise ValueError("PortfolioResult has no ticker attribution")
    daily = result.daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    attribution = result.attribution.copy()
    attribution["Date"] = pd.to_datetime(attribution["Date"])
    rows: list[pd.DataFrame] = []
    for label, (start, end) in periods.items():
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        prior = daily.loc[daily["Date"].lt(start_date), "Equity"]
        start_value = (
            float(prior.iloc[-1])
            if not prior.empty
            else result.summary.initial_capital
        )
        ending = daily.loc[daily["Date"].le(end_date), "Equity"]
        if ending.empty:
            continue
        end_value = float(ending.iloc[-1])
        portfolio_pnl = end_value - start_value
        period = attribution.loc[
            attribution["Date"].between(start_date, end_date)
        ]
        if period.empty:
            continue
        grouped = (
            period.groupby("Ticker", as_index=False)
            .agg(
                OvernightPnL=("OvernightPnL", "sum"),
                IntradayPnL=("IntradayPnL", "sum"),
                GrossPricePnL=("GrossPricePnL", "sum"),
                TransactionCost=("TransactionCost", "sum"),
                NetPnL=("NetPnL", "sum"),
            )
            .sort_values(["NetPnL", "Ticker"], ascending=[False, True])
            .reset_index(drop=True)
        )
        grouped.insert(0, "Period", label)
        grouped["PortfolioStartValue"] = start_value
        grouped["PortfolioEndValue"] = end_value
        grouped["PortfolioNetPnL"] = portfolio_pnl
        grouped["ContributionToPeriodReturnPct"] = (
            grouped["NetPnL"] / start_value * 100
        )
        grouped["ShareOfPortfolioPnLPct"] = np.where(
            abs(portfolio_pnl) > 1e-12,
            grouped["NetPnL"] / portfolio_pnl * 100,
            np.nan,
        )
        grouped["ContributionRank"] = np.arange(1, len(grouped) + 1)
        reconciliation_error = portfolio_pnl - float(grouped["NetPnL"].sum())
        tolerance = 1e-8 * max(abs(end_value), 1.0)
        if abs(reconciliation_error) > tolerance:
            raise RuntimeError(
                f"Period {label} attribution does not reconcile: "
                f"{reconciliation_error}"
            )
        grouped["ReconciliationError"] = reconciliation_error
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def generate_winner_dependence_analysis(
    paths: ProjectPaths,
    settings: ResearchSettings,
    params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    output_dir: str | Path | None = None,
) -> WinnerDependenceArtifacts:
    members, _ = discover_universe(paths, ticker_config_path)
    if not members:
        raise ValueError("No stocks have both price and financial data")
    panel, _ = build_panel(members, settings)
    result = analyze_winner_dependence(panel, params, settings)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "winner_contribution"
            / f"{timestamp}_v6_b_auto200"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    contributions_csv = destination / "ticker_contributions.csv"
    excluded_winners_csv = destination / "excluded_winners.csv"
    scenario_summary_csv = destination / "leave_winner_out_summary.csv"
    scenario_equity_csv = destination / "leave_winner_out_equity.csv"
    daily_attribution_csv = destination / "baseline_daily_attribution.csv"
    manifest_csv = destination / "analysis_manifest.csv"
    atomic_to_csv(result.contributions, contributions_csv, index=False)
    atomic_to_csv(
        result.excluded_winners,
        excluded_winners_csv,
        index=False,
    )
    atomic_to_csv(
        result.scenario_summary,
        scenario_summary_csv,
        index=False,
    )
    atomic_to_csv(
        result.scenario_equity,
        scenario_equity_csv,
        index=False,
    )
    atomic_to_csv(
        result.baseline_daily_attribution,
        daily_attribution_csv,
        index=False,
    )
    atomic_to_csv(result.manifest, manifest_csv, index=False)
    return WinnerDependenceArtifacts(
        output_dir=destination,
        contributions_csv=contributions_csv,
        excluded_winners_csv=excluded_winners_csv,
        scenario_summary_csv=scenario_summary_csv,
        scenario_equity_csv=scenario_equity_csv,
        daily_attribution_csv=daily_attribution_csv,
        manifest_csv=manifest_csv,
    )


def _panel_without_tickers(
    panel: pd.DataFrame,
    excluded: list[str],
    settings: ResearchSettings,
) -> pd.DataFrame:
    if not excluded:
        return panel
    reduced = panel.loc[~panel["Ticker"].isin(excluded)].copy()
    return (
        add_cross_sectional_factors(reduced, settings)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )


def _run_period(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    *,
    start: str,
    end: str,
    record_attribution: bool = False,
) -> tuple[PortfolioResult, PortfolioResult]:
    signal_days = signal_day_panel(
        panel,
        start,
        end,
        settings.rebalance_weekday,
    )
    targets = generate_rebalance_targets(
        score_panel(signal_days, params),
        params,
    )
    strategy = run_portfolio_backtest(
        panel,
        targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        record_attribution=record_attribution,
    )
    benchmark = run_portfolio_backtest(
        panel,
        generate_equal_weight_targets(signal_days),
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    return strategy, benchmark


def _scenario_summary_row(
    scenario: str,
    excluded: list[str],
    period: str,
    strategy: PortfolioResult,
    benchmark: PortfolioResult,
) -> dict[str, object]:
    return {
        "Scenario": scenario,
        "ExcludedTickers": ",".join(excluded),
        "ExcludedCount": len(excluded),
        "Period": period,
        "StartDate": strategy.summary.start_date,
        "EndDate": strategy.summary.end_date,
        "FinalValue": strategy.summary.final_value,
        "StrategyROI": strategy.summary.roi_percent,
        "BenchmarkROI": benchmark.summary.roi_percent,
        "ExcessROI": (
            strategy.summary.roi_percent - benchmark.summary.roi_percent
        ),
        "MaxDrawdown": strategy.summary.max_drawdown_percent,
        "Sharpe": strategy.summary.sharpe_ratio,
        "AnnualizedTurnover": strategy.summary.annualized_turnover,
        "TickerTrades": strategy.summary.ticker_trades,
        "Rebalances": strategy.summary.rebalance_count,
    }


def _add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary.loc[
        summary["Scenario"].eq("BASELINE"),
        ["Period", "StrategyROI", "ExcessROI"],
    ].rename(
        columns={
            "StrategyROI": "BaselineStrategyROI",
            "ExcessROI": "BaselineExcessROI",
        }
    )
    result = summary.merge(baseline, on="Period", how="left")
    result["StrategyROILossVsBaseline"] = (
        result["BaselineStrategyROI"] - result["StrategyROI"]
    )
    result["ExcessROILossVsBaseline"] = (
        result["BaselineExcessROI"] - result["ExcessROI"]
    )
    result["ExcessReturnRetentionPct"] = np.where(
        result["BaselineExcessROI"].abs().gt(1e-12),
        result["ExcessROI"] / result["BaselineExcessROI"] * 100,
        np.nan,
    )
    result["PositiveROI"] = result["StrategyROI"].gt(0)
    result["BeatSameUniverseBenchmark"] = result["ExcessROI"].gt(0)
    return result
