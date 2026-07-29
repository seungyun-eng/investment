from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths

from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .features import (
    TTM_GROWTH_WEIGHTS,
    TTM_VALUE_WEIGHTS,
    build_panel,
)
from .optimization import OptimizationResult, optimize_strategy
from .portfolio import PortfolioResult, run_portfolio_backtest
from .signals import (
    build_daily_recommendations,
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)


def run_research(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    ticker_config_path: str | Path,
) -> dict[str, Any]:
    members, discovery_audit = discover_universe(paths, ticker_config_path)
    if not members:
        raise ValueError("No stocks have both price and financial data")
    panel, data_audit = build_panel(members, settings)
    optimization = optimize_strategy(panel, settings)
    selected_params = optimization.params
    scored_daily = score_panel(panel, selected_params)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_folder = f"{timestamp}_{_safe_label(settings.research_label)}"
    output_dir = (
        paths.results
        / "Cross_Sectional"
        / "rank_signals"
        / run_folder
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, object]] = []
    validation_outputs: dict[str, dict[str, object]] = {}
    for label, (start, end) in settings.validation_periods.items():
        result = _evaluate_period(
            panel,
            scored_daily,
            selected_params,
            settings,
            start,
            end,
        )
        strategy = result["strategy"]
        benchmark = result["benchmark"]
        targets = result["targets"]
        validation_rows.append(
            {
                "Period": label,
                **_comparison_row(strategy, benchmark),
            }
        )
        atomic_to_csv(
            strategy.daily,
            output_dir / f"validation_{label}_equity.csv",
            index=False,
        )
        atomic_to_csv(
            strategy.executions,
            output_dir / f"validation_{label}_executions.csv",
            index=False,
        )
        atomic_to_csv(
            _selection_statistics(panel, targets, start, end),
            output_dir / f"validation_{label}_ticker_stats.csv",
            index=False,
        )
        validation_outputs[label] = result

    latest_end = str(pd.Timestamp(panel["Date"].max()).date())
    live_start = min(
        start for start, _ in settings.validation_periods.values()
    )
    live_signal_days = signal_day_panel(
        panel,
        live_start,
        latest_end,
        settings.rebalance_weekday,
    )
    live_scored_signal_days = score_panel(live_signal_days, selected_params)
    live_targets = generate_rebalance_targets(
        live_scored_signal_days,
        selected_params,
    )
    live_daily = scored_daily.loc[
        scored_daily["Date"].between(live_start, latest_end)
    ]
    daily_recommendations = build_daily_recommendations(
        live_daily,
        live_targets,
        selected_params,
    )
    latest_date = pd.Timestamp(daily_recommendations["Date"].max())
    latest_signals = daily_recommendations.loc[
        daily_recommendations["Date"].eq(latest_date)
    ].copy()
    validation_summary = pd.DataFrame(validation_rows)
    selection_label = settings.selection_validation_label
    final_holdout_labels = [
        label
        for label in settings.validation_periods
        if label != selection_label
    ]
    final_holdout_summary = validation_summary.loc[
        validation_summary["Period"].isin(final_holdout_labels)
    ]
    selection_passed = bool(
        optimization.candidates.iloc[0].get(
            "PassSelectionConstraints",
            True,
        )
    )
    model_passed = bool(
        selection_passed
        and not final_holdout_summary.empty
        and (
            final_holdout_summary["PositiveROI"]
            & final_holdout_summary["BeatEqualWeight"]
        ).all()
    )
    if settings.validation_is_fresh:
        model_status = (
            "VALIDATED"
            if model_passed
            else "RESEARCH_ONLY_VALIDATION_FAILED"
        )
    else:
        model_status = (
            "POST_HOC_DIAGNOSTIC_PASS"
            if model_passed
            else "POST_HOC_DIAGNOSTIC_FAILED"
        )
    latest_signals = _signal_output_columns(latest_signals)
    latest_signals.insert(0, "ModelStatus", model_status)
    signal_history = _signal_output_columns(daily_recommendations)
    signal_history.insert(0, "ModelStatus", model_status)

    combined_audit = discovery_audit.merge(
        data_audit,
        on=["Ticker", "Company"],
        how="left",
    )
    combined_audit["TrainingEligible"] = combined_audit[
        "TrainingEligibleSessions"
    ].fillna(0).gt(0)
    combined_audit["LatestFinancialAgeDays"] = (
        latest_date - pd.to_datetime(combined_audit["FinancialEnd"])
    ).dt.days
    combined_audit["LatestFinancialStale"] = (
        combined_audit["LatestFinancialAgeDays"]
        > settings.max_financial_age_days
    )
    feature_coverage = _financial_feature_coverage(
        panel,
        settings,
        latest_date,
    )

    atomic_to_csv(
        optimization.candidates,
        output_dir / "optimization_candidates.csv",
        index=False,
    )
    atomic_to_csv(
        validation_summary,
        output_dir / "validation_summary.csv",
        index=False,
    )
    atomic_to_csv(
        combined_audit,
        output_dir / "universe_data_audit.csv",
        index=False,
    )
    atomic_to_csv(
        feature_coverage,
        output_dir / "financial_feature_coverage.csv",
        index=False,
    )
    atomic_to_csv(
        latest_signals,
        output_dir / "latest_daily_signals.csv",
        index=False,
    )
    atomic_to_csv(
        signal_history,
        output_dir / "daily_signal_history.csv",
        index=False,
    )
    manifest = _manifest(
        settings,
        selected_params,
        optimization,
        validation_summary,
        combined_audit,
        latest_date,
        output_dir,
        model_status,
    )
    _atomic_json(manifest, output_dir / "selected_strategy.json")
    return {
        "output_dir": output_dir,
        "panel": panel,
        "params": selected_params,
        "validation_summary": validation_summary,
        "latest_signals": latest_signals,
        "data_audit": combined_audit,
        "feature_coverage": feature_coverage,
        "optimization": optimization,
        "validation_outputs": validation_outputs,
        "manifest": manifest,
    }


def generate_latest_signals(
    paths: ProjectPaths,
    settings: ResearchSettings,
    params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    model_start: str = "2025-01-01",
    model_status: str = "UNREVIEWED",
) -> dict[str, Any]:
    members, discovery_audit = discover_universe(paths, ticker_config_path)
    panel, data_audit = build_panel(members, settings)
    scored = score_panel(panel, params)
    latest_end = str(pd.Timestamp(panel["Date"].max()).date())
    signal_days = signal_day_panel(
        panel,
        model_start,
        latest_end,
        settings.rebalance_weekday,
    )
    targets = generate_rebalance_targets(score_panel(signal_days, params), params)
    recommendations = build_daily_recommendations(
        scored.loc[scored["Date"].between(model_start, latest_end)],
        targets,
        params,
    )
    latest_date = pd.Timestamp(recommendations["Date"].max())
    latest = _signal_output_columns(
        recommendations.loc[recommendations["Date"].eq(latest_date)]
    )
    latest.insert(0, "ModelStatus", model_status)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        paths.results
        / "Cross_Sectional"
        / "daily_signals"
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(latest, output_dir / "latest_daily_signals.csv", index=False)
    audit = discovery_audit.merge(
        data_audit,
        on=["Ticker", "Company"],
        how="left",
    )
    atomic_to_csv(audit, output_dir / "universe_data_audit.csv", index=False)
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "latest_price_date": latest_date,
            "model_start": model_start,
            "model_status": model_status,
            "selected_params": params.as_dict(),
            "output_dir": str(output_dir),
        },
        output_dir / "daily_signal_manifest.json",
    )
    return {
        "output_dir": output_dir,
        "latest_signals": latest,
        "latest_date": latest_date,
    }


def load_selected_strategy(path: str | Path) -> StrategyParams:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return StrategyParams.from_dict(raw["selected_params"])


def load_model_status(path: str | Path) -> str:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(raw.get("model_status", "UNREVIEWED"))


def _evaluate_period(
    panel: pd.DataFrame,
    scored_daily: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    start: str,
    end: str,
) -> dict[str, object]:
    signal_days = signal_day_panel(
        panel,
        start,
        end,
        settings.rebalance_weekday,
    )
    scored_signal_days = score_panel(signal_days, params)
    targets = generate_rebalance_targets(scored_signal_days, params)
    benchmark_targets = generate_equal_weight_targets(signal_days)
    strategy = run_portfolio_backtest(
        panel,
        targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    benchmark = run_portfolio_backtest(
        panel,
        benchmark_targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    return {
        "strategy": strategy,
        "benchmark": benchmark,
        "targets": targets,
        "scored_daily": scored_daily.loc[
            scored_daily["Date"].between(start, end)
        ],
    }


def _comparison_row(
    strategy: PortfolioResult,
    benchmark: PortfolioResult,
) -> dict[str, object]:
    return {
        "StartDate": strategy.summary.start_date,
        "EndDate": strategy.summary.end_date,
        "StrategyROI": strategy.summary.roi_percent,
        "EqualWeightUniverseROI": benchmark.summary.roi_percent,
        "ExcessROI": (
            strategy.summary.roi_percent - benchmark.summary.roi_percent
        ),
        "StrategyCAGR": strategy.summary.cagr_percent,
        "BenchmarkCAGR": benchmark.summary.cagr_percent,
        "MaxDrawdown": strategy.summary.max_drawdown_percent,
        "BenchmarkMaxDrawdown": benchmark.summary.max_drawdown_percent,
        "Sharpe": strategy.summary.sharpe_ratio,
        "BenchmarkSharpe": benchmark.summary.sharpe_ratio,
        "AnnualizedTurnover": strategy.summary.annualized_turnover,
        "TickerTrades": strategy.summary.ticker_trades,
        "Rebalances": strategy.summary.rebalance_count,
        "PositiveROI": strategy.summary.roi_percent > 0,
        "BeatEqualWeight": (
            strategy.summary.roi_percent > benchmark.summary.roi_percent
        ),
    }


def _selection_statistics(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    period = panel.loc[panel["Date"].between(start, end)]
    prices = period.groupby("Ticker").agg(
        StartDate=("Date", "min"),
        EndDate=("Date", "max"),
        StartClose=("Close", "first"),
        EndClose=("Close", "last"),
    )
    prices["BuyHoldROI"] = (
        prices["EndClose"] / prices["StartClose"] - 1
    ) * 100
    selected = targets.groupby("Ticker").agg(
        SelectedWeeks=("ModelSelected", "sum"),
        BuySignals=("TradeAction", lambda values: int((values == "BUY").sum())),
        SellSignals=("TradeAction", lambda values: int((values == "SELL").sum())),
        BestRank=("Rank", "min"),
        MeanRank=("Rank", "mean"),
    )
    result = prices.join(selected, how="left").fillna(
        {
            "SelectedWeeks": 0,
            "BuySignals": 0,
            "SellSignals": 0,
        }
    )
    return result.reset_index().sort_values(
        ["SelectedWeeks", "Ticker"],
        ascending=[False, True],
    )


def _signal_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Date",
        "Ticker",
        "Company",
        "DailySignal",
        "TradeAction",
        "ModelSelected",
        "TargetWeight",
        "Rank",
        "AlphaScore",
        "Qualified",
        "Close",
        "Return21",
        "Return63",
        "Return126",
        "Drawdown126",
        "Trend200",
        "MomentumFactor",
        "TrendFactor",
        "GrowthFactor",
        "QualityFactor",
        "RiskControlFactor",
        "EpsTtm",
        "EpsTtmGrowthYoY",
        "EpsTtmGrowthAcceleration",
        "DcfPrice",
        "DcfPriceGrowthYoY",
        "DcfUpside",
        "PeTtm",
        "EbitdaTtm",
        "EbitdaTtmGrowthYoY",
        "EbitdaTtmGrowthAcceleration",
        "EvEbitdaTtm",
        "GrowthAdjustedPe",
        "GrowthAdjustedEvEbitda",
        "FinancialPeriodEnd",
        "FinancialAvailableDate",
        "FinancialAgeDays",
        "FinancialStale",
        "CrossSectionSize",
        "EntryReferencePrice",
        "SignalReferenceReturn",
        "HoldingRebalances",
        "PeakReferencePrice",
        "PeakReferenceReturn",
        "TrailingDrawdown",
        "BestReplacementAlphaScore",
        "ReplacementScoreAdvantage",
        "ProfitExitStreak",
        "EntryBlocked",
        "EntryBlockReason",
        "ExitReason",
        "IsRebalanceSignal",
    ]
    result = frame.loc[:, [column for column in columns if column in frame]].copy()
    return result.sort_values(
        ["ModelSelected", "Rank", "Ticker"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _financial_feature_coverage(
    panel: pd.DataFrame,
    settings: ResearchSettings,
    latest_date: pd.Timestamp,
) -> pd.DataFrame:
    metrics = (
        "EpsTtmGrowthYoY",
        "EpsTtmGrowthAcceleration",
        "DcfPriceGrowthYoY",
        "DcfUpside",
        "EbitdaTtmGrowthYoY",
        "EbitdaTtmGrowthAcceleration",
        "GrowthAdjustedPe",
        "GrowthAdjustedEvEbitda",
        "GrowthFactor",
        "QualityFactor",
    )
    periods = {
        "Train": (settings.train_start, settings.train_end),
        **settings.validation_periods,
        "Latest": (str(latest_date.date()), str(latest_date.date())),
    }
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        eligible = panel.loc[
            panel["Date"].between(start, end) & panel["Eligible"]
        ]
        for metric in metrics:
            non_missing = eligible[metric].notna()
            eligible_rows = len(eligible)
            rows.append(
                {
                    "Period": period,
                    "Metric": metric,
                    "EligibleRows": eligible_rows,
                    "NonMissingRows": int(non_missing.sum()),
                    "CoveragePct": (
                        float(non_missing.mean() * 100)
                        if eligible_rows
                        else np.nan
                    ),
                    "EligibleTickers": int(eligible["Ticker"].nunique()),
                    "NonMissingTickers": int(
                        eligible.loc[non_missing, "Ticker"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _manifest(
    settings: ResearchSettings,
    params: StrategyParams,
    optimization: OptimizationResult,
    validation_summary: pd.DataFrame,
    audit: pd.DataFrame,
    latest_date: pd.Timestamp,
    output_dir: Path,
    model_status: str,
) -> dict[str, object]:
    best = optimization.candidates.iloc[0]
    selection_label = settings.selection_validation_label
    final_holdout_labels = [
        label
        for label in settings.validation_periods
        if label != selection_label
    ]
    selection_metrics = None
    if selection_label is not None:
        selection_metrics = {
            key: _json_value(best[key])
            for key in (
                "SelectionLabel",
                "SelectionROI",
                "SelectionBenchmarkROI",
                "SelectionExcessROI",
                "SelectionCAGR",
                "SelectionMaxDrawdown",
                "SelectionSharpe",
                "SelectionObjective",
                "PassSelectionConstraints",
            )
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "methodology": {
            "training_period": [settings.train_start, settings.train_end],
            "training_only": [settings.train_start, settings.train_end],
            "validation_periods": settings.validation_periods,
            "selection_validation_label": selection_label,
            "validation_used_for_selection": selection_label,
            "final_holdout_labels": final_holdout_labels,
            "signal_timing": "close signal; next available session open execution",
            "rebalance": "last trading session of each W-FRI week",
            "roi_formula": "(final_value / total_injected - 1) * 100",
            "universe": "all non-crypto configured names with both price and quarterly financial files",
            "financial_release_lag_days": settings.financial_release_lag_days,
            "max_financial_age_days": settings.max_financial_age_days,
            "minimum_financial_weight": settings.minimum_financial_weight,
            "financial_feature_mode": settings.financial_feature_mode,
            "financial_feature_weights": {
                "growth": TTM_GROWTH_WEIGHTS,
                "value_quality": TTM_VALUE_WEIGHTS,
            }
            if settings.financial_feature_mode == "ttm_value_momentum"
            else None,
            "dcf_assumptions": {
                "wacc": settings.dcf_wacc,
                "cost_of_equity": settings.dcf_cost_of_equity,
                "short_growth": settings.dcf_short_growth,
                "projection_years": settings.dcf_projection_years,
                "terminal_growth": settings.dcf_terminal_growth,
            },
            "validation_is_fresh": settings.validation_is_fresh,
        },
        "settings": asdict(settings),
        "selected_params": params.as_dict(),
        "model_status": model_status,
        "selection_mode": optimization.selection_mode,
        "selected_candidate": int(best["Candidate"]),
        "training_metrics": {
            key: _json_value(best[key])
            for key in (
                "TrainROI",
                "BenchmarkROI",
                "ExcessROI",
                "TrainCAGR",
                "TrainMaxDrawdown",
                "TrainSharpe",
                "PositiveExcessFolds",
                "Objective",
            )
        },
        "selection_metrics": selection_metrics,
        "validation": validation_summary.to_dict(orient="records"),
        "universe_counts": {
            "included": int((audit["Status"] == "INCLUDED").sum()),
            "training_eligible": int(audit["TrainingEligible"].fillna(False).sum()),
            "latest_financial_stale": int(
                audit["LatestFinancialStale"].fillna(False).sum()
            ),
        },
        "latest_price_date": latest_date,
        "output_dir": str(output_dir),
    }


def _atomic_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_value),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def _json_value(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _safe_label(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    )
    return cleaned or "research"
