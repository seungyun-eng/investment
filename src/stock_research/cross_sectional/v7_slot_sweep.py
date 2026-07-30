from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths

from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .pit_validation import apply_membership_to_panel
from .v7_pit_evaluation import (
    build_v7_source_panel,
    evaluate_period,
    load_ready_tickers,
    normalize_change_membership,
)
from .v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)
from .winner_attribution import summarize_ticker_contributions


@dataclass(frozen=True)
class V7SlotSweepArtifacts:
    output_dir: Path
    slot_summary_csv: Path
    balanced_ranking_csv: Path
    concentration_csv: Path
    equity_csv: Path
    data_audit_csv: Path
    manifest_json: Path


def slot_sweep_params(
    base: StrategyParams,
    top_k: int,
    *,
    exit_buffer: int = 4,
) -> StrategyParams:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if exit_buffer < 0:
        raise ValueError("exit_buffer must be non-negative")
    exit_rank = top_k + exit_buffer
    if exit_rank > base.conviction_exit_rank:
        raise ValueError(
            "top_k plus exit_buffer cannot exceed conviction_exit_rank"
        )
    return replace(
        base,
        top_k=top_k,
        exit_rank=exit_rank,
        profit_rotation_exit_rank=exit_rank,
    )


def spy_buy_and_hold(
    spy: pd.DataFrame,
    *,
    start: str,
    end: str,
    initial_capital: float,
    transaction_cost_bps: float,
) -> tuple[dict[str, float | pd.Timestamp], pd.DataFrame]:
    frame = (
        spy.loc[spy["Date"].between(start, end)]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .copy()
    )
    if frame.empty:
        raise ValueError(f"No SPY prices between {start} and {end}")
    first_open = float(frame.iloc[0]["Open"])
    cost = initial_capital * transaction_cost_bps / 10_000
    invested = initial_capital - cost
    shares = invested / first_open
    frame["Equity"] = shares * pd.to_numeric(
        frame["Close"], errors="raise"
    )
    start_date = pd.Timestamp(frame.iloc[0]["Date"])
    end_date = pd.Timestamp(frame.iloc[-1]["Date"])
    final_value = float(frame.iloc[-1]["Equity"])
    roi = (final_value / initial_capital - 1) * 100
    elapsed_years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    cagr = (
        (final_value / initial_capital) ** (1 / elapsed_years) - 1
    ) * 100
    path = pd.concat(
        [
            pd.DataFrame(
                {
                    "Date": [start_date - pd.Timedelta(nanoseconds=1)],
                    "Equity": [initial_capital],
                }
            ),
            frame[["Date", "Equity"]],
        ],
        ignore_index=True,
    )
    running_peak = path["Equity"].cummax()
    max_drawdown = float(
        (path["Equity"] / running_peak - 1).min() * 100
    )
    returns = path["Equity"].pct_change(fill_method=None).dropna()
    volatility = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / volatility * np.sqrt(252))
        if volatility > 0
        else 0.0
    )
    return (
        {
            "StartDate": start_date,
            "EndDate": end_date,
            "FinalValue": final_value,
            "ROI": roi,
            "CAGR": cagr,
            "MaxDrawdown": max_drawdown,
            "Sharpe": sharpe,
        },
        frame[["Date", "Equity"]].reset_index(drop=True),
    )


def balanced_slot_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby("TopK", as_index=False).agg(
        MedianCAGR=("CAGR", "median"),
        CAGRStd=("CAGR", "std"),
        MinimumSharpe=("Sharpe", "min"),
        MedianSharpe=("Sharpe", "median"),
        WorstMaxDrawdown=("MaxDrawdown", "min"),
        MinimumSPYExcessROI=("SPYExcessROI", "min"),
        PositiveSPYExcessPeriods=("SPYExcessROI", lambda value: int((value > 0).sum())),
    )
    grouped["ReturnRank"] = grouped["MedianCAGR"].rank(
        ascending=False, method="min"
    )
    grouped["StabilityRank"] = grouped["CAGRStd"].rank(
        ascending=True, method="min"
    )
    grouped["SharpeRank"] = grouped["MinimumSharpe"].rank(
        ascending=False, method="min"
    )
    grouped["DrawdownRank"] = grouped["WorstMaxDrawdown"].rank(
        ascending=False, method="min"
    )
    grouped["SPYExcessRank"] = grouped["MinimumSPYExcessROI"].rank(
        ascending=False, method="min"
    )
    grouped["BalancedRankScore"] = grouped[
        [
            "ReturnRank",
            "StabilityRank",
            "SharpeRank",
            "DrawdownRank",
            "SPYExcessRank",
        ]
    ].mean(axis=1)
    grouped["BalancedRank"] = grouped["BalancedRankScore"].rank(
        ascending=True, method="min"
    )
    return grouped.sort_values(
        ["BalancedRank", "TopK"]
    ).reset_index(drop=True)


def run_v7_slot_sweep(
    paths: ProjectPaths,
    settings: ResearchSettings,
    base_params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    frozen_strategy_path: str | Path,
    spy_path: str | Path,
    minimum_top_k: int = 1,
    maximum_top_k: int = 10,
    exit_buffer: int = 4,
    expected_ready_count: int = 575,
    output_dir: str | Path | None = None,
) -> V7SlotSweepArtifacts:
    ready_tickers = load_ready_tickers(
        backfill_status_path,
        expected_count=expected_ready_count,
    )
    members, _ = discover_universe(paths, ticker_config_path)
    discoverable = {member.ticker for member in members}
    missing = sorted(ready_tickers - discoverable)
    if missing:
        raise ValueError(
            "Ready tickers are not discoverable: " + ", ".join(missing)
        )
    warmup_start = str(
        (
            pd.Timestamp(settings.train_start) - pd.DateOffset(years=2)
        ).date()
    )
    base_panel, data_audit = build_v7_source_panel(
        paths,
        members,
        settings,
        ready_tickers=ready_tickers,
        warmup_start=warmup_start,
    )
    observed = add_v7_technical_observations(base_panel)
    membership = normalize_change_membership(
        pd.read_csv(membership_path)
    )
    pit_panel = apply_membership_to_panel(
        observed,
        membership,
        settings,
    )
    technical = add_v7_technical_factors(pit_panel, settings)
    v7_3 = TECHNICAL_VARIANTS[2]
    scoring_panel = scoring_panel_for_variant(technical, v7_3)
    latest_end = str(pd.Timestamp(scoring_panel["Date"].max()).date())
    periods = {
        "TRAIN_2020_2024": (
            settings.train_start,
            min(settings.train_end, latest_end),
        ),
        **{
            label: (start, min(end, latest_end))
            for label, (start, end) in settings.validation_periods.items()
            if start <= latest_end
        },
    }
    periods["FULL_2020_2026"] = (settings.train_start, latest_end)
    reference_dates = pd.DatetimeIndex(
        scoring_panel["Date"].drop_duplicates().sort_values()
    )
    spy = load_sp500_proxy(spy_path)
    spy_results = {
        period: spy_buy_and_hold(
            spy,
            start=start,
            end=end,
            initial_capital=settings.initial_capital,
            transaction_cost_bps=settings.transaction_cost_bps,
        )
        for period, (start, end) in periods.items()
    }

    summary_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    for top_k in range(minimum_top_k, maximum_top_k + 1):
        params = slot_sweep_params(
            base_params,
            top_k,
            exit_buffer=exit_buffer,
        )
        for period, (start, end) in periods.items():
            result = evaluate_period(
                scoring_panel,
                params,
                settings,
                start=start,
                end=end,
                reference_dates=reference_dates,
            )
            strategy = result.strategy.summary
            benchmark = result.benchmark.summary
            spy_summary, spy_equity = spy_results[period]
            summary_rows.append(
                {
                    "TopK": top_k,
                    "ExitRank": params.exit_rank,
                    "Period": period,
                    "StartDate": strategy.start_date,
                    "EndDate": strategy.end_date,
                    "FinalValue": strategy.final_value,
                    "StrategyROI": strategy.roi_percent,
                    "CAGR": strategy.cagr_percent,
                    "MaxDrawdown": strategy.max_drawdown_percent,
                    "Sharpe": strategy.sharpe_ratio,
                    "AnnualizedTurnover": strategy.annualized_turnover,
                    "EqualWeightUniverseROI": (
                        benchmark.roi_percent
                    ),
                    "SPYStartDate": spy_summary["StartDate"],
                    "SPYEndDate": spy_summary["EndDate"],
                    "SPYROI": spy_summary["ROI"],
                    "SPYCAGR": spy_summary["CAGR"],
                    "SPYMaxDrawdown": spy_summary["MaxDrawdown"],
                    "SPYSharpe": spy_summary["Sharpe"],
                    "SPYExcessROI": (
                        strategy.roi_percent - float(spy_summary["ROI"])
                    ),
                }
            )
            contributions = summarize_ticker_contributions(
                result.strategy,
                {period: (start, end)},
            )
            ordered = contributions.reindex(
                contributions["NetPnL"]
                .abs()
                .sort_values(ascending=False)
                .index
            )
            absolute_total = float(ordered["NetPnL"].abs().sum())
            concentration_rows.append(
                {
                    "TopK": top_k,
                    "Period": period,
                    "DistinctHeldTickers": int(
                        contributions["Ticker"].nunique()
                    ),
                    "MeanSelectedCount": float(
                        result.strategy.daily["SelectedCount"].mean()
                    ),
                    "Top1AbsoluteContributionPct": (
                        float(ordered.head(1)["NetPnL"].abs().sum())
                        / absolute_total
                        * 100
                        if absolute_total > 0
                        else np.nan
                    ),
                    "Top3AbsoluteContributionPct": (
                        float(ordered.head(3)["NetPnL"].abs().sum())
                        / absolute_total
                        * 100
                        if absolute_total > 0
                        else np.nan
                    ),
                    "PnLReconciliationError": float(
                        contributions["ReconciliationError"]
                        .abs()
                        .max()
                    ),
                }
            )
            equity = result.strategy.daily.copy()
            equity.insert(0, "Series", "V7_3")
            equity.insert(1, "TopK", top_k)
            equity.insert(2, "Period", period)
            equity_frames.append(equity)
            if top_k == minimum_top_k:
                benchmark_equity = spy_equity.copy()
                benchmark_equity.insert(0, "Series", "SPY_BUY_HOLD")
                benchmark_equity.insert(1, "TopK", 0)
                benchmark_equity.insert(2, "Period", period)
                equity_frames.append(benchmark_equity)

    summary = pd.DataFrame(summary_rows)
    ranking = balanced_slot_ranking(
        summary.loc[~summary["Period"].eq("FULL_2020_2026")]
    )
    concentration = pd.DataFrame(concentration_rows)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v7_slot_sweep"
            / f"{timestamp}_v7_3_top1_to_{maximum_top_k}"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "slot_summary_csv": destination / "slot_summary.csv",
        "balanced_ranking_csv": destination / "balanced_ranking.csv",
        "concentration_csv": destination / "concentration.csv",
        "equity_csv": destination / "equity.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(summary, outputs["slot_summary_csv"], index=False)
    atomic_to_csv(ranking, outputs["balanced_ranking_csv"], index=False)
    atomic_to_csv(
        concentration, outputs["concentration_csv"], index=False
    )
    atomic_to_csv(
        pd.concat(equity_frames, ignore_index=True),
        outputs["equity_csv"],
        index=False,
    )
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    frozen_payload = json.loads(
        Path(frozen_strategy_path).read_text(encoding="utf-8")
    )
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "V7-3 position-count sweep versus SPY buy-and-hold",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "v6_b_unchanged": True,
            "variant": v7_3.name,
            "technical_components": v7_3.components,
            "top_k_values": list(
                range(minimum_top_k, maximum_top_k + 1)
            ),
            "exit_rank_rule": f"top_k + {exit_buffer}",
            "mechanical_parameter_changes": (
                "top_k, exit_rank, and profit_rotation_exit_rank only"
            ),
            "balanced_ranking": (
                "equal-weight mean rank of median CAGR, CAGR stability, "
                "minimum period Sharpe, worst MDD, and minimum SPY excess ROI"
            ),
            "balanced_ranking_periods": (
                "TRAIN_2020_2024, 2025, and 2026; the overlapping "
                "FULL_2020_2026 result is reported but excluded from ranking"
            ),
            "spy_source": str(Path(spy_path).resolve()),
            "spy_source_sha256": _sha256(Path(spy_path)),
            "spy_adjustment": (
                "Adj Open and Adj Close; dividend/split-adjusted buy-and-hold"
            ),
            "spy_execution": (
                "buy first period session at adjusted open with 10bps "
                "entry cost; hold through final adjusted close"
            ),
            "frozen_v6_candidate": frozen_payload.get(
                "selected_candidate"
            ),
            "frozen_strategy_sha256": _sha256(
                Path(frozen_strategy_path)
            ),
            "ready_ticker_count": len(ready_tickers),
            "financial_point_in_time": False,
            "caveats": [
                "Macrotrends financials are current restated history.",
                "The 575-name panel excludes unavailable acquired/delisted names.",
                "All slot counts are evaluated after observing V6, 2025, and 2026.",
                "Choosing the best slot count adds another post-hoc selection step.",
            ],
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return V7SlotSweepArtifacts(
        output_dir=destination,
        **outputs,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
