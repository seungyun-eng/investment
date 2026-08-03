from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.indicators import normalize_price_columns
from stock_research.io_utils import atomic_to_csv, read_csv_fallback
from stock_research.paths import ProjectPaths
from stock_research.tsla_integrated.data import (
    load_equity_financials,
    load_equity_prices,
)

from .config import ResearchSettings, StrategyParams
from .data import UniverseMember, discover_universe
from .features import (
    FINANCIAL_SIGNAL_COLUMNS,
    add_cross_sectional_factors,
    build_equity_features,
)
from .pit_validation import (
    apply_membership_to_panel,
    causal_signal_day_panel,
)
from .portfolio import PortfolioResult, run_portfolio_backtest
from .signals import (
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
)
from .winner_attribution import summarize_ticker_contributions


@dataclass(frozen=True)
class PeriodEvaluation:
    strategy: PortfolioResult
    benchmark: PortfolioResult
    signal_days: pd.DataFrame
    targets: pd.DataFrame


@dataclass(frozen=True)
class V7PitArtifacts:
    output_dir: Path
    data_audit_csv: Path
    membership_coverage_csv: Path
    signal_coverage_csv: Path
    financial_lag_audit_csv: Path
    period_summary_csv: Path
    v6_comparison_csv: Path
    ticker_contributions_csv: Path
    wdc_dependence_csv: Path
    executions_csv: Path
    selected_signals_csv: Path
    equity_csv: Path
    manifest_json: Path


def load_ready_tickers(
    status_path: str | Path,
    *,
    expected_count: int | None = None,
) -> set[str]:
    status = pd.read_csv(status_path)
    required = {"Ticker", "V6Ready"}
    missing = sorted(required - set(status.columns))
    if missing:
        raise ValueError(f"Backfill status is missing columns: {missing}")
    ready_flag = (
        status["V6Ready"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    tickers = set(
        status.loc[ready_flag, "Ticker"].astype(str).str.upper().str.strip()
    )
    tickers.discard("")
    if expected_count is not None and len(tickers) != expected_count:
        raise ValueError(
            f"Expected {expected_count} V6-ready tickers, found {len(tickers)}"
        )
    return tickers


def normalize_change_membership(
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Use data symbols while retaining historical ticker identity for audit."""

    if "AsOfDate" not in membership:
        raise ValueError("Membership requires AsOfDate")
    symbol_column = (
        "DataSymbol" if "DataSymbol" in membership else "Ticker"
    )
    if symbol_column not in membership:
        raise ValueError("Membership requires Ticker or DataSymbol")
    frame = membership.copy()
    frame["AsOfDate"] = pd.to_datetime(frame["AsOfDate"], errors="raise")
    if "Ticker" in frame:
        frame["HistoricalTicker"] = (
            frame["Ticker"].astype(str).str.upper().str.strip()
        )
    else:
        frame["HistoricalTicker"] = ""
    frame["Ticker"] = (
        frame[symbol_column].astype(str).str.upper().str.strip()
    )
    frame = frame.loc[
        frame["Ticker"].ne("") & frame["Ticker"].ne("NAN")
    ].copy()
    if "Selected" in frame:
        selected = (
            frame["Selected"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        frame = frame.loc[selected].copy()
    sort_columns = ["AsOfDate"]
    if "Rank" in frame:
        frame["Rank"] = pd.to_numeric(frame["Rank"], errors="coerce")
        sort_columns.append("Rank")
    sort_columns.extend(["Ticker", "HistoricalTicker"])
    frame = (
        frame.sort_values(sort_columns)
        .drop_duplicates(["AsOfDate", "Ticker"], keep="first")
        .reset_index(drop=True)
    )
    if "Rank" not in frame:
        frame["Rank"] = frame.groupby("AsOfDate").cumcount() + 1
    frame["Selected"] = True
    return frame


def load_raw_company_prices(
    company_dir: str | Path,
    *,
    fallback_path: str | Path | None = None,
    earliest_date: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load and normalize all raw price fragments without technical indicators."""

    directory = Path(company_dir)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.csv")):
            try:
                normalized = normalize_price_columns(
                    read_csv_fallback(path)
                )
                frames.append(_stable_price_columns(normalized))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path.name}: {exc}")
    if frames:
        prices = (
            pd.concat(frames, ignore_index=True)
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )
        source_kind = "RAW_NORMALIZED"
        source_path = str(directory)
    elif fallback_path is not None:
        source = Path(fallback_path)
        prices = load_equity_prices(source)
        source_kind = "PROCESSED_FALLBACK"
        source_path = str(source)
    else:
        detail = "; ".join(errors) if errors else "no CSV files"
        raise ValueError(f"No usable price data in {directory}: {detail}")
    if earliest_date is not None:
        prices = prices.loc[
            prices["Date"].ge(pd.Timestamp(earliest_date))
        ].reset_index(drop=True)
    if prices.empty:
        raise ValueError(f"No price rows remain for {directory}")
    return prices, source_kind, source_path


def build_v7_source_panel(
    paths: ProjectPaths,
    members: list[UniverseMember],
    settings: ResearchSettings,
    *,
    ready_tickers: set[str],
    warmup_start: str,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    selected = [
        member for member in members if member.ticker in ready_tickers
    ]
    for index, member in enumerate(selected, start=1):
        audit: dict[str, object] = {
            "Ticker": member.ticker,
            "Company": member.company,
            "RequestedReady": True,
            "BuildStatus": "FAILED",
            "Error": "",
        }
        try:
            prices, source_kind, source_path = load_raw_company_prices(
                paths.raw_prices / member.company,
                fallback_path=member.price_path,
                earliest_date=warmup_start,
            )
            financials = load_equity_financials(member.financial_path)
            equity = build_equity_features(
                prices,
                financials,
                ticker=member.ticker,
                company=member.company,
                settings=settings,
            )
            train_mask = equity["Date"].between(
                settings.train_start,
                settings.train_end,
            )
            audit.update(
                {
                    "BuildStatus": "INCLUDED",
                    "PriceSourceKind": source_kind,
                    "PriceSource": source_path,
                    "PriceStart": prices["Date"].min(),
                    "PriceEnd": prices["Date"].max(),
                    "PriceRows": len(prices),
                    "FinancialStart": financials["Date"].min(),
                    "FinancialEnd": financials["Date"].max(),
                    "FinancialRows": len(financials),
                    "TrainingSessions": int(train_mask.sum()),
                    "TrainingEligibleSessions": int(
                        (train_mask & equity["Eligible"]).sum()
                    ),
                    "FirstEligibleDate": equity.loc[
                        equity["Eligible"], "Date"
                    ].min(),
                }
            )
            frames.append(equity)
        except Exception as exc:  # noqa: BLE001
            audit["Error"] = str(exc)
        audits.append(audit)
        if progress_every and (
            index % progress_every == 0 or index == len(selected)
        ):
            included = sum(
                row["BuildStatus"] == "INCLUDED" for row in audits
            )
            print(
                f"DATA {index}/{len(selected)} processed; "
                f"{included} included"
            )
    if not frames:
        raise ValueError("No V7 source equities could be built")
    panel = add_cross_sectional_factors(
        pd.concat(frames, ignore_index=True),
        settings,
    )
    return (
        panel.sort_values(["Date", "Ticker"]).reset_index(drop=True),
        pd.DataFrame(audits).sort_values("Ticker").reset_index(drop=True),
    )


def evaluate_period(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    *,
    start: str,
    end: str,
    reference_dates: pd.DatetimeIndex,
    record_attribution: bool = True,
) -> PeriodEvaluation:
    signal_days = causal_signal_day_panel(
        panel,
        start=start,
        end=end,
        reference_market_dates=reference_dates,
        rebalance_weekday=settings.rebalance_weekday,
    )
    if signal_days.empty:
        raise ValueError(f"No causal signal dates between {start} and {end}")
    targets = generate_rebalance_targets(
        score_panel(signal_days, params),
        params,
        force_universe_exit=settings.force_universe_exit,
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
    return PeriodEvaluation(
        strategy=strategy,
        benchmark=benchmark,
        signal_days=signal_days,
        targets=targets,
    )


def build_financial_lag_audit(
    panel: pd.DataFrame,
    settings: ResearchSettings,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, (start, end) in periods.items():
        frame = panel.loc[panel["Date"].between(start, end)].copy()
        available = frame["FinancialAvailableDate"].notna()
        observed_lag = (
            frame.loc[available, "FinancialAvailableDate"]
            - frame.loc[available, "FinancialPeriodEnd"]
        ).dt.days
        stale = (
            pd.to_numeric(frame["FinancialAgeDays"], errors="coerce")
            > settings.max_financial_age_days
        )
        stale_signals = frame.loc[
            stale, list(FINANCIAL_SIGNAL_COLUMNS)
        ]
        rows.append(
            {
                "Period": label,
                "Rows": len(frame),
                "Tickers": int(frame["Ticker"].nunique()),
                "RowsWithFinancials": int(available.sum()),
                "FinancialLagDaysMin": observed_lag.min(),
                "FinancialLagDaysMedian": observed_lag.median(),
                "FinancialLagDaysMax": observed_lag.max(),
                "ExpectedFinancialLagDays": (
                    settings.financial_release_lag_days
                ),
                "AvailabilityLookAheadViolations": int(
                    (
                        available
                        & frame["FinancialAvailableDate"].gt(frame["Date"])
                    ).sum()
                ),
                "StaleRows": int(stale.sum()),
                "StaleSignalNonNullCells": int(
                    stale_signals.notna().sum().sum()
                ),
                "MaxFinancialAgeDays": settings.max_financial_age_days,
                "FinancialDataIsPointInTime": False,
            }
        )
    return pd.DataFrame(rows)


def build_membership_coverage(
    membership: pd.DataFrame,
    ready_tickers: set[str],
    data_audit: pd.DataFrame,
    *,
    maximum_price_gap_days: int = 7,
) -> pd.DataFrame:
    included = data_audit.loc[
        data_audit["BuildStatus"].eq("INCLUDED")
    ].copy()
    included_tickers = set(included["Ticker"])
    price_start = included.set_index("Ticker")["PriceStart"].map(
        pd.Timestamp
    )
    price_end = included.set_index("Ticker")["PriceEnd"].map(pd.Timestamp)
    rows: list[dict[str, object]] = []
    for date, group in membership.groupby("AsOfDate", sort=True):
        tickers = set(group["Ticker"])
        ready = tickers & ready_tickers
        built = tickers & included_tickers
        price_covered = {
            ticker
            for ticker in built
            if (
                price_start.get(ticker, pd.Timestamp.max)
                <= pd.Timestamp(date)
                and price_end.get(ticker, pd.Timestamp.min)
                >= (
                    pd.Timestamp(date)
                    - pd.Timedelta(days=maximum_price_gap_days)
                )
            )
        }
        rows.append(
            {
                "AsOfDate": pd.Timestamp(date),
                "IndexMembers": len(tickers),
                "ReadyMembers": len(ready),
                "BuiltMembers": len(built),
                "PriceCoveredOnDate": len(price_covered),
                "MaximumPriceGapDays": maximum_price_gap_days,
                "ReadyCoveragePct": (
                    len(ready) / len(tickers) * 100 if tickers else np.nan
                ),
                "PriceCoveragePct": (
                    len(price_covered) / len(tickers) * 100
                    if tickers
                    else np.nan
                ),
                "MissingReadyTickers": ",".join(sorted(tickers - ready)),
            }
        )
    return pd.DataFrame(rows)


def build_signal_coverage(
    signal_days_by_period: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, frame in signal_days_by_period.items():
        for date, group in frame.groupby("Date", sort=True):
            member = group["UniverseMember"].fillna(False)
            eligible = group["Eligible"].fillna(False)
            fresh = ~group["FinancialStale"].fillna(True)
            rows.append(
                {
                    "Period": period,
                    "SignalDate": pd.Timestamp(date),
                    "PanelRows": len(group),
                    "ReadyMembersWithPriceRow": int(member.sum()),
                    "EligibleMembers": int(eligible.sum()),
                    "EligibleFreshFinancialMembers": int(
                        (eligible & fresh).sum()
                    ),
                    "EligibleStaleFinancialMembers": int(
                        (eligible & ~fresh).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_v7_pit_evaluation(
    paths: ProjectPaths,
    settings: ResearchSettings,
    params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    frozen_strategy_path: str | Path,
    expected_ready_count: int = 575,
    output_dir: str | Path | None = None,
) -> V7PitArtifacts:
    ready_tickers = load_ready_tickers(
        backfill_status_path,
        expected_count=expected_ready_count,
    )
    members, discovery_audit = discover_universe(
        paths, ticker_config_path
    )
    discoverable = {member.ticker for member in members}
    missing_discovery = sorted(ready_tickers - discoverable)
    if missing_discovery:
        raise ValueError(
            "Ready tickers are not discoverable: "
            + ", ".join(missing_discovery)
        )

    warmup_start = str(
        (
            pd.Timestamp(settings.train_start) - pd.DateOffset(years=2)
        ).date()
    )
    panel, data_audit = build_v7_source_panel(
        paths,
        members,
        settings,
        ready_tickers=ready_tickers,
        warmup_start=warmup_start,
    )
    raw_membership = pd.read_csv(membership_path)
    membership = normalize_change_membership(raw_membership)
    pit_panel = apply_membership_to_panel(panel, membership, settings)

    latest_end = str(pd.Timestamp(pit_panel["Date"].max()).date())
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
    reference_dates = pd.DatetimeIndex(
        pit_panel["Date"].drop_duplicates().sort_values()
    )

    results: dict[str, PeriodEvaluation] = {}
    summary_rows: list[dict[str, object]] = []
    contribution_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    execution_frames: list[pd.DataFrame] = []
    selected_signal_frames: list[pd.DataFrame] = []
    for label, (start, end) in periods.items():
        result = evaluate_period(
            pit_panel,
            params,
            settings,
            start=start,
            end=end,
            reference_dates=reference_dates,
        )
        results[label] = result
        summary_rows.append(_period_summary_row(label, result))
        contribution_frames.append(
            summarize_ticker_contributions(
                result.strategy,
                {label: (start, end)},
            )
        )
        equity = result.strategy.daily.copy()
        equity.insert(0, "Period", label)
        equity_frames.append(equity)
        executions = result.strategy.executions.copy()
        executions.insert(0, "Period", label)
        execution_frames.append(executions)
        selected = result.targets.loc[
            result.targets["ModelSelected"].fillna(False)
            | result.targets["TradeAction"].isin(["BUY", "SELL"])
        ].copy()
        selected.insert(0, "Period", label)
        selected_signal_frames.append(selected)

    summary = pd.DataFrame(summary_rows)
    contributions = pd.concat(contribution_frames, ignore_index=True)
    wdc_dependence = _run_wdc_dependence(
        pit_panel,
        params,
        settings,
        periods,
        reference_dates,
        results,
        contributions,
    )
    signal_coverage = build_signal_coverage(
        {label: result.signal_days for label, result in results.items()}
    )
    financial_audit = build_financial_lag_audit(
        pit_panel,
        settings,
        periods,
    )
    membership_coverage = build_membership_coverage(
        membership,
        ready_tickers,
        data_audit,
    )

    frozen_payload = json.loads(
        Path(frozen_strategy_path).read_text(encoding="utf-8")
    )
    comparison = _build_v6_comparison(summary, frozen_payload)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v7_pit_575"
            / f"{timestamp}_frozen_v6_b_1931"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "data_audit_csv": destination / "data_audit.csv",
        "membership_coverage_csv": (
            destination / "membership_coverage.csv"
        ),
        "signal_coverage_csv": destination / "signal_coverage.csv",
        "financial_lag_audit_csv": (
            destination / "financial_lag_audit.csv"
        ),
        "period_summary_csv": destination / "period_summary.csv",
        "v6_comparison_csv": destination / "v6_comparison.csv",
        "ticker_contributions_csv": (
            destination / "ticker_contributions.csv"
        ),
        "wdc_dependence_csv": destination / "wdc_dependence.csv",
        "executions_csv": destination / "executions.csv",
        "selected_signals_csv": destination / "selected_signals.csv",
        "equity_csv": destination / "equity.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    atomic_to_csv(
        membership_coverage,
        outputs["membership_coverage_csv"],
        index=False,
    )
    atomic_to_csv(
        signal_coverage, outputs["signal_coverage_csv"], index=False
    )
    atomic_to_csv(
        financial_audit,
        outputs["financial_lag_audit_csv"],
        index=False,
    )
    atomic_to_csv(summary, outputs["period_summary_csv"], index=False)
    atomic_to_csv(comparison, outputs["v6_comparison_csv"], index=False)
    atomic_to_csv(
        contributions,
        outputs["ticker_contributions_csv"],
        index=False,
    )
    atomic_to_csv(
        wdc_dependence,
        outputs["wdc_dependence_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(execution_frames, ignore_index=True),
        outputs["executions_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(selected_signal_frames, ignore_index=True),
        outputs["selected_signals_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(equity_frames, ignore_index=True),
        outputs["equity_csv"],
        index=False,
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "task": "V7-PIT 575-name frozen V6-B diagnostic",
        "model_status": "POST_HOC_DIAGNOSTIC_PASS",
        "validation_is_fresh": False,
        "ready_ticker_count_requested": len(ready_tickers),
        "panel_ticker_count": int(panel["Ticker"].nunique()),
        "membership_snapshot_count": int(
            membership["AsOfDate"].nunique()
        ),
        "strategy_candidate": frozen_payload.get("selected_candidate"),
        "strategy_params": params.as_dict(),
        "settings": frozen_payload.get("settings", {}),
        "strategy_sha256": _sha256(Path(frozen_strategy_path)),
        "membership_sha256": _sha256(Path(membership_path)),
        "backfill_status_sha256": _sha256(Path(backfill_status_path)),
        "warmup_start": warmup_start,
        "price_loading": (
            "normalize and merge raw company CSV fragments; use processed "
            "file only when no raw fragment is usable"
        ),
        "financial_timing": (
            f"quarter-end plus {settings.financial_release_lag_days} days; "
            f"signals blank after {settings.max_financial_age_days} days"
        ),
        "signal_rule": (
            "known final exchange session of each W-FRI week close"
        ),
        "execution_rule": "next available trading session open",
        "transaction_cost_bps": settings.transaction_cost_bps,
        "roi_formula": "(final_value / total_injected - 1) * 100",
        "universe_mechanics": (
            "entries are eligible only in the latest known S&P membership "
            "snapshot; existing holdings retain frozen V6-B exit logic"
        ),
        "survivorship_caveat": (
            "Only 575 names with local price and financial data are usable. "
            "Missing acquired/delisted constituents remain excluded."
        ),
        "financial_point_in_time": False,
        "financial_caveat": (
            "Macrotrends values are current restated history keyed by fiscal "
            "quarter-end. The 45-day lag is an approximation, not true PIT."
        ),
        "selection_caveat": (
            "V6-B was selected after 2,000 trials and 2025/2026 were already "
            "observed; all comparisons remain post-hoc diagnostics."
        ),
        "discovery_audit_rows": len(discovery_audit),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    _atomic_json(manifest, outputs["manifest_json"])
    return V7PitArtifacts(
        output_dir=destination,
        **outputs,
    )


def _stable_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) < 5:
        raise ValueError("Normalized price frame has fewer than five columns")
    positional = {
        frame.columns[0]: "Date",
        frame.columns[1]: "Close",
        frame.columns[2]: "Open",
        frame.columns[3]: "High",
        frame.columns[4]: "Low",
    }
    if len(frame.columns) > 5:
        positional[frame.columns[5]] = "Volume"
    result = frame.rename(columns=positional)
    for column in ("Close", "Open", "High", "Low", "Volume"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return (
        result.dropna(subset=["Date", "Close", "Open"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def _period_summary_row(
    label: str,
    result: PeriodEvaluation,
) -> dict[str, object]:
    strategy = result.strategy.summary
    benchmark = result.benchmark.summary
    return {
        "Period": label,
        "StartDate": strategy.start_date,
        "EndDate": strategy.end_date,
        "FinalValue": strategy.final_value,
        "StrategyROI": strategy.roi_percent,
        "BenchmarkROI": benchmark.roi_percent,
        "ExcessROI": strategy.roi_percent - benchmark.roi_percent,
        "CAGR": strategy.cagr_percent,
        "MaxDrawdown": strategy.max_drawdown_percent,
        "Sharpe": strategy.sharpe_ratio,
        "BenchmarkMaxDrawdown": benchmark.max_drawdown_percent,
        "BenchmarkSharpe": benchmark.sharpe_ratio,
        "AnnualizedTurnover": strategy.annualized_turnover,
        "TickerTrades": strategy.ticker_trades,
        "Rebalances": strategy.rebalance_count,
        "SignalDates": int(result.signal_days["Date"].nunique()),
    }


def _run_wdc_dependence(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    periods: dict[str, tuple[str, str]],
    reference_dates: pd.DatetimeIndex,
    baseline_results: dict[str, PeriodEvaluation],
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    reduced = (
        add_cross_sectional_factors(
            panel.loc[~panel["Ticker"].eq("WDC")].copy(),
            settings,
        )
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for label, (start, end) in periods.items():
        baseline = baseline_results[label]
        no_wdc = evaluate_period(
            reduced,
            params,
            settings,
            start=start,
            end=end,
            reference_dates=reference_dates,
            record_attribution=False,
        )
        wdc = contributions.loc[
            contributions["Period"].eq(label)
            & contributions["Ticker"].eq("WDC")
        ]
        wdc_pnl = float(wdc["NetPnL"].sum()) if not wdc.empty else 0.0
        wdc_return = (
            float(wdc["ContributionToPeriodReturnPct"].sum())
            if not wdc.empty
            else 0.0
        )
        baseline_excess = (
            baseline.strategy.summary.roi_percent
            - baseline.benchmark.summary.roi_percent
        )
        no_wdc_excess = (
            no_wdc.strategy.summary.roi_percent
            - no_wdc.benchmark.summary.roi_percent
        )
        rows.append(
            {
                "Period": label,
                "WDCNetPnL": wdc_pnl,
                "WDCContributionToReturnPct": wdc_return,
                "BaselineStrategyROI": (
                    baseline.strategy.summary.roi_percent
                ),
                "NoWDCStrategyROI": no_wdc.strategy.summary.roi_percent,
                "StrategyROIDeltaWithoutWDC": (
                    no_wdc.strategy.summary.roi_percent
                    - baseline.strategy.summary.roi_percent
                ),
                "BaselineBenchmarkROI": (
                    baseline.benchmark.summary.roi_percent
                ),
                "NoWDCBenchmarkROI": no_wdc.benchmark.summary.roi_percent,
                "BaselineExcessROI": baseline_excess,
                "NoWDCExcessROI": no_wdc_excess,
                "ExcessRetentionWithoutWDCPct": (
                    no_wdc_excess / baseline_excess * 100
                    if abs(baseline_excess) > 1e-12
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_v6_comparison(
    v7_summary: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> pd.DataFrame:
    training = dict(frozen_payload["training_metrics"])
    v6_rows = {
        "TRAIN_2020_2024": {
            "V6StrategyROI": training["TrainROI"],
            "V6BenchmarkROI": training["BenchmarkROI"],
            "V6ExcessROI": training["ExcessROI"],
            "V6MaxDrawdown": training["TrainMaxDrawdown"],
            "V6Sharpe": training["TrainSharpe"],
        }
    }
    for row in frozen_payload.get("validation", []):
        v6_rows[str(row["Period"])] = {
            "V6StrategyROI": row["StrategyROI"],
            "V6BenchmarkROI": row["EqualWeightUniverseROI"],
            "V6ExcessROI": row["ExcessROI"],
            "V6MaxDrawdown": row["MaxDrawdown"],
            "V6Sharpe": row["Sharpe"],
        }
    rows: list[dict[str, object]] = []
    for record in v7_summary.to_dict(orient="records"):
        period = str(record["Period"])
        baseline = v6_rows.get(period)
        if baseline is None:
            continue
        rows.append(
            {
                "Period": period,
                **baseline,
                "V7StrategyROI": record["StrategyROI"],
                "V7BenchmarkROI": record["BenchmarkROI"],
                "V7ExcessROI": record["ExcessROI"],
                "V7MaxDrawdown": record["MaxDrawdown"],
                "V7Sharpe": record["Sharpe"],
                "StrategyROIDeltaV7MinusV6": (
                    record["StrategyROI"] - baseline["V6StrategyROI"]
                ),
                "ExcessROIDeltaV7MinusV6": (
                    record["ExcessROI"] - baseline["V6ExcessROI"]
                ),
                "V6ExcessRetainedPct": (
                    record["ExcessROI"] / baseline["V6ExcessROI"] * 100
                    if abs(float(baseline["V6ExcessROI"])) > 1e-12
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


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
