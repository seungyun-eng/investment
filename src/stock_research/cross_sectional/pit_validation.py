from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths
from stock_research.tickers import load_tickers

from .config import ResearchSettings, StrategyParams
from .data import UniverseMember, discover_universe
from .features import add_cross_sectional_factors, build_panel
from .portfolio import PortfolioResult, run_portfolio_backtest
from .signals import (
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)

MEMBERSHIP_REQUIRED_COLUMNS = {
    "AsOfDate",
    "Ticker",
}
DELISTING_REQUIRED_COLUMNS = {
    "Ticker",
    "EffectiveDate",
    "DelistingReturn",
}
FROZEN_REPORTED_BASELINE = {
    "TRAIN_2020_2024": {
        "StrategyROI": 1671.0,
        "BenchmarkROI": 219.0,
        "ExcessROI": 1452.0,
    },
    "2025": {
        "StrategyROI": 282.76,
        "BenchmarkROI": 31.97,
        "ExcessROI": 250.79,
    },
    "2026": {
        "StrategyROI": 60.32,
        "BenchmarkROI": 19.03,
        "ExcessROI": 41.30,
    },
}


@dataclass(frozen=True)
class PitDiagnosticArtifacts:
    output_dir: Path
    local_universe_audit_csv: Path
    requested_comparison_csv: Path
    scenario_summary_csv: Path
    annual_membership_csv: Path
    wdc_audit_csv: Path
    signal_date_comparison_csv: Path
    scenario_equity_csv: Path
    data_readiness_csv: Path
    delisting_policy_csv: Path
    manifest_json: Path


def causal_weekly_signal_dates(
    reference_market_dates: Iterable[object],
    *,
    start: str,
    end: str,
    rebalance_weekday: int = 4,
) -> pd.DatetimeIndex:
    """Return the known exchange-calendar final session of each signal week."""

    dates = pd.Series(pd.to_datetime(list(reference_market_dates)))
    dates = dates.dropna().drop_duplicates().sort_values()
    if dates.empty:
        return pd.DatetimeIndex([])
    weekday_names = ("MON", "TUE", "WED", "THU", "FRI")
    weeks = dates.dt.to_period(f"W-{weekday_names[rebalance_weekday]}")
    final_session = dates.groupby(weeks).max()
    week_end = final_session.index.to_timestamp(how="end").normalize()
    completed = week_end <= pd.Timestamp(dates.max()).normalize()
    final_session = final_session.loc[completed]
    final_session = final_session.loc[final_session.between(start, end)]
    return pd.DatetimeIndex(final_session.to_numpy())


def causal_signal_day_panel(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    reference_market_dates: Iterable[object],
    rebalance_weekday: int = 4,
) -> pd.DataFrame:
    signal_dates = causal_weekly_signal_dates(
        reference_market_dates,
        start=start,
        end=end,
        rebalance_weekday=rebalance_weekday,
    )
    period = panel.loc[panel["Date"].between(start, end)].copy()
    return period.loc[period["Date"].isin(signal_dates)].copy()


def load_membership(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_membership(frame)


def validate_membership(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(MEMBERSHIP_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"PIT membership is missing required columns: {missing}"
        )
    result = frame.copy()
    result["AsOfDate"] = pd.to_datetime(
        result["AsOfDate"],
        errors="raise",
    )
    result["Ticker"] = result["Ticker"].astype(str).str.upper()
    if "Selected" in result:
        result = result.loc[result["Selected"].fillna(False)].copy()
    if result.duplicated(["AsOfDate", "Ticker"]).any():
        raise ValueError("Duplicate ticker in a PIT membership snapshot")
    if "Rank" not in result:
        result["Rank"] = result.groupby("AsOfDate").cumcount() + 1
    result["Rank"] = pd.to_numeric(result["Rank"], errors="raise")
    result["_UniverseMember"] = True
    return result.sort_values(["AsOfDate", "Rank", "Ticker"]).reset_index(
        drop=True
    )


def load_delisting_events(
    path: str | Path,
    *,
    missing_return_policy: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return apply_delisting_return_policy(
        frame,
        missing_return_policy=missing_return_policy,
    )


def apply_delisting_return_policy(
    events: pd.DataFrame,
    *,
    missing_return_policy: str,
) -> pd.DataFrame:
    required_without_return = {"Ticker", "EffectiveDate"}
    missing = sorted(required_without_return - set(events.columns))
    if missing:
        raise ValueError(
            f"Delisting events are missing required columns: {missing}"
        )
    policy = missing_return_policy.upper()
    if policy not in {
        "ERROR",
        "ZERO",
        "TOTAL_LOSS",
        "EXCHANGE_HAIRCUT",
    }:
        raise ValueError(f"Unsupported missing delisting policy: {policy}")
    result = events.copy()
    result["Ticker"] = result["Ticker"].astype(str).str.upper()
    result["EffectiveDate"] = pd.to_datetime(
        result["EffectiveDate"],
        errors="raise",
    )
    if "DelistingReturn" not in result:
        result["DelistingReturn"] = np.nan
    result["DelistingReturn"] = pd.to_numeric(
        result["DelistingReturn"],
        errors="coerce",
    )
    category = result.get(
        "DelistingCategory",
        pd.Series("", index=result.index),
    ).astype(str).str.upper()
    bankruptcy = category.isin(
        {"BANKRUPTCY", "LIQUIDATION", "WORTHLESS"}
    )
    result.loc[bankruptcy, "DelistingReturn"] = -1.0
    missing_return = result["DelistingReturn"].isna()
    if missing_return.any() and policy == "ERROR":
        tickers = result.loc[missing_return, "Ticker"].tolist()
        raise ValueError(
            f"Missing observed delisting returns for: {tickers}"
        )
    if policy == "ZERO":
        result.loc[missing_return, "DelistingReturn"] = 0.0
    elif policy == "TOTAL_LOSS":
        result.loc[missing_return, "DelistingReturn"] = -1.0
    elif policy == "EXCHANGE_HAIRCUT":
        exchange = result.get(
            "Exchange",
            pd.Series("", index=result.index),
        ).astype(str).str.upper()
        result.loc[missing_return, "DelistingReturn"] = np.where(
            exchange.loc[missing_return].str.contains("NASDAQ"),
            -0.55,
            -0.30,
        )
    invalid = (
        ~np.isfinite(result["DelistingReturn"])
        | result["DelistingReturn"].lt(-1.0)
    )
    if invalid.any():
        raise ValueError(
            "Resolved delisting returns must be finite and at least -1"
        )
    result["MissingReturnPolicy"] = policy
    result["ReturnWasImputed"] = missing_return & ~bankruptcy
    return result.sort_values(
        ["EffectiveDate", "Ticker"]
    ).reset_index(drop=True)


def apply_membership_to_panel(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    settings: ResearchSettings,
    *,
    delisting_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    members = validate_membership(membership)
    snapshots = pd.DataFrame(
        {"AsOfDate": members["AsOfDate"].drop_duplicates().sort_values()}
    )
    dates = pd.DataFrame(
        {"Date": pd.to_datetime(panel["Date"].drop_duplicates())}
    ).sort_values("Date")
    dated = pd.merge_asof(
        dates,
        snapshots,
        left_on="Date",
        right_on="AsOfDate",
        direction="backward",
    )
    frame = panel.merge(dated, on="Date", how="left", validate="many_to_one")
    member_columns = members[
        ["AsOfDate", "Ticker", "Rank", "_UniverseMember"]
    ].rename(columns={"Rank": "UniverseRank"})
    frame = frame.merge(
        member_columns,
        on=["AsOfDate", "Ticker"],
        how="left",
        validate="many_to_one",
    )
    frame["UniverseMember"] = frame["_UniverseMember"].fillna(False)
    frame = frame.drop(columns=["_UniverseMember"])
    if delisting_events is not None and not delisting_events.empty:
        first_delisting = (
            delisting_events.groupby("Ticker", as_index=False)[
                "EffectiveDate"
            ]
            .min()
            .rename(columns={"EffectiveDate": "DelistingEffectiveDate"})
        )
        frame = frame.merge(
            first_delisting,
            on="Ticker",
            how="left",
            validate="many_to_one",
        )
        frame["UniverseMember"] &= (
            frame["DelistingEffectiveDate"].isna()
            | frame["Date"].lt(frame["DelistingEffectiveDate"])
        )
    frame["EligibleBeforeUniverse"] = frame["Eligible"]
    frame["Eligible"] &= frame["UniverseMember"]
    return (
        add_cross_sectional_factors(frame, settings)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )


def build_annual_liquidity_membership(
    panel: pd.DataFrame,
    as_of_dates: Iterable[object],
    *,
    target_size: int = 200,
    lookback_sessions: int = 63,
    minimum_sessions: int = 42,
    minimum_price: float = 5.0,
    minimum_market_cap: float = 1_000_000_000.0,
    minimum_dollar_volume: float = 10_000_000.0,
    minimum_ipo_age_years: int = 2,
    source: str = "LOCAL_SURVIVORS_63D_PROXY",
    is_point_in_time: bool = False,
    includes_delisted: bool = False,
) -> pd.DataFrame:
    """Build annual snapshots using only observations before each as-of date."""

    if "Volume" not in panel:
        raise ValueError("Panel must contain Volume for liquidity ranking")
    frame = panel.sort_values(["Ticker", "Date"]).copy()
    frame["DollarVolumeObservation"] = (
        pd.to_numeric(frame["Close"], errors="coerce")
        * pd.to_numeric(frame["Volume"], errors="coerce")
    )
    first_dates = frame.groupby("Ticker")["Date"].min()
    snapshots: list[pd.DataFrame] = []
    for value in sorted(pd.to_datetime(list(as_of_dates))):
        history = frame.loc[frame["Date"].lt(value)]
        trailing = history.groupby("Ticker", group_keys=False).tail(
            lookback_sessions
        )
        if trailing.empty:
            continue
        liquidity = trailing.groupby("Ticker").agg(
            LookbackSessions=("DollarVolumeObservation", "count"),
            DollarVolume=("DollarVolumeObservation", "median"),
            LastPriceDate=("Date", "max"),
            LastClose=("Close", "last"),
        )
        share_values = (
            pd.to_numeric(trailing["Shares"], errors="coerce")
            if "Shares" in trailing
            else pd.Series(np.nan, index=trailing.index)
        )
        shares = (
            trailing.loc[
                share_values.gt(0)
            ]
            .groupby("Ticker")["Shares"]
            .last()
        )
        liquidity["Shares"] = shares
        liquidity["MarketCap"] = (
            liquidity["LastClose"] * liquidity["Shares"] * 1_000_000.0
        )
        liquidity["SharesUnitAssumption"] = (
            "Macrotrends shares outstanding in millions"
        )
        liquidity["FirstLocalPriceDate"] = first_dates
        liquidity["LocalHistoryAgeDays"] = (
            value - liquidity["FirstLocalPriceDate"]
        ).dt.days
        liquidity["MarketCapAvailable"] = liquidity["MarketCap"].notna()
        liquidity["EligibleForProxy"] = (
            liquidity["LookbackSessions"].ge(minimum_sessions)
            & liquidity["LastClose"].ge(minimum_price)
            & liquidity["DollarVolume"].ge(minimum_dollar_volume)
            & liquidity["MarketCap"].ge(minimum_market_cap)
            & liquidity["LocalHistoryAgeDays"].ge(
                round(minimum_ipo_age_years * 365.25)
            )
        )
        selected = (
            liquidity.loc[liquidity["EligibleForProxy"]]
            .sort_values(
                ["DollarVolume", "MarketCap", "Ticker"],
                ascending=[False, False, True],
                na_position="last",
            )
            .head(target_size)
            .reset_index()
        )
        selected["AsOfDate"] = value
        selected["Rank"] = np.arange(1, len(selected) + 1)
        selected["Selected"] = True
        selected["Source"] = source
        selected["IsPointInTime"] = is_point_in_time
        selected["IncludesDelisted"] = includes_delisted
        selected["SelectionLookbackSessions"] = lookback_sessions
        selected["MinimumIPOAgeYears"] = minimum_ipo_age_years
        snapshots.append(selected)
    if not snapshots:
        raise ValueError("No annual membership snapshots were produced")
    return pd.concat(snapshots, ignore_index=True)


def generate_pit_diagnostic(
    paths: ProjectPaths,
    settings: ResearchSettings,
    params: StrategyParams,
    *,
    current_ticker_config_path: str | Path,
    supplemental_ticker_config_paths: Iterable[str | Path] = (),
    pit_membership_path: str | Path | None = None,
    delisting_events_path: str | Path | None = None,
    missing_delisting_return_policy: str = "ERROR",
    output_dir: str | Path | None = None,
) -> PitDiagnosticArtifacts:
    config_paths = [
        Path(current_ticker_config_path),
        *(Path(path) for path in supplemental_ticker_config_paths),
    ]
    local_universe_audit = audit_local_price_universe(paths, config_paths)
    allowed_tickers = set(
        local_universe_audit.loc[
            local_universe_audit["IncludedIndividualEquity"], "Ticker"
        ].dropna()
    )
    members, discovery_audit = _discover_combined_universe(
        paths,
        config_paths,
    )
    members = [member for member in members if member.ticker in allowed_tickers]
    panel, data_audit = build_panel(members, settings)
    current_members, _ = discover_universe(
        paths,
        current_ticker_config_path,
    )
    current_tickers = {member.ticker for member in current_members}
    current_panel = panel.loc[panel["Ticker"].isin(current_tickers)].copy()
    current_panel = add_cross_sectional_factors(current_panel, settings)
    reference_dates = pd.DatetimeIndex(
        panel["Date"].drop_duplicates().sort_values()
    )
    latest_end = str(pd.Timestamp(panel["Date"].max()).date())
    as_of_dates = pd.to_datetime(
        [
            f"{year}-01-01"
            for year in range(
                pd.Timestamp(settings.train_start).year,
                pd.Timestamp(latest_end).year + 1,
            )
        ]
    )
    annual_membership = build_annual_liquidity_membership(
        panel,
        as_of_dates,
    )
    proxy_panel = apply_membership_to_panel(
        panel,
        annual_membership,
        settings,
    )

    delisting_events = None
    if delisting_events_path is not None:
        delisting_events = load_delisting_events(
            delisting_events_path,
            missing_return_policy=missing_delisting_return_policy,
        )

    scenario_frames: list[tuple[str, pd.DataFrame, str, bool]] = [
        (
            "BASELINE_CURRENT_SNAPSHOT_NONCAUSAL",
            current_panel,
            "COVERAGE_MAX_RETROSPECTIVE",
            False,
        ),
        (
            "CURRENT_SNAPSHOT_CAUSAL",
            current_panel,
            "REFERENCE_CALENDAR_LAST_SESSION",
            False,
        ),
        (
            "LOCAL_ANNUAL_SURVIVOR_PROXY_CAUSAL",
            proxy_panel,
            "REFERENCE_CALENDAR_LAST_SESSION",
            False,
        ),
    ]
    true_pit_executed = False
    missing_pit_tickers: list[str] = []
    if pit_membership_path is not None:
        pit_membership = load_membership(pit_membership_path)
        missing_pit_tickers = sorted(
            set(pit_membership["Ticker"]) - set(panel["Ticker"])
        )
        if not missing_pit_tickers:
            pit_panel = apply_membership_to_panel(
                panel,
                pit_membership,
                settings,
                delisting_events=delisting_events,
            )
            scenario_frames.append(
                (
                    "LICENSED_PIT_CAUSAL",
                    pit_panel,
                    "REFERENCE_CALENDAR_LAST_SESSION",
                    True,
                )
            )
            true_pit_executed = True

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
    summary_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    wdc_rows: list[dict[str, object]] = []
    signal_dates: dict[tuple[str, str], pd.DatetimeIndex] = {}
    for scenario, scenario_panel, signal_rule, is_true_pit in scenario_frames:
        for period, (start, end) in periods.items():
            result = _evaluate_period(
                scenario_panel,
                params,
                settings,
                start=start,
                end=end,
                reference_dates=reference_dates,
                causal=signal_rule == "REFERENCE_CALENDAR_LAST_SESSION",
                delisting_events=delisting_events,
            )
            strategy = result["strategy"]
            benchmark = result["benchmark"]
            targets = result["targets"]
            dates = pd.DatetimeIndex(result["signal_days"]["Date"].unique())
            signal_dates[(scenario, period)] = dates
            summary_rows.append(
                {
                    "Scenario": scenario,
                    "Period": period,
                    "IsTruePIT": is_true_pit,
                    "SignalRule": signal_rule,
                    "StartDate": strategy.summary.start_date,
                    "EndDate": strategy.summary.end_date,
                    "StrategyROI": strategy.summary.roi_percent,
                    "BenchmarkROI": benchmark.summary.roi_percent,
                    "ExcessROI": (
                        strategy.summary.roi_percent
                        - benchmark.summary.roi_percent
                    ),
                    "CAGR": strategy.summary.cagr_percent,
                    "MaxDrawdown": strategy.summary.max_drawdown_percent,
                    "Sharpe": strategy.summary.sharpe_ratio,
                    "SignalCount": len(dates),
                    "TickerTrades": strategy.summary.ticker_trades,
                    "DelistingEventsApplied": (
                        len(strategy.delistings)
                        if strategy.delistings is not None
                        else 0
                    ),
                }
            )
            equity = strategy.daily.copy()
            equity.insert(0, "Scenario", scenario)
            equity.insert(1, "Period", period)
            equity_frames.append(equity)
            wdc_rows.append(
                _wdc_period_audit(
                    scenario_panel,
                    targets,
                    scenario=scenario,
                    period=period,
                    start=start,
                    end=end,
                    market_dates=reference_dates,
                )
            )

    summary = _add_baseline_comparison(pd.DataFrame(summary_rows))
    requested_comparison = build_requested_comparison(summary)
    wdc_audit = pd.DataFrame(wdc_rows)
    signal_comparison = _signal_date_comparison(
        signal_dates,
        periods,
    )
    readiness = pd.DataFrame(
        [
            {
                "GeneratedAt": datetime.now(UTC).isoformat(),
                "IdentificationStatus": (
                    "TRUE_PIT_EXECUTED"
                    if true_pit_executed
                    else "NOT_IDENTIFIED_LOCAL_PROXY_ONLY"
                ),
                "TruePITExecuted": true_pit_executed,
                "LocalConfiguredEquities": len(members),
                "CurrentSnapshotEquities": len(current_tickers),
                "LocalPanelTickers": int(panel["Ticker"].nunique()),
                "PITMembershipProvided": pit_membership_path is not None,
                "DelistingFileProvided": delisting_events_path is not None,
                "MissingPITPanelTickers": len(missing_pit_tickers),
                "FinancialDataIsPointInTime": False,
                "FinancialLimitation": (
                    "Macrotrends current restated values keyed by fiscal "
                    "quarter-end; 45-day lag is an approximation"
                ),
                "LocalProxyLimitation": (
                    "annual ranking only among locally stored current "
                    "survivors; no delisted securities"
                ),
            }
        ]
    )
    delisting_policy = _delisting_policy_table()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "pit_diagnostic"
            / f"{timestamp}_v6_b_task2"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    scenario_summary_csv = destination / "scenario_summary.csv"
    local_universe_audit_csv = destination / "local_universe_audit.csv"
    requested_comparison_csv = destination / "requested_comparison.csv"
    annual_membership_csv = destination / "annual_membership.csv"
    wdc_audit_csv = destination / "wdc_audit.csv"
    signal_date_comparison_csv = (
        destination / "signal_date_comparison.csv"
    )
    scenario_equity_csv = destination / "scenario_equity.csv"
    data_readiness_csv = destination / "data_readiness.csv"
    delisting_policy_csv = destination / "delisting_policy.csv"
    manifest_json = destination / "manifest.json"
    atomic_to_csv(
        local_universe_audit, local_universe_audit_csv, index=False
    )
    atomic_to_csv(
        requested_comparison, requested_comparison_csv, index=False
    )
    atomic_to_csv(summary, scenario_summary_csv, index=False)
    atomic_to_csv(
        annual_membership,
        annual_membership_csv,
        index=False,
    )
    atomic_to_csv(wdc_audit, wdc_audit_csv, index=False)
    atomic_to_csv(
        signal_comparison,
        signal_date_comparison_csv,
        index=False,
    )
    atomic_to_csv(
        pd.concat(equity_frames, ignore_index=True),
        scenario_equity_csv,
        index=False,
    )
    atomic_to_csv(readiness, data_readiness_csv, index=False)
    atomic_to_csv(delisting_policy, delisting_policy_csv, index=False)
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_status": "POST_HOC_DIAGNOSTIC_PASS",
            "validation_is_fresh": False,
            "task": "Task 2 point-in-time universe diagnostic",
            "true_pit_executed": true_pit_executed,
            "missing_pit_panel_tickers": missing_pit_tickers,
            "current_snapshot_ticker_count": len(current_tickers),
            "combined_local_ticker_count": int(panel["Ticker"].nunique()),
            "raw_price_folder_count": len(local_universe_audit),
            "clean_individual_equity_count": int(
                local_universe_audit["IncludedIndividualEquity"].sum()
            ),
            "signal_rule_baseline": (
                "coverage-max day selected with full-week hindsight"
            ),
            "signal_rule_diagnostic": (
                "last exchange-calendar session in W-FRI week"
            ),
            "delisting_timing": (
                "forced settlement before regular-open rebalance on the "
                "first market session on/after EffectiveDate"
            ),
            "delisting_cost": (
                "no additional turnover cost; observed/imputed delisting "
                "return is treated as net settlement"
            ),
            "financial_point_in_time": False,
            "survivorship_warning": (
                "Local annual proxy is not a point-in-time universe and "
                "cannot identify the magnitude of survivorship bias"
            ),
            "frozen_reported_baseline": FROZEN_REPORTED_BASELINE,
            "baseline_reproduction_warning": (
                "The baseline is copied from the pre-registered prompt. "
                "The refreshed-data rerun is retained separately in "
                "scenario_summary.csv and must not replace the denominator."
            ),
            "discovery_rows": len(discovery_audit),
            "data_audit_rows": len(data_audit),
            "output_dir": str(destination),
        },
        manifest_json,
    )
    return PitDiagnosticArtifacts(
        output_dir=destination,
        local_universe_audit_csv=local_universe_audit_csv,
        requested_comparison_csv=requested_comparison_csv,
        scenario_summary_csv=scenario_summary_csv,
        annual_membership_csv=annual_membership_csv,
        wdc_audit_csv=wdc_audit_csv,
        signal_date_comparison_csv=signal_date_comparison_csv,
        scenario_equity_csv=scenario_equity_csv,
        data_readiness_csv=data_readiness_csv,
        delisting_policy_csv=delisting_policy_csv,
        manifest_json=manifest_json,
    )


def build_requested_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    proxy = summary.loc[
        summary["Scenario"].eq("LOCAL_ANNUAL_SURVIVOR_PROXY_CAUSAL")
    ].set_index("Period")
    labels = {
        "TRAIN_2020_2024": "2020-2024",
        "2025": "2025",
        "2026": "2026 YTD",
    }
    rows: list[dict[str, object]] = []
    for period, baseline in FROZEN_REPORTED_BASELINE.items():
        rows.append(
            {
                "Scenario": "BASELINE_CURRENT_200_RETROSPECTIVE",
                "Period": labels[period],
                **baseline,
                "BaselineExcessRetentionPct": 100.0,
            }
        )
        dynamic = proxy.loc[period]
        rows.append(
            {
                "Scenario": "LOCAL_ANNUAL_SURVIVOR_PROXY_CAUSAL",
                "Period": labels[period],
                "StrategyROI": dynamic["StrategyROI"],
                "BenchmarkROI": dynamic["BenchmarkROI"],
                "ExcessROI": dynamic["ExcessROI"],
                "BaselineExcessRetentionPct": (
                    dynamic["ExcessROI"] / baseline["ExcessROI"] * 100
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_local_price_universe(
    paths: ProjectPaths,
    config_paths: Iterable[Path],
) -> pd.DataFrame:
    """Classify every raw-price folder before building historical snapshots."""

    by_company: dict[str, str] = {}
    for path in config_paths:
        for ticker, config in load_tickers(path).items():
            by_company[config.display_name] = ticker.upper()
    explicit_exclusions = {
        "BITCOIN": ("EXCLUDED_CRYPTO", "CRYPTOCURRENCY"),
        "XRP": ("EXCLUDED_CRYPTO", "CRYPTOCURRENCY"),
        "S&P500": ("EXCLUDED_INDEX_OR_ETF", "INDEX_OR_ETF"),
        "SPY": ("EMPTY_FOLDER", "INDEX_OR_ETF"),
    }
    rows: list[dict[str, object]] = []
    for folder in sorted(paths.raw_prices.iterdir(), key=lambda value: value.name):
        if not folder.is_dir():
            continue
        csv_files = sorted(folder.glob("*.csv"))
        key = folder.name.upper()
        reason, security_type = ("", "INDIVIDUAL_EQUITY")
        if key in explicit_exclusions:
            reason, security_type = explicit_exclusions[key]
        elif not csv_files:
            reason, security_type = ("EMPTY_FOLDER", "UNKNOWN")
        ticker = by_company.get(folder.name, "")
        if not ticker and not reason:
            reason = "NO_TICKER_MAPPING"
            security_type = "UNKNOWN"
        rows.append(
            {
                "Folder": folder.name,
                "Ticker": ticker,
                "CsvCount": len(csv_files),
                "SecurityType": security_type,
                "ExclusionReason": reason,
                "IncludedIndividualEquity": not bool(reason),
            }
        )
    return pd.DataFrame(rows)


def _discover_combined_universe(
    paths: ProjectPaths,
    config_paths: Iterable[Path],
) -> tuple[list[UniverseMember], pd.DataFrame]:
    by_ticker: dict[str, UniverseMember] = {}
    audits: list[pd.DataFrame] = []
    for path in config_paths:
        members, audit = discover_universe(paths, path)
        by_ticker.update({member.ticker: member for member in members})
        current = audit.copy()
        current.insert(0, "TickerConfig", str(path))
        audits.append(current)
    return (
        sorted(by_ticker.values(), key=lambda member: member.ticker),
        pd.concat(audits, ignore_index=True),
    )


def _evaluate_period(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    *,
    start: str,
    end: str,
    reference_dates: pd.DatetimeIndex,
    causal: bool,
    delisting_events: pd.DataFrame | None,
) -> dict[str, object]:
    if causal:
        signal_days = causal_signal_day_panel(
            panel,
            start=start,
            end=end,
            reference_market_dates=reference_dates,
            rebalance_weekday=settings.rebalance_weekday,
        )
    else:
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
    period_delistings = (
        delisting_events.loc[
            delisting_events["Ticker"].isin(panel["Ticker"].unique())
        ]
        if delisting_events is not None
        else None
    )
    strategy = run_portfolio_backtest(
        panel,
        targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        delisting_events=period_delistings,
    )
    benchmark = run_portfolio_backtest(
        panel,
        generate_equal_weight_targets(signal_days),
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        delisting_events=period_delistings,
    )
    return {
        "strategy": strategy,
        "benchmark": benchmark,
        "targets": targets,
        "signal_days": signal_days,
    }


def _add_baseline_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary.loc[
        summary["Scenario"].eq(
            "BASELINE_CURRENT_SNAPSHOT_NONCAUSAL"
        ),
        ["Period", "ExcessROI"],
    ].rename(columns={"ExcessROI": "BaselineExcessROI"})
    result = summary.merge(baseline, on="Period", how="left")
    result["ExcessLossVsBaseline"] = (
        result["BaselineExcessROI"] - result["ExcessROI"]
    )
    result["ExcessRetentionVsBaselinePct"] = np.where(
        result["BaselineExcessROI"].abs().gt(1e-12),
        result["ExcessROI"] / result["BaselineExcessROI"] * 100,
        np.nan,
    )
    result["AtLeastHalfTrainingExcessLost"] = (
        result["Period"].eq("TRAIN_2020_2024")
        & result["ExcessRetentionVsBaselinePct"].lt(50)
    )
    result["SurvivorshipBiasPrimarySourceVerdict"] = np.where(
        ~result["IsTruePIT"],
        "NOT_IDENTIFIED_NON_PIT_SCENARIO",
        np.where(
            result["AtLeastHalfTrainingExcessLost"],
            "YES_HALF_OR_MORE_TRAIN_EXCESS_LOST",
            "NO_LESS_THAN_HALF_TRAIN_EXCESS_LOST",
        ),
    )
    return result


def _wdc_period_audit(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    scenario: str,
    period: str,
    start: str,
    end: str,
    market_dates: pd.DatetimeIndex,
) -> dict[str, object]:
    wdc_panel = panel.loc[
        panel["Ticker"].eq("WDC") & panel["Date"].between(start, end)
    ]
    wdc_targets = targets.loc[
        targets["Ticker"].eq("WDC")
        & targets["Date"].between(start, end)
    ].sort_values("Date")
    selected = wdc_targets.loc[
        wdc_targets.get(
            "ModelSelected",
            pd.Series(False, index=wdc_targets.index),
        ).fillna(False)
    ]
    buys = wdc_targets.loc[
        wdc_targets.get(
            "TradeAction",
            pd.Series("", index=wdc_targets.index),
        ).eq("BUY")
    ]
    first_buy = (
        pd.Timestamp(buys["Date"].iloc[0]) if not buys.empty else pd.NaT
    )
    first_execution = pd.NaT
    if pd.notna(first_buy):
        position = int(market_dates.searchsorted(first_buy, side="right"))
        if position < len(market_dates):
            first_execution = pd.Timestamp(market_dates[position])
    universe_rank = (
        pd.to_numeric(wdc_panel["UniverseRank"], errors="coerce")
        if "UniverseRank" in wdc_panel
        else pd.Series(np.nan, index=wdc_panel.index)
    )
    return {
        "Scenario": scenario,
        "Period": period,
        "WDCUniverseMemberAny": bool(
            wdc_panel.get(
                "UniverseMember",
                pd.Series(True, index=wdc_panel.index),
            ).fillna(False).any()
        ),
        "WDCUniverseRankMin": (
            universe_rank.min()
            if not wdc_panel.empty
            else np.nan
        ),
        "WDCSelectedSignalCount": len(selected),
        "WDCFirstSelectedSignal": (
            selected["Date"].iloc[0] if not selected.empty else pd.NaT
        ),
        "WDCFirstBuySignal": first_buy,
        "WDCFirstExecutionDate": first_execution,
    }


def _signal_date_comparison(
    signal_dates: dict[tuple[str, str], pd.DatetimeIndex],
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_name = "BASELINE_CURRENT_SNAPSHOT_NONCAUSAL"
    for (scenario, period), dates in signal_dates.items():
        baseline = signal_dates[(baseline_name, period)]
        added = dates.difference(baseline)
        removed = baseline.difference(dates)
        rows.append(
            {
                "Scenario": scenario,
                "Period": period,
                "Start": periods[period][0],
                "End": periods[period][1],
                "BaselineSignalCount": len(baseline),
                "ScenarioSignalCount": len(dates),
                "AddedVsBaselineCount": len(added),
                "RemovedVsBaselineCount": len(removed),
                "SameSignalDates": len(added) == 0 and len(removed) == 0,
                "AddedDates": ",".join(
                    str(pd.Timestamp(date).date()) for date in added
                ),
                "RemovedDates": ",".join(
                    str(pd.Timestamp(date).date()) for date in removed
                ),
            }
        )
    return pd.DataFrame(rows)


def _delisting_policy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Case": "OBSERVED_RETURN",
                "AppliedReturn": "vendor DelistingReturn",
                "Use": "primary result",
            },
            {
                "Case": "BANKRUPTCY_LIQUIDATION_WORTHLESS",
                "AppliedReturn": "-1.00",
                "Use": "forced total loss",
            },
            {
                "Case": "MISSING_TOTAL_LOSS_STRESS",
                "AppliedReturn": "-1.00",
                "Use": "conservative sensitivity",
            },
            {
                "Case": "MISSING_EXCHANGE_HAIRCUT",
                "AppliedReturn": "NASDAQ -0.55; other -0.30",
                "Use": "explicit imputation sensitivity, not observed fact",
            },
            {
                "Case": "MISSING_ZERO_RETURN",
                "AppliedReturn": "0.00",
                "Use": "optimistic upper-bound sensitivity",
            },
        ]
    )


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
