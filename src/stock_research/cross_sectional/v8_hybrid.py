from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths

from .config import ResearchSettings
from .data import discover_universe
from .pit_validation import (
    apply_membership_to_panel,
    causal_signal_day_panel,
)
from .portfolio import PortfolioResult, run_portfolio_backtest
from .v7_pit_evaluation import (
    build_v7_source_panel,
    load_ready_tickers,
    normalize_change_membership,
)
from .v7_slot_sweep import spy_buy_and_hold
from .v7_technical import (
    add_v7_technical_factors,
    add_v7_technical_observations,
)
from .v7_trade_report import build_execution_ledger
from .winner_attribution import summarize_ticker_contributions


@dataclass(frozen=True)
class V8HybridConfig:
    core_slots: int = 7
    core_total_weight: float = 0.70
    core_market_cap_rank: int = 100
    core_minimum_market_cap: float = 10_000_000_000
    core_minimum_hold_quarters: int = 4
    inflection_slots: int = 2
    inflection_scout_weight: float = 0.05
    inflection_confirm_weight: float = 0.10
    inflection_max_weight: float = 0.15
    inflection_confirm_weeks: int = 4
    inflection_max_weeks: int = 8
    inflection_minimum_hold_weeks: int = 13
    inflection_market_cap_rank: int = 300
    inflection_minimum_market_cap: float = 2_000_000_000

    @property
    def core_position_weight(self) -> float:
        return self.core_total_weight / self.core_slots

    def __post_init__(self) -> None:
        if self.core_slots < 1 or self.inflection_slots < 1:
            raise ValueError("V8 sleeves require positive slot counts")
        if not 0 < self.core_total_weight < 1:
            raise ValueError("core_total_weight must be between zero and one")
        maximum = (
            self.core_total_weight
            + self.inflection_slots * self.inflection_max_weight
        )
        if maximum > 1 + 1e-9:
            raise ValueError("V8 maximum target weights exceed 100%")


@dataclass(frozen=True)
class V8Scenario:
    label: str
    excluded_tickers: tuple[str, ...]
    result: PortfolioResult
    targets: pd.DataFrame
    execution_ledger: pd.DataFrame
    trade_events: pd.DataFrame
    positions: pd.DataFrame
    contributions: pd.DataFrame


@dataclass(frozen=True)
class V8HybridArtifacts:
    output_dir: Path
    report_html: Path
    summary_csv: Path
    period_summary_csv: Path
    trade_events_csv: Path
    position_ledger_csv: Path
    execution_ledger_csv: Path
    contributions_csv: Path
    targets_csv: Path
    equity_csv: Path
    data_audit_csv: Path
    manifest_json: Path


def add_v8_scores(
    panel: pd.DataFrame,
    config: V8HybridConfig,
) -> pd.DataFrame:
    """Add fixed V8 core and fundamental-inflection scores."""

    frame = panel.copy()
    eligible = frame["Eligible"].fillna(False)
    # Macrotrends expresses historical shares outstanding in millions.
    shares = pd.to_numeric(frame.get("Shares"), errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    frame["MarketCap"] = close * shares * 1_000_000
    frame["MarketCapRank"] = (
        frame["MarketCap"]
        .where(eligible)
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )

    acceleration_inputs = {
        "EpsTtmGrowthYoY": 0.25,
        "EpsTtmGrowthAcceleration": 0.15,
        "EbitdaTtmGrowthYoY": 0.25,
        "EbitdaTtmGrowthAcceleration": 0.15,
        "DcfPriceGrowthYoY": 0.20,
    }
    ranked_acceleration = {
        column: _centered_rank(frame, column, eligible)
        for column in acceleration_inputs
    }
    frame["FundamentalAccelerationScore"] = _weighted_available_mean(
        ranked_acceleration,
        acceleration_inputs,
    )
    frame["InflectionTechnicalScore"] = frame[
        ["MAFactor", "MACDFactor", "OBVFactor"]
    ].mean(axis=1)
    frame["CoreScore"] = (
        0.35 * pd.to_numeric(frame["GrowthFactor"], errors="coerce")
        + 0.35 * pd.to_numeric(frame["QualityFactor"], errors="coerce")
        + 0.15 * pd.to_numeric(frame["MAFactor"], errors="coerce")
        + 0.10 * pd.to_numeric(frame["MomentumFactor"], errors="coerce")
        + 0.05
        * pd.to_numeric(frame["RiskControlFactor"], errors="coerce")
    )
    frame["InflectionScore"] = (
        0.45
        * pd.to_numeric(
            frame["FundamentalAccelerationScore"],
            errors="coerce",
        )
        + 0.15 * pd.to_numeric(frame["QualityFactor"], errors="coerce")
        + 0.20
        * pd.to_numeric(
            frame["InflectionTechnicalScore"],
            errors="coerce",
        )
        + 0.15 * pd.to_numeric(frame["MomentumFactor"], errors="coerce")
        + 0.05
        * pd.to_numeric(frame["RiskControlFactor"], errors="coerce")
    )
    frame["CoreQualified"] = (
        eligible
        & frame["MarketCapRank"].le(config.core_market_cap_rank)
        & frame["MarketCap"].ge(config.core_minimum_market_cap)
        & frame["GrowthFactor"].ge(0)
        & frame["QualityFactor"].ge(0)
        & frame["Trend200"].ge(-0.10)
    )
    positive_growth = (
        frame[
            [
                "EpsTtmGrowthYoY",
                "EbitdaTtmGrowthYoY",
                "DcfPriceGrowthYoY",
            ]
        ]
        .gt(0.15)
        .any(axis=1)
    )
    frame["InflectionQualified"] = (
        eligible
        & frame["MarketCapRank"].le(config.inflection_market_cap_rank)
        & frame["MarketCap"].ge(config.inflection_minimum_market_cap)
        & frame["FundamentalAccelerationScore"].ge(0.10)
        & frame["Trend200"].ge(0)
        & frame["Return126"].ge(0.10)
        & positive_growth
    )
    frame["CoreRank"] = (
        frame["CoreScore"]
        .where(frame["CoreQualified"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    frame["InflectionRank"] = (
        frame["InflectionScore"]
        .where(frame["InflectionQualified"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    return frame


def generate_v8_targets(
    signal_days: pd.DataFrame,
    config: V8HybridConfig,
) -> pd.DataFrame:
    """Create sparse trade-date targets for the fixed V8 hybrid rules."""

    frame = signal_days.sort_values(["Date", "Ticker"]).copy()
    signal_dates = pd.DatetimeIndex(
        frame["Date"].drop_duplicates().sort_values()
    )
    quarter_review_dates = set(
        pd.Series(signal_dates, index=signal_dates)
        .groupby(signal_dates.to_period("Q"))
        .max()
        .tolist()
    )
    core: dict[str, dict[str, object]] = {}
    inflection: dict[str, dict[str, object]] = {}
    last_weights: dict[str, float] = {}
    records: list[pd.DataFrame] = []

    for date, group in frame.groupby("Date", sort=True):
        date = pd.Timestamp(date)
        indexed = group.set_index("Ticker", drop=False)
        actions: dict[str, str] = {}
        reasons: dict[str, str] = {}
        exit_reasons: dict[str, str] = {}
        prior_sleeves = {
            **{ticker: "CORE" for ticker in core},
            **{ticker: "INFLECTION" for ticker in inflection},
        }

        core_review = not core or date in quarter_review_dates
        if core_review:
            for ticker in list(core):
                state = core[ticker]
                state["QuartersHeld"] = int(
                    state.get("QuartersHeld", 0)
                ) + 1
                row = _ticker_row(indexed, ticker)
                exit_reason = _core_exit_reason(row, state, config)
                if exit_reason:
                    core.pop(ticker)
                    actions[ticker] = "SELL"
                    exit_reasons[ticker] = exit_reason
                    reasons[ticker] = _core_sell_reason(
                        row,
                        exit_reason,
                    )
            candidates = group.loc[group["CoreQualified"]].sort_values(
                ["CoreRank", "Ticker"]
            )
            for row in candidates.itertuples(index=False):
                ticker = str(row.Ticker)
                if len(core) >= config.core_slots:
                    break
                if ticker in core or ticker in inflection:
                    continue
                core[ticker] = {
                    "EntryDate": date,
                    "QuartersHeld": 1,
                }
                actions[ticker] = "BUY"
                reasons[ticker] = _core_buy_reason(pd.Series(row._asdict()))

        for ticker in list(inflection):
            state = inflection[ticker]
            state["WeeksHeld"] = int(state.get("WeeksHeld", 0)) + 1
            row = _ticker_row(indexed, ticker)
            qualified = _row_bool(row, "InflectionQualified")
            state["ConsecutiveConfirmations"] = (
                int(state.get("ConsecutiveConfirmations", 0)) + 1
                if qualified
                else 0
            )
            exit_reason = _inflection_exit_reason(row, state, config)
            if exit_reason:
                inflection.pop(ticker)
                actions[ticker] = "SELL"
                exit_reasons[ticker] = exit_reason
                reasons[ticker] = _inflection_sell_reason(
                    row,
                    exit_reason,
                )
                continue
            old_stage = int(state.get("Stage", 1))
            confirmations = int(state["ConsecutiveConfirmations"])
            new_stage = old_stage
            if confirmations >= config.inflection_max_weeks:
                new_stage = 3
            elif confirmations >= config.inflection_confirm_weeks:
                new_stage = max(new_stage, 2)
            if new_stage > old_stage:
                state["Stage"] = new_stage
                actions[ticker] = "SCALE_UP"
                reasons[ticker] = _scale_reason(
                    row,
                    new_stage,
                    confirmations,
                )

        candidates = group.loc[group["InflectionQualified"]].sort_values(
            ["InflectionRank", "Ticker"]
        )
        for row in candidates.itertuples(index=False):
            ticker = str(row.Ticker)
            if len(inflection) >= config.inflection_slots:
                break
            if ticker in inflection or ticker in core:
                continue
            inflection[ticker] = {
                "EntryDate": date,
                "WeeksHeld": 1,
                "ConsecutiveConfirmations": 1,
                "Stage": 1,
            }
            actions[ticker] = "BUY"
            reasons[ticker] = _inflection_buy_reason(
                pd.Series(row._asdict())
            )

        weights = {
            ticker: config.core_position_weight for ticker in core
        }
        stage_weights = {
            1: config.inflection_scout_weight,
            2: config.inflection_confirm_weight,
            3: config.inflection_max_weight,
        }
        weights.update(
            {
                ticker: stage_weights[int(state["Stage"])]
                for ticker, state in inflection.items()
            }
        )
        if sum(weights.values()) > 1 + 1e-9:
            raise RuntimeError("V8 targets exceed 100%")
        changed = set(weights) != set(last_weights) or any(
            abs(weight - last_weights.get(ticker, 0.0)) > 1e-12
            for ticker, weight in weights.items()
        )
        if not changed:
            continue

        tickers = sorted(set(weights) | set(last_weights))
        output_rows: list[dict[str, object]] = []
        for ticker in tickers:
            row = _ticker_row(indexed, ticker)
            base = row.to_dict() if row is not None else {
                "Date": date,
                "Ticker": ticker,
            }
            sleeve = (
                "CORE"
                if ticker in core
                else "INFLECTION"
                if ticker in inflection
                else prior_sleeves.get(ticker, "UNKNOWN")
            )
            action = actions.get(ticker, "HOLD")
            base.update(
                {
                    "Date": date,
                    "SignalDate": date,
                    "Ticker": ticker,
                    "Sleeve": sleeve,
                    "TargetWeight": weights.get(ticker, 0.0),
                    "TradeAction": action,
                    "Reason": reasons.get(
                        ticker,
                        "기존 장기 투자 논리가 유지되어 보유.",
                    ),
                    "ExitReason": exit_reasons.get(ticker, ""),
                    "CoreQuartersHeld": (
                        core.get(ticker, {}).get("QuartersHeld")
                    ),
                    "InflectionWeeksHeld": (
                        inflection.get(ticker, {}).get("WeeksHeld")
                    ),
                    "InflectionStage": (
                        inflection.get(ticker, {}).get("Stage")
                    ),
                }
            )
            output_rows.append(base)
        records.append(pd.DataFrame(output_rows))
        last_weights = weights

    if not records:
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "TargetWeight",
                "TradeAction",
            ]
        )
    return pd.concat(records, ignore_index=True)


def run_v8_scenario(
    panel: pd.DataFrame,
    settings: ResearchSettings,
    config: V8HybridConfig,
    *,
    start: str,
    end: str,
    reference_dates: pd.DatetimeIndex,
    label: str,
    excluded_tickers: tuple[str, ...] = (),
) -> V8Scenario:
    scenario_panel = panel.copy()
    if excluded_tickers:
        scenario_panel["Eligible"] &= ~scenario_panel["Ticker"].isin(
            excluded_tickers
        )
    scored = add_v8_scores(scenario_panel, config)
    signal_days = causal_signal_day_panel(
        scored,
        start=start,
        end=end,
        reference_market_dates=reference_dates,
        rebalance_weekday=settings.rebalance_weekday,
    )
    targets = generate_v8_targets(signal_days, config)
    result = run_portfolio_backtest(
        scored,
        targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        record_attribution=True,
    )
    executions, _ = build_execution_ledger(
        scored,
        targets,
        result,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    events = build_v8_trade_events(targets, executions)
    positions = build_v8_position_ledger(
        events,
        executions,
        result,
        scored,
        end_date=end,
    )
    contributions = summarize_ticker_contributions(
        result,
        {label: (start, end)},
    )
    return V8Scenario(
        label=label,
        excluded_tickers=excluded_tickers,
        result=result,
        targets=targets,
        execution_ledger=executions,
        trade_events=events,
        positions=positions,
        contributions=contributions,
    )


def run_v8_hybrid(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    spy_path: str | Path,
    frozen_strategy_path: str | Path,
    config: V8HybridConfig | None = None,
    expected_ready_count: int = 575,
    v7_slot_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> V8HybridArtifacts:
    config = config or V8HybridConfig()
    ready_tickers = load_ready_tickers(
        backfill_status_path,
        expected_count=expected_ready_count,
    )
    members, _ = discover_universe(paths, ticker_config_path)
    missing = sorted(
        ready_tickers - {member.ticker for member in members}
    )
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
    latest_end = str(pd.Timestamp(technical["Date"].max()).date())
    reference_dates = pd.DatetimeIndex(
        technical["Date"].drop_duplicates().sort_values()
    )
    base = run_v8_scenario(
        technical,
        settings,
        config,
        start=settings.train_start,
        end=latest_end,
        reference_dates=reference_dates,
        label="V8_HYBRID",
    )
    top_winner = str(
        base.contributions.sort_values(
            "NetPnL",
            ascending=False,
        ).iloc[0]["Ticker"]
    )
    ex_wdc = run_v8_scenario(
        technical,
        settings,
        config,
        start=settings.train_start,
        end=latest_end,
        reference_dates=reference_dates,
        label="V8_EX_WDC",
        excluded_tickers=("WDC",),
    )
    if top_winner == "WDC":
        ex_top = V8Scenario(
            label="V8_EX_TOP_WINNER_WDC",
            excluded_tickers=ex_wdc.excluded_tickers,
            result=ex_wdc.result,
            targets=ex_wdc.targets,
            execution_ledger=ex_wdc.execution_ledger,
            trade_events=ex_wdc.trade_events,
            positions=ex_wdc.positions,
            contributions=ex_wdc.contributions.assign(
                Period="V8_EX_TOP_WINNER_WDC"
            ),
        )
    else:
        ex_top = run_v8_scenario(
            technical,
            settings,
            config,
            start=settings.train_start,
            end=latest_end,
            reference_dates=reference_dates,
            label=f"V8_EX_TOP_WINNER_{top_winner}",
            excluded_tickers=(top_winner,),
        )
    scenarios = (base, ex_wdc, ex_top)

    spy = load_sp500_proxy(spy_path)
    spy_summary, spy_equity = spy_buy_and_hold(
        spy,
        start=settings.train_start,
        end=latest_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    summary = build_v8_summary(
        scenarios,
        spy_summary,
        v7_slot_summary_path=v7_slot_summary_path,
    )
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
        "FULL_2020_2026": (settings.train_start, latest_end),
    }
    period_summary = build_period_summary(
        scenarios,
        spy_equity,
        periods,
        initial_capital=settings.initial_capital,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v8_hybrid"
            / f"{timestamp}_fixed_core_inflection"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "report_html": destination / "v8_hybrid_report.html",
        "summary_csv": destination / "summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "trade_events_csv": destination / "trade_events.csv",
        "position_ledger_csv": destination / "position_ledger.csv",
        "execution_ledger_csv": destination / "execution_ledger.csv",
        "contributions_csv": destination / "contributions.csv",
        "targets_csv": destination / "targets.csv",
        "equity_csv": destination / "equity.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "manifest_json": destination / "manifest.json",
    }
    scenario_frames = {
        "trade_events_csv": [],
        "position_ledger_csv": [],
        "execution_ledger_csv": [],
        "contributions_csv": [],
        "targets_csv": [],
        "equity_csv": [],
    }
    for scenario in scenarios:
        for key, frame in (
            ("trade_events_csv", scenario.trade_events),
            ("position_ledger_csv", scenario.positions),
            ("execution_ledger_csv", scenario.execution_ledger),
            ("contributions_csv", scenario.contributions),
            ("targets_csv", scenario.targets),
            ("equity_csv", scenario.result.daily),
        ):
            labeled = frame.copy()
            labeled.insert(0, "Scenario", scenario.label)
            scenario_frames[key].append(labeled)
    spy_labeled = spy_equity.copy()
    spy_labeled.insert(0, "Scenario", "SPY_BUY_HOLD")
    scenario_frames["equity_csv"].append(spy_labeled)
    atomic_to_csv(summary, outputs["summary_csv"], index=False)
    atomic_to_csv(
        period_summary,
        outputs["period_summary_csv"],
        index=False,
    )
    for key, frames in scenario_frames.items():
        atomic_to_csv(
            pd.concat(frames, ignore_index=True),
            outputs[key],
            index=False,
        )
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    _atomic_text(
        outputs["report_html"],
        render_v8_report(
            summary,
            period_summary,
            base,
            top_winner=top_winner,
            config=config,
        ),
    )
    _atomic_json(
        outputs["manifest_json"],
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "Fixed-rule V8 hybrid core plus inflection prototype",
            "model_status": "POST_HOC_PROTOTYPE",
            "validation_is_fresh": False,
            "optimization_performed": False,
            "config": asdict(config),
            "core_score": (
                "35% Growth + 35% Quality + 15% MA + 10% Momentum "
                "+ 5% RiskControl"
            ),
            "inflection_score": (
                "45% fundamental acceleration + 15% Quality + "
                "20% MA/MACD/OBV + 15% Momentum + 5% RiskControl"
            ),
            "execution_rule": (
                "causal final W-FRI session close signal, next session open"
            ),
            "transaction_cost_bps": settings.transaction_cost_bps,
            "top_winner": top_winner,
            "financial_point_in_time": False,
            "frozen_v6_v7_unchanged": True,
            "frozen_strategy_sha256": _sha256(
                Path(frozen_strategy_path)
            ),
            "backfill_status_sha256": _sha256(
                Path(backfill_status_path)
            ),
            "membership_sha256": _sha256(Path(membership_path)),
            "caveats": [
                (
                    "Macrotrends financials are restated current history, "
                    "not true point-in-time statements."
                ),
                (
                    "The 575-name historical S&P proxy omits unavailable "
                    "acquired and delisted constituents."
                ),
                (
                    "All 2025 and 2026 outcomes were already observed; "
                    "this is not a fresh out-of-sample validation."
                ),
                (
                    "Industry-theme, management-guidance, estimate-revision, "
                    "and first-seen filing data are not yet available, so "
                    "this is a numerical proxy rather than a complete "
                    "Druckenmiller-style process."
                ),
                (
                    "WDC and top-winner exclusions are diagnostics and were "
                    "not used to tune the fixed rules."
                ),
            ],
            "outputs": {
                name: str(path) for name, path in outputs.items()
            },
        },
    )
    return V8HybridArtifacts(output_dir=destination, **outputs)


def build_v8_trade_events(
    targets: pd.DataFrame,
    executions: pd.DataFrame,
) -> pd.DataFrame:
    actions = targets.loc[
        targets["TradeAction"].isin(["BUY", "SELL", "SCALE_UP"])
    ].copy()
    execution_actions = executions.loc[
        executions["SignalTradeAction"].isin(
            ["BUY", "SELL", "SCALE_UP"]
        )
    ].copy()
    execution_columns = [
        "SignalDate",
        "ExecutionDate",
        "Ticker",
        "ExecutionSide",
        "ExecutionType",
        "PreTradeEquity",
        "ExecutionPrice",
        "SharesTraded",
        "SharesAfter",
        "GrossTradeAmount",
        "NotionalAfter",
        "AllocatedTransactionCost",
    ]
    result = actions.merge(
        execution_actions[execution_columns],
        left_on=["Date", "Ticker"],
        right_on=["SignalDate", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    preferred = [
        "Date",
        "ExecutionDate",
        "Ticker",
        "Company",
        "Sleeve",
        "TradeAction",
        "Reason",
        "ExitReason",
        "TargetWeight",
        "PreTradeEquity",
        "ExecutionPrice",
        "SharesTraded",
        "SharesAfter",
        "GrossTradeAmount",
        "NotionalAfter",
        "AllocatedTransactionCost",
        "MarketCap",
        "MarketCapRank",
        "CoreScore",
        "CoreRank",
        "InflectionScore",
        "InflectionRank",
        "FundamentalAccelerationScore",
        "InflectionTechnicalScore",
        "EpsTtmGrowthYoY",
        "EpsTtmGrowthAcceleration",
        "EbitdaTtmGrowthYoY",
        "EbitdaTtmGrowthAcceleration",
        "DcfPriceGrowthYoY",
        "DcfUpside",
        "PeTtm",
        "EvEbitdaTtm",
        "Return126",
        "Trend200",
    ]
    return result.loc[
        :, [column for column in preferred if column in result]
    ].sort_values(["ExecutionDate", "Ticker"]).reset_index(drop=True)


def build_v8_position_ledger(
    events: pd.DataFrame,
    executions: pd.DataFrame,
    result: PortfolioResult,
    panel: pd.DataFrame,
    *,
    end_date: str,
) -> pd.DataFrame:
    attribution = result.attribution
    if attribution is None:
        raise ValueError("V8 position ledger requires ticker attribution")
    attribution = attribution.copy()
    attribution["Date"] = pd.to_datetime(attribution["Date"])
    latest = (
        panel.loc[panel["Date"].le(pd.Timestamp(end_date))]
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .set_index("Ticker")
    )
    execution_groups = {
        ticker: group.sort_values("ExecutionDate")
        for ticker, group in executions.groupby("Ticker")
    }
    open_entries: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for event in events.sort_values(
        ["ExecutionDate", "TradeAction", "Ticker"]
    ).itertuples(index=False):
        values = pd.Series(event._asdict())
        ticker = str(event.Ticker)
        if event.TradeAction == "BUY":
            open_entries[ticker] = values
        elif event.TradeAction == "SELL" and ticker in open_entries:
            rows.append(
                _v8_position_record(
                    ticker,
                    open_entries.pop(ticker),
                    values,
                    execution_groups.get(ticker, pd.DataFrame()),
                    attribution,
                )
            )
    for ticker, entry in open_entries.items():
        last = latest.loc[ticker] if ticker in latest.index else pd.Series()
        rows.append(
            _v8_position_record(
                ticker,
                entry,
                None,
                execution_groups.get(ticker, pd.DataFrame()),
                attribution,
                mark_date=pd.Timestamp(end_date),
                mark_price=pd.to_numeric(
                    last.get("Close"),
                    errors="coerce",
                ),
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["EntryExecutionDate", "Ticker"]
    ).reset_index(drop=True)


def build_v8_summary(
    scenarios: tuple[V8Scenario, ...],
    spy_summary: dict[str, object],
    *,
    v7_slot_summary_path: str | Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        summary = scenario.result.summary
        closed = scenario.positions.loc[
            scenario.positions["Status"].eq("CLOSED")
        ]
        rows.append(
            {
                "Strategy": scenario.label,
                "StartDate": summary.start_date,
                "EndDate": summary.end_date,
                "FinalValue": summary.final_value,
                "ROI": summary.roi_percent,
                "CAGR": summary.cagr_percent,
                "Sharpe": summary.sharpe_ratio,
                "MaxDrawdown": summary.max_drawdown_percent,
                "AnnualizedTurnover": summary.annualized_turnover,
                "ModelBuys": int(
                    scenario.trade_events["TradeAction"].eq("BUY").sum()
                ),
                "ModelSells": int(
                    scenario.trade_events["TradeAction"].eq("SELL").sum()
                ),
                "ScaleUps": int(
                    scenario.trade_events["TradeAction"]
                    .eq("SCALE_UP")
                    .sum()
                ),
                "DistinctTickers": int(
                    scenario.positions["Ticker"].nunique()
                ),
                "MedianClosedHoldingDays": (
                    float(closed["HoldingCalendarDays"].median())
                    if not closed.empty
                    else np.nan
                ),
                "ClosedLossRate": (
                    float(closed["NetPnL"].lt(0).mean() * 100)
                    if not closed.empty
                    else np.nan
                ),
                "ExcludedTickers": ",".join(
                    scenario.excluded_tickers
                ),
            }
        )
    rows.append(
        {
            "Strategy": "SPY_BUY_HOLD",
            "StartDate": spy_summary["StartDate"],
            "EndDate": spy_summary["EndDate"],
            "FinalValue": spy_summary["FinalValue"],
            "ROI": spy_summary["ROI"],
            "CAGR": spy_summary["CAGR"],
            "Sharpe": spy_summary["Sharpe"],
            "MaxDrawdown": spy_summary["MaxDrawdown"],
            "AnnualizedTurnover": 0.0,
            "ModelBuys": 1,
            "ModelSells": 0,
            "ScaleUps": 0,
            "DistinctTickers": 1,
            "MedianClosedHoldingDays": np.nan,
            "ClosedLossRate": np.nan,
            "ExcludedTickers": "",
        }
    )
    if v7_slot_summary_path is not None:
        v7 = pd.read_csv(v7_slot_summary_path)
        selected = v7.loc[
            v7["TopK"].eq(5)
            & v7["Period"].eq("FULL_2020_2026")
        ]
        if len(selected) == 1:
            row = selected.iloc[0]
            rows.append(
                {
                    "Strategy": "V7_3_TOP5_REFERENCE",
                    "StartDate": row["StartDate"],
                    "EndDate": row["EndDate"],
                    "FinalValue": row["FinalValue"],
                    "ROI": row["StrategyROI"],
                    "CAGR": row["CAGR"],
                    "Sharpe": row["Sharpe"],
                    "MaxDrawdown": row["MaxDrawdown"],
                    "AnnualizedTurnover": row["AnnualizedTurnover"],
                    "ModelBuys": np.nan,
                    "ModelSells": np.nan,
                    "ScaleUps": np.nan,
                    "DistinctTickers": np.nan,
                    "MedianClosedHoldingDays": 57.0,
                    "ClosedLossRate": 10.95,
                    "ExcludedTickers": "",
                }
            )
    return pd.DataFrame(rows)


def build_period_summary(
    scenarios: tuple[V8Scenario, ...],
    spy_equity: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
    *,
    initial_capital: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        for scenario in scenarios:
            rows.append(
                {
                    "Period": period,
                    "Strategy": scenario.label,
                    **_path_metrics(
                        scenario.result.daily[["Date", "Equity"]],
                        start,
                        end,
                        initial_capital=initial_capital,
                    ),
                }
            )
        rows.append(
            {
                "Period": period,
                "Strategy": "SPY_BUY_HOLD",
                **_path_metrics(
                    spy_equity[["Date", "Equity"]],
                    start,
                    end,
                    initial_capital=initial_capital,
                ),
            }
        )
    return pd.DataFrame(rows)


def render_v8_report(
    summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    base: V8Scenario,
    *,
    top_winner: str,
    config: V8HybridConfig,
) -> str:
    base_summary = summary.loc[
        summary["Strategy"].eq("V8_HYBRID")
    ].iloc[0]
    current = base.positions.loc[base.positions["Status"].eq("OPEN")][
        [
            "Ticker",
            "Sleeve",
            "EntryExecutionDate",
            "HoldingCalendarDays",
            "InitialAllocation",
            "MarkPrice",
            "PriceReturn",
            "NetPnL",
            "EntryReason",
        ]
    ].copy()
    recent = base.trade_events.tail(30)[
        [
            "ExecutionDate",
            "Ticker",
            "Sleeve",
            "TradeAction",
            "GrossTradeAmount",
            "ExecutionPrice",
            "Reason",
        ]
    ].copy()
    caveat = (
        "이 결과는 최적화하지 않은 고정 규칙의 사후 프로토타입입니다. "
        "Macrotrends 재무의 비-PIT성, 575개 생존 데이터 공백, 이미 관찰한 "
        "2025·2026 결과 때문에 독립 OOS 성과가 아닙니다."
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V8 Hybrid Prototype</title><style>
body{{font-family:"Segoe UI","Malgun Gothic",sans-serif;background:#0e1117;
color:#e7eaf0;margin:0}}main{{max-width:1450px;margin:auto;padding:28px}}
h1{{margin-bottom:6px}}h2{{margin-top:30px}}.note{{color:#aab3c2;line-height:1.6}}
.warning{{background:#2c2414;border:1px solid #705b28;padding:14px;border-radius:10px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
.card,.panel{{background:#171c25;border:1px solid #2c3340;border-radius:11px;padding:15px}}
.label{{color:#98a2b1;font-size:12px}}.value{{font-size:23px;font-weight:700;margin-top:7px}}
.table{{overflow:auto;max-height:650px}}table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border-bottom:1px solid #2c3340;padding:9px;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:#222936}}td:last-child,th:last-child{{text-align:left;white-space:normal;min-width:360px}}
</style></head><body><main>
<h1>V8 Hybrid · 장기 복리주 + 변곡점 단계매수</h1>
<p class="note">코어 {config.core_slots}개 {config.core_total_weight:.0%} ·
인플렉션 최대 {config.inflection_slots}개, 종목당
{config.inflection_scout_weight:.0%}→{config.inflection_confirm_weight:.0%}→
{config.inflection_max_weight:.0%}</p>
<p class="warning">{html.escape(caveat)}</p>
<div class="cards">
{_html_card("최종 평가액", float(base_summary["FinalValue"]), money=True)}
{_html_card("ROI", float(base_summary["ROI"]))}
{_html_card("CAGR", float(base_summary["CAGR"]))}
{_html_card("Sharpe", float(base_summary["Sharpe"]), percent=False)}
{_html_card("MDD", float(base_summary["MaxDrawdown"]))}
{_html_card("중앙 보유일", float(base_summary["MedianClosedHoldingDays"]), percent=False)}
{_html_card("거래 종목", float(base_summary["DistinctTickers"]), percent=False)}
{_html_card("최대 기여주", top_winner, raw=True)}
</div>
<h2>전략·WDC 제외·최대승자 제외·SPY 비교</h2>
<div class="panel table">{summary.to_html(index=False, border=0)}</div>
<h2>기간별 연속 계좌 성과</h2>
<div class="panel table">{period_summary.to_html(index=False, border=0)}</div>
<h2>현재 보유</h2><div class="panel table">{current.to_html(index=False, border=0)}</div>
<h2>최근 모델 행동</h2><div class="panel table">{recent.to_html(index=False, border=0)}</div>
</main></body></html>"""


def _centered_rank(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(frame.get(column), errors="coerce").where(eligible)
    return (
        values.groupby(frame["Date"]).rank(pct=True, method="average")
        - 0.5
    )


def _weighted_available_mean(
    values: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    index = next(iter(values.values())).index
    numerator = pd.Series(0.0, index=index)
    denominator = pd.Series(0.0, index=index)
    for column, weight in weights.items():
        value = values[column]
        numerator += value.fillna(0.0) * weight
        denominator += value.notna().astype(float) * weight
    return numerator / denominator.replace(0, np.nan)


def _ticker_row(
    indexed: pd.DataFrame,
    ticker: str,
) -> pd.Series | None:
    if ticker not in indexed.index:
        return None
    row = indexed.loc[ticker]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def _row_number(row: pd.Series | None, column: str) -> float:
    if row is None:
        return np.nan
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def _row_bool(row: pd.Series | None, column: str) -> bool:
    return bool(row is not None and bool(row.get(column, False)))


def _core_exit_reason(
    row: pd.Series | None,
    state: dict[str, object],
    config: V8HybridConfig,
) -> str | None:
    severe = bool(
        _row_number(row, "EpsTtmGrowthYoY") < -0.20
        and _row_number(row, "EbitdaTtmGrowthYoY") < -0.20
        and _row_number(row, "Trend200") < -0.15
    )
    if severe:
        return "CORE_SEVERE_THESIS_BREAK"
    if int(state.get("QuartersHeld", 0)) < config.core_minimum_hold_quarters:
        return None
    persistent = bool(
        _row_number(row, "GrowthFactor") < -0.20
        and _row_number(row, "QualityFactor") < -0.15
    )
    return "CORE_QUALITY_GROWTH_BREAK" if persistent else None


def _inflection_exit_reason(
    row: pd.Series | None,
    state: dict[str, object],
    config: V8HybridConfig,
) -> str | None:
    severe = bool(
        _row_number(row, "EpsTtmGrowthYoY") < 0
        and _row_number(row, "EbitdaTtmGrowthYoY") < 0
        and _row_number(row, "Trend200") < -0.15
        and _row_number(row, "Return126") < -0.20
    )
    if severe:
        return "INFLECTION_SEVERE_THESIS_BREAK"
    if int(state.get("WeeksHeld", 0)) < config.inflection_minimum_hold_weeks:
        return None
    persistent = bool(
        _row_number(row, "GrowthFactor") < -0.15
        and _row_number(row, "InflectionTechnicalScore") < -0.15
    )
    return "INFLECTION_GROWTH_TREND_BREAK" if persistent else None


def _core_buy_reason(row: pd.Series) -> str:
    return (
        f"대형 복리주 후보: 시총 순위 {_fmt_rank(row.get('MarketCapRank'))}, "
        f"CoreScore {_fmt(row.get('CoreScore'))}. 재무 성장 "
        f"{_fmt(row.get('GrowthFactor'))}, 가치·품질 "
        f"{_fmt(row.get('QualityFactor'))}, EPS TTM 성장 "
        f"{_pct(row.get('EpsTtmGrowthYoY'))}, EBITDA 성장 "
        f"{_pct(row.get('EbitdaTtmGrowthYoY'))}."
    )


def _core_sell_reason(row: pd.Series | None, code: str) -> str:
    return (
        f"단순 순위 하락이 아니라 장기 투자 논리 훼손({code}). "
        f"EPS 성장 {_pct(_row_number(row, 'EpsTtmGrowthYoY'))}, "
        f"EBITDA 성장 {_pct(_row_number(row, 'EbitdaTtmGrowthYoY'))}, "
        f"200일 추세 {_pct(_row_number(row, 'Trend200'))}."
    )


def _inflection_buy_reason(row: pd.Series) -> str:
    return (
        f"변곡점 탐색 매수: InflectionRank "
        f"{_fmt_rank(row.get('InflectionRank'))}, 점수 "
        f"{_fmt(row.get('InflectionScore'))}, 재무 가속 "
        f"{_fmt(row.get('FundamentalAccelerationScore'))}, "
        f"126일 수익률 {_pct(row.get('Return126'))}. "
        "처음 5%만 진입하고 확인될 때만 확대."
    )


def _scale_reason(
    row: pd.Series | None,
    stage: int,
    confirmations: int,
) -> str:
    target = {2: "10%", 3: "15%"}[stage]
    return (
        f"재무 변곡점과 가격 확인이 {confirmations}주 연속 유지되어 "
        f"목표비중을 {target}로 확대. InflectionScore "
        f"{_fmt(_row_number(row, 'InflectionScore'))}."
    )


def _inflection_sell_reason(row: pd.Series | None, code: str) -> str:
    return (
        f"가격 하락만이 아니라 성장·추세 논리 훼손({code}). "
        f"EPS 성장 {_pct(_row_number(row, 'EpsTtmGrowthYoY'))}, "
        f"EBITDA 성장 {_pct(_row_number(row, 'EbitdaTtmGrowthYoY'))}, "
        f"126일 수익률 {_pct(_row_number(row, 'Return126'))}."
    )


def _v8_position_record(
    ticker: str,
    entry: pd.Series,
    exit_event: pd.Series | None,
    executions: pd.DataFrame,
    attribution: pd.DataFrame,
    *,
    mark_date: pd.Timestamp | None = None,
    mark_price: float | None = None,
) -> dict[str, object]:
    entry_date = pd.Timestamp(entry["ExecutionDate"])
    end = (
        pd.Timestamp(exit_event["ExecutionDate"])
        if exit_event is not None
        else pd.Timestamp(mark_date)
    )
    relevant_exec = executions.loc[
        executions["ExecutionDate"].between(entry_date, end)
    ]
    pnl = attribution.loc[
        attribution["Ticker"].eq(ticker)
        & attribution["Date"].between(entry_date, end)
    ]
    exit_price = (
        float(exit_event["ExecutionPrice"])
        if exit_event is not None
        else float(mark_price)
    )
    entry_price = float(entry["ExecutionPrice"])
    return {
        "Ticker": ticker,
        "Company": entry.get("Company", ticker),
        "Sleeve": entry.get("Sleeve", ""),
        "Status": "CLOSED" if exit_event is not None else "OPEN",
        "EntrySignalDate": pd.Timestamp(entry["Date"]),
        "EntryExecutionDate": entry_date,
        "ExitOrMarkDate": end,
        "HoldingCalendarDays": int((end - entry_date).days),
        "EntryPrice": entry_price,
        "ExitPrice": (
            exit_price if exit_event is not None else np.nan
        ),
        "MarkPrice": (
            exit_price if exit_event is None else np.nan
        ),
        "PriceReturn": (
            exit_price / entry_price - 1
            if entry_price > 0 and pd.notna(exit_price)
            else np.nan
        ),
        "InitialAllocation": float(entry["NotionalAfter"]),
        "TotalBought": float(
            relevant_exec.loc[
                relevant_exec["TradeNotional"].gt(0),
                "GrossTradeAmount",
            ].sum()
        ),
        "TotalSold": float(
            relevant_exec.loc[
                relevant_exec["TradeNotional"].lt(0),
                "GrossTradeAmount",
            ].sum()
        ),
        "TransactionCost": float(
            relevant_exec["AllocatedTransactionCost"].sum()
        ),
        "NetPnL": float(pnl["NetPnL"].sum()),
        "EntryReason": entry.get("Reason", ""),
        "ExitReasonCode": (
            exit_event.get("ExitReason", "")
            if exit_event is not None
            else ""
        ),
        "ExitReason": (
            exit_event.get("Reason", "")
            if exit_event is not None
            else "장기 투자 논리가 유지되어 계속 보유."
        ),
    }


def _path_metrics(
    equity: pd.DataFrame,
    start: str,
    end: str,
    *,
    initial_capital: float,
) -> dict[str, object]:
    frame = equity.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date")
    period = frame.loc[frame["Date"].between(start, end)].copy()
    if period.empty:
        return {
            "StartDate": pd.NaT,
            "EndDate": pd.NaT,
            "StartValue": np.nan,
            "EndValue": np.nan,
            "ROI": np.nan,
            "CAGR": np.nan,
            "Sharpe": np.nan,
            "MaxDrawdown": np.nan,
        }
    prior = frame.loc[frame["Date"].lt(pd.Timestamp(start)), "Equity"]
    start_value = float(prior.iloc[-1]) if not prior.empty else initial_capital
    end_value = float(period["Equity"].iloc[-1])
    roi = (end_value / start_value - 1) * 100
    elapsed = max(
        (
            pd.Timestamp(period["Date"].iloc[-1])
            - pd.Timestamp(period["Date"].iloc[0])
        ).days
        / 365.25,
        1 / 365.25,
    )
    cagr = (end_value / start_value) ** (1 / elapsed) * 100 - 100
    path = pd.concat(
        [
            pd.Series([start_value]),
            period["Equity"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    returns = path.pct_change(fill_method=None).dropna()
    volatility = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / volatility * np.sqrt(252))
        if volatility > 0
        else 0.0
    )
    drawdown = path / path.cummax() - 1
    return {
        "StartDate": period["Date"].iloc[0],
        "EndDate": period["Date"].iloc[-1],
        "StartValue": start_value,
        "EndValue": end_value,
        "ROI": roi,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDrawdown": float(drawdown.min() * 100),
    }


def _fmt(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):.3f}" if pd.notna(numeric) else "N/A"


def _fmt_rank(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{int(numeric)}위" if pd.notna(numeric) else "N/A"


def _pct(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):+.1%}" if pd.notna(numeric) else "N/A"


def _html_card(
    label: str,
    value: float | str,
    *,
    money: bool = False,
    percent: bool = True,
    raw: bool = False,
) -> str:
    if raw:
        rendered = str(value)
    elif money:
        rendered = f"${float(value):,.0f}"
    elif percent:
        rendered = f"{float(value):+.2f}%"
    else:
        rendered = f"{float(value):,.2f}"
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(rendered)}</div></div>'
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=path.stem + "_",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
    )


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
