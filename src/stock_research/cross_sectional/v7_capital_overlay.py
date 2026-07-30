from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths

from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .pit_validation import apply_membership_to_panel, causal_signal_day_panel
from .portfolio import PortfolioSummary, PreparedMarket, prepare_market
from .signals import generate_rebalance_targets, score_panel
from .v7_pit_evaluation import (
    build_v7_source_panel,
    load_ready_tickers,
    normalize_change_membership,
)
from .v7_slot_sweep import slot_sweep_params, spy_buy_and_hold
from .v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)


@dataclass(frozen=True)
class OverlayCandidate:
    candidate_id: int
    cash_gate_quantile: float | None
    strong_gate_quantile: float | None
    strong_long_gross: float
    short_gross: float
    short_score_max: float
    short_count: int = 5
    max_gross: float = 2.0


@dataclass(frozen=True)
class ResolvedOverlayCandidate:
    candidate: OverlayCandidate
    cash_gate_score: float
    strong_gate_score: float


@dataclass(frozen=True)
class SignedBacktestResult:
    daily: pd.DataFrame
    executions: pd.DataFrame
    summary: PortfolioSummary
    ruined: bool
    total_funding_cost: float
    total_short_borrow_cost: float


@dataclass(frozen=True)
class V7CapitalOverlayArtifacts:
    output_dir: Path
    candidate_summary_csv: Path
    period_summary_csv: Path
    equity_csv: Path
    executions_csv: Path
    signal_exposure_csv: Path
    short_selections_csv: Path
    data_audit_csv: Path
    manifest_json: Path


@dataclass(frozen=True)
class _PeriodContext:
    scored: pd.DataFrame
    v7_targets: pd.DataFrame
    market: PreparedMarket


def signal_confidence(v7_targets: pd.DataFrame) -> pd.Series:
    """Mean AlphaScore of the five virtual V7-3 holdings on each signal day."""

    selected = v7_targets.loc[
        v7_targets["ModelSelected"].fillna(False)
    ].copy()
    if selected.empty:
        return pd.Series(dtype=float, name="LongConfidence")
    confidence = (
        pd.to_numeric(selected["AlphaScore"], errors="coerce")
        .groupby(selected["Date"])
        .mean()
        .sort_index()
    )
    confidence.name = "LongConfidence"
    return confidence


def resolve_candidate(
    candidate: OverlayCandidate,
    training_confidence: pd.Series,
) -> ResolvedOverlayCandidate:
    clean = pd.to_numeric(training_confidence, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Training confidence has no finite observations")
    cash_gate = (
        -np.inf
        if candidate.cash_gate_quantile is None
        else float(clean.quantile(candidate.cash_gate_quantile))
    )
    strong_gate = (
        -np.inf
        if candidate.strong_gate_quantile is None
        else float(clean.quantile(candidate.strong_gate_quantile))
    )
    if strong_gate < cash_gate:
        strong_gate = cash_gate
    return ResolvedOverlayCandidate(
        candidate=candidate,
        cash_gate_score=cash_gate,
        strong_gate_score=strong_gate,
    )


def generate_candidate_grid(
    *,
    cash_gate_quantiles: Iterable[float | None],
    strong_gate_quantiles: Iterable[float | None],
    strong_long_gross_values: Iterable[float],
    short_gross_values: Iterable[float],
    short_score_max_values: Iterable[float],
    short_count: int,
    max_gross: float,
) -> list[OverlayCandidate]:
    if short_count < 1:
        raise ValueError("short_count must be positive")
    rows: list[OverlayCandidate] = []
    seen: set[tuple[object, ...]] = set()

    def add(
        cash_q: float | None,
        strong_q: float | None,
        long_gross: float,
        short_gross: float,
        short_score_max: float,
    ) -> None:
        effective_strong_q = None if long_gross == 1.0 else strong_q
        effective_short_max = (
            min(short_score_max_values)
            if short_gross == 0
            else short_score_max
        )
        key = (
            cash_q,
            effective_strong_q,
            float(long_gross),
            float(short_gross),
            float(effective_short_max),
        )
        if key in seen:
            return
        if long_gross + short_gross > max_gross + 1e-12:
            return
        seen.add(key)
        rows.append(
            OverlayCandidate(
                candidate_id=len(rows),
                cash_gate_quantile=cash_q,
                strong_gate_quantile=effective_strong_q,
                strong_long_gross=float(long_gross),
                short_gross=float(short_gross),
                short_score_max=float(effective_short_max),
                short_count=short_count,
                max_gross=max_gross,
            )
        )

    add(None, None, 1.0, 0.0, min(short_score_max_values))
    for cash_q in cash_gate_quantiles:
        for strong_q in strong_gate_quantiles:
            for long_gross in strong_long_gross_values:
                for short_gross in short_gross_values:
                    for short_score_max in short_score_max_values:
                        add(
                            cash_q,
                            strong_q,
                            float(long_gross),
                            float(short_gross),
                            float(short_score_max),
                        )
    return rows


def build_overlay_targets(
    scored_signal_days: pd.DataFrame,
    v7_targets: pd.DataFrame,
    resolved: ResolvedOverlayCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scale frozen V7-3 longs and optionally short strict bottom-five names."""

    candidate = resolved.candidate
    target_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    short_rows: list[dict[str, object]] = []
    targets_by_date = {
        pd.Timestamp(date): group
        for date, group in v7_targets.groupby("Date", sort=True)
    }
    for date, group in scored_signal_days.groupby("Date", sort=True):
        date = pd.Timestamp(date)
        base = targets_by_date.get(date)
        if base is None:
            continue
        long_rows = base.loc[base["ModelSelected"].fillna(False)].copy()
        confidence = float(
            pd.to_numeric(long_rows["AlphaScore"], errors="coerce").mean()
        )
        if not np.isfinite(confidence):
            confidence = -np.inf
        if confidence < resolved.cash_gate_score:
            long_gross = 0.0
            regime = "CASH_GATE"
        elif confidence >= resolved.strong_gate_score:
            long_gross = candidate.strong_long_gross
            regime = "STRONG"
        else:
            long_gross = 1.0
            regime = "NORMAL"

        short_pool = group.loc[
            group["Eligible"].fillna(False)
            & pd.to_numeric(group["AlphaScore"], errors="coerce").le(
                candidate.short_score_max
            )
            & pd.to_numeric(group["Trend200"], errors="coerce").lt(0)
            & pd.to_numeric(group["Return126"], errors="coerce").lt(0)
            & ~group["Ticker"].isin(long_rows["Ticker"])
        ].copy()
        short_pool = short_pool.sort_values(
            ["AlphaScore", "Ticker"],
            ascending=[True, True],
        ).head(candidate.short_count)
        actual_short_gross = (
            candidate.short_gross if not short_pool.empty else 0.0
        )
        if long_gross + actual_short_gross > candidate.max_gross + 1e-12:
            raise ValueError("Resolved target exceeds candidate max_gross")

        if long_gross > 0 and not long_rows.empty:
            long_weight = long_gross / len(long_rows)
            for ticker in long_rows["Ticker"].astype(str):
                target_rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "TargetWeight": long_weight,
                        "Side": "LONG",
                    }
                )
        if actual_short_gross > 0:
            short_weight = -actual_short_gross / len(short_pool)
            for rank, row in enumerate(
                short_pool.itertuples(index=False),
                start=1,
            ):
                target_rows.append(
                    {
                        "Date": date,
                        "Ticker": str(row.Ticker),
                        "TargetWeight": short_weight,
                        "Side": "SHORT",
                    }
                )
                short_rows.append(
                    {
                        "Date": date,
                        "Ticker": str(row.Ticker),
                        "ShortRank": rank,
                        "AlphaScore": float(row.AlphaScore),
                        "Trend200": float(row.Trend200),
                        "Return126": float(row.Return126),
                    }
                )
        if long_gross == 0 and actual_short_gross == 0:
            target_rows.append(
                {
                    "Date": date,
                    "Ticker": str(group.iloc[0]["Ticker"]),
                    "TargetWeight": 0.0,
                    "Side": "CASH",
                }
            )
        exposure_rows.append(
            {
                "Date": date,
                "LongConfidence": (
                    confidence if np.isfinite(confidence) else np.nan
                ),
                "CashGateScore": resolved.cash_gate_score,
                "StrongGateScore": resolved.strong_gate_score,
                "Regime": regime,
                "LongGross": long_gross,
                "ShortGross": actual_short_gross,
                "GrossExposureTarget": long_gross + actual_short_gross,
                "NetExposureTarget": long_gross - actual_short_gross,
                "LongCount": len(long_rows) if long_gross > 0 else 0,
                "ShortCount": len(short_pool) if actual_short_gross > 0 else 0,
            }
        )
    return (
        pd.DataFrame(target_rows),
        pd.DataFrame(exposure_rows),
        pd.DataFrame(short_rows),
    )


def run_signed_overlay_backtest(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    start: str,
    end: str,
    initial_capital: float,
    transaction_cost_bps: float,
    funding_annual_rate: float,
    short_borrow_annual_rate: float,
    prepared_market: PreparedMarket | None = None,
) -> SignedBacktestResult:
    """Execute signed weekly targets at the next session open with carry costs."""

    market = prepared_market or prepare_market(panel, start=start, end=end)
    if market.start != start or market.end != end:
        raise ValueError("Prepared market period does not match requested period")
    dates = market.dates
    tickers = market.tickers
    ticker_positions = {
        str(ticker): position for position, ticker in enumerate(tickers)
    }
    execution_schedule: dict[pd.Timestamp, tuple[pd.Timestamp, np.ndarray]] = {}
    date_positions = {date: position for position, date in enumerate(dates)}
    for signal_date, group in targets.groupby("Date", sort=True):
        signal_date = pd.Timestamp(signal_date)
        position = date_positions.get(signal_date)
        if position is None or position + 1 >= len(dates):
            continue
        weights = np.zeros(len(tickers), dtype=float)
        for row in group.itertuples(index=False):
            ticker_position = ticker_positions.get(str(row.Ticker))
            if ticker_position is not None:
                weights[ticker_position] += float(row.TargetWeight)
        execution_schedule[pd.Timestamp(dates[position + 1])] = (
            signal_date,
            weights,
        )

    open_values = market.open_prices.to_numpy(dtype=float, copy=False)
    close_values = market.valuation_prices.to_numpy(dtype=float, copy=False)
    fallback_values = market.fallback_trade_prices.to_numpy(
        dtype=float,
        copy=False,
    )
    shares = np.zeros(len(tickers), dtype=float)
    cash = float(initial_capital)
    previous_close = np.zeros(len(tickers), dtype=float)
    previous_date: pd.Timestamp | None = None
    total_turnover = 0.0
    total_funding_cost = 0.0
    total_short_borrow_cost = 0.0
    ticker_trades = 0
    rebalance_count = 0
    ruined = False
    daily_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []
    cost_rate = transaction_cost_bps / 10_000

    for position, date in enumerate(dates):
        date = pd.Timestamp(date)
        if ruined:
            daily_rows.append(
                {
                    "Date": date,
                    "Equity": 0.0,
                    "Cash": 0.0,
                    "GrossExposure": 0.0,
                    "NetExposure": 0.0,
                    "LongCount": 0,
                    "ShortCount": 0,
                    "FundingCost": 0.0,
                    "ShortBorrowCost": 0.0,
                    "Ruined": True,
                }
            )
            continue

        carry_days = (
            max((date - previous_date).days, 0)
            if previous_date is not None
            else 0
        )
        funding_cost = (
            max(-cash, 0.0)
            * funding_annual_rate
            * carry_days
            / 365.25
        )
        short_notional = float(
            np.sum(
                np.where(
                    shares < 0,
                    np.abs(shares * previous_close),
                    0.0,
                )
            )
        )
        short_borrow_cost = (
            short_notional
            * short_borrow_annual_rate
            * carry_days
            / 365.25
        )
        cash -= funding_cost + short_borrow_cost
        total_funding_cost += funding_cost
        total_short_borrow_cost += short_borrow_cost

        open_row = np.where(
            np.isnan(open_values[position]),
            fallback_values[position],
            open_values[position],
        )
        close_row = close_values[position]
        open_for_valuation = np.where(np.isnan(open_row), 0.0, open_row)
        close_for_valuation = np.where(np.isnan(close_row), 0.0, close_row)
        transaction_cost = 0.0
        if date in execution_schedule:
            signal_date, target = execution_schedule[date]
            tradable = (open_row > 0) & np.isfinite(open_row)
            target = np.where(tradable, target, 0.0)
            pre_trade_equity = cash + float(
                np.sum(shares * open_for_valuation)
            )
            if pre_trade_equity <= 0:
                ruined = True
                cash = 0.0
                shares[:] = 0.0
            else:
                current_notional = shares * open_for_valuation
                desired_notional = target * pre_trade_equity
                changes = desired_notional - current_notional
                turnover = float(np.abs(changes).sum())
                transaction_cost = turnover * cost_rate
                net_equity = pre_trade_equity - transaction_cost
                if net_equity <= 0:
                    ruined = True
                    cash = 0.0
                    shares[:] = 0.0
                else:
                    desired_after_cost = target * net_equity
                    changed = (
                        np.abs(changes)
                        > max(pre_trade_equity, 1.0) * 1e-8
                    )
                    ticker_trades += int(changed.sum())
                    rebalance_count += 1
                    total_turnover += turnover
                    shares = np.divide(
                        desired_after_cost,
                        open_row,
                        out=np.zeros_like(desired_after_cost),
                        where=tradable,
                    )
                    cash = net_equity - float(desired_after_cost.sum())
                    execution_rows.append(
                        {
                            "ExecutionDate": date,
                            "SignalDate": signal_date,
                            "PreTradeEquity": pre_trade_equity,
                            "Turnover": turnover,
                            "TransactionCost": transaction_cost,
                            "FundingCostSincePriorSession": funding_cost,
                            "ShortBorrowCostSincePriorSession": (
                                short_borrow_cost
                            ),
                            "LongGrossTarget": float(
                                target[target > 0].sum()
                            ),
                            "ShortGrossTarget": float(
                                -target[target < 0].sum()
                            ),
                            "GrossTarget": float(np.abs(target).sum()),
                            "NetTarget": float(target.sum()),
                            "LongCount": int((target > 0).sum()),
                            "ShortCount": int((target < 0).sum()),
                        }
                    )
        equity = (
            0.0
            if ruined
            else cash + float(np.sum(shares * close_for_valuation))
        )
        if equity <= 0:
            ruined = True
            equity = 0.0
            cash = 0.0
            shares[:] = 0.0
        long_notional = float(
            np.sum(np.where(shares > 0, shares * close_for_valuation, 0.0))
        )
        ending_short_notional = float(
            np.sum(
                np.where(
                    shares < 0,
                    np.abs(shares * close_for_valuation),
                    0.0,
                )
            )
        )
        gross = (
            (long_notional + ending_short_notional) / equity
            if equity > 0
            else 0.0
        )
        net = (
            (long_notional - ending_short_notional) / equity
            if equity > 0
            else 0.0
        )
        daily_rows.append(
            {
                "Date": date,
                "Equity": equity,
                "Cash": cash,
                "GrossExposure": gross,
                "NetExposure": net,
                "LongCount": int((shares > 0).sum()),
                "ShortCount": int((shares < 0).sum()),
                "FundingCost": funding_cost,
                "ShortBorrowCost": short_borrow_cost,
                "Ruined": ruined,
            }
        )
        previous_close = close_for_valuation
        previous_date = date

    daily = pd.DataFrame(daily_rows)
    summary = _summarize_signed(
        daily,
        initial_capital=initial_capital,
        total_turnover=total_turnover,
        ticker_trades=ticker_trades,
        rebalance_count=rebalance_count,
    )
    return SignedBacktestResult(
        daily=daily,
        executions=pd.DataFrame(execution_rows),
        summary=summary,
        ruined=ruined,
        total_funding_cost=total_funding_cost,
        total_short_borrow_cost=total_short_borrow_cost,
    )


def run_v7_capital_overlay(
    paths: ProjectPaths,
    settings: ResearchSettings,
    base_params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    frozen_strategy_path: str | Path,
    spy_path: str | Path,
    overlay_config: dict[str, object],
    expected_ready_count: int = 575,
    output_dir: str | Path | None = None,
) -> V7CapitalOverlayArtifacts:
    ready_tickers = load_ready_tickers(
        backfill_status_path,
        expected_count=expected_ready_count,
    )
    members, _ = discover_universe(paths, ticker_config_path)
    missing = sorted(ready_tickers - {member.ticker for member in members})
    if missing:
        raise ValueError(
            "Ready tickers are not discoverable: " + ", ".join(missing)
        )
    warmup_start = str(
        (pd.Timestamp(settings.train_start) - pd.DateOffset(years=2)).date()
    )
    base_panel, data_audit = build_v7_source_panel(
        paths,
        members,
        settings,
        ready_tickers=ready_tickers,
        warmup_start=warmup_start,
    )
    observed = add_v7_technical_observations(base_panel)
    membership = normalize_change_membership(pd.read_csv(membership_path))
    pit_panel = apply_membership_to_panel(observed, membership, settings)
    technical = add_v7_technical_factors(pit_panel, settings)
    v7_3 = TECHNICAL_VARIANTS[2]
    scoring_panel = scoring_panel_for_variant(technical, v7_3)
    v7_params = slot_sweep_params(base_params, 5, exit_buffer=4)
    latest_end = str(pd.Timestamp(scoring_panel["Date"].max()).date())
    reference_dates = pd.DatetimeIndex(
        scoring_panel["Date"].drop_duplicates().sort_values()
    )

    periods: dict[str, tuple[str, str]] = {
        "TRAIN_2020_2024": (
            settings.train_start,
            min(settings.train_end, latest_end),
        )
    }
    for index, (start, end) in enumerate(settings.training_folds, start=1):
        periods[f"TRAIN_FOLD_{index}"] = (start, min(end, latest_end))
    for label, (start, end) in settings.validation_periods.items():
        if start <= latest_end:
            periods[label] = (start, min(end, latest_end))
    periods["FULL_2020_2026"] = (settings.train_start, latest_end)

    contexts = {
        label: _build_period_context(
            scoring_panel,
            v7_params,
            settings,
            start=start,
            end=end,
            reference_dates=reference_dates,
        )
        for label, (start, end) in periods.items()
    }
    training_confidence = signal_confidence(
        contexts["TRAIN_2020_2024"].v7_targets
    )
    candidates = generate_candidate_grid(
        cash_gate_quantiles=overlay_config["cash_gate_quantiles"],
        strong_gate_quantiles=overlay_config["strong_gate_quantiles"],
        strong_long_gross_values=overlay_config["strong_long_gross_values"],
        short_gross_values=overlay_config["short_gross_values"],
        short_score_max_values=overlay_config["short_score_max_values"],
        short_count=int(overlay_config["short_count"]),
        max_gross=float(overlay_config["max_gross"]),
    )
    funding_rate = float(overlay_config["funding_annual_rate"])
    borrow_rate = float(overlay_config["short_borrow_annual_rate"])
    minimum_mdd = float(overlay_config["minimum_train_mdd"])
    minimum_positive_folds = int(overlay_config["minimum_positive_folds"])
    minimum_worst_fold_roi = float(
        overlay_config["minimum_worst_fold_roi"]
    )
    train_labels = [
        label for label in periods if label.startswith("TRAIN_FOLD_")
    ]
    candidate_rows: list[dict[str, object]] = []
    resolved_by_id: dict[int, ResolvedOverlayCandidate] = {}
    for index, candidate in enumerate(candidates, start=1):
        resolved = resolve_candidate(candidate, training_confidence)
        resolved_by_id[candidate.candidate_id] = resolved
        metrics: dict[str, SignedBacktestResult] = {}
        for label in ["TRAIN_2020_2024", *train_labels]:
            context = contexts[label]
            targets, _, _ = build_overlay_targets(
                context.scored,
                context.v7_targets,
                resolved,
            )
            start, end = periods[label]
            metrics[label] = run_signed_overlay_backtest(
                scoring_panel,
                targets,
                start=start,
                end=end,
                initial_capital=settings.initial_capital,
                transaction_cost_bps=settings.transaction_cost_bps,
                funding_annual_rate=funding_rate,
                short_borrow_annual_rate=borrow_rate,
                prepared_market=context.market,
            )
        train = metrics["TRAIN_2020_2024"]
        fold_rois = [
            metrics[label].summary.roi_percent for label in train_labels
        ]
        fold_cagrs = [
            metrics[label].summary.cagr_percent for label in train_labels
        ]
        positive_folds = int(sum(value > 0 for value in fold_rois))
        pass_constraints = bool(
            not train.ruined
            and train.summary.max_drawdown_percent >= minimum_mdd
            and positive_folds >= minimum_positive_folds
            and min(fold_rois) >= minimum_worst_fold_roi
        )
        candidate_rows.append(
            {
                **asdict(candidate),
                "CashGateScore": resolved.cash_gate_score,
                "StrongGateScore": resolved.strong_gate_score,
                "TrainROI": train.summary.roi_percent,
                "TrainCAGR": train.summary.cagr_percent,
                "TrainSharpe": train.summary.sharpe_ratio,
                "TrainMaxDrawdown": train.summary.max_drawdown_percent,
                "TrainAnnualizedTurnover": (
                    train.summary.annualized_turnover
                ),
                "TrainFundingCost": train.total_funding_cost,
                "TrainShortBorrowCost": train.total_short_borrow_cost,
                "MedianFoldROI": float(np.median(fold_rois)),
                "WorstFoldROI": float(min(fold_rois)),
                "MedianFoldCAGR": float(np.median(fold_cagrs)),
                "PositiveFolds": positive_folds,
                "Ruined": train.ruined,
                "PassConstraints": pass_constraints,
            }
        )
        if index % 20 == 0 or index == len(candidates):
            print(f"OVERLAY {index}/{len(candidates)} candidates evaluated")

    candidate_summary = pd.DataFrame(candidate_rows)
    valid = candidate_summary.loc[candidate_summary["PassConstraints"]].copy()
    if valid.empty:
        raise RuntimeError("No overlay candidate passed the training constraints")
    selected_id = int(
        valid.sort_values(
            ["TrainCAGR", "MedianFoldCAGR", "TrainMaxDrawdown"],
            ascending=[False, False, False],
        ).iloc[0]["candidate_id"]
    )
    max_train_id = int(
        candidate_summary.sort_values(
            ["TrainCAGR", "MedianFoldCAGR"],
            ascending=[False, False],
        ).iloc[0]["candidate_id"]
    )
    baseline_id = 0
    report_ids = {
        "V7_3_BASELINE": baseline_id,
        "V7_3_OVERLAY_SELECTED": selected_id,
    }
    if max_train_id not in report_ids.values():
        report_ids["MAX_TRAIN_UNCONSTRAINED"] = max_train_id

    spy = load_sp500_proxy(spy_path)
    period_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    execution_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    short_frames: list[pd.DataFrame] = []
    evaluation_labels = [
        "TRAIN_2020_2024",
        *[
            label
            for label in settings.validation_periods
            if label in contexts
        ],
        "FULL_2020_2026",
    ]
    for series_name, candidate_id in report_ids.items():
        resolved = resolved_by_id[candidate_id]
        for label in evaluation_labels:
            context = contexts[label]
            targets, exposure, short_selection = build_overlay_targets(
                context.scored,
                context.v7_targets,
                resolved,
            )
            start, end = periods[label]
            result = run_signed_overlay_backtest(
                scoring_panel,
                targets,
                start=start,
                end=end,
                initial_capital=settings.initial_capital,
                transaction_cost_bps=settings.transaction_cost_bps,
                funding_annual_rate=funding_rate,
                short_borrow_annual_rate=borrow_rate,
                prepared_market=context.market,
            )
            spy_summary, _ = spy_buy_and_hold(
                spy,
                start=start,
                end=end,
                initial_capital=settings.initial_capital,
                transaction_cost_bps=settings.transaction_cost_bps,
            )
            summary = result.summary
            period_rows.append(
                {
                    "Series": series_name,
                    "CandidateID": candidate_id,
                    "Period": label,
                    "StartDate": summary.start_date,
                    "EndDate": summary.end_date,
                    "ROI": summary.roi_percent,
                    "CAGR": summary.cagr_percent,
                    "Sharpe": summary.sharpe_ratio,
                    "MaxDrawdown": summary.max_drawdown_percent,
                    "AnnualizedTurnover": summary.annualized_turnover,
                    "FundingCost": result.total_funding_cost,
                    "ShortBorrowCost": result.total_short_borrow_cost,
                    "Ruined": result.ruined,
                    "SPYROI": spy_summary["ROI"],
                    "SPYCAGR": spy_summary["CAGR"],
                    "SPYExcessROI": (
                        summary.roi_percent - float(spy_summary["ROI"])
                    ),
                }
            )
            equity = result.daily.copy()
            equity.insert(0, "Series", series_name)
            equity.insert(1, "CandidateID", candidate_id)
            equity.insert(2, "Period", label)
            equity_frames.append(equity)
            executions = result.executions.copy()
            executions.insert(0, "Series", series_name)
            executions.insert(1, "CandidateID", candidate_id)
            executions.insert(2, "Period", label)
            execution_frames.append(executions)
            exposure.insert(0, "Series", series_name)
            exposure.insert(1, "CandidateID", candidate_id)
            exposure.insert(2, "Period", label)
            exposure_frames.append(exposure)
            if not short_selection.empty:
                short_selection.insert(0, "Series", series_name)
                short_selection.insert(1, "CandidateID", candidate_id)
                short_selection.insert(2, "Period", label)
                short_frames.append(short_selection)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v7_capital_overlay"
            / f"{timestamp}_v7_3_roi_first"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_summary_csv": destination / "candidate_summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "equity_csv": destination / "equity.csv",
        "executions_csv": destination / "executions.csv",
        "signal_exposure_csv": destination / "signal_exposure.csv",
        "short_selections_csv": destination / "short_selections.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(candidate_summary, outputs["candidate_summary_csv"], index=False)
    atomic_to_csv(pd.DataFrame(period_rows), outputs["period_summary_csv"], index=False)
    atomic_to_csv(pd.concat(equity_frames, ignore_index=True), outputs["equity_csv"], index=False)
    atomic_to_csv(
        pd.concat(execution_frames, ignore_index=True),
        outputs["executions_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(exposure_frames, ignore_index=True),
        outputs["signal_exposure_csv"],
        index=False,
    )
    atomic_to_csv(
        (
            pd.concat(short_frames, ignore_index=True)
            if short_frames
            else pd.DataFrame(
                columns=[
                    "Series",
                    "CandidateID",
                    "Period",
                    "Date",
                    "Ticker",
                    "ShortRank",
                    "AlphaScore",
                    "Trend200",
                    "Return126",
                ]
            )
        ),
        outputs["short_selections_csv"],
        index=False,
    )
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    frozen_payload = json.loads(
        Path(frozen_strategy_path).read_text(encoding="utf-8")
    )
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "V7-3 ROI-first capital allocation overlay",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "v7_3_stock_selection_unchanged": True,
            "variant": v7_3.name,
            "v7_params": v7_params.as_dict(),
            "selected_candidate_id": selected_id,
            "maximum_train_candidate_id": max_train_id,
            "selection_rule": (
                "maximum 2020-2024 CAGR among candidates passing predeclared "
                "drawdown, fold, and no-ruin constraints"
            ),
            "overlay_config": overlay_config,
            "candidate_count": len(candidates),
            "funding_rule": (
                "negative cash charged at configured annual rate by calendar day"
            ),
            "short_rule": (
                "up to five lowest AlphaScore eligible names with both "
                "Trend200 and Return126 below zero"
            ),
            "signal_rule": "known final exchange session of each W-FRI week close",
            "execution_rule": "next available trading session open",
            "transaction_cost_bps": settings.transaction_cost_bps,
            "frozen_v6_candidate": frozen_payload.get("selected_candidate"),
            "frozen_strategy_sha256": _sha256(Path(frozen_strategy_path)),
            "ready_ticker_count": len(ready_tickers),
            "financial_point_in_time": False,
            "caveats": [
                "Macrotrends financials are current restated history.",
                "The 575-name panel excludes unavailable acquired/delisted names.",
                "2025 and 2026 were already observed and are not fresh OOS.",
                "This additional overlay grid adds another multiple-testing layer.",
                "Short fills, borrow availability, recalls, and locate fees are approximated.",
                "Leveraged exposure is modeled as notional leverage, not a specific ETF.",
            ],
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return V7CapitalOverlayArtifacts(output_dir=destination, **outputs)


def _build_period_context(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    *,
    start: str,
    end: str,
    reference_dates: pd.DatetimeIndex,
) -> _PeriodContext:
    signal_days = causal_signal_day_panel(
        panel,
        start=start,
        end=end,
        reference_market_dates=reference_dates,
        rebalance_weekday=settings.rebalance_weekday,
    )
    scored = score_panel(signal_days, params)
    targets = generate_rebalance_targets(scored, params)
    return _PeriodContext(
        scored=scored,
        v7_targets=targets,
        market=prepare_market(panel, start=start, end=end),
    )


def _summarize_signed(
    daily: pd.DataFrame,
    *,
    initial_capital: float,
    total_turnover: float,
    ticker_trades: int,
    rebalance_count: int,
) -> PortfolioSummary:
    start_date = pd.Timestamp(daily.iloc[0]["Date"])
    end_date = pd.Timestamp(daily.iloc[-1]["Date"])
    final_value = float(daily.iloc[-1]["Equity"])
    roi = (final_value / initial_capital - 1) * 100
    years = max((end_date - start_date).days / 365.25, 1 / 252)
    cagr = (
        ((final_value / initial_capital) ** (1 / years) - 1) * 100
        if final_value > 0
        else -100.0
    )
    running_peak = daily["Equity"].cummax()
    max_drawdown = float(
        (daily["Equity"] / running_peak.replace(0, np.nan) - 1)
        .fillna(-1.0)
        .min()
        * 100
    )
    returns = daily["Equity"].pct_change(fill_method=None).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    volatility = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / volatility * np.sqrt(252))
        if volatility > 0
        else 0.0
    )
    turnover_multiple = total_turnover / initial_capital
    return PortfolioSummary(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        total_injected=initial_capital,
        final_value=final_value,
        roi_percent=roi,
        cagr_percent=cagr,
        max_drawdown_percent=max_drawdown,
        sharpe_ratio=sharpe,
        turnover_multiple=turnover_multiple,
        annualized_turnover=turnover_multiple / years,
        ticker_trades=ticker_trades,
        rebalance_count=rebalance_count,
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
