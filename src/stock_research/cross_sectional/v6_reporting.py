from __future__ import annotations

import html
import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths

from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .features import build_panel
from .portfolio import run_portfolio_backtest
from .signals import (
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)

VARIANT_LABELS = {
    "V5": "V5 기준",
    "V6-A": "V6-A 승자 보유",
    "V6-B": "V6-B 교체 문턱",
    "V6-C": "V6-C 과열 차단",
    "V6-D": "V6-D 위험관리",
    "V6-BC": "V6-B+C 교체·과열",
    "V6-ALL": "V6 결합형",
}


@dataclass(frozen=True)
class V6ComparisonArtifacts:
    output_dir: Path
    summary_csv: Path
    sensitivity_csv: Path
    equity_csv: Path
    events_csv: Path
    ledger_csv: Path
    html_report: Path


def frozen_v6_variants(
    base: StrategyParams,
) -> dict[str, StrategyParams]:
    """Return single-rule ablations and one combined V6 using frozen V5 weights."""

    wider_exit_rank = max(10, base.exit_rank)
    return {
        "V5": base,
        "V6-A": replace(
            base,
            profit_rotation_exit_rank=wider_exit_rank,
            profit_rotation_confirmation_rebalances=2,
        ),
        "V6-B": replace(
            base,
            replacement_score_advantage=0.05,
        ),
        "V6-C": replace(
            base,
            overheated_entry_enabled=True,
            overheated_return126=1.0,
            overheated_trend200=0.50,
            overheated_drawdown126_floor=-0.03,
        ),
        "V6-D": replace(
            base,
            hard_stop_return=-0.20,
            trailing_stop_enabled=True,
            trailing_stop_activation_gain=0.20,
            trailing_stop_drawdown=-0.15,
        ),
        "V6-BC": replace(
            base,
            replacement_score_advantage=0.05,
            overheated_entry_enabled=True,
            overheated_return126=1.0,
            overheated_trend200=0.50,
            overheated_drawdown126_floor=-0.03,
        ),
        "V6-ALL": replace(
            base,
            profit_rotation_exit_rank=wider_exit_rank,
            profit_rotation_confirmation_rebalances=2,
            replacement_score_advantage=0.05,
            overheated_entry_enabled=True,
            overheated_return126=1.0,
            overheated_trend200=0.50,
            overheated_drawdown126_floor=-0.03,
            hard_stop_return=-0.20,
            trailing_stop_enabled=True,
            trailing_stop_activation_gain=0.20,
            trailing_stop_drawdown=-0.15,
        ),
    }


def generate_v6_comparison(
    paths: ProjectPaths,
    settings: ResearchSettings,
    base_params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    output_dir: str | Path | None = None,
) -> V6ComparisonArtifacts:
    members, _ = discover_universe(paths, ticker_config_path)
    if not members:
        raise ValueError("No stocks have both price and financial data")
    panel, _ = build_panel(members, settings)
    latest_date = pd.Timestamp(panel["Date"].max())
    latest_end = str(latest_date.date())
    live_start = min(start for start, _ in settings.validation_periods.values())
    variants = frozen_v6_variants(base_params)

    summary_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    ledger_frames: list[pd.DataFrame] = []

    for variant, params in variants.items():
        for period, (start, configured_end) in settings.validation_periods.items():
            end = min(configured_end, latest_end)
            result = _evaluate_period(panel, params, settings, start, end)
            summary_rows.append(
                _summary_row(
                    variant,
                    period,
                    result["strategy"].summary,
                    result["benchmark"].summary,
                )
            )

        live_signal_days = signal_day_panel(
            panel,
            live_start,
            latest_end,
            settings.rebalance_weekday,
        )
        live_targets = generate_rebalance_targets(
            score_panel(live_signal_days, params),
            params,
        )
        live_result = run_portfolio_backtest(
            panel,
            live_targets,
            start=live_start,
            end=latest_end,
            initial_capital=settings.initial_capital,
            transaction_cost_bps=settings.transaction_cost_bps,
        )
        live_benchmark = run_portfolio_backtest(
            panel,
            generate_equal_weight_targets(live_signal_days),
            start=live_start,
            end=latest_end,
            initial_capital=settings.initial_capital,
            transaction_cost_bps=settings.transaction_cost_bps,
        )
        ledger = build_position_ledger(
            panel,
            live_targets,
            variant=variant,
            latest_date=latest_date,
        )
        events = build_event_table(live_targets, ledger, variant=variant)
        summary_rows.append(
            {
                **_summary_row(
                    variant,
                    "2025-2026",
                    live_result.summary,
                    live_benchmark.summary,
                ),
                **_position_summary(ledger),
                "EntryBlocks": int(
                    live_targets.get(
                        "EntryBlocked",
                        pd.Series(False, index=live_targets.index),
                    )
                    .fillna(False)
                    .sum()
                ),
            }
        )
        equity = live_result.daily.copy()
        equity["Variant"] = variant
        equity["NormalizedEquity"] = (
            equity["Equity"] / float(equity["Equity"].iloc[0]) * 100
        )
        equity_frames.append(equity)
        event_frames.append(events)
        ledger_frames.append(ledger)

    summary = pd.DataFrame(summary_rows)
    summary = _add_v5_deltas(summary)
    sensitivity = _replacement_hurdle_sensitivity(
        panel,
        settings,
        base_params,
        latest_end=latest_end,
        live_start=live_start,
    )
    equity = pd.concat(equity_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    ledger = pd.concat(ledger_frames, ignore_index=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v6_comparison"
            / f"{timestamp}_frozen_v5_ablation"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "v6_variant_summary.csv"
    sensitivity_path = destination / "v6_b_threshold_sensitivity.csv"
    equity_path = destination / "v6_equity_curves.csv"
    events_path = destination / "v6_rebalance_events.csv"
    ledger_path = destination / "v6_position_ledger.csv"
    html_path = destination / "v6_rebalancing_comparison.html"
    atomic_to_csv(summary, summary_path, index=False)
    atomic_to_csv(sensitivity, sensitivity_path, index=False)
    atomic_to_csv(equity, equity_path, index=False)
    atomic_to_csv(events, events_path, index=False)
    atomic_to_csv(ledger, ledger_path, index=False)
    _write_html_report(
        html_path,
        summary=summary,
        sensitivity=sensitivity,
        equity=equity,
        events=events,
        ledger=ledger,
        latest_date=latest_date,
    )
    return V6ComparisonArtifacts(
        output_dir=destination,
        summary_csv=summary_path,
        sensitivity_csv=sensitivity_path,
        equity_csv=equity_path,
        events_csv=events_path,
        ledger_csv=ledger_path,
        html_report=html_path,
    )


def build_position_ledger(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    variant: str,
    latest_date: pd.Timestamp,
) -> pd.DataFrame:
    prices = panel.loc[:, ["Date", "Ticker", "Open", "Close"]].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    price_rows = prices.set_index(["Date", "Ticker"])
    market_dates = pd.DatetimeIndex(prices["Date"].drop_duplicates()).sort_values()
    next_dates = {
        market_dates[index]: market_dates[index + 1]
        for index in range(len(market_dates) - 1)
    }
    latest_close = (
        prices.loc[prices["Date"].eq(latest_date)]
        .set_index("Ticker")["Close"]
        .to_dict()
    )
    state: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    closed: list[dict[str, object]] = []
    event_rows = targets.loc[
        targets["TradeAction"].isin(["BUY", "SELL"])
    ].sort_values(["Date", "TradeAction", "Ticker"])
    for row in event_rows.itertuples(index=False):
        signal_date = pd.Timestamp(row.Date)
        execution_date = next_dates.get(signal_date)
        execution_price = _execution_price(
            price_rows,
            signal_date,
            execution_date,
            str(row.Ticker),
        )
        ticker = str(row.Ticker)
        if row.TradeAction == "BUY":
            counts[ticker] = counts.get(ticker, 0) + 1
            state[ticker] = {
                "Variant": variant,
                "PositionId": f"{ticker}-{counts[ticker]:02d}",
                "Ticker": ticker,
                "EntrySignalDate": signal_date,
                "EntryExecutionDate": execution_date,
                "EntryExecutionPrice": execution_price,
                "EntryRank": _optional_number(row, "Rank"),
                "EntryAlphaScore": _optional_number(row, "AlphaScore"),
            }
            continue
        entry = state.get(ticker)
        if entry is None:
            continue
        if execution_date is None or execution_price is None:
            entry["PendingExitSignalDate"] = signal_date
            entry["PendingExitReason"] = getattr(row, "ExitReason", None)
            continue
        state.pop(ticker)
        entry_price = entry["EntryExecutionPrice"]
        execution_return = _price_return(execution_price, entry_price)
        mark_price = _optional_float(latest_close.get(ticker))
        closed.append(
            {
                **entry,
                "Status": "CLOSED",
                "ExitSignalDate": signal_date,
                "ExitExecutionDate": execution_date,
                "ExitExecutionPrice": execution_price,
                "ExitReason": getattr(row, "ExitReason", None),
                "ExecutionPriceReturn": execution_return,
                "LatestMarkPrice": mark_price,
                "HoldToLatestReturn": _price_return(mark_price, entry_price),
            }
        )
    for ticker, entry in state.items():
        mark_price = _optional_float(latest_close.get(ticker))
        entry_price = entry["EntryExecutionPrice"]
        pending_exit_date = entry.get("PendingExitSignalDate")
        closed.append(
            {
                **entry,
                "Status": (
                    "PENDING_EXIT"
                    if pending_exit_date is not None
                    else (
                        "OPEN"
                        if entry["EntryExecutionDate"] is not None
                        else "PENDING"
                    )
                ),
                "ExitSignalDate": pending_exit_date,
                "ExitExecutionDate": None,
                "ExitExecutionPrice": None,
                "ExitReason": entry.get("PendingExitReason"),
                "ExecutionPriceReturn": _price_return(mark_price, entry_price),
                "LatestMarkPrice": mark_price,
                "HoldToLatestReturn": _price_return(mark_price, entry_price),
            }
        )
    columns = [
        "Variant",
        "PositionId",
        "Ticker",
        "Status",
        "EntrySignalDate",
        "EntryExecutionDate",
        "EntryExecutionPrice",
        "EntryRank",
        "EntryAlphaScore",
        "ExitSignalDate",
        "ExitExecutionDate",
        "ExitExecutionPrice",
        "ExitReason",
        "ExecutionPriceReturn",
        "LatestMarkPrice",
        "HoldToLatestReturn",
    ]
    return pd.DataFrame(closed, columns=columns).sort_values(
        ["EntrySignalDate", "Ticker"]
    )


def build_event_table(
    targets: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    events = targets.loc[
        targets["TradeAction"].isin(["BUY", "SELL"])
    ].copy()
    events.insert(0, "Variant", variant)
    buy_outcomes = ledger.set_index(["Ticker", "EntrySignalDate"])[
        "ExecutionPriceReturn"
    ].to_dict()
    sell_outcomes = ledger.loc[ledger["Status"].eq("CLOSED")].set_index(
        ["Ticker", "ExitSignalDate"]
    )["ExecutionPriceReturn"].to_dict()
    hold_outcomes = ledger.loc[ledger["Status"].eq("CLOSED")].set_index(
        ["Ticker", "ExitSignalDate"]
    )["HoldToLatestReturn"].to_dict()
    events["PositionReturn"] = [
        (
            buy_outcomes.get((ticker, pd.Timestamp(date)))
            if action == "BUY"
            else sell_outcomes.get((ticker, pd.Timestamp(date)))
        )
        for ticker, date, action in zip(
            events["Ticker"],
            events["Date"],
            events["TradeAction"],
            strict=True,
        )
    ]
    events["HoldToLatestReturn"] = [
        (
            None
            if action == "BUY"
            else hold_outcomes.get((ticker, pd.Timestamp(date)))
        )
        for ticker, date, action in zip(
            events["Ticker"],
            events["Date"],
            events["TradeAction"],
            strict=True,
        )
    ]
    desired = [
        "Variant",
        "Date",
        "Ticker",
        "Company",
        "TradeAction",
        "Rank",
        "AlphaScore",
        "Close",
        "SignalReferenceReturn",
        "ExitReason",
        "PositionReturn",
        "HoldToLatestReturn",
        "ReplacementScoreAdvantage",
        "ProfitExitStreak",
        "EntryBlocked",
        "EntryBlockReason",
    ]
    return events.loc[:, [column for column in desired if column in events]]


def _evaluate_period(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    start: str,
    end: str,
) -> dict[str, Any]:
    signal_days = signal_day_panel(
        panel,
        start,
        end,
        settings.rebalance_weekday,
    )
    targets = generate_rebalance_targets(score_panel(signal_days, params), params)
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
        generate_equal_weight_targets(signal_days),
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    return {"strategy": strategy, "benchmark": benchmark, "targets": targets}


def _replacement_hurdle_sensitivity(
    panel: pd.DataFrame,
    settings: ResearchSettings,
    base_params: StrategyParams,
    *,
    latest_end: str,
    live_start: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    thresholds = (0.0, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10)
    for threshold in thresholds:
        params = replace(
            base_params,
            replacement_score_advantage=threshold,
        )
        for period, (start, configured_end) in settings.validation_periods.items():
            end = min(configured_end, latest_end)
            result = _evaluate_period(panel, params, settings, start, end)
            rows.append(
                {
                    "Threshold": threshold,
                    "Period": period,
                    "StrategyROI": result["strategy"].summary.roi_percent,
                    "MaxDrawdown": (
                        result["strategy"].summary.max_drawdown_percent
                    ),
                    "Sharpe": result["strategy"].summary.sharpe_ratio,
                    "AnnualizedTurnover": (
                        result["strategy"].summary.annualized_turnover
                    ),
                }
            )
        signal_days = signal_day_panel(
            panel,
            live_start,
            latest_end,
            settings.rebalance_weekday,
        )
        targets = generate_rebalance_targets(
            score_panel(signal_days, params),
            params,
        )
        result = run_portfolio_backtest(
            panel,
            targets,
            start=live_start,
            end=latest_end,
            initial_capital=settings.initial_capital,
            transaction_cost_bps=settings.transaction_cost_bps,
        )
        rows.append(
            {
                "Threshold": threshold,
                "Period": "2025-2026",
                "StrategyROI": result.summary.roi_percent,
                "MaxDrawdown": result.summary.max_drawdown_percent,
                "Sharpe": result.summary.sharpe_ratio,
                "AnnualizedTurnover": result.summary.annualized_turnover,
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[
        result["Threshold"].eq(0.0),
        ["Period", "StrategyROI"],
    ].rename(columns={"StrategyROI": "V5ROI"})
    result = result.merge(baseline, on="Period", how="left")
    result["ROIvsV5"] = result["StrategyROI"] - result["V5ROI"]
    return result


def _summary_row(
    variant: str,
    period: str,
    strategy: Any,
    benchmark: Any,
) -> dict[str, object]:
    return {
        "Variant": variant,
        "VariantLabel": VARIANT_LABELS[variant],
        "Period": period,
        "StrategyROI": strategy.roi_percent,
        "BenchmarkROI": benchmark.roi_percent,
        "ExcessROI": strategy.roi_percent - benchmark.roi_percent,
        "MaxDrawdown": strategy.max_drawdown_percent,
        "Sharpe": strategy.sharpe_ratio,
        "AnnualizedTurnover": strategy.annualized_turnover,
        "TickerTrades": strategy.ticker_trades,
        "Rebalances": strategy.rebalance_count,
    }


def _position_summary(ledger: pd.DataFrame) -> dict[str, object]:
    closed = ledger.loc[ledger["Status"].eq("CLOSED")]
    returns = pd.to_numeric(
        closed["ExecutionPriceReturn"],
        errors="coerce",
    )
    return {
        "ClosedPositions": len(closed),
        "LossPositions": int(returns.lt(0).sum()),
        "WorstPositionReturn": (
            float(returns.min() * 100)
            if returns.notna().any()
            else np.nan
        ),
        "MeanPositionReturn": (
            float(returns.mean() * 100)
            if returns.notna().any()
            else np.nan
        ),
    }


def _add_v5_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        summary.loc[
            summary["Variant"].eq("V5"),
            ["Period", "StrategyROI", "MaxDrawdown"],
        ]
        .rename(
            columns={
                "StrategyROI": "V5ROI",
                "MaxDrawdown": "V5MaxDrawdown",
            }
        )
    )
    result = summary.merge(baseline, on="Period", how="left")
    result["ROIvsV5"] = result["StrategyROI"] - result["V5ROI"]
    result["DrawdownVsV5"] = (
        result["MaxDrawdown"] - result["V5MaxDrawdown"]
    )
    return result


def _execution_price(
    price_rows: pd.DataFrame,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp | None,
    ticker: str,
) -> float | None:
    if execution_date is None:
        return None
    try:
        value = price_rows.loc[(execution_date, ticker), "Open"]
    except KeyError:
        value = np.nan
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    numeric = _optional_float(value)
    if numeric is not None and numeric > 0:
        return numeric
    try:
        fallback = price_rows.loc[(signal_date, ticker), "Close"]
    except KeyError:
        return None
    if isinstance(fallback, pd.Series):
        fallback = fallback.iloc[0]
    return _optional_float(fallback)


def _price_return(
    exit_price: float | None,
    entry_price: float | None,
) -> float | None:
    if (
        exit_price is None
        or entry_price is None
        or not np.isfinite(exit_price)
        or not np.isfinite(entry_price)
        or entry_price <= 0
    ):
        return None
    return exit_price / entry_price - 1


def _optional_float(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else None


def _optional_number(row: object, column: str) -> float | None:
    return _optional_float(getattr(row, column, None))


def _write_html_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    equity: pd.DataFrame,
    events: pd.DataFrame,
    ledger: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> None:
    figure = go.Figure()
    for variant, variant_label in VARIANT_LABELS.items():
        frame = equity.loc[equity["Variant"].eq(variant)]
        figure.add_trace(
            go.Scatter(
                x=frame["Date"],
                y=frame["NormalizedEquity"],
                mode="lines",
                name=variant_label,
                line={"width": 4 if variant == "V5" else 2},
            )
        )
    figure.update_layout(
        title="2025–2026 연속 운용 자산곡선 · 시작값 100",
        xaxis_title="날짜",
        yaxis_title="정규화 자산",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 55, "r": 25, "t": 60, "b": 90},
    )
    chart = plot(
        figure,
        output_type="div",
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    table = summary.loc[
        summary["Period"].isin(["2025", "2026"]),
        [
            "VariantLabel",
            "Period",
            "StrategyROI",
            "ROIvsV5",
            "MaxDrawdown",
            "Sharpe",
            "AnnualizedTurnover",
        ],
    ].copy()
    for column in (
        "StrategyROI",
        "ROIvsV5",
        "MaxDrawdown",
        "Sharpe",
        "AnnualizedTurnover",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    table.columns = [
        "변형",
        "기간",
        "ROI %",
        "V5 대비 %p",
        "최대낙폭 %",
        "Sharpe",
        "연환산 회전율",
    ]
    sensitivity_table = sensitivity.loc[
        sensitivity["Period"].eq("2025-2026"),
        [
            "Threshold",
            "StrategyROI",
            "ROIvsV5",
            "MaxDrawdown",
            "Sharpe",
            "AnnualizedTurnover",
        ],
    ].copy()
    sensitivity_table["Threshold"] = sensitivity_table["Threshold"].map(
        lambda value: f"{value:.3f}"
    )
    for column in sensitivity_table.columns[1:]:
        sensitivity_table[column] = pd.to_numeric(
            sensitivity_table[column],
            errors="coerce",
        ).round(2)
    sensitivity_table.columns = [
        "점수 문턱",
        "연속 ROI %",
        "V5 대비 %p",
        "최대낙폭 %",
        "Sharpe",
        "연환산 회전율",
    ]
    event_payload = _event_payload(events)
    ledger_payload = _ledger_payload(ledger)
    body = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V6 frozen V5 rebalancing comparison</title>
<style>
:root {{ color-scheme: light; --bg:#f5f7fb; --card:#fff; --text:#172033;
  --muted:#667085; --line:#d8dee9; --blue:#2563eb; --green:#16803c;
  --red:#c62828; --amber:#a15c00; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font-family:Inter,"Noto Sans KR",system-ui,sans-serif; }}
main {{ max-width:1500px; margin:auto; padding:24px; }}
h1 {{ margin:0 0 6px; font-size:28px; }}
h2 {{ margin:0 0 14px; font-size:20px; }}
.sub {{ color:var(--muted); margin:0 0 20px; }}
.card {{ background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:18px; margin:14px 0; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--line);
  text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
select,input,button {{ font:inherit; }}
.controls {{ display:grid; grid-template-columns:minmax(180px,260px) 1fr;
  gap:16px; align-items:end; margin-bottom:16px; }}
label {{ display:grid; gap:6px; color:var(--muted); }}
select {{ padding:9px; border:1px solid var(--line); border-radius:8px;
  background:var(--card); color:var(--text); }}
.event-date {{ font-weight:700; margin:10px 0; }}
.flow {{ display:grid; grid-template-columns:1fr auto 1fr; gap:16px;
  align-items:center; }}
.node {{ border-left:5px solid var(--blue); background:#f3f6fb;
  padding:12px; margin:8px 0; }}
.node.in {{ border-left-color:var(--green); }}
.node.loss {{ border-left-color:var(--red); }}
.node strong {{ display:flex; justify-content:space-between; gap:12px; }}
.meta {{ color:var(--muted); font-size:13px; margin-top:4px; }}
.arrow {{ color:var(--muted); text-align:center; font-size:24px; }}
.positions td.loss {{ color:var(--red); font-weight:700; }}
.positions td.gain {{ color:var(--green); font-weight:700; }}
@media(max-width:700px) {{
  main {{ padding:12px; }}
  .controls,.flow {{ grid-template-columns:1fr; }}
  .arrow {{ transform:rotate(90deg); }}
}}
</style>
</head>
<body>
<main>
<h1>V6 리밸런싱 규칙 비교</h1>
<p class="sub">V5 팩터 가중치와 진입 필터 고정 · 최신 가격
{html.escape(str(latest_date.date()))} · 사후진단</p>
<section class="card">{chart}</section>
<section class="card">
<h2>연도별 성과</h2>
<div class="table-wrap">{table.to_html(index=False, border=0)}</div>
</section>
<section class="card">
<h2>V6-B 교체 문턱 민감도 · 2025–2026 연속 운용</h2>
<div class="table-wrap">{sensitivity_table.to_html(index=False, border=0)}</div>
</section>
<section class="card">
<h2>리밸런싱 탐색</h2>
<div class="controls">
<label>변형<select id="variant"></select></label>
<label>이벤트 <span id="event-count"></span>
<input id="event-range" type="range" min="0" value="0" step="1"></label>
</div>
<div id="event-date" class="event-date"></div>
<div class="flow">
<div><h3>매도</h3><div id="outgoing"></div></div>
<div class="arrow">→</div>
<div><h3>매수</h3><div id="incoming"></div></div>
</div>
</section>
<section class="card">
<h2>포지션 결과</h2>
<div id="positions" class="table-wrap positions"></div>
</section>
</main>
<script>
const labels={json.dumps(VARIANT_LABELS, ensure_ascii=False)};
const eventData={json.dumps(event_payload, ensure_ascii=False)};
const ledgerData={json.dumps(ledger_payload, ensure_ascii=False)};
const variant=document.getElementById("variant");
const range=document.getElementById("event-range");
Object.keys(labels).forEach(key=>{{
  const option=document.createElement("option");
  option.value=key; option.textContent=labels[key]; variant.append(option);
}});
function pct(value) {{
  if(value===null || value===undefined || Number.isNaN(value)) return "–";
  return `${{value>=0?"+":""}}${{(value*100).toFixed(2)}}%`;
}}
function node(item,type) {{
  const loss=(item.positionReturn??0)<0;
  return `<div class="node ${{type}} ${{loss?"loss":""}}">
    <strong><span>${{item.ticker}}</span>
    <span>${{pct(item.positionReturn)}}</span></strong>
    <div class="meta">${{item.rank===null?"순위 밖":item.rank+"위"}} ·
    신호수익 ${{pct(item.signalReturn)}} · ${{item.reason||"진입"}}</div>
    ${{type==="out"?`<div class="meta">계속 보유 시
    ${{pct(item.holdToLatest)}}</div>`:""}}
    </div>`;
}}
function renderPositions(key) {{
  const rows=ledgerData[key]||[];
  let out="<table><thead><tr><th>종목</th><th>진입</th><th>청산/상태</th>"+
    "<th>실현·평가수익</th><th>최신까지 보유</th><th>사유</th></tr></thead><tbody>";
  rows.forEach(row=>{{
    const cls=(row.positionReturn??0)<0?"loss":"gain";
    out+=`<tr><td>${{row.positionId}}</td><td>${{row.entryDate}}</td>
      <td>${{row.exitDate||row.status}}</td><td class="${{cls}}">
      ${{pct(row.positionReturn)}}</td><td>${{pct(row.holdToLatest)}}</td>
      <td>${{row.reason||"–"}}</td></tr>`;
  }});
  document.getElementById("positions").innerHTML=out+"</tbody></table>";
}}
function renderEvent() {{
  const rows=eventData[variant.value]||[];
  if(!rows.length) return;
  const index=Math.min(Number(range.value),rows.length-1);
  const event=rows[index];
  range.max=rows.length-1;
  document.getElementById("event-count").textContent=`${{index+1}} / ${{rows.length}}`;
  document.getElementById("event-date").textContent=event.date;
  document.getElementById("outgoing").innerHTML=
    event.sells.map(item=>node(item,"out")).join("")||"<div class='meta'>없음</div>";
  document.getElementById("incoming").innerHTML=
    event.buys.map(item=>node(item,"in")).join("")||"<div class='meta'>없음</div>";
}}
variant.addEventListener("change",()=>{{range.value=0;renderEvent();
  renderPositions(variant.value);}});
range.addEventListener("input",renderEvent);
variant.value="V6-ALL"; renderEvent(); renderPositions(variant.value);
</script>
</body>
</html>"""
    _atomic_text(path, body)


def _event_payload(events: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    payload: dict[str, list[dict[str, object]]] = {}
    for variant, variant_rows in events.groupby("Variant", sort=False):
        event_list: list[dict[str, object]] = []
        for date, group in variant_rows.groupby("Date", sort=True):
            def values(
                event_group: pd.DataFrame,
                action: str,
            ) -> list[dict[str, object]]:
                rows = event_group.loc[
                    event_group["TradeAction"].eq(action)
                ]
                return [
                    {
                        "ticker": str(row.Ticker),
                        "rank": _json_number(row.Rank),
                        "positionReturn": _json_number(row.PositionReturn),
                        "holdToLatest": _json_number(row.HoldToLatestReturn),
                        "signalReturn": _json_number(row.SignalReferenceReturn),
                        "reason": (
                            None
                            if pd.isna(row.ExitReason)
                            else str(row.ExitReason)
                        ),
                    }
                    for row in rows.itertuples(index=False)
                ]

            event_list.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "sells": values(group, "SELL"),
                    "buys": values(group, "BUY"),
                }
            )
        payload[str(variant)] = event_list
    return payload


def _ledger_payload(ledger: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    payload: dict[str, list[dict[str, object]]] = {}
    for variant, rows in ledger.groupby("Variant", sort=False):
        payload[str(variant)] = [
            {
                "positionId": str(row.PositionId),
                "entryDate": _json_date(row.EntryExecutionDate),
                "exitDate": _json_date(row.ExitExecutionDate),
                "status": str(row.Status),
                "positionReturn": _json_number(row.ExecutionPriceReturn),
                "holdToLatest": _json_number(row.HoldToLatestReturn),
                "reason": (
                    None if pd.isna(row.ExitReason) else str(row.ExitReason)
                ),
            }
            for row in rows.itertuples(index=False)
        ]
    return payload


def _json_number(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else None


def _json_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
