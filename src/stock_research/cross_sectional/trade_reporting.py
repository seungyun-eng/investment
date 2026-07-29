from __future__ import annotations

import html
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Line,
    Polygon,
    PolyLine,
    Rect,
    String,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths
from stock_research.tsla_integrated.data import load_equity_prices

FACTOR_COLUMNS = (
    "MomentumFactor",
    "TrendFactor",
    "GrowthFactor",
    "QualityFactor",
    "RiskControlFactor",
)
FACTOR_LABELS = {
    "MomentumFactor": "모멘텀",
    "TrendFactor": "추세",
    "GrowthFactor": "재무 성장",
    "QualityFactor": "가치·품질",
    "RiskControlFactor": "위험통제",
}
ACTION_LABELS = {
    "BUY": "매수",
    "SELL": "매도",
    "HOLD": "보유",
    "WATCH": "관찰",
    "AVOID": "제외",
}
EXIT_LABELS = {
    "PROFITABLE_ROTATION": "수익 순환매",
    "CONVICTION_BREAKDOWN": "투자근거 훼손",
    "HARD_STOP": "하드스톱",
    "TRAILING_STOP": "추적 손절",
    "MISSING_REFERENCE_EXIT": "기준가격 누락",
    "RANK_OR_FILTER_EXIT": "순위·필터 이탈",
}


@dataclass(frozen=True)
class TradeReportArtifacts:
    output_dir: Path
    html_report: Path
    pdf_report: Path
    position_ledger: Path
    trade_events: Path
    weekly_signals: Path
    reconciliation: Path


def latest_loss_protected_strategy(results_root: Path) -> Path:
    candidates = list(
        (
            results_root / "Cross_Sectional" / "rank_signals"
        ).glob("*loss_protected_v5*/selected_strategy.json")
    )
    if not candidates:
        raise FileNotFoundError(
            "No loss_protected_v5 selected_strategy.json was found."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def generate_v5_trade_report(
    paths: ProjectPaths,
    strategy_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> TradeReportArtifacts:
    strategy_path = Path(strategy_path).expanduser().resolve()
    strategy_dir = strategy_path.parent
    manifest = json.loads(strategy_path.read_text(encoding="utf-8"))
    history_path = strategy_dir / "daily_signal_history.csv"
    audit_path = strategy_dir / "universe_data_audit.csv"
    validation_path = strategy_dir / "validation_summary.csv"
    for required in (history_path, audit_path, validation_path):
        if not required.exists():
            raise FileNotFoundError(f"Required V5 artifact is missing: {required}")

    history = _load_signal_history(history_path)
    audit = pd.read_csv(audit_path)
    validation = pd.read_csv(validation_path)
    prices = _load_report_prices(paths, audit, history)
    enriched = add_signal_strengths(history)
    events = _trade_events(enriched)
    events, reconciliation = attach_execution_prices(events, prices)
    latest_date = _latest_complete_date(enriched)
    ledger = build_position_ledger(events, prices, latest_date=latest_date)
    events = attach_position_outcomes(events, ledger)
    weekly = enriched.loc[enriched["IsRebalanceSignal"]].copy()

    label = str(manifest.get("settings", {}).get("research_label", "v5"))
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "trade_reports"
            / f"{label}_{latest_date.date()}"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)

    ledger_path = destination / "v5_position_ledger.csv"
    event_path = destination / "v5_trade_events.csv"
    weekly_path = destination / "v5_weekly_signal_strength.csv"
    reconciliation_path = destination / "v5_price_reconciliation.csv"
    atomic_to_csv(ledger, ledger_path, index=False)
    atomic_to_csv(events, event_path, index=False)
    atomic_to_csv(weekly[_weekly_output_columns(weekly)], weekly_path, index=False)
    atomic_to_csv(reconciliation, reconciliation_path, index=False)

    html_path = destination / "v5_trade_signal_report.html"
    pdf_path = destination / "v5_trade_signal_report.pdf"
    _write_html_report(
        html_path,
        manifest=manifest,
        validation=validation,
        history=enriched,
        prices=prices,
        events=events,
        ledger=ledger,
        reconciliation=reconciliation,
    )
    _write_pdf_report(
        pdf_path,
        manifest=manifest,
        validation=validation,
        history=enriched,
        prices=prices,
        events=events,
        ledger=ledger,
        reconciliation=reconciliation,
    )
    return TradeReportArtifacts(
        output_dir=destination,
        html_report=html_path,
        pdf_report=pdf_path,
        position_ledger=ledger_path,
        trade_events=event_path,
        weekly_signals=weekly_path,
        reconciliation=reconciliation_path,
    )


def add_signal_strengths(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert centered cross-sectional scores into intuitive 0-100 strengths."""

    result = frame.copy()
    alpha = pd.to_numeric(result["AlphaScore"], errors="coerce")
    result["CompositeStrength"] = ((alpha + 0.5) * 100).clip(0, 100)
    size = pd.to_numeric(result["CrossSectionSize"], errors="coerce")
    rank = pd.to_numeric(result["Rank"], errors="coerce")
    denominator = (size - 1).where(size.gt(1))
    result["RankStrength"] = (
        100 * (1 - (rank - 1) / denominator)
    ).clip(0, 100)
    for column in FACTOR_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}Strength"] = ((values + 0.5) * 100).clip(0, 100)
    return result


def attach_execution_prices(
    events: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the next union-market session open used by the portfolio engine."""

    result = events.sort_values(["Date", "Ticker"]).reset_index(drop=True).copy()
    market_dates = pd.DatetimeIndex(prices["Date"].dropna().unique()).sort_values()
    if market_dates.empty:
        raise ValueError("No price dates are available for execution matching.")
    lookup = prices.set_index(["Ticker", "Date"]).sort_index()
    reconciliation_rows: list[dict[str, object]] = []
    execution_dates: list[pd.Timestamp | pd.NaT] = []
    execution_prices: list[float] = []
    price_sources: list[str] = []
    for row in result.itertuples(index=False):
        signal_date = pd.Timestamp(row.Date)
        next_positions = np.flatnonzero(market_dates > signal_date)
        if len(next_positions) == 0:
            execution_dates.append(pd.NaT)
            execution_prices.append(np.nan)
            price_sources.append("NO_FUTURE_SESSION")
            continue
        execution_date = pd.Timestamp(market_dates[next_positions[0]])
        ticker = str(row.Ticker)
        open_price = _lookup_number(lookup, ticker, execution_date, "Open")
        if pd.notna(open_price) and open_price > 0:
            execution_price = float(open_price)
            source = "NEXT_SESSION_OPEN"
        else:
            prior = prices.loc[
                prices["Ticker"].eq(ticker)
                & prices["Date"].lt(execution_date)
                & prices["Close"].notna()
            ].sort_values("Date")
            execution_price = (
                float(prior["Close"].iloc[-1]) if not prior.empty else np.nan
            )
            source = "PRIOR_CLOSE_FALLBACK"
        source_close = _lookup_number(lookup, ticker, signal_date, "Close")
        signal_close = float(row.Close) if pd.notna(row.Close) else np.nan
        difference = (
            source_close - signal_close
            if pd.notna(source_close) and pd.notna(signal_close)
            else np.nan
        )
        reconciliation_rows.append(
            {
                "SignalDate": signal_date,
                "Ticker": ticker,
                "SignalHistoryClose": signal_close,
                "ProcessedPriceClose": source_close,
                "CloseDifference": difference,
                "CloseMatches": bool(
                    pd.notna(difference)
                    and math.isclose(
                        source_close,
                        signal_close,
                        rel_tol=1e-9,
                        abs_tol=1e-8,
                    )
                ),
                "ExecutionDate": execution_date,
                "ExecutionPrice": execution_price,
                "ExecutionPriceSource": source,
            }
        )
        execution_dates.append(execution_date)
        execution_prices.append(execution_price)
        price_sources.append(source)
    result["ExecutionDate"] = execution_dates
    result["ExecutionPrice"] = execution_prices
    result["ExecutionPriceSource"] = price_sources
    return result, pd.DataFrame(reconciliation_rows)


def build_position_ledger(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    latest_date: pd.Timestamp,
) -> pd.DataFrame:
    """Pair full-entry BUY and full-exit SELL signals into position episodes."""

    open_positions: dict[str, pd.Series] = {}
    positions: list[dict[str, object]] = []
    counters: dict[str, int] = {}
    ordered = events.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    for _, event in ordered.iterrows():
        ticker = str(event["Ticker"])
        action = str(event["TradeAction"])
        if action == "BUY":
            if ticker in open_positions:
                raise ValueError(f"Duplicate BUY without SELL for {ticker}.")
            open_positions[ticker] = event
            continue
        if action != "SELL":
            continue
        entry = open_positions.pop(ticker, None)
        if entry is None:
            raise ValueError(f"SELL without an open BUY for {ticker}.")
        counters[ticker] = counters.get(ticker, 0) + 1
        positions.append(
            _position_row(
                ticker,
                counters[ticker],
                entry,
                event,
                status="CLOSED",
            )
        )

    latest_prices = (
        prices.loc[prices["Date"].le(latest_date)]
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .set_index("Ticker")
    )
    for ticker, entry in sorted(open_positions.items()):
        counters[ticker] = counters.get(ticker, 0) + 1
        if ticker in latest_prices.index:
            latest = latest_prices.loc[ticker]
            mark_price = float(latest["Close"])
            mark_date = pd.Timestamp(latest["Date"])
        else:
            mark_price = np.nan
            mark_date = latest_date
        mark = entry.copy()
        mark["Date"] = mark_date
        mark["ExecutionDate"] = mark_date
        mark["ExecutionPrice"] = mark_price
        mark["Close"] = mark_price
        mark["TradeAction"] = "OPEN"
        mark["ExitReason"] = ""
        positions.append(
            _position_row(
                ticker,
                counters[ticker],
                entry,
                mark,
                status="OPEN",
            )
        )
    ledger = pd.DataFrame(positions)
    if ledger.empty:
        return ledger
    return ledger.sort_values(
        ["EntrySignalDate", "Ticker", "PositionNumber"]
    ).reset_index(drop=True)


def attach_position_outcomes(
    events: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Attach paired entry/exit or latest-mark P/L to each position event."""

    result = events.copy()
    result["PositionId"] = ""
    result["PositionStatus"] = ""
    for column in (
        "PositionEntryPrice",
        "PositionExitOrMarkPrice",
        "PositionPnLPerShare",
        "PositionReturn",
        "PositionHoldingDays",
    ):
        result[column] = np.nan
    for position in ledger.itertuples(index=False):
        buy_mask = (
            result["Ticker"].eq(position.Ticker)
            & result["TradeAction"].eq("BUY")
            & result["Date"].eq(pd.Timestamp(position.EntrySignalDate))
        )
        if pd.notna(position.ExitSignalDate):
            sell_mask = (
                result["Ticker"].eq(position.Ticker)
                & result["TradeAction"].eq("SELL")
                & result["Date"].eq(pd.Timestamp(position.ExitSignalDate))
            )
        else:
            sell_mask = pd.Series(False, index=result.index)
        mask = buy_mask | sell_mask
        result.loc[mask, "PositionId"] = position.PositionId
        result.loc[mask, "PositionStatus"] = position.Status
        result.loc[mask, "PositionEntryPrice"] = position.EntryExecutionPrice
        result.loc[mask, "PositionExitOrMarkPrice"] = (
            position.ExitExecutionPrice
        )
        result.loc[mask, "PositionPnLPerShare"] = position.PerSharePnL
        result.loc[mask, "PositionReturn"] = position.ExecutionPriceReturn
        result.loc[mask, "PositionHoldingDays"] = position.HoldingCalendarDays
    return result


def _load_signal_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in (
        "Close",
        "Rank",
        "AlphaScore",
        "TargetWeight",
        "Return21",
        "Return63",
        "Return126",
        "Trend200",
        "SignalReferenceReturn",
        "HoldingRebalances",
        "CrossSectionSize",
        *FACTOR_COLUMNS,
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "ModelSelected",
        "Qualified",
        "FinancialStale",
        "IsRebalanceSignal",
    ):
        frame[column] = (
            frame[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )
    frame["ExitReason"] = frame["ExitReason"].fillna("").astype(str)
    return frame.dropna(subset=["Date"]).sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)


def _latest_complete_date(frame: pd.DataFrame) -> pd.Timestamp:
    coverage = frame.groupby("Date")["Ticker"].nunique()
    if coverage.empty:
        raise ValueError("Signal history contains no dated ticker rows.")
    maximum = int(coverage.max())
    return pd.Timestamp(coverage.loc[coverage.eq(maximum)].index.max())


def _report_label(manifest: dict[str, Any]) -> str:
    research_label = str(
        manifest.get("settings", {}).get("research_label", "")
    ).casefold()
    if research_label.startswith("v6_b"):
        return "V6-B"
    if research_label.startswith("v6_"):
        return "V6"
    if "v5" in research_label or "loss_protected" in research_label:
        return "V5"
    return "Strategy"


def _load_report_prices(
    paths: ProjectPaths,
    audit: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    start = pd.Timestamp(history["Date"].min()) - pd.Timedelta(days=10)
    end = pd.Timestamp(history["Date"].max()) + pd.Timedelta(days=10)
    frames: list[pd.DataFrame] = []
    included = audit.loc[audit["Status"].eq("INCLUDED")]
    for row in included.itertuples(index=False):
        path = paths.processed / str(row.PriceFile)
        if not path.exists():
            raise FileNotFoundError(f"Price file referenced by V5 is missing: {path}")
        frame = load_equity_prices(path)
        frame = frame.loc[frame["Date"].between(start, end)].copy()
        frame["Ticker"] = str(row.Ticker)
        frame["Company"] = str(row.Company)
        frames.append(frame[["Date", "Ticker", "Company", "Open", "Close"]])
    if not frames:
        raise ValueError("No included price files were loaded.")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


def _trade_events(history: pd.DataFrame) -> pd.DataFrame:
    events = history.loc[
        history["TradeAction"].isin(["BUY", "SELL"])
    ].copy()
    columns = [
        "Date",
        "Ticker",
        "Company",
        "TradeAction",
        "DailySignal",
        "Close",
        "Rank",
        "AlphaScore",
        "CompositeStrength",
        "RankStrength",
        "Return126",
        "Trend200",
        "SignalReferenceReturn",
        "HoldingRebalances",
        "ExitReason",
        "FinancialStale",
        *FACTOR_COLUMNS,
        *(f"{column}Strength" for column in FACTOR_COLUMNS),
    ]
    return events[columns].sort_values(["Date", "Ticker"]).reset_index(drop=True)


def _position_row(
    ticker: str,
    position_number: int,
    entry: pd.Series,
    exit_event: pd.Series,
    *,
    status: str,
) -> dict[str, object]:
    entry_price = _series_number(entry, "ExecutionPrice")
    exit_price = _series_number(exit_event, "ExecutionPrice")
    execution_return = (
        exit_price / entry_price - 1
        if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0
        else np.nan
    )
    entry_execution_date = pd.Timestamp(entry["ExecutionDate"])
    exit_execution_date = pd.Timestamp(exit_event["ExecutionDate"])
    return {
        "PositionId": f"{ticker}-{position_number:02d}",
        "PositionNumber": position_number,
        "Ticker": ticker,
        "Company": str(entry["Company"]),
        "Status": status,
        "EntrySignalDate": pd.Timestamp(entry["Date"]),
        "EntryExecutionDate": entry_execution_date,
        "EntrySignalClose": _series_number(entry, "Close"),
        "EntryExecutionPrice": entry_price,
        "EntryRank": _series_number(entry, "Rank"),
        "EntryAlphaScore": _series_number(entry, "AlphaScore"),
        "EntryCompositeStrength": _series_number(entry, "CompositeStrength"),
        "ExitSignalDate": (
            pd.Timestamp(exit_event["Date"]) if status == "CLOSED" else pd.NaT
        ),
        "ExitExecutionDate": exit_execution_date,
        "ExitSignalClose": _series_number(exit_event, "Close"),
        "ExitExecutionPrice": exit_price,
        "ExitRank": _series_number(exit_event, "Rank"),
        "ExitAlphaScore": _series_number(exit_event, "AlphaScore"),
        "ExitCompositeStrength": _series_number(
            exit_event,
            "CompositeStrength",
        ),
        "ExitReason": str(exit_event.get("ExitReason", "") or ""),
        "SignalReferenceReturn": _series_number(
            exit_event,
            "SignalReferenceReturn",
        ),
        "ExecutionPriceReturn": execution_return,
        "PerSharePnL": (
            exit_price - entry_price
            if pd.notna(entry_price) and pd.notna(exit_price)
            else np.nan
        ),
        "HoldingCalendarDays": (
            exit_execution_date - entry_execution_date
        ).days,
        "HoldingRebalancesAtExit": _series_number(
            exit_event,
            "HoldingRebalances",
        ),
        **{
            f"Entry{column}Strength": _series_number(
                entry,
                f"{column}Strength",
            )
            for column in FACTOR_COLUMNS
        },
        **{
            f"Exit{column}Strength": _series_number(
                exit_event,
                f"{column}Strength",
            )
            for column in FACTOR_COLUMNS
        },
    }


def _weekly_output_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "Date",
        "Ticker",
        "Company",
        "DailySignal",
        "TradeAction",
        "ModelSelected",
        "TargetWeight",
        "Qualified",
        "Rank",
        "AlphaScore",
        "CompositeStrength",
        "RankStrength",
        "Close",
        "Return126",
        "Trend200",
        "SignalReferenceReturn",
        "HoldingRebalances",
        "ExitReason",
        "FinancialStale",
        *FACTOR_COLUMNS,
        *(f"{column}Strength" for column in FACTOR_COLUMNS),
    ]
    return [column for column in preferred if column in frame.columns]


def _write_html_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.DataFrame,
    events: pd.DataFrame,
    ledger: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> None:
    latest_date = _latest_complete_date(history)
    report_label = _report_label(manifest)
    latest = history.loc[history["Date"].eq(latest_date)].sort_values(
        ["ModelSelected", "Rank"],
        ascending=[False, True],
        na_position="last",
    )
    open_positions = ledger.loc[ledger["Status"].eq("OPEN")]
    closed = ledger.loc[ledger["Status"].eq("CLOSED")]
    summary = _position_summary(ledger)
    selected = manifest["selected_params"]
    generated = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    traded_tickers = sorted(events["Ticker"].unique())
    charts: list[str] = []
    for index, ticker in enumerate(traded_tickers):
        figure = _ticker_figure(
            ticker,
            prices.loc[prices["Ticker"].eq(ticker)],
            history.loc[history["Ticker"].eq(ticker)],
            events.loc[events["Ticker"].eq(ticker)],
            ledger.loc[ledger["Ticker"].eq(ticker)],
        )
        charts.append(
            "<section class='ticker-section'>"
            f"<h3>{html.escape(ticker)} · "
            f"{html.escape(_company_name(history, ticker))}</h3>"
            + figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"responsive": True, "displaylogo": False},
            )
            + _html_event_table(events.loc[events["Ticker"].eq(ticker)])
            + "</section>"
        )

    validation_table = validation.copy()
    validation_table = validation_table[
        [
            "Period",
            "StartDate",
            "EndDate",
            "StrategyROI",
            "EqualWeightUniverseROI",
            "ExcessROI",
            "MaxDrawdown",
            "Sharpe",
            "AnnualizedTurnover",
        ]
    ]
    validation_table.columns = [
        "기간",
        "시작",
        "종료",
        "전략 ROI",
        "동일가중 ROI",
        "초과 ROI",
        "최대낙폭",
        "Sharpe",
        "연환산 회전율",
    ]
    for column in ("전략 ROI", "동일가중 ROI", "초과 ROI", "최대낙폭"):
        validation_table[column] = pd.to_numeric(
            validation_table[column], errors="coerce"
        ).map(_percent_points)
    validation_table["Sharpe"] = pd.to_numeric(
        validation_table["Sharpe"], errors="coerce"
    ).map(lambda value: _number(value, 2))
    validation_table["연환산 회전율"] = pd.to_numeric(
        validation_table["연환산 회전율"], errors="coerce"
    ).map(lambda value: _number(value, 2))

    current_table = latest.loc[
        latest["ModelSelected"] | latest["DailySignal"].eq("BUY_WATCH"),
        [
            "Ticker",
            "Company",
            "DailySignal",
            "TargetWeight",
            "Rank",
            "AlphaScore",
            "CompositeStrength",
            "SignalReferenceReturn",
            "HoldingRebalances",
        ],
    ].copy()
    current_table.columns = [
        "종목",
        "회사",
        "일일 신호",
        "목표비중",
        "순위",
        "점수",
        "강도",
        "진입기준 손익",
        "보유 리밸런싱",
    ]
    current_table["목표비중"] = current_table["목표비중"].map(_percent)
    current_table["순위"] = current_table["순위"].map(_integer)
    current_table["점수"] = current_table["점수"].map(
        lambda value: _number(value, 4)
    )
    current_table["강도"] = current_table["강도"].map(
        lambda value: _number(value, 1)
    )
    current_table["진입기준 손익"] = current_table["진입기준 손익"].map(
        _percent
    )

    ledger_table = _ledger_display(ledger)
    mismatches = int((~reconciliation["CloseMatches"]).sum())
    html_document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report_label)} 매매·신호 강도 리포트</title>
<style>
:root {{ --ink:#142033; --muted:#607087; --blue:#3157d5; --green:#0f9f6e;
        --red:#d64545; --amber:#d68a00; --line:#dce3ee; --panel:#fff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:"Segoe UI","Malgun Gothic",sans-serif;
       color:var(--ink); background:#f3f6fa; }}
main {{ max-width:1540px; margin:0 auto; padding:30px; }}
h1 {{ margin:0 0 8px; font-size:32px; }}
h2 {{ margin:0 0 18px; }} h3 {{ margin:0 0 10px; }}
.meta {{ color:var(--muted); margin-bottom:22px; }}
.warning {{ border-left:5px solid var(--amber); background:#fff8e8; }}
.panel,.ticker-section {{ background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:22px; margin-bottom:22px;
  box-shadow:0 4px 14px rgba(20,32,51,.05); }}
.cards {{ display:grid; grid-template-columns:repeat(5,minmax(130px,1fr));
  gap:12px; margin-bottom:22px; }}
.card {{ background:#17243a; color:#fff; border-radius:12px; padding:16px; }}
.card .label {{ color:#aebbd0; font-size:13px; }}
.card .value {{ font-size:25px; font-weight:700; margin-top:5px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#edf2f8; color:#33445d; position:sticky; top:0; }}
th,td {{ border-bottom:1px solid #e5eaf1; padding:8px 9px; text-align:left;
         white-space:nowrap; }}
td:nth-child(n+4) {{ text-align:right; }}
.pill {{ display:inline-block; border-radius:999px; padding:2px 8px;
         background:#eaf0ff; color:#294bbd; }}
ul {{ line-height:1.65; }}
@media(max-width:900px) {{
  main {{ padding:16px; }} .grid {{ grid-template-columns:1fr; }}
  .cards {{ grid-template-columns:repeat(2,1fr); }}
}}
@media print {{
  body {{ background:white; }} main {{ max-width:none; padding:0; }}
  .panel,.ticker-section {{ box-shadow:none; break-inside:avoid; }}
}}
</style></head><body><main>
<h1>{html.escape(report_label)} 매매·신호 강도 리포트</h1>
<div class="meta">모델 상태: <span class="pill">{html.escape(str(manifest["model_status"]))}</span>
 · 신호 기간: {history["Date"].min():%Y-%m-%d} ~ {latest_date:%Y-%m-%d}
 · 생성: {html.escape(generated)}</div>
<div class="panel warning"><b>해석 주의.</b> {html.escape(report_label)}는 사후 진단 모델입니다.
 신호일 종가로 결정하고 다음 거래일 시가에 체결한 것으로 재구성했습니다.
 아래 개별 포지션 수익률은 체결가격 기준 단순 가격수익률이며, 주간 동일비중
 조정과 거래비용이 포함된 포트폴리오 기여수익률은 아닙니다.</div>
<div class="cards">
  <div class="card"><div class="label">완전 진입</div><div class="value">{int((events["TradeAction"] == "BUY").sum())}</div></div>
  <div class="card"><div class="label">완전 청산</div><div class="value">{len(closed)}</div></div>
  <div class="card"><div class="label">현재 보유</div><div class="value">{len(open_positions)}</div></div>
  <div class="card"><div class="label">종료 포지션 승률</div><div class="value">{_percent(summary["WinRate"])}</div></div>
  <div class="card"><div class="label">종료 포지션 평균 수익</div><div class="value">{_percent(summary["MeanReturn"])}</div></div>
</div>
<div class="grid">
 <div class="panel"><h2>선택된 모델 구조</h2>
 <ul>
  <li>모멘텀 {selected["momentum_weight"]:.1%}, 추세 {selected["trend_weight"]:.1%},
      재무 성장 {selected["growth_weight"]:.1%}, 가치·품질 {selected["quality_weight"]:.1%},
      위험통제 {selected["risk_control_weight"]:.1%}</li>
  <li>상위 {selected["top_k"]}종목 동일비중, 일반 퇴출 순위 {selected["exit_rank"]}위</li>
  <li>수익 순환매 +{selected["minimum_exit_gain"]:.0%},
      투자근거 훼손 순위 {selected["conviction_exit_rank"]}위 밖,
      하드스톱 {selected["hard_stop_return"]:.0%}</li>
  <li>강도 0–100 = (종합 AlphaScore + 0.5) × 100</li>
 </ul></div>
 <div class="panel"><h2>가격 연결 검증</h2>
 <ul>
  <li>매매 이벤트 {len(reconciliation)}건 중 종가 불일치 {mismatches}건</li>
  <li>다음 거래일 시가 사용: {(events["ExecutionPriceSource"] == "NEXT_SESSION_OPEN").sum()}건</li>
  <li>전일 종가 대체: {(events["ExecutionPriceSource"] == "PRIOR_CLOSE_FALLBACK").sum()}건</li>
  <li>마지막 가격 날짜: {latest_date:%Y-%m-%d}</li>
 </ul></div>
</div>
<div class="panel"><h2>검증 구간 성과</h2>
<div class="table-wrap">{validation_table.to_html(index=False, escape=True)}</div></div>
<div class="panel"><h2>최신 보유·진입 대기 신호</h2>
<div class="table-wrap">{current_table.to_html(index=False, escape=True)}</div></div>
<div class="panel"><h2>포지션 원장</h2>
<div class="table-wrap">{ledger_table.to_html(index=False, escape=True)}</div></div>
<div class="panel"><h2>리포트 읽는 법</h2><ul>
<li>가격 차트의 파란 ▲는 진입, 초록 ▼는 수익 청산, 빨간 ▼는 손실 청산입니다.</li>
<li>매도 마커에 마우스를 올리면 매수가·매도가·주당 손익·실제 수익률·
    보유기간·매도 사유가 표시됩니다.</li>
<li>강도 50은 관찰 종목군의 중앙 수준, 100에 가까울수록 상대 종합점수가 높습니다.</li>
<li>종목별 강도 선은 주간 리밸런싱 신호일만 표시합니다.</li>
<li>HOLD 중에도 목표 1/3 비중으로 맞추기 위한 수량 조정 주문은 발생할 수 있습니다.</li>
<li>2025와 2026 검증 포트폴리오는 각각 구간 시작 시점에 초기화되므로,
    아래 연속 포지션 원장과 검증 ROI를 직접 합산하면 안 됩니다.</li>
</ul></div>
<h2>종목별 가격·매매·신호 강도</h2>
{''.join(charts)}
</main></body></html>"""
    _atomic_write_text(path, html_document)


def _ticker_figure(
    ticker: str,
    prices: pd.DataFrame,
    history: pd.DataFrame,
    events: pd.DataFrame,
    ledger: pd.DataFrame,
) -> go.Figure:
    weekly = history.loc[history["IsRebalanceSignal"]].copy()
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.08,
        subplot_titles=("주가와 실제 다음 시가 체결", "주간 종합·요인 강도"),
    )
    figure.add_trace(
        go.Scatter(
            x=prices["Date"],
            y=prices["Close"],
            mode="lines",
            name="종가",
            line={"color": "#3157d5", "width": 1.7},
            hovertemplate="%{x|%Y-%m-%d}<br>종가 %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for position in ledger.itertuples(index=False):
        profitable = float(position.ExecutionPriceReturn) >= 0
        color = "#0f9f6e" if profitable else "#d64545"
        status_label = "매도가" if position.Status == "CLOSED" else "현재가"
        detail = (
            f"{position.PositionId}<br>"
            f"매수가 {position.EntryExecutionPrice:,.2f}<br>"
            f"{status_label} {position.ExitExecutionPrice:,.2f}<br>"
            f"주당 손익 {position.PerSharePnL:+,.2f}<br>"
            f"수익률 {position.ExecutionPriceReturn:+.2%}<br>"
            f"보유 {position.HoldingCalendarDays}일"
        )
        figure.add_trace(
            go.Scatter(
                x=[position.EntryExecutionDate, position.ExitExecutionDate],
                y=[position.EntryExecutionPrice, position.ExitExecutionPrice],
                mode="lines",
                name="수익 구간" if profitable else "손실 구간",
                showlegend=False,
                line={"color": color, "width": 3, "dash": "dot"},
                text=[detail, detail],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    buys = events.loc[events["TradeAction"].eq("BUY")]
    if not buys.empty:
        figure.add_trace(
            go.Scatter(
                x=buys["ExecutionDate"],
                y=buys["ExecutionPrice"],
                mode="markers+text",
                text=["매수"] * len(buys),
                textposition="top center",
                name="매수",
                marker={"color": "#2563eb", "symbol": "triangle-up", "size": 12},
                customdata=buys[
                    ["Rank", "CompositeStrength", "PositionId"]
                ].to_numpy(),
                hovertemplate=(
                    "%{customdata[2]}<br>%{x|%Y-%m-%d}<br>"
                    "매수가 %{y:,.2f}<br>진입 순위 %{customdata[0]:.0f}<br>"
                    "진입 강도 %{customdata[1]:.1f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    sells = events.loc[events["TradeAction"].eq("SELL")].copy()
    sells["ExitReasonDisplay"] = sells["ExitReason"].map(
        lambda value: EXIT_LABELS.get(str(value), str(value))
    )
    for profitable, color, name in (
        (True, "#0f9f6e", "수익 매도"),
        (False, "#d64545", "손실 매도"),
    ):
        condition = (
            sells["PositionReturn"].ge(0)
            if profitable
            else sells["PositionReturn"].lt(0)
        )
        points = sells.loc[condition]
        if points.empty:
            continue
        points = points.copy()
        points["PositionPnLDisplay"] = points["PositionPnLPerShare"].map(
            lambda value: _signed_number(value, 2)
        )
        points["SellRankDisplay"] = points["Rank"].map(
            lambda value: _integer(value) or "순위 없음"
        )
        points["SellStrengthDisplay"] = points["CompositeStrength"].map(
            lambda value: _number(value, 1)
        )
        labels = points["PositionReturn"].map(
            lambda value: f"매도 {float(value):+.1%}"
        )
        custom = points[
            [
                "PositionId",
                "PositionEntryPrice",
                "ExecutionPrice",
                "PositionPnLDisplay",
                "PositionReturn",
                "PositionHoldingDays",
                "ExitReasonDisplay",
                "SellRankDisplay",
                "SellStrengthDisplay",
            ]
        ].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=points["ExecutionDate"],
                y=points["ExecutionPrice"],
                mode="markers+text",
                text=labels,
                textposition="bottom center",
                name=name,
                marker={"color": color, "symbol": "triangle-down", "size": 13},
                customdata=custom,
                hovertemplate=(
                    "%{customdata[0]}<br>%{x|%Y-%m-%d}<br>"
                    "매수가 %{customdata[1]:,.2f}<br>"
                    "매도가 %{customdata[2]:,.2f}<br>"
                    "주당 손익 %{customdata[3]}<br>"
                    "실제 수익률 %{customdata[4]:+.2%}<br>"
                    "보유기간 %{customdata[5]:.0f}일<br>"
                    "매도 사유 %{customdata[6]}<br>"
                    "매도시 순위 %{customdata[7]}<br>"
                    "매도시 강도 %{customdata[8]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    open_positions = ledger.loc[ledger["Status"].eq("OPEN")]
    if not open_positions.empty:
        open_positions = open_positions.copy()
        open_positions["PerSharePnLDisplay"] = open_positions[
            "PerSharePnL"
        ].map(lambda value: _signed_number(value, 2))
        figure.add_trace(
            go.Scatter(
                x=open_positions["ExitExecutionDate"],
                y=open_positions["ExitExecutionPrice"],
                mode="markers+text",
                text=open_positions["ExecutionPriceReturn"].map(
                    lambda value: f"현재 {float(value):+.1%}"
                ),
                textposition="top center",
                name="현재 평가",
                marker={
                    "color": open_positions["ExecutionPriceReturn"].map(
                        lambda value: (
                            "#0f9f6e" if float(value) >= 0 else "#d64545"
                        )
                    ),
                    "symbol": "circle",
                    "size": 10,
                },
                customdata=open_positions[
                    [
                        "PositionId",
                        "EntryExecutionPrice",
                        "ExitExecutionPrice",
                        "PerSharePnLDisplay",
                        "ExecutionPriceReturn",
                        "HoldingCalendarDays",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}<br>%{x|%Y-%m-%d}<br>"
                    "매수가 %{customdata[1]:,.2f}<br>"
                    "현재가 %{customdata[2]:,.2f}<br>"
                    "주당 평가손익 %{customdata[3]}<br>"
                    "평가수익률 %{customdata[4]:+.2%}<br>"
                    "보유기간 %{customdata[5]:.0f}일<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=weekly["Date"],
            y=weekly["CompositeStrength"],
            mode="lines+markers",
            name="종합 강도",
            line={"color": "#17243a", "width": 2.2},
            marker={"size": 4},
            customdata=weekly[["Rank", "DailySignal"]].to_numpy(),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>강도 %{y:.1f}<br>"
                "순위 %{customdata[0]:.0f}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    colors_by_factor = {
        "MomentumFactor": "#2563eb",
        "TrendFactor": "#7c3aed",
        "GrowthFactor": "#0f9f6e",
        "QualityFactor": "#d97706",
        "RiskControlFactor": "#dc2626",
    }
    for column in FACTOR_COLUMNS:
        figure.add_trace(
            go.Scatter(
                x=weekly["Date"],
                y=weekly[f"{column}Strength"],
                mode="lines",
                name=FACTOR_LABELS[column],
                line={"color": colors_by_factor[column], "width": 1},
                visible="legendonly",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.add_hline(
        y=50,
        line={"color": "#aab5c5", "width": 1, "dash": "dot"},
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="가격", row=1, col=1)
    figure.update_yaxes(title_text="강도", range=[0, 100], row=2, col=1)
    figure.update_layout(
        height=620,
        template="plotly_white",
        hovermode="closest",
        margin={"l": 65, "r": 25, "t": 55, "b": 45},
        legend={"orientation": "h", "y": -0.12},
    )
    return figure


def _html_event_table(events: pd.DataFrame) -> str:
    shown = events[
        [
            "Date",
            "ExecutionDate",
            "TradeAction",
            "PositionEntryPrice",
            "ExecutionPrice",
            "PositionPnLPerShare",
            "PositionReturn",
            "Rank",
            "AlphaScore",
            "CompositeStrength",
            "ExitReason",
        ]
    ].copy()
    shown["Date"] = shown["Date"].dt.strftime("%Y-%m-%d")
    shown["ExecutionDate"] = shown["ExecutionDate"].dt.strftime("%Y-%m-%d")
    shown["TradeAction"] = shown["TradeAction"].map(ACTION_LABELS)
    buy_rows = shown["TradeAction"].eq(ACTION_LABELS["BUY"])
    shown.loc[buy_rows, "PositionEntryPrice"] = shown.loc[
        buy_rows,
        "ExecutionPrice",
    ]
    shown.loc[buy_rows, "PositionPnLPerShare"] = np.nan
    shown.loc[buy_rows, "PositionReturn"] = np.nan
    shown["PositionEntryPrice"] = shown["PositionEntryPrice"].map(
        lambda value: _number(value, 2)
    )
    shown["ExecutionPrice"] = shown["ExecutionPrice"].map(
        lambda value: _number(value, 2)
    )
    shown["PositionPnLPerShare"] = shown["PositionPnLPerShare"].map(
        lambda value: _signed_number(value, 2)
    )
    shown["PositionReturn"] = shown["PositionReturn"].map(_signed_percent)
    shown["Rank"] = shown["Rank"].map(_integer)
    shown["AlphaScore"] = shown["AlphaScore"].map(
        lambda value: _number(value, 4)
    )
    shown["CompositeStrength"] = shown["CompositeStrength"].map(
        lambda value: _number(value, 1)
    )
    shown["ExitReason"] = shown["ExitReason"].map(
        lambda value: EXIT_LABELS.get(str(value), str(value))
    )
    shown.columns = [
        "신호일",
        "체결일",
        "행동",
        "매수가",
        "체결가",
        "주당 손익",
        "실제 수익률",
        "순위",
        "점수",
        "강도",
        "매도 사유",
    ]
    return (
        "<div class='table-wrap'>"
        + shown.to_html(index=False, escape=True)
        + "</div>"
    )


def _write_pdf_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.DataFrame,
    events: pd.DataFrame,
    ledger: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> None:
    regular_font, bold_font = _register_korean_fonts()
    report_label = _report_label(manifest)
    report_title = f"{report_label} 매매·신호 강도 리포트"
    page_size = landscape(A4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        prefix=path.stem + "_",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        document = SimpleDocTemplate(
            str(temporary),
            pagesize=page_size,
            rightMargin=14 * mm,
            leftMargin=14 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
            title=report_title,
            author="stock-research",
        )
        styles = _pdf_styles(regular_font, bold_font)
        story: list[Any] = []
        latest_date = _latest_complete_date(history)
        closed = ledger.loc[ledger["Status"].eq("CLOSED")]
        open_positions = ledger.loc[ledger["Status"].eq("OPEN")]
        summary = _position_summary(ledger)
        story.extend(
            [
                Paragraph(report_title, styles["TitleKo"]),
                Paragraph(
                    (
                        f"모델 상태: {manifest['model_status']}  |  "
                        f"신호 기간: {history['Date'].min():%Y-%m-%d} - "
                        f"{latest_date:%Y-%m-%d}"
                    ),
                    styles["Meta"],
                ),
                Spacer(1, 5 * mm),
                _pdf_summary_cards(
                    [
                        ("완전 진입", str(int((events["TradeAction"] == "BUY").sum()))),
                        ("완전 청산", str(len(closed))),
                        ("현재 보유", str(len(open_positions))),
                        ("종료 승률", _percent(summary["WinRate"])),
                        ("종료 평균수익", _percent(summary["MeanReturn"])),
                    ],
                    regular_font,
                    bold_font,
                ),
                Spacer(1, 5 * mm),
                Paragraph("핵심 해석", styles["HeadingKo"]),
                Paragraph(
                    (
                        f"{report_label}는 사후 진단 모델이다. 신호일 종가로 결정하고 다음 "
                        "거래일 시가에 체결한다. 개별 포지션 수익률은 완전 진입과 "
                        "완전 청산 사이의 단순 가격수익률이며, 주간 동일비중 조정과 "
                        "거래비용이 포함된 포트폴리오 기여수익률은 아니다."
                    ),
                    styles["BodyKo"],
                ),
                Spacer(1, 3 * mm),
                Paragraph("검증 구간 성과", styles["HeadingKo"]),
                _pdf_validation_table(validation, regular_font, bold_font),
                Spacer(1, 4 * mm),
                Paragraph("매도 사유 집계", styles["HeadingKo"]),
                _pdf_exit_reason_table(events, regular_font, bold_font),
                PageBreak(),
                Paragraph("포지션 원장", styles["TitleSmall"]),
                Paragraph(
                    "OPEN 행의 청산일·청산가는 최신 평가일과 종가를 뜻한다.",
                    styles["Meta"],
                ),
                Spacer(1, 2 * mm),
                _pdf_ledger_table(ledger, regular_font, bold_font),
                PageBreak(),
                Paragraph(
                    f"최신 신호 - {latest_date:%Y-%m-%d}",
                    styles["TitleSmall"],
                ),
                _pdf_latest_signal_table(
                    history.loc[history["Date"].eq(latest_date)],
                    regular_font,
                    bold_font,
                ),
                Spacer(1, 5 * mm),
                Paragraph("가격 연결 검증", styles["HeadingKo"]),
                Paragraph(
                    (
                        f"매매 이벤트 {len(reconciliation)}건 중 신호 이력 종가와 "
                        f"가격 파일 종가 불일치는 "
                        f"{int((~reconciliation['CloseMatches']).sum())}건이다. "
                        f"전일 종가 대체 체결은 "
                        f"{int((events['ExecutionPriceSource'] == 'PRIOR_CLOSE_FALLBACK').sum())}건이다."
                    ),
                    styles["BodyKo"],
                ),
            ]
        )
        traded_tickers = sorted(events["Ticker"].unique())
        for ticker in traded_tickers:
            ticker_prices = prices.loc[prices["Ticker"].eq(ticker)]
            ticker_history = history.loc[history["Ticker"].eq(ticker)]
            ticker_events = events.loc[events["Ticker"].eq(ticker)]
            story.extend(
                [
                    PageBreak(),
                    Paragraph(
                        f"{ticker} · {_company_name(history, ticker)}",
                        styles["TitleSmall"],
                    ),
                    _pdf_ticker_chart(
                        ticker_prices,
                        ticker_history,
                        ticker_events,
                        regular_font,
                    ),
                    Spacer(1, 3 * mm),
                    _pdf_ticker_event_table(
                        ticker_events,
                        regular_font,
                        bold_font,
                    ),
                    Spacer(1, 3 * mm),
                    Paragraph(
                        "강도 50은 상대평가 중앙 수준이다. ▲는 실제 다음 시가 "
                        "진입, ▼는 실제 다음 시가 완전 청산이다.",
                        styles["Meta"],
                    ),
                ]
            )
        document.build(
            story,
            onFirstPage=lambda canvas, doc: _pdf_footer(
                canvas, doc, regular_font, report_title
            ),
            onLaterPages=lambda canvas, doc: _pdf_footer(
                canvas, doc, regular_font, report_title
            ),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pdf_ticker_chart(
    prices: pd.DataFrame,
    history: pd.DataFrame,
    events: pd.DataFrame,
    font_name: str,
) -> Drawing:
    width = 252 * mm
    height = 112 * mm
    drawing = Drawing(width, height)
    left = 14 * mm
    right = width - 8 * mm
    top = height - 8 * mm
    price_bottom = 42 * mm
    strength_top = 34 * mm
    strength_bottom = 9 * mm
    price_values = pd.to_numeric(prices["Close"], errors="coerce")
    valid_prices = prices.loc[price_values.notna()].copy()
    if valid_prices.empty:
        drawing.add(
            String(
                width / 2,
                height / 2,
                "표시할 가격 데이터가 없습니다.",
                textAnchor="middle",
                fontName=font_name,
                fontSize=10,
            )
        )
        return drawing
    dates = pd.to_datetime(valid_prices["Date"])
    start = dates.min()
    end = dates.max()
    day_span = max((end - start).days, 1)
    minimum = float(valid_prices["Close"].min())
    maximum = float(valid_prices["Close"].max())
    padding = max((maximum - minimum) * 0.08, maximum * 0.01, 0.01)
    y_min = max(minimum - padding, 0)
    y_max = maximum + padding

    def x_value(value: object) -> float:
        return left + (
            (pd.Timestamp(value) - start).days / day_span * (right - left)
        )

    def price_y(value: float) -> float:
        return price_bottom + (
            (value - y_min) / max(y_max - y_min, 1e-12)
        ) * (top - price_bottom)

    drawing.add(Rect(left, price_bottom, right - left, top - price_bottom,
                     fillColor=colors.HexColor("#f9fbfe"),
                     strokeColor=colors.HexColor("#d8e0ea")))
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = price_bottom + fraction * (top - price_bottom)
        value = y_min + fraction * (y_max - y_min)
        drawing.add(Line(left, y, right, y, strokeColor=colors.HexColor("#e6ebf2")))
        drawing.add(
            String(
                left - 2 * mm,
                y - 1.5 * mm,
                f"{value:,.1f}",
                textAnchor="end",
                fontName=font_name,
                fontSize=6.5,
                fillColor=colors.HexColor("#607087"),
            )
        )
    points = [
        (x_value(date), price_y(float(close)))
        for date, close in zip(valid_prices["Date"], valid_prices["Close"])
    ]
    drawing.add(
        PolyLine(
            points,
            strokeColor=colors.HexColor("#3157d5"),
            strokeWidth=1.4,
        )
    )
    for row in events.itertuples(index=False):
        if pd.isna(row.ExecutionDate) or pd.isna(row.ExecutionPrice):
            continue
        x = x_value(row.ExecutionDate)
        y = price_y(float(row.ExecutionPrice))
        if row.TradeAction == "BUY":
            drawing.add(
                Polygon(
                    [x, y + 3.5 * mm, x - 2.5 * mm, y, x + 2.5 * mm, y],
                    fillColor=colors.HexColor("#2563eb"),
                    strokeColor=colors.white,
                )
            )
            label_y = y + 5 * mm
        else:
            profitable = (
                pd.notna(row.PositionReturn)
                and float(row.PositionReturn) >= 0
            )
            drawing.add(
                Polygon(
                    [x, y - 3.5 * mm, x - 2.5 * mm, y, x + 2.5 * mm, y],
                    fillColor=colors.HexColor(
                        "#0f9f6e" if profitable else "#d64545"
                    ),
                    strokeColor=colors.white,
                )
            )
            label_y = y - 7 * mm
        drawing.add(
            String(
                x,
                label_y,
                (
                    f"매도 {float(row.PositionReturn):+.1%}"
                    if row.TradeAction == "SELL"
                    and pd.notna(row.PositionReturn)
                    else ACTION_LABELS[row.TradeAction]
                ),
                textAnchor="middle",
                fontName=font_name,
                fontSize=6.5,
            )
        )

    drawing.add(Rect(left, strength_bottom, right - left,
                     strength_top - strength_bottom,
                     fillColor=colors.HexColor("#fbfcfe"),
                     strokeColor=colors.HexColor("#d8e0ea")))
    for strength in (0, 50, 100):
        y = strength_bottom + strength / 100 * (
            strength_top - strength_bottom
        )
        drawing.add(Line(left, y, right, y,
                         strokeColor=colors.HexColor("#e0e6ef"),
                         strokeDashArray=[2, 2] if strength == 50 else None))
        drawing.add(
            String(
                left - 2 * mm,
                y - 1.5 * mm,
                str(strength),
                textAnchor="end",
                fontName=font_name,
                fontSize=6.5,
                fillColor=colors.HexColor("#607087"),
            )
        )
    weekly = history.loc[
        history["IsRebalanceSignal"] & history["CompositeStrength"].notna()
    ]
    strength_points = [
        (
            x_value(date),
            strength_bottom + float(strength) / 100
            * (strength_top - strength_bottom),
        )
        for date, strength in zip(weekly["Date"], weekly["CompositeStrength"])
    ]
    if len(strength_points) >= 2:
        drawing.add(
            PolyLine(
                strength_points,
                strokeColor=colors.HexColor("#17243a"),
                strokeWidth=1.4,
            )
        )
    for x, y in strength_points:
        drawing.add(
            Circle(
                x,
                y,
                0.7 * mm,
                fillColor=colors.HexColor("#17243a"),
                strokeColor=None,
            )
        )
    drawing.add(
        String(
            left,
            height - 4 * mm,
            "가격",
            fontName=font_name,
            fontSize=7,
            fillColor=colors.HexColor("#607087"),
        )
    )
    drawing.add(
        String(
            left,
            strength_top + 2 * mm,
            "종합 강도",
            fontName=font_name,
            fontSize=7,
            fillColor=colors.HexColor("#607087"),
        )
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        date = start + pd.Timedelta(days=int(day_span * fraction))
        x = x_value(date)
        drawing.add(Line(x, strength_bottom, x, strength_bottom - 1.5 * mm,
                         strokeColor=colors.HexColor("#8e9caf")))
        drawing.add(
            String(
                x,
                strength_bottom - 5 * mm,
                date.strftime("%Y-%m"),
                textAnchor="middle",
                fontName=font_name,
                fontSize=6.5,
                fillColor=colors.HexColor("#607087"),
            )
        )
    return drawing


def _pdf_styles(regular_font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "TitleKo": ParagraphStyle(
            "TitleKo",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=24,
            leading=30,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#17243a"),
        ),
        "TitleSmall": ParagraphStyle(
            "TitleSmall",
            parent=base["Heading1"],
            fontName=bold_font,
            fontSize=18,
            leading=23,
            textColor=colors.HexColor("#17243a"),
        ),
        "HeadingKo": ParagraphStyle(
            "HeadingKo",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=12,
            leading=16,
            spaceAfter=5,
            textColor=colors.HexColor("#274066"),
        ),
        "BodyKo": ParagraphStyle(
            "BodyKo",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#26364c"),
        ),
        "Meta": ParagraphStyle(
            "Meta",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=8,
            leading=12,
            textColor=colors.HexColor("#607087"),
        ),
        "Cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=6.6,
            leading=8.2,
            alignment=TA_LEFT,
        ),
        "CellCenter": ParagraphStyle(
            "CellCenter",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=6.6,
            leading=8.2,
            alignment=TA_CENTER,
        ),
    }


def _register_korean_fonts() -> tuple[str, str]:
    font_root = Path(os.environ.get("WINDIR", "")) / "Fonts"
    regular = font_root / "malgun.ttf"
    bold = font_root / "malgunbd.ttf"
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError(
            "Malgun Gothic fonts are required to create the Korean PDF."
        )
    regular_name = "V5Malgun"
    bold_name = "V5MalgunBold"
    if regular_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(regular_name, str(regular)))
    if bold_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(bold_name, str(bold)))
    return regular_name, bold_name


def _pdf_summary_cards(
    items: list[tuple[str, str]],
    regular_font: str,
    bold_font: str,
) -> Table:
    data = [
        [
            Paragraph(label, ParagraphStyle(
                f"card-label-{index}",
                fontName=regular_font,
                fontSize=7,
                textColor=colors.HexColor("#b9c5d8"),
                alignment=TA_CENTER,
            ))
            for index, (label, _) in enumerate(items)
        ],
        [
            Paragraph(value, ParagraphStyle(
                f"card-value-{index}",
                fontName=bold_font,
                fontSize=16,
                textColor=colors.white,
                alignment=TA_CENTER,
                leading=19,
            ))
            for index, (_, value) in enumerate(items)
        ],
    ]
    table = Table(data, colWidths=[49 * mm] * len(items), rowHeights=[8 * mm, 12 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#17243a")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#17243a")),
                ("INNERGRID", (0, 0), (-1, -1), 1, colors.HexColor("#2b3b55")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _pdf_validation_table(
    validation: pd.DataFrame,
    regular_font: str,
    bold_font: str,
) -> Table:
    headers = [
        "기간",
        "시작",
        "종료",
        "전략 ROI",
        "동일가중 ROI",
        "초과 ROI",
        "최대낙폭",
        "Sharpe",
        "연환산 회전율",
    ]
    rows = [headers]
    for row in validation.itertuples(index=False):
        rows.append(
            [
                str(row.Period),
                str(row.StartDate),
                str(row.EndDate),
                _percent_points(row.StrategyROI),
                _percent_points(row.EqualWeightUniverseROI),
                _percent_points(row.ExcessROI),
                _percent_points(row.MaxDrawdown),
                _number(row.Sharpe, 2),
                _number(row.AnnualizedTurnover, 2),
            ]
        )
    return _styled_pdf_table(
        rows,
        [18, 29, 29, 25, 31, 25, 25, 20, 31],
        regular_font,
        bold_font,
        font_size=7.5,
    )


def _pdf_exit_reason_table(
    events: pd.DataFrame,
    regular_font: str,
    bold_font: str,
) -> Table:
    counts = (
        events.loc[events["TradeAction"].eq("SELL"), "ExitReason"]
        .value_counts()
        .rename_axis("Reason")
        .reset_index(name="Count")
    )
    rows = [["매도 사유", "횟수"]]
    rows.extend(
        [
            [EXIT_LABELS.get(str(row.Reason), str(row.Reason)), str(row.Count)]
            for row in counts.itertuples(index=False)
        ]
    )
    return _styled_pdf_table(
        rows,
        [65, 25],
        regular_font,
        bold_font,
        font_size=8,
    )


def _pdf_ledger_table(
    ledger: pd.DataFrame,
    regular_font: str,
    bold_font: str,
) -> LongTable:
    rows = [
        [
            "ID",
            "상태",
            "진입 신호",
            "진입 체결",
            "진입가",
            "진입 순위",
            "진입 강도",
            "청산 신호",
            "청산·평가일",
            "청산·평가가",
            "가격수익",
            "주당손익",
            "보유일",
            "매도 사유",
        ]
    ]
    for row in ledger.itertuples(index=False):
        rows.append(
            [
                row.PositionId,
                row.Status,
                _date(row.EntrySignalDate),
                _date(row.EntryExecutionDate),
                _number(row.EntryExecutionPrice, 2),
                _integer(row.EntryRank),
                _number(row.EntryCompositeStrength, 1),
                _date(row.ExitSignalDate),
                _date(row.ExitExecutionDate),
                _number(row.ExitExecutionPrice, 2),
                _percent(row.ExecutionPriceReturn),
                _signed_number(row.PerSharePnL, 2),
                str(row.HoldingCalendarDays),
                EXIT_LABELS.get(str(row.ExitReason), str(row.ExitReason)),
            ]
        )
    table = LongTable(
        rows,
        repeatRows=1,
        colWidths=[
            18 * mm,
            13 * mm,
            19 * mm,
            19 * mm,
            16 * mm,
            15 * mm,
            16 * mm,
            19 * mm,
            21 * mm,
            18 * mm,
            17 * mm,
            17 * mm,
            13 * mm,
            28 * mm,
        ],
    )
    table.setStyle(_pdf_table_style(regular_font, bold_font, 6.3))
    return table


def _pdf_latest_signal_table(
    latest: pd.DataFrame,
    regular_font: str,
    bold_font: str,
) -> Table:
    selected = latest.loc[
        latest["ModelSelected"] | latest["DailySignal"].eq("BUY_WATCH")
    ].sort_values(["ModelSelected", "Rank"], ascending=[False, True])
    rows = [[
        "종목",
        "회사",
        "신호",
        "목표비중",
        "순위",
        "점수",
        "강도",
        "진입기준 손익",
        "보유 리밸런싱",
    ]]
    for row in selected.itertuples(index=False):
        rows.append(
            [
                row.Ticker,
                row.Company,
                row.DailySignal,
                _percent(row.TargetWeight),
                _integer(row.Rank),
                _number(row.AlphaScore, 4),
                _number(row.CompositeStrength, 1),
                _percent(row.SignalReferenceReturn),
                _integer(row.HoldingRebalances),
            ]
        )
    return _styled_pdf_table(
        rows,
        [19, 45, 30, 22, 16, 20, 18, 29, 30],
        regular_font,
        bold_font,
        font_size=7,
    )


def _pdf_ticker_event_table(
    events: pd.DataFrame,
    regular_font: str,
    bold_font: str,
) -> Table:
    rows = [[
        "신호일",
        "체결일",
        "행동",
        "체결가",
        "순위",
        "점수",
        "강도",
        "모멘텀",
        "추세",
        "성장",
        "품질",
        "위험통제",
        "주당손익",
        "실제수익",
        "사유",
    ]]
    for row in events.itertuples(index=False):
        rows.append(
            [
                _date(row.Date),
                _date(row.ExecutionDate),
                ACTION_LABELS.get(row.TradeAction, row.TradeAction),
                _number(row.ExecutionPrice, 2),
                _integer(row.Rank),
                _number(row.AlphaScore, 4),
                _number(row.CompositeStrength, 1),
                _number(row.MomentumFactorStrength, 1),
                _number(row.TrendFactorStrength, 1),
                _number(row.GrowthFactorStrength, 1),
                _number(row.QualityFactorStrength, 1),
                _number(row.RiskControlFactorStrength, 1),
                (
                    _signed_number(row.PositionPnLPerShare, 2)
                    if row.TradeAction == "SELL"
                    else ""
                ),
                (
                    _signed_percent(row.PositionReturn)
                    if row.TradeAction == "SELL"
                    else ""
                ),
                EXIT_LABELS.get(str(row.ExitReason), str(row.ExitReason)),
            ]
        )
    return _styled_pdf_table(
        rows,
        [18, 18, 14, 17, 12, 18, 15, 18, 15, 15, 15, 18, 18, 18, 27],
        regular_font,
        bold_font,
        font_size=6.2,
    )


def _styled_pdf_table(
    rows: list[list[str]],
    widths_mm: list[float],
    regular_font: str,
    bold_font: str,
    *,
    font_size: float,
) -> Table:
    table = Table(
        rows,
        colWidths=[width * mm for width in widths_mm],
        repeatRows=1,
    )
    table.setStyle(_pdf_table_style(regular_font, bold_font, font_size))
    return table


def _pdf_table_style(
    regular_font: str,
    bold_font: str,
    font_size: float,
) -> TableStyle:
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), regular_font),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 2),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#253a59")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e2")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor("#f7f9fc"),
            ]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def _pdf_footer(
    canvas: Any,
    document: Any,
    font_name: str,
    report_title: str,
) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8e0ea"))
    canvas.line(14 * mm, 10 * mm, landscape(A4)[0] - 14 * mm, 10 * mm)
    canvas.setFont(font_name, 7)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(14 * mm, 6.5 * mm, report_title)
    canvas.drawRightString(
        landscape(A4)[0] - 14 * mm,
        6.5 * mm,
        f"{document.page}",
    )
    canvas.restoreState()


def _position_summary(ledger: pd.DataFrame) -> dict[str, float]:
    closed = ledger.loc[
        ledger["Status"].eq("CLOSED")
        & ledger["ExecutionPriceReturn"].notna()
    ]
    returns = pd.to_numeric(closed["ExecutionPriceReturn"], errors="coerce")
    return {
        "WinRate": float(returns.gt(0).mean()) if not returns.empty else np.nan,
        "MeanReturn": float(returns.mean()) if not returns.empty else np.nan,
        "MedianReturn": float(returns.median()) if not returns.empty else np.nan,
    }


def _ledger_display(ledger: pd.DataFrame) -> pd.DataFrame:
    shown = ledger[
        [
            "PositionId",
            "Company",
            "Status",
            "EntrySignalDate",
            "EntryExecutionDate",
            "EntryExecutionPrice",
            "EntryRank",
            "EntryCompositeStrength",
            "ExitSignalDate",
            "ExitExecutionDate",
            "ExitExecutionPrice",
            "ExecutionPriceReturn",
            "PerSharePnL",
            "HoldingCalendarDays",
            "ExitReason",
        ]
    ].copy()
    for column in (
        "EntrySignalDate",
        "EntryExecutionDate",
        "ExitSignalDate",
        "ExitExecutionDate",
    ):
        shown[column] = shown[column].map(_date)
    for column in ("EntryExecutionPrice", "ExitExecutionPrice"):
        shown[column] = shown[column].map(lambda value: _number(value, 2))
    shown["EntryRank"] = shown["EntryRank"].map(_integer)
    shown["EntryCompositeStrength"] = shown["EntryCompositeStrength"].map(
        lambda value: _number(value, 1)
    )
    shown["ExecutionPriceReturn"] = shown["ExecutionPriceReturn"].map(_percent)
    shown["PerSharePnL"] = shown["PerSharePnL"].map(
        lambda value: _signed_number(value, 2)
    )
    shown["ExitReason"] = shown["ExitReason"].map(
        lambda value: EXIT_LABELS.get(str(value), str(value))
    )
    shown.columns = [
        "포지션",
        "회사",
        "상태",
        "진입 신호일",
        "진입 체결일",
        "진입가",
        "진입 순위",
        "진입 강도",
        "청산 신호일",
        "청산·평가일",
        "청산·평가가",
        "가격수익",
        "주당손익",
        "보유일",
        "매도 사유",
    ]
    return shown


def _company_name(history: pd.DataFrame, ticker: str) -> str:
    names = history.loc[history["Ticker"].eq(ticker), "Company"].dropna()
    return str(names.iloc[0]) if not names.empty else ""


def _lookup_number(
    lookup: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    column: str,
) -> float:
    try:
        value = lookup.loc[(ticker, date), column]
    except KeyError:
        return np.nan
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _series_number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column, np.nan), errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def _atomic_write_text(path: Path, content: str) -> None:
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


def _date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _number(value: object, decimals: int) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(numeric) else f"{float(numeric):,.{decimals}f}"


def _signed_number(value: object, decimals: int) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(numeric) else f"{float(numeric):+,.{decimals}f}"


def _integer(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(numeric) else f"{round(float(numeric))}"


def _percent(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(numeric) else f"{float(numeric):.2%}"


def _signed_percent(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(numeric) else f"{float(numeric):+.2%}"


def _percent_points(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(numeric) else f"{float(numeric):.2f}%"
