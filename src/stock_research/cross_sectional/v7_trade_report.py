from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from dataclasses import dataclass
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
from .portfolio import PortfolioResult, prepare_market
from .v7_pit_evaluation import (
    build_v7_source_panel,
    evaluate_period,
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


TECHNICAL_COMPONENTS = ("MAFactor", "MACDFactor", "OBVFactor")


@dataclass(frozen=True)
class V7TradeReportArtifacts:
    output_dir: Path
    report_html: Path
    trade_events_csv: Path
    position_ledger_csv: Path
    execution_ledger_csv: Path
    weekly_holdings_csv: Path
    equity_csv: Path
    reconciliation_csv: Path
    data_audit_csv: Path
    manifest_json: Path


def run_v7_trade_report(
    paths: ProjectPaths,
    settings: ResearchSettings,
    base_params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    frozen_strategy_path: str | Path,
    spy_path: str | Path,
    top_k: int = 5,
    exit_buffer: int = 4,
    expected_ready_count: int = 575,
    output_dir: str | Path | None = None,
) -> V7TradeReportArtifacts:
    """Build a continuous V7-3 trade report with exact execution amounts."""

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
    company_by_ticker = {
        member.ticker: member.company for member in members
    }
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
    params = slot_sweep_params(
        base_params,
        top_k,
        exit_buffer=exit_buffer,
    )
    reference_dates = pd.DatetimeIndex(
        scoring_panel["Date"].drop_duplicates().sort_values()
    )
    evaluation = evaluate_period(
        scoring_panel,
        params,
        settings,
        start=settings.train_start,
        end=latest_end,
        reference_dates=reference_dates,
        record_attribution=True,
    )
    strategy = evaluation.strategy
    execution_ledger, reconciliation = build_execution_ledger(
        scoring_panel,
        evaluation.targets,
        strategy,
        start=settings.train_start,
        end=latest_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    trade_events = build_trade_events(
        evaluation.targets,
        execution_ledger,
        params,
        company_by_ticker=company_by_ticker,
    )
    positions = build_position_ledger(
        trade_events,
        execution_ledger,
        strategy,
        latest_prices=scoring_panel,
        end_date=latest_end,
        company_by_ticker=company_by_ticker,
    )
    weekly_holdings = build_weekly_holdings(
        evaluation.targets,
        company_by_ticker,
    )

    spy = load_sp500_proxy(spy_path)
    spy_summary, spy_equity = spy_buy_and_hold(
        spy,
        start=settings.train_start,
        end=latest_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    equity = strategy.daily.copy()
    equity = equity.merge(
        spy_equity.rename(columns={"Equity": "SPYEquity"}),
        on="Date",
        how="left",
    )
    equity["SPYEquity"] = equity["SPYEquity"].ffill()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v7_trade_reports"
            / f"{timestamp}_v7_3_top{top_k}"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "report_html": destination / "v7_3_trade_report.html",
        "trade_events_csv": destination / "trade_events.csv",
        "position_ledger_csv": destination / "position_ledger.csv",
        "execution_ledger_csv": destination / "execution_ledger.csv",
        "weekly_holdings_csv": destination / "weekly_holdings.csv",
        "equity_csv": destination / "equity.csv",
        "reconciliation_csv": destination / "reconciliation.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(trade_events, outputs["trade_events_csv"], index=False)
    atomic_to_csv(positions, outputs["position_ledger_csv"], index=False)
    atomic_to_csv(
        execution_ledger,
        outputs["execution_ledger_csv"],
        index=False,
    )
    atomic_to_csv(
        weekly_holdings,
        outputs["weekly_holdings_csv"],
        index=False,
    )
    atomic_to_csv(equity, outputs["equity_csv"], index=False)
    atomic_to_csv(
        reconciliation,
        outputs["reconciliation_csv"],
        index=False,
    )
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    report = render_trade_report(
        trade_events=trade_events,
        positions=positions,
        executions=execution_ledger,
        weekly_holdings=weekly_holdings,
        equity=equity,
        strategy=strategy,
        spy_summary=spy_summary,
        params=params,
        generated_at=datetime.now(UTC),
    )
    _atomic_text(outputs["report_html"], report)
    frozen_payload = json.loads(
        Path(frozen_strategy_path).read_text(encoding="utf-8")
    )
    _atomic_json(
        outputs["manifest_json"],
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "V7-3 continuous trade ledger and reason report",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "variant": v7_3.name,
            "technical_components": list(v7_3.components),
            "top_k": top_k,
            "exit_rank": params.exit_rank,
            "exit_rank_rule": f"top_k + {exit_buffer}",
            "start": settings.train_start,
            "end": latest_end,
            "initial_capital": settings.initial_capital,
            "transaction_cost_bps": settings.transaction_cost_bps,
            "execution_rule": (
                "causal final W-FRI session close signal, next session open"
            ),
            "roi_formula": (
                "(final_value / total_injected - 1) * 100"
            ),
            "strategy_summary": strategy.summary.as_dict(),
            "spy_summary": spy_summary,
            "frozen_v6_candidate": frozen_payload.get(
                "selected_candidate",
                frozen_payload.get("selected_candidate_id"),
            ),
            "frozen_strategy_sha256": _sha256(
                Path(frozen_strategy_path)
            ),
            "backfill_status_sha256": _sha256(
                Path(backfill_status_path)
            ),
            "membership_sha256": _sha256(Path(membership_path)),
            "spy_sha256": _sha256(Path(spy_path)),
            "ready_ticker_count": len(ready_tickers),
            "financial_point_in_time": False,
            "caveats": [
                (
                    "Macrotrends financials are current restated history, "
                    "not true point-in-time statements."
                ),
                (
                    "The 575-name dataset still omits unavailable acquired "
                    "and delisted constituents."
                ),
                (
                    "V6-B, 2025, 2026, and the choice of five slots were "
                    "already observed; this report is post-hoc."
                ),
                (
                    "Reported backtest performance may be upward biased by "
                    "survivorship, multiple testing, and post-hoc selection."
                ),
            ],
            "outputs": {
                name: str(path) for name, path in outputs.items()
            },
        },
    )
    return V7TradeReportArtifacts(
        output_dir=destination,
        **outputs,
    )


def build_execution_ledger(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    result: PortfolioResult,
    *,
    start: str,
    end: str,
    initial_capital: float,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay portfolio executions and expose exact ticker-level amounts."""

    market = prepare_market(panel, start=start, end=end)
    target_schedule = _target_schedule(
        targets.loc[targets["Date"].between(start, end)],
        market.dates,
    )
    open_values = market.open_prices.to_numpy(dtype=float, copy=False)
    close_values = market.valuation_prices.to_numpy(dtype=float, copy=False)
    fallback_values = market.fallback_trade_prices.to_numpy(
        dtype=float,
        copy=False,
    )
    tickers = market.tickers
    shares = np.zeros(len(tickers), dtype=float)
    cash = float(initial_capital)
    cost_rate = transaction_cost_bps / 10_000
    records: list[dict[str, object]] = []
    daily_checks: list[dict[str, object]] = []
    aggregate_by_date = (
        result.executions.set_index("ExecutionDate")
        if not result.executions.empty
        else pd.DataFrame()
    )
    result_daily = result.daily.set_index("Date")

    for position, date in enumerate(market.dates):
        raw_open = open_values[position]
        fallback_used = np.isnan(raw_open)
        open_row = np.where(
            fallback_used,
            fallback_values[position],
            raw_open,
        )
        open_for_valuation = np.where(np.isnan(open_row), 0.0, open_row)
        close_row = np.where(
            np.isnan(close_values[position]),
            0.0,
            close_values[position],
        )
        if date in target_schedule:
            signal_date, signal_rows = target_schedule[date]
            indexed = signal_rows.set_index("Ticker", drop=False)
            target = (
                indexed["TargetWeight"]
                .astype(float)
                .reindex(tickers, fill_value=0.0)
                .to_numpy()
            )
            tradable = (open_row > 0) & ~np.isnan(open_row)
            requested_target_sum = float(target.sum())
            untradable_target = float(target[~tradable].sum())
            if untradable_target > 0:
                target = np.where(tradable, target, 0.0)
                total_target = float(target.sum())
                if total_target > 0:
                    target = (
                        target
                        / total_target
                        * requested_target_sum
                    )
            before_shares = shares.copy()
            pre_trade_equity = cash + float(
                np.sum(before_shares * open_for_valuation)
            )
            before_notional = before_shares * open_for_valuation
            desired_before_cost = target * pre_trade_equity
            requested_changes = desired_before_cost - before_notional
            turnover = float(np.abs(requested_changes).sum())
            transaction_cost = turnover * cost_rate
            net_equity = max(pre_trade_equity - transaction_cost, 0.0)
            after_notional = target * net_equity
            after_shares = np.divide(
                after_notional,
                open_row,
                out=np.zeros_like(after_notional),
                where=tradable,
            )
            trade_notional = after_notional - before_notional
            changed = np.abs(requested_changes) > (
                max(pre_trade_equity, 1.0) * 1e-8
            )
            allocated_cost = np.zeros(len(tickers), dtype=float)
            if turnover > 0:
                allocated_cost = (
                    np.abs(requested_changes)
                    / turnover
                    * transaction_cost
                )
            for ticker_index in np.flatnonzero(changed):
                ticker = str(tickers[ticker_index])
                signal = (
                    indexed.loc[ticker]
                    if ticker in indexed.index
                    else None
                )
                if isinstance(signal, pd.DataFrame):
                    signal = signal.iloc[-1]
                before = float(before_notional[ticker_index])
                after = float(after_notional[ticker_index])
                delta = float(trade_notional[ticker_index])
                records.append(
                    {
                        "SignalDate": signal_date,
                        "ExecutionDate": date,
                        "Ticker": ticker,
                        "ExecutionSide": (
                            "BUY" if delta > 0 else "SELL"
                        ),
                        "ExecutionType": _execution_type(before, after),
                        "SignalTradeAction": _signal_value(
                            signal,
                            "TradeAction",
                            "REBALANCE",
                        ),
                        "ExitReason": _signal_value(
                            signal,
                            "ExitReason",
                            "",
                        ),
                        "TargetWeight": float(target[ticker_index]),
                        "PreTradeEquity": pre_trade_equity,
                        "ExecutionPrice": float(open_row[ticker_index]),
                        "UsedPriorCloseFallback": bool(
                            fallback_used[ticker_index]
                        ),
                        "SharesBefore": float(
                            before_shares[ticker_index]
                        ),
                        "SharesTraded": float(
                            after_shares[ticker_index]
                            - before_shares[ticker_index]
                        ),
                        "SharesAfter": float(after_shares[ticker_index]),
                        "NotionalBefore": before,
                        "TradeNotional": delta,
                        "GrossTradeAmount": abs(delta),
                        "NotionalAfter": after,
                        "RequestedTurnoverBasis": abs(
                            float(requested_changes[ticker_index])
                        ),
                        "AllocatedTransactionCost": float(
                            allocated_cost[ticker_index]
                        ),
                    }
                )
            shares = after_shares
            cash = net_equity - float(after_notional.sum())
            if not aggregate_by_date.empty and date in aggregate_by_date.index:
                expected = aggregate_by_date.loc[date]
                if isinstance(expected, pd.DataFrame):
                    expected = expected.iloc[-1]
                daily_checks.append(
                    {
                        "CheckType": "EXECUTION",
                        "Date": date,
                        "EquityError": (
                            pre_trade_equity
                            - float(expected["PreTradeEquity"])
                        ),
                        "TurnoverError": (
                            turnover - float(expected["Turnover"])
                        ),
                        "CostError": (
                            transaction_cost
                            - float(expected["TransactionCost"])
                        ),
                    }
                )
        equity = cash + float(np.sum(shares * close_row))
        expected_equity = float(result_daily.loc[date, "Equity"])
        daily_checks.append(
            {
                "CheckType": "DAILY_EQUITY",
                "Date": date,
                "EquityError": equity - expected_equity,
                "TurnoverError": np.nan,
                "CostError": np.nan,
            }
        )

    ledger = pd.DataFrame(records)
    checks = pd.DataFrame(daily_checks)
    maximum_error = (
        checks[["EquityError", "TurnoverError", "CostError"]]
        .abs()
        .max()
        .max()
    )
    if pd.notna(maximum_error) and float(maximum_error) > 1e-6:
        raise RuntimeError(
            "Ticker execution replay does not reconcile with portfolio "
            f"engine; maximum error={maximum_error}"
        )
    return ledger, checks


def build_trade_events(
    targets: pd.DataFrame,
    executions: pd.DataFrame,
    params: StrategyParams,
    *,
    company_by_ticker: dict[str, str] | None = None,
) -> pd.DataFrame:
    events = targets.loc[
        targets["TradeAction"].isin(["BUY", "SELL"])
    ].copy()
    events["QualifiedCount"] = (
        targets["Qualified"]
        .fillna(False)
        .groupby(targets["Date"])
        .transform("sum")
        .loc[events.index]
        .astype(int)
    )
    merge_columns = [
        "SignalDate",
        "ExecutionDate",
        "Ticker",
        "ExecutionSide",
        "ExecutionType",
        "TargetWeight",
        "PreTradeEquity",
        "ExecutionPrice",
        "SharesTraded",
        "SharesAfter",
        "TradeNotional",
        "GrossTradeAmount",
        "NotionalAfter",
        "AllocatedTransactionCost",
        "UsedPriorCloseFallback",
    ]
    events = events.merge(
        executions.loc[
            executions["SignalTradeAction"].isin(["BUY", "SELL"]),
            merge_columns,
        ],
        left_on=["Date", "Ticker", "TradeAction"],
        right_on=["SignalDate", "Ticker", "ExecutionSide"],
        how="left",
        validate="one_to_one",
    )
    events["Company"] = events["Ticker"].map(company_by_ticker or {}).fillna(
        events.get("Company", events["Ticker"])
    )
    events = add_factor_contributions(events, params)
    events["Reason"] = events.apply(
        lambda row: explain_trade(row, params),
        axis=1,
    )
    preferred = [
        "Date",
        "ExecutionDate",
        "Ticker",
        "Company",
        "TradeAction",
        "ExecutionType",
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
        "Rank",
        "QualifiedCount",
        "AlphaScore",
        "SignalReferenceReturn",
        "HoldingRebalances",
        "BestReplacementAlphaScore",
        "ReplacementScoreAdvantage",
        "Trend200",
        "Return126",
        "MomentumContribution",
        "MAContribution",
        "MACDContribution",
        "OBVContribution",
        "GrowthContribution",
        "QualityContribution",
        "RiskContribution",
        "TopFactorContributions",
        "EpsTtm",
        "EpsTtmGrowthYoY",
        "DcfPrice",
        "DcfPriceGrowthYoY",
        "DcfUpside",
        "PeTtm",
        "EbitdaTtm",
        "EbitdaTtmGrowthYoY",
        "EvEbitdaTtm",
        "UsedPriorCloseFallback",
    ]
    return events.loc[
        :,
        [column for column in preferred if column in events],
    ].sort_values(["ExecutionDate", "TradeAction", "Ticker"]).reset_index(
        drop=True
    )


def add_factor_contributions(
    frame: pd.DataFrame,
    params: StrategyParams,
) -> pd.DataFrame:
    result = frame.copy()
    technical_weight = params.trend_weight / len(TECHNICAL_COMPONENTS)
    contribution_specs = {
        "MomentumContribution": (
            "MomentumFactor",
            params.momentum_weight,
            "모멘텀",
        ),
        "MAContribution": ("MAFactor", technical_weight, "MA 추세"),
        "MACDContribution": (
            "MACDFactor",
            technical_weight,
            "MACD",
        ),
        "OBVContribution": ("OBVFactor", technical_weight, "OBV 수급"),
        "GrowthContribution": (
            "GrowthFactor",
            params.growth_weight,
            "재무 성장",
        ),
        "QualityContribution": (
            "QualityFactor",
            params.quality_weight,
            "재무 가치·품질",
        ),
        "RiskContribution": (
            "RiskControlFactor",
            params.risk_control_weight,
            "위험 통제",
        ),
    }
    for output, (source, weight, _) in contribution_specs.items():
        result[output] = (
            pd.to_numeric(result.get(source), errors="coerce").fillna(0.0)
            * weight
        )

    contribution_columns = list(contribution_specs)
    labels = {
        output: label for output, (_, _, label) in contribution_specs.items()
    }

    def top_contributions(row: pd.Series) -> str:
        ordered = sorted(
            (
                (labels[column], float(row[column]))
                for column in contribution_columns
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ", ".join(
            f"{label} {value:+.3f}" for label, value in ordered[:3]
        )

    result["TopFactorContributions"] = result.apply(
        top_contributions,
        axis=1,
    )
    return result


def explain_trade(row: pd.Series, params: StrategyParams) -> str:
    action = str(row.get("TradeAction", ""))
    if action == "BUY":
        rank = _format_integer(row.get("Rank"))
        count = _format_integer(row.get("QualifiedCount"))
        score = _format_number(row.get("AlphaScore"), 3)
        trend = _format_percent(row.get("Trend200"))
        momentum = _format_percent(row.get("Return126"))
        financial = _financial_reason(row)
        return (
            f"진입 가능 조건 통과(200일 추세 {trend}, 126일 수익률 "
            f"{momentum}). 적격 {count}개 중 {rank}위, AlphaScore "
            f"{score}. 주요 점수 기여: "
            f"{row.get('TopFactorContributions', '')}. {financial}"
        )

    reason = str(row.get("ExitReason", "") or "")
    reference_return = _format_percent(row.get("SignalReferenceReturn"))
    rank = _format_integer(row.get("Rank"), missing="순위 밖")
    advantage = _format_number(
        row.get("ReplacementScoreAdvantage"),
        3,
        missing="N/A",
    )
    held = _format_integer(row.get("HoldingRebalances"))
    if reason == "PROFITABLE_ROTATION":
        return (
            f"수익 {reference_return}로 최소 이익 "
            f"{params.minimum_exit_gain:.0%}를 넘었고, 순위 {rank}로 "
            f"회전 기준 {params.exit_rank}위 밖. 대체 후보 점수 우위 "
            f"{advantage}(필요 {params.replacement_score_advantage:.3f})가 "
            "확인되어 더 높은 점수 종목으로 교체."
        )
    if reason == "HARD_STOP":
        return (
            f"신호 종가 기준 손실 {reference_return}가 하드스톱 "
            f"{params.hard_stop_return:.0%} 이하로 내려가 손실 제한 매도."
        )
    if reason == "CONVICTION_BREAKDOWN":
        return (
            f"{held}회 리밸런싱 보유 후에도 수익 {reference_return}, "
            f"순위 {rank}, 200일 추세 {_format_percent(row.get('Trend200'))}, "
            f"126일 수익률 {_format_percent(row.get('Return126'))}로 "
            "확신 붕괴 조건을 동시에 충족해 매도."
        )
    if reason == "TRAILING_STOP":
        return (
            f"수익 보호용 추적 손절 발동. 신호 기준 수익 "
            f"{reference_return}, 고점 대비 하락 "
            f"{_format_percent(row.get('TrailingDrawdown'))}."
        )
    if reason == "MISSING_REFERENCE_EXIT":
        return (
            f"진입 기준가 누락 상태에서 순위 {rank} 또는 진입 필터 "
            "이탈이 확인되어 데이터 안전 규칙으로 매도."
        )
    return (
        f"순위 {rank} 또는 진입 필터 이탈로 매도. 신호 기준 수익 "
        f"{reference_return}; 원인 코드 {reason or '미기록'}."
    )


def build_position_ledger(
    trade_events: pd.DataFrame,
    executions: pd.DataFrame,
    result: PortfolioResult,
    *,
    latest_prices: pd.DataFrame,
    end_date: str,
    company_by_ticker: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pair model entries/exits and attribute exact portfolio PnL."""

    attribution = (
        result.attribution.copy()
        if result.attribution is not None
        else pd.DataFrame()
    )
    if not attribution.empty:
        attribution["Date"] = pd.to_datetime(attribution["Date"])
    execution_by_ticker = {
        ticker: group.sort_values("ExecutionDate").copy()
        for ticker, group in executions.groupby("Ticker")
    }
    latest = (
        latest_prices.loc[
            latest_prices["Date"].le(pd.Timestamp(end_date))
        ]
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .set_index("Ticker")
    )
    open_positions: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for event in trade_events.sort_values(
        ["ExecutionDate", "TradeAction", "Ticker"]
    ).itertuples(index=False):
        ticker = str(event.Ticker)
        if event.TradeAction == "BUY":
            open_positions[ticker] = pd.Series(event._asdict())
            continue
        if event.TradeAction != "SELL" or ticker not in open_positions:
            continue
        entry = open_positions.pop(ticker)
        rows.append(
            _position_record(
                ticker,
                entry,
                pd.Series(event._asdict()),
                execution_by_ticker.get(ticker, pd.DataFrame()),
                attribution,
                company_by_ticker or {},
            )
        )
    for ticker, entry in open_positions.items():
        last = latest.loc[ticker] if ticker in latest.index else pd.Series()
        rows.append(
            _position_record(
                ticker,
                entry,
                None,
                execution_by_ticker.get(ticker, pd.DataFrame()),
                attribution,
                company_by_ticker or {},
                mark_date=pd.Timestamp(end_date),
                mark_price=pd.to_numeric(
                    last.get("Close"),
                    errors="coerce",
                ),
            )
        )
    result_frame = pd.DataFrame(rows)
    if result_frame.empty:
        return result_frame
    return result_frame.sort_values(
        ["EntryExecutionDate", "Ticker"]
    ).reset_index(drop=True)


def build_weekly_holdings(
    targets: pd.DataFrame,
    company_by_ticker: dict[str, str],
) -> pd.DataFrame:
    holdings = targets.loc[targets["TargetWeight"].gt(0)].copy()
    columns = [
        "Date",
        "Ticker",
        "Company",
        "TargetWeight",
        "Rank",
        "AlphaScore",
        "SignalReferenceReturn",
        "HoldingRebalances",
        "TradeAction",
    ]
    holdings["Company"] = holdings["Ticker"].map(company_by_ticker).fillna(
        holdings.get("Company", holdings["Ticker"])
    )
    return holdings.loc[
        :, [column for column in columns if column in holdings]
    ].sort_values(["Date", "TargetWeight", "Ticker"]).reset_index(drop=True)


def render_trade_report(
    *,
    trade_events: pd.DataFrame,
    positions: pd.DataFrame,
    executions: pd.DataFrame,
    weekly_holdings: pd.DataFrame,
    equity: pd.DataFrame,
    strategy: PortfolioResult,
    spy_summary: dict[str, object],
    params: StrategyParams,
    generated_at: datetime,
) -> str:
    summary = strategy.summary
    closed = positions.loc[positions["Status"].eq("CLOSED")]
    open_positions = positions.loc[positions["Status"].eq("OPEN")]
    win_rate = (
        float(closed["AttributedNetPnL"].gt(0).mean() * 100)
        if not closed.empty
        else np.nan
    )
    unique_tickers = int(positions["Ticker"].nunique())
    event_table = _display_event_table(trade_events)
    position_table = _display_position_table(positions)
    current_table = _display_current_table(open_positions)
    year_table = _display_year_table(trade_events, positions)
    chart_data = equity.loc[
        :, ["Date", "Equity", "SPYEquity"]
    ].dropna(subset=["Date", "Equity"])
    chart_json = json.dumps(
        [
            {
                "date": pd.Timestamp(row.Date).strftime("%Y-%m-%d"),
                "strategy": round(float(row.Equity), 2),
                "spy": (
                    round(float(row.SPYEquity), 2)
                    if pd.notna(row.SPYEquity)
                    else None
                ),
            }
            for row in chart_data.itertuples(index=False)
        ],
        ensure_ascii=False,
    )
    caveat = (
        "주의: 이 숫자는 생존편향·2,000회 다중검정·2025/2026 "
        "사후관찰·5종목 선택의 추가 최적화로 상방 오염될 수 있습니다. "
        "Macrotrends 재무도 완전한 point-in-time 데이터가 아닙니다."
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V7-3 5종목 매매 리포트</title>
<style>
:root{{--bg:#0e1117;--panel:#171b23;--line:#2a303b;--text:#e7eaf0;
--muted:#9aa4b2;--blue:#72b7ff;--green:#55d187;--red:#ff7891;--gold:#f4c95d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);
font-family:"Segoe UI","Malgun Gothic",sans-serif}} .wrap{{max-width:1520px;
margin:0 auto;padding:28px}} h1{{margin:0 0 8px;font-size:30px}} h2{{margin-top:34px}}
.sub,.note{{color:var(--muted);line-height:1.6}} .warning{{padding:14px 16px;
background:#2a2011;border:1px solid #6b5324;border-radius:10px;color:#f4d58a}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
gap:12px;margin:20px 0}} .card{{background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:16px}} .label{{font-size:12px;color:var(--muted);
margin-bottom:8px}} .value{{font-size:24px;font-weight:700}} .positive{{color:var(--green)}}
.negative{{color:var(--red)}} .panel{{background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:16px;margin:14px 0}} .table-wrap{{overflow:auto;max-height:680px}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th{{position:sticky;top:0;
background:#202631;color:#cbd3df;z-index:1}} th,td{{padding:9px 10px;
border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:nth-child(6),td:nth-child(6){{white-space:normal;min-width:360px;text-align:left}}
th:first-child,td:first-child,th:nth-child(3),td:nth-child(3){{text-align:left}}
tr:hover{{background:#1d2430}} input,select{{background:#111722;color:var(--text);
border:1px solid var(--line);border-radius:7px;padding:9px;margin-right:8px}}
canvas{{width:100%;height:320px}} .legend{{display:flex;gap:20px;color:var(--muted);
font-size:12px}} .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;
margin-right:6px}} code{{color:#d4bfff}} @media(max-width:700px){{.wrap{{padding:14px}}
th:nth-child(6),td:nth-child(6){{min-width:260px}}}}
</style>
</head>
<body><main class="wrap">
<h1>V7-3 · 5종목 실제 매매 리포트</h1>
<div class="sub">연속 운용 {summary.start_date:%Y-%m-%d} → {summary.end_date:%Y-%m-%d}
· 금요일 종가 신호 → 다음 거래일 시가 체결 · 거래비용 10bps ·
생성 {generated_at.astimezone(UTC):%Y-%m-%d %H:%M UTC}</div>
<p class="warning">{html.escape(caveat)}</p>
<section class="cards">
{_card("초기 자금", _money(summary.initial_capital))}
{_card("최종 평가액", _money(summary.final_value), summary.final_value >= summary.initial_capital)}
{_card("V7-3 순수익률", f"{summary.roi_percent:+.2f}%", summary.roi_percent >= 0)}
{_card("SPY 순수익률", f"{float(spy_summary['ROI']):+.2f}%", float(spy_summary['ROI']) >= 0)}
{_card("연환산 Sharpe", f"{summary.sharpe_ratio:.2f}")}
{_card("최대낙폭", f"{summary.max_drawdown_percent:.2f}%", False)}
{_card("완료 포지션", f"{len(closed):,}건")}
{_card("포지션 승률", f"{win_rate:.1f}%" if pd.notna(win_rate) else "N/A")}
{_card("거래 종목", f"{unique_tickers:,}개")}
{_card("현재 보유", f"{len(open_positions):,}개")}
</section>
<section class="panel">
<h2>돈의 파이 변화</h2>
<div class="legend"><span><i class="dot" style="background:#72b7ff"></i>V7-3</span>
<span><i class="dot" style="background:#9aa4b2"></i>SPY Buy & Hold</span></div>
<canvas id="equityChart"></canvas>
</section>
<section><h2>현재 보유 종목</h2><p class="note">아직 매도 신호가 나오지 않은 포지션입니다.</p>
<div class="panel table-wrap">{current_table}</div></section>
<section><h2>연도별 매매 요약</h2><div class="panel table-wrap">{year_table}</div></section>
<section><h2>전체 보유기간 원장</h2>
<p class="note">중간 주간 리밸런싱을 포함한 종목별 일일 기여 손익을 합산한 값입니다.
단순 매수가×최종주식수 계산보다 실제 포트폴리오 손익에 더 가깝습니다.</p>
<div class="panel table-wrap">{position_table}</div></section>
<section><h2>매수·매도 신호와 이유</h2>
<p class="note">매수의 ‘주요 점수 기여’는 각 팩터값×동결 가중치입니다.
재무 성장 {params.growth_weight:.1%}, 가치·품질 {params.quality_weight:.1%},
기술 {params.trend_weight:.1%}, 모멘텀 {params.momentum_weight:.1%},
위험 통제 {params.risk_control_weight:.1%}. 기술 비중은 MA/MACD/OBV에 균등 배분됩니다.</p>
<div><input id="tickerFilter" placeholder="티커 검색">
<select id="actionFilter"><option value="">전체 행동</option>
<option value="BUY">BUY</option><option value="SELL">SELL</option></select></div>
<div class="panel table-wrap" id="eventsTable">{event_table}</div></section>
<section><h2>파일 설명</h2><div class="panel note">
<code>trade_events.csv</code>: 모델의 BUY/SELL 이벤트와 이유·체결금액<br>
<code>position_ledger.csv</code>: 진입부터 청산/현재까지 보유기간과 손익<br>
<code>execution_ledger.csv</code>: 매주 비중 재조정을 포함한 모든 실제 주문 수량·금액<br>
<code>weekly_holdings.csv</code>: 각 주 신호일의 5종목 목표 포트폴리오<br>
<code>reconciliation.csv</code>: 종목별 체결 재현과 원 백테스트의 수치 일치 검증
</div></section>
</main>
<script>
const data={chart_json};
const canvas=document.getElementById("equityChart"),ctx=canvas.getContext("2d");
function draw(){{
 const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=320;
 canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 ctx.clearRect(0,0,w,h);const pad={{l:56,r:18,t:18,b:30}};
 const vals=data.flatMap(d=>[d.strategy,d.spy]).filter(v=>v!=null);
 const min=Math.min(...vals),max=Math.max(...vals),span=max-min||1;
 const x=i=>pad.l+i/(data.length-1)*(w-pad.l-pad.r);
 const y=v=>pad.t+(max-v)/span*(h-pad.t-pad.b);
 ctx.strokeStyle="#2a303b";ctx.lineWidth=1;
 for(let i=0;i<5;i++){{const yy=pad.t+i*(h-pad.t-pad.b)/4;ctx.beginPath();
 ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();}}
 function line(key,color){{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
 let started=false;data.forEach((d,i)=>{{if(d[key]==null)return;const xx=x(i),yy=y(d[key]);
 if(!started){{ctx.moveTo(xx,yy);started=true}}else ctx.lineTo(xx,yy)}});ctx.stroke()}}
 line("spy","#9aa4b2");line("strategy","#72b7ff");
 ctx.fillStyle="#9aa4b2";ctx.font="11px Segoe UI";
 ctx.fillText("$"+Math.round(max).toLocaleString(),4,pad.t+4);
 ctx.fillText("$"+Math.round(min).toLocaleString(),4,h-pad.b);
}}
draw();window.addEventListener("resize",draw);
const tf=document.getElementById("tickerFilter"),af=document.getElementById("actionFilter");
function filter(){{
 const ticker=tf.value.toUpperCase(),action=af.value;
 document.querySelectorAll("#eventsTable tbody tr").forEach(row=>{{
  const text=row.textContent.toUpperCase();
  row.style.display=(text.includes(ticker)&&(!action||text.includes(action)))?"":"none";
 }});
}} tf.addEventListener("input",filter);af.addEventListener("change",filter);
</script></body></html>"""


def _position_record(
    ticker: str,
    entry: pd.Series,
    exit_event: pd.Series | None,
    ticker_executions: pd.DataFrame,
    attribution: pd.DataFrame,
    company_by_ticker: dict[str, str],
    *,
    mark_date: pd.Timestamp | None = None,
    mark_price: float | None = None,
) -> dict[str, object]:
    entry_date = pd.Timestamp(entry["ExecutionDate"])
    exit_date = (
        pd.Timestamp(exit_event["ExecutionDate"])
        if exit_event is not None
        else pd.Timestamp(mark_date)
    )
    relevant_exec = ticker_executions.loc[
        ticker_executions["ExecutionDate"].between(entry_date, exit_date)
    ]
    relevant_pnl = (
        attribution.loc[
            attribution["Ticker"].eq(ticker)
            & attribution["Date"].between(entry_date, exit_date)
        ]
        if not attribution.empty
        else pd.DataFrame()
    )
    exit_price = (
        float(exit_event["ExecutionPrice"])
        if exit_event is not None
        else float(mark_price)
    )
    entry_price = float(entry["ExecutionPrice"])
    price_return = (
        exit_price / entry_price - 1
        if entry_price > 0 and pd.notna(exit_price)
        else np.nan
    )
    attributed_pnl = (
        float(relevant_pnl["NetPnL"].sum())
        if not relevant_pnl.empty
        else np.nan
    )
    return {
        "Ticker": ticker,
        "Company": company_by_ticker.get(
            ticker,
            entry.get("Company", ticker),
        ),
        "Status": "CLOSED" if exit_event is not None else "OPEN",
        "EntrySignalDate": pd.Timestamp(entry["Date"]),
        "EntryExecutionDate": entry_date,
        "ExitSignalDate": (
            pd.Timestamp(exit_event["Date"])
            if exit_event is not None
            else pd.NaT
        ),
        "ExitOrMarkDate": exit_date,
        "HoldingCalendarDays": int((exit_date - entry_date).days),
        "HoldingRebalanceExecutions": len(relevant_exec),
        "EntryPrice": entry_price,
        "ExitOrMarkPrice": exit_price,
        "PriceReturn": price_return,
        "InitialShares": float(entry["SharesAfter"]),
        "InitialAllocation": float(entry["NotionalAfter"]),
        "ExitProceeds": (
            float(exit_event["GrossTradeAmount"])
            if exit_event is not None
            else np.nan
        ),
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
        "AttributedNetPnL": attributed_pnl,
        "AttributedReturnOnInitialAllocation": (
            attributed_pnl / float(entry["NotionalAfter"])
            if float(entry["NotionalAfter"]) > 0
            and pd.notna(attributed_pnl)
            else np.nan
        ),
        "EntryRank": entry.get("Rank"),
        "EntryAlphaScore": entry.get("AlphaScore"),
        "EntryReason": entry.get("Reason", ""),
        "ExitReasonCode": (
            exit_event.get("ExitReason", "")
            if exit_event is not None
            else ""
        ),
        "ExitReason": (
            exit_event.get("Reason", "")
            if exit_event is not None
            else "아직 V7-3 매도 조건이 충족되지 않아 보유 중."
        ),
    }


def _target_schedule(
    targets: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]]:
    schedule: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
    date_positions = {date: index for index, date in enumerate(dates)}
    for signal_date, group in targets.groupby("Date", sort=True):
        signal = pd.Timestamp(signal_date)
        position = date_positions.get(signal)
        if position is None or position + 1 >= len(dates):
            continue
        schedule[pd.Timestamp(dates[position + 1])] = (
            signal,
            group.copy(),
        )
    return schedule


def _signal_value(
    signal: pd.Series | None,
    column: str,
    default: object,
) -> object:
    if signal is None or column not in signal or pd.isna(signal[column]):
        return default
    return signal[column]


def _execution_type(before: float, after: float) -> str:
    tolerance = 1e-10
    if abs(before) <= tolerance and after > tolerance:
        return "OPEN"
    if before > tolerance and abs(after) <= tolerance:
        return "CLOSE"
    if after > before:
        return "INCREASE"
    return "DECREASE"


def _financial_reason(row: pd.Series) -> str:
    parts: list[str] = []
    values = (
        ("EPS TTM 성장", "EpsTtmGrowthYoY", True),
        ("DCF 상승여력", "DcfUpside", True),
        ("DCF 성장", "DcfPriceGrowthYoY", True),
        ("EBITDA 성장", "EbitdaTtmGrowthYoY", True),
        ("P/E", "PeTtm", False),
        ("EV/EBITDA", "EvEbitdaTtm", False),
    )
    for label, column, percent in values:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(value):
            continue
        rendered = f"{float(value):+.1%}" if percent else f"{float(value):.1f}배"
        parts.append(f"{label} {rendered}")
    return (
        "당시 재무: " + ", ".join(parts) + "."
        if parts
        else "당시 사용 가능한 재무값은 결측 또는 stale 처리됨."
    )


def _display_event_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    columns = [
        "ExecutionDate",
        "Ticker",
        "Company",
        "TradeAction",
        "GrossTradeAmount",
        "Reason",
        "ExecutionPrice",
        "SharesTraded",
        "TargetWeight",
        "Rank",
        "AlphaScore",
    ]
    shown = shown.loc[:, [column for column in columns if column in shown]]
    shown = shown.rename(
        columns={
            "ExecutionDate": "체결일",
            "Ticker": "티커",
            "Company": "회사",
            "TradeAction": "행동",
            "GrossTradeAmount": "체결금액",
            "Reason": "이유",
            "ExecutionPrice": "체결가",
            "SharesTraded": "주식수 변화",
            "TargetWeight": "목표비중",
            "Rank": "순위",
            "AlphaScore": "Alpha",
        }
    )
    return _to_html_table(
        shown,
        date_columns=("체결일",),
        money_columns=("체결금액", "체결가"),
        percent_columns=("목표비중",),
        number_columns=("주식수 변화", "Alpha"),
    )


def _display_position_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    columns = [
        "Ticker",
        "Status",
        "EntryExecutionDate",
        "ExitOrMarkDate",
        "HoldingCalendarDays",
        "InitialAllocation",
        "TotalBought",
        "TotalSold",
        "PriceReturn",
        "AttributedNetPnL",
        "AttributedReturnOnInitialAllocation",
        "ExitReasonCode",
    ]
    shown = shown.loc[:, columns].rename(
        columns={
            "Ticker": "티커",
            "Status": "상태",
            "EntryExecutionDate": "매수일",
            "ExitOrMarkDate": "매도/평가일",
            "HoldingCalendarDays": "보유일",
            "InitialAllocation": "최초투입",
            "TotalBought": "누적매수",
            "TotalSold": "누적매도",
            "PriceReturn": "가격수익률",
            "AttributedNetPnL": "기여손익",
            "AttributedReturnOnInitialAllocation": "투입대비기여수익",
            "ExitReasonCode": "매도코드",
        }
    )
    return _to_html_table(
        shown,
        date_columns=("매수일", "매도/평가일"),
        money_columns=("최초투입", "누적매수", "누적매도", "기여손익"),
        percent_columns=("가격수익률", "투입대비기여수익"),
    )


def _display_current_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    columns = [
        "Ticker",
        "Company",
        "EntryExecutionDate",
        "InitialAllocation",
        "ExitOrMarkPrice",
        "PriceReturn",
        "AttributedNetPnL",
        "EntryReason",
    ]
    shown = shown.loc[:, columns].rename(
        columns={
            "Ticker": "티커",
            "Company": "회사",
            "EntryExecutionDate": "매수일",
            "InitialAllocation": "최초투입",
            "ExitOrMarkPrice": "현재가",
            "PriceReturn": "가격수익률",
            "AttributedNetPnL": "기여손익",
            "EntryReason": "매수 이유",
        }
    )
    return _to_html_table(
        shown,
        date_columns=("매수일",),
        money_columns=("최초투입", "현재가", "기여손익"),
        percent_columns=("가격수익률",),
    )


def _display_year_table(
    events: pd.DataFrame,
    positions: pd.DataFrame,
) -> str:
    event_year = events.copy()
    event_year["연도"] = pd.to_datetime(
        event_year["ExecutionDate"]
    ).dt.year
    event_summary = event_year.groupby("연도").agg(
        매수=("TradeAction", lambda values: int((values == "BUY").sum())),
        매도=("TradeAction", lambda values: int((values == "SELL").sum())),
        매매금액=("GrossTradeAmount", "sum"),
    )
    position_year = positions.copy()
    position_year["연도"] = pd.to_datetime(
        position_year["ExitOrMarkDate"]
    ).dt.year
    position_summary = position_year.groupby("연도").agg(
        실현포지션=(
            "Status",
            lambda values: int((values == "CLOSED").sum()),
        ),
        포지션기여손익=("AttributedNetPnL", "sum"),
    )
    shown = event_summary.join(position_summary, how="outer").fillna(0)
    shown.index = shown.index.astype(int)
    return _to_html_table(
        shown.reset_index(),
        money_columns=("매매금액", "포지션기여손익"),
    )


def _to_html_table(
    frame: pd.DataFrame,
    *,
    date_columns: tuple[str, ...] = (),
    money_columns: tuple[str, ...] = (),
    percent_columns: tuple[str, ...] = (),
    number_columns: tuple[str, ...] = (),
) -> str:
    shown = frame.copy()
    for column in date_columns:
        if column in shown:
            shown[column] = pd.to_datetime(shown[column]).dt.strftime(
                "%Y-%m-%d"
            )
    for column in money_columns:
        if column in shown:
            shown[column] = shown[column].map(
                lambda value: (
                    _money(float(value)) if pd.notna(value) else ""
                )
            )
    for column in percent_columns:
        if column in shown:
            shown[column] = shown[column].map(
                lambda value: (
                    f"{float(value):+.2%}" if pd.notna(value) else ""
                )
            )
    for column in number_columns:
        if column in shown:
            shown[column] = shown[column].map(
                lambda value: (
                    f"{float(value):,.3f}" if pd.notna(value) else ""
                )
            )
    return shown.to_html(
        index=False,
        border=0,
        escape=True,
        classes="data-table",
    )


def _card(label: str, value: str, positive: bool | None = None) -> str:
    css = (
        " positive"
        if positive is True
        else " negative"
        if positive is False
        else ""
    )
    return (
        '<div class="card"><div class="label">'
        f"{html.escape(label)}</div><div class=\"value{css}\">"
        f"{html.escape(value)}</div></div>"
    )


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _format_integer(value: object, missing: str = "N/A") -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return str(int(round(float(numeric)))) if pd.notna(numeric) else missing


def _format_number(
    value: object,
    digits: int,
    missing: str = "N/A",
) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):.{digits}f}" if pd.notna(numeric) else missing


def _format_percent(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):+.1%}" if pd.notna(numeric) else "N/A"


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
    raise TypeError(f"Object of type {type(value).__name__} is not serializable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
