from __future__ import annotations

"""Dashboard backtest engine: wires the price-only panel (dashboard-managed
tickers, no Macrotrends dependency) through the exact same scoring/targeting/
execution machinery used everywhere else this session (V7-3 technical
variant, SEC-filing score, PIT-Reconstructed Growth/Quality, next-session-
open execution), and shapes the result into JSON-serializable dashboard
payloads: equity curve, holdings-over-time, buy/sell trade log with reasons,
a top-15 ranked table, and a per-ticker trade/ROI summary.

Policy is the known16 frozen baseline with only top_k varied (per the
session's finding that top_k=3 concentrates risk in very few names on a
broader universe) -- every other field (filing_weight, hard_stop,
minimum_hold, replacement_advantage) and the base V7-3 factor weights are
unchanged from the production known16 config.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.cross_sectional.config import ResearchSettings, StrategyParams
from stock_research.cross_sectional.filing_signals import add_filing_factors, merge_filing_features
from stock_research.cross_sectional.filing_v7_optimization import (
    FilingV7Policy,
    generate_filing_v7_targets,
)
from stock_research.cross_sectional.point_in_time import reconstruct_growth_quality_pit_safe
from stock_research.cross_sectional.portfolio import prepare_market, run_portfolio_backtest
from stock_research.cross_sectional.sec_filings import split_adjust_share_series
from stock_research.cross_sectional.signals import (
    monthly_weight_reset_targets,
    signal_day_panel,
)
from stock_research.cross_sectional.v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)
from stock_research.cross_sectional.winner_attribution import summarize_ticker_contributions
from stock_research.dashboard import data_collection, state as dashboard_state
from stock_research.dashboard.price_panel import DashboardMember, build_price_only_panel
from stock_research.paths import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_PARAMS_CONFIG_PATH = REPO_ROOT / "config/cross_sectional/filing_v7_optimization.json"
TECHNICAL_VARIANT_NAME = "V7_3_MA_MACD_OBV_SLOT5"
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_BPS = 10.0
EXECUTION_POLICY_NAME = "WEEKLY_SIGNAL_MONTHLY_FIRST_WEIGHT_RESET"
MINIMUM_PRICE_HISTORY_SESSIONS = 200
# Calendar-day buffer kept before the requested start date so every ticker
# still has a full 200-session lookback for Trend200/eligibility once the
# panel is truncated there (see build_price_only_panel's warmup_start).
WARMUP_BUFFER_DAYS = 400

# Same frozen-baseline fields as the known16 production policy; only top_k
# is a dashboard setting.
_POLICY_BASE = dict(
    filing_weight=0.25,
    minimum_filing_coverage=6,
    filing_quality_floor=-0.15,
    red_flag_veto=True,
    exit_rank_buffer=4,
    hard_stop_return=-0.35,
    minimum_hold_rebalances=4,
    replacement_score_advantage=0.05,
)


def _base_params() -> StrategyParams:
    config = json.loads(BASE_PARAMS_CONFIG_PATH.read_text(encoding="utf-8"))
    return StrategyParams.from_dict(dict(config["base_v7_params"]))


def _policy(top_k: int) -> FilingV7Policy:
    return FilingV7Policy(top_k=top_k, **_POLICY_BASE)


def _members(paths: ProjectPaths, state: "dashboard_state.DashboardState") -> list[DashboardMember]:
    return [
        DashboardMember(t.ticker, t.company, paths.stock_root / t.price_path)
        for t in state.tickers
    ]


def _ensure_filing_data(paths: ProjectPaths, state: "dashboard_state.DashboardState") -> Path:
    path = (
        paths.processed
        / "SEC Filings"
        / data_collection.DASHBOARD_FILING_LABEL
        / "point_in_time_features.csv"
    )
    if not path.exists():
        data_collection.sync_filings_for_dashboard(paths, state)
    return path


def _settings(train_start: str, train_end: str, oos_start: str, oos_end: str, cross_section_size: int) -> ResearchSettings:
    return ResearchSettings(
        train_start=train_start,
        train_end=train_end,
        validation_periods={"Dashboard": (oos_start, oos_end)},
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        minimum_price_history_sessions=MINIMUM_PRICE_HISTORY_SESSIONS,
        minimum_cross_section_size=cross_section_size,
        pit_strict=True,
    )


def _build_factored_panel(
    members: list[DashboardMember],
    settings: ResearchSettings,
    filing_features: pd.DataFrame,
    *,
    warmup_start: str,
) -> pd.DataFrame:
    panel = build_price_only_panel(members, settings, warmup_start=warmup_start)
    observed = add_v7_technical_observations(panel)
    technical = add_v7_technical_factors(observed, settings)
    variant = next(v for v in TECHNICAL_VARIANTS if v.name == TECHNICAL_VARIANT_NAME)
    v7_panel = scoring_panel_for_variant(technical, variant)
    merged = merge_filing_features(v7_panel, filing_features)
    merged = reconstruct_growth_quality_pit_safe(
        merged, minimum_cross_section_size=settings.minimum_cross_section_size
    )
    return add_filing_factors(merged, minimum_cross_section_size=settings.minimum_cross_section_size)


@dataclass
class ScoredPanel:
    """The scored weekly cross-section -- every ticker, every signal date,
    with the final AlphaScore and every factor that feeds it still present
    as its own column. `run_backtest` layers a portfolio simulation on top
    of this; signal-health diagnostics (Rank IC, Universe EW benchmark) only
    need this part, not the simulation, so it's split out to avoid making
    every caller pay for (or duplicate) the panel-building step."""

    members: list[DashboardMember]
    settings: ResearchSettings
    factored: pd.DataFrame
    scored: pd.DataFrame
    targets: pd.DataFrame
    top_k: int


def build_scored_panel(
    paths: ProjectPaths,
    state: "dashboard_state.DashboardState",
    *,
    start: str,
    end: str,
    top_k: int | None = None,
    filing_features_override: pd.DataFrame | None = None,
) -> ScoredPanel:
    top_k = top_k or state.top_k
    members = _members(paths, state)
    if len(members) < 2:
        raise ValueError("Need at least 2 tickers in the universe to backtest")
    cross_section_size = min(8, max(2, len(members) // 2))

    warmup_start = str((pd.Timestamp(start) - pd.Timedelta(days=WARMUP_BUFFER_DAYS)).date())
    train_end = str((pd.Timestamp(start) - pd.Timedelta(days=1)).date())
    settings = _settings(warmup_start, train_end, start, end, cross_section_size)

    filing_features = (
        filing_features_override.copy()
        if filing_features_override is not None
        else pd.read_csv(_ensure_filing_data(paths, state))
    )

    factored = _build_factored_panel(members, settings, filing_features, warmup_start=warmup_start)
    signal_days = signal_day_panel(factored, start, end, settings.rebalance_weekday)
    if signal_days.empty:
        raise ValueError(f"No trading data in range {start}..{end}")
    policy = _policy(top_k)
    base_params = _base_params()
    scored, targets = generate_filing_v7_targets(signal_days, base_params, policy)
    return ScoredPanel(members=members, settings=settings, factored=factored, scored=scored, targets=targets, top_k=top_k)


def run_backtest(
    paths: ProjectPaths,
    state: "dashboard_state.DashboardState",
    *,
    start: str,
    end: str,
    top_k: int | None = None,
) -> dict[str, object]:
    panel = build_scored_panel(paths, state, start=start, end=end, top_k=top_k)
    return simulate_portfolio(panel, start=start, end=end)


def simulate_portfolio(panel: ScoredPanel, *, start: str, end: str) -> dict[str, object]:
    """The portfolio-simulation half of `run_backtest`, split out so a
    caller that already has a `ScoredPanel` (e.g. because it also needs the
    raw scored frame for signal-health diagnostics) doesn't have to rebuild
    the panel a second time to get the NAV/summary."""

    members, targets, top_k = panel.members, panel.targets, panel.top_k
    execution_targets = monthly_weight_reset_targets(targets)

    market = prepare_market(panel.factored, start=start, end=end)
    result = run_portfolio_backtest(
        pd.DataFrame(),
        execution_targets,
        start=start,
        end=end,
        initial_capital=panel.settings.initial_capital,
        transaction_cost_bps=panel.settings.transaction_cost_bps,
        prepared_market=market,
        record_attribution=True,
    )

    latest_signal_date = pd.Timestamp(targets["Date"].max())
    latest_execution = execution_targets.loc[
        execution_targets["Date"].eq(latest_signal_date)
    ]
    latest_execution_reason = (
        str(latest_execution["ExecutionReason"].iloc[0])
        if not latest_execution.empty
        else None
    )

    return {
        "meta": {
            "start": start,
            "end": end,
            "latest_signal_date": _clean(targets["Date"].max()),
            "top_k": top_k,
            "tickers": sorted(m.ticker for m in members),
            "policy": {**_POLICY_BASE, "top_k": top_k},
            "signal_frequency": "WEEKLY",
            "weight_reset_frequency": "MONTHLY_FIRST_SIGNAL",
            "immediate_membership_changes": True,
            "execution_policy": EXECUTION_POLICY_NAME,
            "latest_signal_execution_reason": latest_execution_reason,
        },
        "summary": _summary_payload(result),
        "equity_curve": _equity_curve_payload(result),
        "holdings_timeline": _holdings_timeline_payload(result),
        "trades": _trades_payload(targets),
        "top15_latest": _top15_payload(targets, member_company={m.ticker: m.company for m in members}),
        "ticker_summary": _ticker_summary_payload(targets, result),
    }


def _clean(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating,)):
        return _clean(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _summary_payload(result) -> dict[str, object]:
    s = result.summary
    return {k: _clean(v) for k, v in s.as_dict().items()}


def _equity_curve_payload(result) -> list[dict[str, object]]:
    daily = result.daily.copy()
    daily["CashWeight"] = np.where(daily["Equity"] > 0, daily["Cash"] / daily["Equity"], 1.0)
    return [
        {
            "date": _clean(row.Date),
            "equity": _clean(row.Equity),
            "cash": _clean(row.Cash),
            "cash_weight": _clean(row.CashWeight),
            "selected_count": _clean(row.SelectedCount),
        }
        for row in daily.itertuples(index=False)
    ]


def _holdings_timeline_payload(result) -> list[dict[str, object]]:
    if result.attribution is None or result.attribution.empty:
        return []
    attribution = result.attribution
    held = attribution.loc[attribution["EndingWeight"].abs() > 1e-6]
    return [
        {
            "date": _clean(row.Date),
            "ticker": row.Ticker,
            "weight": _clean(row.EndingWeight),
        }
        for row in held.itertuples(index=False)
    ]


_EXIT_REASON_TEXT = {
    "UNIVERSE_EXIT": "SEC 공시에서 계속기업 불확실성/중대한 내부통제 취약점/재무제표 재작성 중 하나가 새로 발견되어 즉시 강제 매도됨.",
    "HARD_STOP": "진입가 대비 손실이 -35% 손절 기준에 도달해 즉시 매도됨.",
    "TRAILING_STOP": "고점 대비 트레일링 스톱 기준에 도달해 매도됨.",
    "PROFITABLE_ROTATION": "수익 중이었으나 순위가 밀렸고, 더 높은 점수의 대체 후보가 나타나 교체 매도됨.",
    "CONVICTION_BREAKDOWN": "순위가 크게 하락하고 추세·모멘텀이 동시에 무너져 강제 매도됨.",
    "MISSING_REFERENCE_EXIT": "진입가 기록 없이 자격을 상실해 매도됨.",
}

_FACTOR_LABELS = {
    "MomentumFactor": "모멘텀",
    "TrendFactor": "추세",
    "GrowthFactor": "성장성(SEC 공시)",
    "QualityFactor": "재무품질(SEC 공시)",
    "RiskControlFactor": "저변동성/낙폭관리",
}


def _buy_reason(row: pd.Series) -> str:
    factor_values = {
        label: row.get(col)
        for col, label in _FACTOR_LABELS.items()
        if pd.notna(row.get(col))
    }
    top_factors = sorted(factor_values.items(), key=lambda kv: kv[1], reverse=True)[:2]
    factor_text = ", ".join(f"{label}({value:+.2f})" for label, value in top_factors)
    rank = row.get("Rank")
    score = row.get("AlphaScore")
    filing_score = row.get("FilingScore")
    parts = [f"상대순위 {int(rank)}위" if pd.notna(rank) else "순위 정보 없음"]
    if pd.notna(score):
        parts.append(f"Final Score {score:+.3f}")
    if factor_text:
        parts.append(f"주요 근거: {factor_text}")
    if pd.notna(filing_score):
        parts.append(f"SEC 공시 점수 {filing_score:+.3f}")
    return " / ".join(parts) + " 로 편입됨."


def _sell_reason(row: pd.Series) -> str:
    exit_reason = row.get("ExitReason")
    base = _EXIT_REASON_TEXT.get(str(exit_reason), f"매도 사유: {exit_reason}")
    ref_return = row.get("SignalReferenceReturn")
    if pd.notna(ref_return):
        base += f" (매도 시점 수익률 {ref_return:+.1%})"
    return base


def _trades_payload(targets: pd.DataFrame) -> list[dict[str, object]]:
    trades = targets.loc[targets["TradeAction"].isin(["BUY", "SELL"])].sort_values(["Date", "Ticker"])
    rows: list[dict[str, object]] = []
    for row_dict in trades.to_dict(orient="records"):
        series = pd.Series(row_dict)
        action = series["TradeAction"]
        reason = _buy_reason(series) if action == "BUY" else _sell_reason(series)
        rows.append(
            {
                "date": _clean(series["Date"]),
                "ticker": series["Ticker"],
                "action": action,
                "close": _clean(series.get("Close")),
                "rank": _clean(series.get("Rank")),
                "score": _clean(series.get("AlphaScore")),
                "exit_reason": series.get("ExitReason") if pd.notna(series.get("ExitReason")) else None,
                "reason": reason,
            }
        )
    return rows


def _top15_payload(targets: pd.DataFrame, member_company: dict[str, str]) -> list[dict[str, object]]:
    latest_date = targets["Date"].max()
    if pd.isna(latest_date):
        return []
    latest = targets.loc[targets["Date"].eq(latest_date)].copy()
    latest["RankSort"] = latest["Rank"].fillna(10_000)
    latest = latest.sort_values("RankSort").head(15)
    rows = []
    for row_dict in latest.to_dict(orient="records"):
        ticker = row_dict["Ticker"]
        model_selected = bool(row_dict.get("ModelSelected", False))
        trade_action = row_dict.get("TradeAction")
        qualified = bool(row_dict.get("Qualified", False))
        rank = row_dict.get("Rank")
        if model_selected:
            status = "HOLD"
        elif qualified and pd.notna(rank) and trade_action == "WATCH":
            status = "WATCHLIST"
        elif qualified:
            status = "QUALIFIED"
        else:
            status = "NOT_QUALIFIED"
        rows.append(
            {
                "rank": _clean(row_dict.get("Rank")),
                "ticker": ticker,
                "company": member_company.get(ticker, ticker),
                "final_score": _clean(row_dict.get("AlphaScore")),
                "v7_score": _clean(row_dict.get("BaseV7Score") if "BaseV7Score" in row_dict else None),
                "filing_score": _clean(row_dict.get("FilingScore")),
                "status": status,
                "target_weight": _clean(row_dict.get("TargetWeight")),
            }
        )
    return rows


def _ticker_summary_payload(targets: pd.DataFrame, result) -> list[dict[str, object]]:
    if result.attribution is None or result.attribution.empty:
        pnl_by_ticker: dict[str, float] = {}
        cash_flow_by_ticker: dict[str, dict[str, float]] = {}
    else:
        pnl_by_ticker = result.attribution.groupby("Ticker")["NetPnL"].sum().to_dict()
        cash_flow_by_ticker = {}
        for attribution_ticker, attribution in result.attribution.groupby("Ticker"):
            cash_flow = pd.to_numeric(
                attribution["TradeCashFlow"], errors="coerce"
            ).fillna(0.0)
            trade_notional = pd.to_numeric(
                attribution["TradeNotional"], errors="coerce"
            ).fillna(0.0)
            trade_shares = pd.to_numeric(
                attribution["TradeShares"], errors="coerce"
            ).fillna(0.0)
            bought_shares = float(trade_shares.clip(lower=0).sum())
            bought_notional = float(trade_notional.clip(lower=0).sum())
            latest = attribution.sort_values("Date").iloc[-1]
            cash_flow_by_ticker[str(attribution_ticker)] = {
                "gross_buys": float(-cash_flow.clip(upper=0).sum()),
                "net_sales": float(cash_flow.clip(lower=0).sum()),
                "ending_value": float(latest.get("EndingNotional", 0.0)),
                "average_buy_price": (
                    bought_notional / bought_shares
                    if bought_shares > 0
                    else math.nan
                ),
            }

    rows: list[dict[str, object]] = []
    for ticker, group in targets.sort_values("Date").groupby("Ticker"):
        trades = group.loc[group["TradeAction"].isin(["BUY", "SELL"])]
        buys = trades.loc[trades["TradeAction"].eq("BUY")]
        sells = trades.loc[trades["TradeAction"].eq("SELL")]
        if buys.empty and sells.empty:
            continue
        open_entry = None
        completed: list[dict[str, object]] = []
        for row_dict in trades.to_dict(orient="records"):
            if row_dict["TradeAction"] == "BUY":
                open_entry = row_dict
            elif row_dict["TradeAction"] == "SELL" and open_entry is not None:
                entry_price = open_entry.get("Close")
                exit_price = row_dict.get("Close")
                roi = (exit_price / entry_price - 1) if entry_price else None
                completed.append(
                    {
                        "entry_date": open_entry["Date"],
                        "exit_date": row_dict["Date"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "roi_pct": roi,
                        "exit_reason": row_dict.get("ExitReason"),
                    }
                )
                open_entry = None
        net_pnl = pnl_by_ticker.get(ticker, 0.0)
        capital = cash_flow_by_ticker.get(
            str(ticker),
            {
                "gross_buys": 0.0,
                "net_sales": 0.0,
                "ending_value": 0.0,
                "average_buy_price": math.nan,
            },
        )
        gross_buys = capital["gross_buys"]
        recovered_and_held = capital["net_sales"] + capital["ending_value"]
        aggregate_roi_pct = (
            (recovered_and_held / gross_buys - 1.0) * 100.0
            if gross_buys > 0
            else None
        )
        rows.append(
            {
                "ticker": ticker,
                "completed_trades": len(completed),
                "still_held": open_entry is not None,
                "net_pnl": _clean(net_pnl),
                "gross_buys": _clean(gross_buys),
                "net_sales": _clean(capital["net_sales"]),
                "ending_value": _clean(capital["ending_value"]),
                "recovered_and_held": _clean(recovered_and_held),
                "aggregate_roi_pct": _clean(aggregate_roi_pct),
                "average_buy_price": _clean(capital["average_buy_price"]),
                "trades": [
                    {
                        "entry_date": _clean(t["entry_date"]),
                        "exit_date": _clean(t["exit_date"]),
                        "entry_price": _clean(t["entry_price"]),
                        "exit_price": _clean(t["exit_price"]),
                        "roi_pct": _clean(t["roi_pct"]),
                        "exit_reason": t["exit_reason"],
                        "hold_days": (
                            (t["exit_date"] - t["entry_date"]).days
                            if pd.notna(t["entry_date"]) and pd.notna(t["exit_date"])
                            else None
                        ),
                    }
                    for t in completed
                ],
            }
        )
    rows.sort(key=lambda r: (r["net_pnl"] if r["net_pnl"] is not None else 0.0), reverse=True)
    return rows


def _trailing_twelve_month_eps(filing_features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Real, PIT-safe trailing-12-month EPS from filed NetIncome/DilutedShares.

    Most US filers never file a standalone Q4 10-Q -- Q4 only ever appears
    bundled into the annual 10-K (FY total). A naive "sum of the last 4
    QUARTERLY rows" therefore silently skips every Q4 and wraps around into
    next year's Q1, understating TTM net income for most of each fiscal
    year. Q4 is derived here as FY total minus the filed Q1+Q2+Q3, then a
    continuous 4-quarter rolling sum is taken over the reconstructed
    quarterly series. Display-only: not part of any trading signal.
    """
    empty = pd.DataFrame(columns=["AvailableDate", "EpsTtm", "EarningsGrowthYoY"])
    sub = filing_features.loc[filing_features["Ticker"].astype(str).str.upper().eq(ticker)].copy()
    if sub.empty:
        return empty
    sub["PeriodOfReport"] = pd.to_datetime(sub["PeriodOfReport"], errors="coerce")
    sub["AvailableDate"] = pd.to_datetime(sub["AvailableDate"], errors="coerce")
    sub = sub.sort_values("PeriodOfReport")
    sub["NetIncome"] = pd.to_numeric(sub["NetIncome"], errors="coerce")
    sub["DilutedShares"] = split_adjust_share_series(pd.to_numeric(sub["DilutedShares"], errors="coerce"))

    quarterly = sub.loc[
        sub["PeriodKind"].eq("QUARTERLY"),
        ["PeriodOfReport", "AvailableDate", "NetIncome", "DilutedShares"],
    ].copy()
    annual = sub.loc[sub["PeriodKind"].eq("ANNUAL")]
    derived_q4_rows = []
    for _, arow in annual.iterrows():
        fy_end = arow["PeriodOfReport"]
        preceding = quarterly.loc[
            quarterly["PeriodOfReport"].between(fy_end - pd.Timedelta(days=280), fy_end - pd.Timedelta(days=1))
        ]
        if len(preceding) == 3 and pd.notna(arow["NetIncome"]) and preceding["NetIncome"].notna().all():
            derived_q4_rows.append(
                {
                    "PeriodOfReport": fy_end,
                    "AvailableDate": arow["AvailableDate"],
                    "NetIncome": arow["NetIncome"] - preceding["NetIncome"].sum(),
                    "DilutedShares": arow["DilutedShares"],
                }
            )
    all_quarters = pd.concat([quarterly, pd.DataFrame(derived_q4_rows)], ignore_index=True)
    all_quarters = all_quarters.sort_values("PeriodOfReport").drop_duplicates("PeriodOfReport", keep="last")
    all_quarters["NetIncomeTtm"] = all_quarters["NetIncome"].rolling(4, min_periods=4).sum()
    all_quarters["EpsTtm"] = all_quarters["NetIncomeTtm"] / all_quarters["DilutedShares"]

    growth = sub[["PeriodOfReport", "NetIncomeGrowthYoYFiled"]]
    merged = pd.merge_asof(
        all_quarters[["AvailableDate", "PeriodOfReport", "EpsTtm"]].sort_values("PeriodOfReport"),
        growth.sort_values("PeriodOfReport"),
        on="PeriodOfReport",
        direction="backward",
    )
    return merged[["AvailableDate", "EpsTtm", "NetIncomeGrowthYoYFiled"]].rename(
        columns={"NetIncomeGrowthYoYFiled": "EarningsGrowthYoY"}
    ).dropna(subset=["AvailableDate"]).sort_values("AvailableDate")


def _attach_pe_peg(ticker_panel: pd.DataFrame, filing_features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    eps_series = _trailing_twelve_month_eps(filing_features, ticker)
    result = ticker_panel.sort_values("Date").copy()
    if eps_series.empty:
        result["PE"] = np.nan
        result["PEG"] = np.nan
        return result
    result = pd.merge_asof(
        result, eps_series, left_on="Date", right_on="AvailableDate", direction="backward"
    )
    result["PE"] = (result["Close"] / result["EpsTtm"]).where(result["EpsTtm"] > 0)
    result["PEG"] = (result["PE"] / (result["EarningsGrowthYoY"] * 100)).where(
        result["EarningsGrowthYoY"] > 0
    )
    return result.drop(columns=["AvailableDate_y", "EpsTtm", "EarningsGrowthYoY"], errors="ignore").rename(
        columns={"AvailableDate_x": "AvailableDate"}
    )


def get_ticker_history(
    paths: ProjectPaths,
    state: "dashboard_state.DashboardState",
    ticker: str,
    *,
    start: str,
    end: str,
) -> dict[str, object]:
    members = _members(paths, state)
    cross_section_size = min(8, max(2, len(members) // 2))
    warmup_start = str((pd.Timestamp(start) - pd.Timedelta(days=WARMUP_BUFFER_DAYS)).date())
    train_end = str((pd.Timestamp(start) - pd.Timedelta(days=1)).date())
    settings = _settings(warmup_start, train_end, start, end, cross_section_size)
    filing_path = _ensure_filing_data(paths, state)
    filing_features = pd.read_csv(filing_path)
    factored = _build_factored_panel(members, settings, filing_features, warmup_start=warmup_start)

    ticker_panel = factored.loc[
        factored["Ticker"].eq(ticker) & factored["Date"].between(start, end)
    ].sort_values("Date")
    ticker_panel = _attach_pe_peg(ticker_panel, filing_features, ticker)
    columns = [
        "Date",
        "Close",
        "PE",
        "PEG",
        "MomentumFactor",
        "TrendFactor",
        "GrowthFactor",
        "QualityFactor",
        "RiskControlFactor",
        "FilingDurabilityFactor",
        "FilingBalanceSheetFactor",
        "FilingDisclosureSafetyFactor",
        "FilingFundamentalFactor",
        "FiledRevenueGrowthYoY",
        "FiledNetIncomeGrowthYoY",
        "FiledEbitdaGrowthYoY",
        "FiledEbitdaMargin",
        "FiledGrossMargin",
        "FiledOperatingMargin",
        "FiledFreeCashFlowMargin",
        "FiledShareGrowthYoY",
        "FiledResearchAndDevelopmentToRevenue",
        "FiledStockCompensationToRevenue",
        "FiledNetDebtToAssets",
        "FiledDebtToEquity",
        "FiledInterestCoverage",
        "FilingCoverageCount",
        "AvailableDate",
        "Form",
    ]
    available = [c for c in columns if c in ticker_panel.columns]
    # Filing rows repeat between filings (forward-filled); keep only the
    # signal-week cadence to avoid an overwhelming daily series.
    weekly = signal_day_panel(ticker_panel, start, end, settings.rebalance_weekday)
    records = weekly[available].to_dict(orient="records")
    return {"ticker": ticker, "history": [{k: _clean(v) for k, v in r.items()} for r in records]}
