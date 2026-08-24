from __future__ import annotations

"""Live TSLA recommendation: V7.3 (TSLA-only, frozen Candidate 342) + SPY V2
tactical-defense overlay -- the "symmetric" combination validated all session
in research/tsla_unlevered_v2_overlay_v1/. Replaces the cycle-profit
(SINGLE_70_SELL_50_TECHNICAL) rule as the app's live TSLA signal after
head-to-head validation found V7.3+V2 wins ~69% of 59 monthly-cohort 12-month
windows with a consistently smaller drawdown in every single comparison run,
and no reliable regime signal (RSI/distance-from-SMA200) exists to justify
switching between the two strategies.

Unlike the cycle strategy this is a plain target-WEIGHT rule with no path-
dependent state (no cycle anchor, no fired-drawdown bookkeeping) -- today's
target is simply sign(State) * V2Exposure, compared against whatever TSLA
position is actually currently held.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stock_research.io_utils import read_csv_fallback
from stock_research.macro_momentum_sp500.tactical_defense import evaluate_defense
from stock_research.tsla_integrated.config import load_config
from stock_research.tsla_integrated.data import (
    load_macro_predictions,
    load_tsla_financials,
    load_tsla_prices,
)
from stock_research.tsla_integrated.features import build_integrated_features
from stock_research.tsla_integrated.optimization import params_from_candidate_row
from stock_research.tsla_integrated.portfolio import run_integrated_backtest
from stock_research.tsla_integrated.strategy import generate_integrated_signals
from stock_research.paths import ProjectPaths

STRATEGY_NAME = "V73_TSLA_PLUS_V2"
CANDIDATE_ID = 342
# A move smaller than this is not worth a rebalance order at reference close.
MINIMUM_REBALANCE_FRACTION = 0.05


@dataclass(frozen=True)
class LiveV73V2Position:
    cash: float
    shares: float = 0.0
    average_cost: float | None = None


def compute_today_signal(paths: ProjectPaths) -> dict[str, Any]:
    """Today's State (LONG/SHORT/CASH) from the frozen V7.3 signal plus
    today's SPY V2 exposure -- exactly the same construction validated in
    run_cycle_vs_v2_start_year_cohorts.py::build_v73_history, just taking
    the latest row instead of looping per calendar year.

    Uses the dashboard's daily-refreshed Dashboard Data/Prices/TSLA.csv,
    not the legacy Processed Data/Tesla_지표포함.csv -- that file is only
    ever updated by manually clicking a button in RUN_PROJECT_FIXED.ipynb
    (no automation refreshes it), so it silently went 12+ business days
    stale and blocked every live recommendation behind the freshness gate.
    build_integrated_features computes its own SMA/RSI/MACD from raw
    OHLCV, so the legacy file's pre-computed indicator columns were never
    actually needed -- load_equity_prices reads any CSV in the same
    positional Date,Close,Open,High,Low,Volume layout, which is exactly
    how data_collection.download_price_history writes the dashboard file."""
    price_path = paths.stock_root / "Dashboard Data" / "Prices" / "TSLA.csv"
    financial_path = paths.financial_raw / "TSLA_financials_Q.xlsx"
    macro_path = paths.results / "SP500" / "macro_momentum_sp500" / "oos_predictions_20260725_143133_110892.csv"
    # Same staleness problem as the legacy TSLA price file: Macro
    # Data/SPY Adjusted Historical Data.csv is a manually-refreshed feed
    # that goes stale silently. Dashboard Data/Benchmarks/SPY_adjusted.csv
    # is the same dividend-adjusted series the live "매크로경보" (V2
    # tactical-defense) screen already uses, and it IS kept current by the
    # dashboard's own daily refresh (see dashboard/macro_alert.py).
    spy_path = paths.stock_root / "Dashboard Data" / "Benchmarks" / "SPY_adjusted.csv"
    filing_path = paths.repo_root / "artifacts" / "tesla_v4_1_historical_validation" / "tsla_filing_metrics_full.csv"
    panel_path = paths.repo_root / "artifacts" / "tesla_v4_1_historical_validation" / "tesla_v4_1_panel.csv"
    _, settings = load_config(paths.repo_root / "config" / "tsla_integrated" / "research.json")

    panel = read_csv_fallback(panel_path, parse_dates=["Date"])
    extra_macro = panel[["Date", "HighYieldSpread", "YieldCurve10Y2Y"]].rename(
        columns={"HighYieldSpread": "HY_Spread", "YieldCurve10Y2Y": "YieldCurve"}
    )
    features = build_integrated_features(
        load_tsla_prices(price_path),
        load_tsla_financials(financial_path),
        load_macro_predictions(macro_path),
        financial_release_lag_days=settings.financial_release_lag_days,
        filing_features=read_csv_fallback(filing_path),
        ticker="TSLA",
        extra_macro=extra_macro,
    )
    candidates = read_csv_fallback(
        paths.repo_root / "artifacts" / "tsla_annual_roi_floor" / "unlevered_candidate_annual_metrics.csv"
    )
    params = params_from_candidate_row(candidates.loc[candidates["CandidateID"].eq(CANDIDATE_ID)].iloc[0])
    signals = generate_integrated_signals(features, params)
    result = run_integrated_backtest(
        signals,
        params,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        slippage_bps=settings.slippage_bps,
        annual_short_borrow_bps=settings.annual_short_borrow_bps,
    )
    latest = result.daily.iloc[-1]

    spy = read_csv_fallback(spy_path, parse_dates=["Date"])[["Date", "Close"]]
    defense = evaluate_defense(spy)
    v2 = pd.DataFrame({"Date": pd.to_datetime(defense.dates), "V2Exposure": defense.exposure})
    v2_row = v2.loc[v2["Date"] <= latest["Date"]].iloc[-1]

    state = str(latest["State"])
    direction = 1.0 if state == "LONG" else -1.0 if state == "SHORT" else 0.0
    v2_exposure = float(v2_row["V2Exposure"])
    signal_row = signals.loc[signals["Date"].eq(latest["Date"])].iloc[-1]
    explanation = _explain_state(state, signal_row, params)
    return {
        "Date": pd.Timestamp(latest["Date"]),
        "Close": float(latest["Close"]),
        "BaseState": state,
        "V2Exposure": v2_exposure,
        "V2AsOf": pd.Timestamp(v2_row["Date"]).date().isoformat(),
        "TargetWeight": direction * v2_exposure,
        "Explanation": explanation,
    }


def _explain_state(state: str, row: "pd.Series[Any]", params: Any) -> dict[str, Any]:
    """Plain-language + numeric breakdown of why V7.3's base state is what
    it is today: composite score vs the buy/sell/short thresholds, and
    which of the technical component's two drivers (RSI regime, 50-day
    trend direction) is pulling it down or up. Display-only -- does not
    change the decision itself, just explains the one already made."""
    composite = float(row["CompositeScore"])
    technical = float(row["TechnicalScore"])
    rsi = float(row["RSI14"])
    trend50 = float(row["Trend50"])
    macd = float(row["MACD"])
    macd_signal = float(row["MACDSignal"])
    overbought = rsi >= params.rsi_overbought
    oversold = rsi <= params.rsi_oversold
    trend_up = trend50 > 0
    macd_bullish = macd > macd_signal

    notes: list[str] = []
    if overbought:
        notes.append(f"RSI {rsi:.1f}로 과매수 구간(≥{params.rsi_overbought:.1f})이라 기술점수가 깎였습니다")
    elif oversold:
        notes.append(f"RSI {rsi:.1f}로 과매도 구간(≤{params.rsi_oversold:.1f})이라 기술점수가 올라갔습니다")
    notes.append(f"50일 추세가 {'플러스' if trend_up else '마이너스'}({trend50:+.1%})입니다")
    notes.append(f"MACD가 시그널선보다 {'위' if macd_bullish else '아래'}에 있습니다")

    if state == "LONG":
        headline = f"종합점수 {composite:.3f}가 매수 임계값 {params.buy_threshold:.3f}을 넘어서 LONG입니다."
    elif state == "SHORT":
        headline = f"종합점수 {composite:.3f}가 공매도 임계값 {params.short_threshold:.3f} 아래로 떨어져 SHORT입니다."
    else:
        headline = (
            f"종합점수 {composite:.3f}가 매도 임계값 {params.sell_threshold:.3f}보다 낮아 "
            f"(공매도 임계값 {params.short_threshold:.3f}까지는 아니라서) CASH입니다."
        )

    return {
        "Headline": headline,
        "Notes": notes,
        "CompositeScore": composite,
        "BuyThreshold": float(params.buy_threshold),
        "SellThreshold": float(params.sell_threshold),
        "ShortThreshold": float(params.short_threshold),
        "TechnicalScore": technical,
        "RSI14": rsi,
        "RSIOverbought": float(params.rsi_overbought),
        "RSIOversold": float(params.rsi_oversold),
        "Trend50": trend50,
        "MACDBullish": macd_bullish,
    }


def recommend_next_session(signal: dict[str, Any], position: LiveV73V2Position) -> dict[str, Any]:
    """Compare the currently-held TSLA position against today's target
    weight and return a next-open plan, without placing an order."""
    close = float(signal["Close"])
    equity = position.cash + position.shares * close
    if equity <= 0:
        raise ValueError("Account equity must be positive.")
    current_weight = position.shares * close / equity
    target_weight = float(signal["TargetWeight"])
    delta = target_weight - current_weight

    action = "HOLD"
    side: str | None = None
    reason = "Current position is already within tolerance of today's target weight."
    if abs(delta) < MINIMUM_REBALANCE_FRACTION:
        pass
    elif delta > 0:
        action = "BUY_TO_TARGET_NEXT_OPEN"
        side = "BUY"
        reason = (
            f"Target weight {target_weight:+.0%} (State={signal['BaseState']}, "
            f"V2 exposure {signal['V2Exposure']:.0%}) exceeds current weight "
            f"{current_weight:+.0%}."
        )
    else:
        action = "SELL_TO_TARGET_NEXT_OPEN" if target_weight >= 0 else "SELL_SHORT_TO_TARGET_NEXT_OPEN"
        side = "SELL"
        reason = (
            f"Target weight {target_weight:+.0%} (State={signal['BaseState']}, "
            f"V2 exposure {signal['V2Exposure']:.0%}) is below current weight "
            f"{current_weight:+.0%}."
        )

    target_notional = equity * target_weight
    delta_notional = equity * delta
    return {
        "Strategy": STRATEGY_NAME,
        "SignalAsOfClose": pd.Timestamp(signal["Date"]).date().isoformat(),
        "ReferenceClose": close,
        "Action": action,
        "OrderSide": side,
        "OrderEquityFraction": abs(delta),
        "OrderNotionalAtReferenceClose": abs(delta_notional),
        "EstimatedSharesAtReferenceClose": abs(delta_notional) / close if close > 0 else 0.0,
        "PositionBeforeFraction": current_weight,
        "TargetWeight": target_weight,
        "TargetNotionalAtReferenceClose": target_notional,
        "Reason": reason,
        "Composition": {
            "BaseState": signal["BaseState"],
            "V2Exposure": signal["V2Exposure"],
            "V2AsOf": signal["V2AsOf"],
        },
        "Explanation": signal.get("Explanation"),
    }
