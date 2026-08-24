"""Macro alert card for the Alpha Desk dashboard.

Turns the frozen SPY tactical-defense rule (V2) into the payload the app shows
under its macro-alert category: the current exposure, whether today's evaluation
changed it, the concrete per-ticker action that follows, and the numbers the
next transition is waiting on.

The rule is recomputed from full SPY history on every run -- there is no stored
alert state to drift. `latest_today.json["macro_alert"]` is therefore a pure
function of the price file and the frozen parameters.

Scope: this de-risks the cross-sectional model portfolio only. It never routes
into SPY, and it does not touch the separate TSLA card.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests

from stock_research.dashboard.data_collection import YAHOO_CHART_URL
from stock_research.io_utils import atomic_to_csv
from stock_research.macro_momentum_sp500.tactical_defense import (
    TUNED_DEFENSE,
    DefenseParams,
    DefenseState,
    evaluate_defense,
)
from stock_research.paths import ProjectPaths

BENCHMARK_TICKER = "SPY"
RULE_ID = "SPY_TACTICAL_DEFENSE_V2_TUNED"
HISTORY_LIMIT = 12
#: Not a real cap -- large enough to cover the full SPY history the rule was
#: evaluated on, so the dashboard's zoomable chart gets every day, not just
#: a recent window. Mirrors the JS-side CHART_LIMIT in
#: alpha-desk-cloud/cloudflare-monitor/src/macro-alert.js.
CHART_LIMIT = 3000

#: Exposure change -> what the desk has to do that day.
EVENT_HEADLINES: dict[str, str] = {
    "LIQUIDATE": "매크로 방어 발동 · 모델 보유 전량 청산",
    "QUICK_RESTORE": "가짜신호 확인 · 100% 원복",
    "SCALE_IN": "단계 재매수",
    "RESTORE": "150일선 회복 확정 · 100% 원복",
}
EVENT_SEVERITY: dict[str, str] = {
    "LIQUIDATE": "critical",
    "SCALE_IN": "action",
    "QUICK_RESTORE": "action",
    "RESTORE": "action",
}


def spy_price_path(paths: ProjectPaths) -> Path:
    return paths.stock_root / "Dashboard Data" / "Benchmarks" / "SPY_adjusted.csv"


def refresh_spy_adjusted_close(destination: Path) -> dict[str, object]:
    """Download SPY *dividend-adjusted* daily closes.

    The shared `download_price_history` writes raw closes in a fixed positional
    column order that other loaders depend on, so this keeps its own file rather
    than adding a column there. Adjusted closes are not cosmetic here: the rule
    was validated on them, and an unadjusted series shifts RSI enough to flip a
    tier (observed on 2026-03-23, 30% vs 80%).
    """

    response = requests.get(
        YAHOO_CHART_URL.format(ticker=BENCHMARK_TICKER),
        params={
            "period1": 0,
            "period2": int(time.time()),
            "interval": "1d",
            "events": "div,splits",
        },
        headers={"User-Agent": "Mozilla/5.0 (macro alert price downloader)"},
        timeout=30,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance error for {BENCHMARK_TICKER}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("No SPY price data returned")
    result = results[0]
    timestamps = result.get("timestamp")
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps or not adjusted:
        raise ValueError("SPY response carried no adjusted closes")
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
            "Close": adjusted,
        }
    ).dropna()
    frame = frame.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(frame, destination, index=False)
    return {
        "rows": len(frame),
        "end_date": str(frame["Date"].max().date()) if not frame.empty else None,
        "path": str(destination),
    }


def load_spy_frame(paths: ProjectPaths, *, download: bool = True) -> pd.DataFrame:
    """Read the adjusted SPY series the rule was validated on."""

    destination = spy_price_path(paths)
    if download:
        refresh_spy_adjusted_close(destination)
    if not destination.exists():
        raise FileNotFoundError(
            f"SPY adjusted price file is missing: {destination}. "
            "Run with download=True once to create it."
        )
    frame = pd.read_csv(destination)
    return frame[["Date", "Close"]].dropna()


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), digits)


def _chart(state: DefenseState) -> list[dict[str, Any]]:
    """Daily {date, close, moving_average, rsi} series for the dashboard's
    zoomable price/RSI chart. Mirrors buildChart() in the JS-side
    alpha-desk-cloud/cloudflare-monitor/src/macro-alert.js."""

    start = max(len(state.dates) - CHART_LIMIT, 0)
    return [
        {
            "date": state.dates[index],
            "close": _round(state.close[index], 2),
            "moving_average": _round(state.moving_average[index], 2),
            "rsi": _round(state.rsi[index], 1),
        }
        for index in range(start, len(state.dates))
    ]


def _watch(state: DefenseState, params: DefenseParams) -> dict[str, Any]:
    """What the rule needs to see next, in numbers the user can check daily."""

    tail = state.tail
    close = float(state.close[-1])
    average = state.moving_average[-1]
    deviation = None if np.isnan(average) else float(close / average - 1)
    latest_rsi = None if np.isnan(state.rsi[-1]) else float(state.rsi[-1])

    if state.phase == "INVESTED":
        return {
            "waiting_for": "LIQUIDATE",
            "condition": (
                f"종가가 150일선 아래로 {params.sell_confirm}일 연속 확정되면 전량 청산"
            ),
            "consecutive_days_below": tail.below_streak,
            "days_required": params.sell_confirm,
            "days_remaining": max(params.sell_confirm - tail.below_streak, 0),
            "moving_average": _round(average, 2),
            "distance_to_moving_average": _round(deviation),
            "rsi": _round(latest_rsi, 1),
        }

    quick_recovery_days_left = None
    quick_recovery_price = None
    if (
        params.quick_recovery_depth is not None
        and not tail.rsi_bought
        and tail.days_since_sell is not None
        and tail.sell_depth is not None
        and tail.sell_depth > params.quick_recovery_depth
    ):
        quick_recovery_days_left = max(
            params.quick_recovery_window - tail.days_since_sell, 0
        )
        quick_recovery_price = tail.sell_price

    next_tier = None
    if state.current_exposure < params.rsi_high_exposure:
        next_tier = {"rsi_below": params.rsi_high, "exposure": params.rsi_high_exposure}
    elif state.current_exposure < params.rsi_low_exposure:
        next_tier = {"rsi_below": params.rsi_low, "exposure": params.rsi_low_exposure}

    return {
        "waiting_for": "RESTORE",
        "condition": (
            f"종가가 150일선 위로 {params.restore_confirm}일 연속 확정되면 100% 원복"
        ),
        "consecutive_days_above": tail.restore_streak,
        "days_required": params.restore_confirm,
        "days_remaining": max(params.restore_confirm - tail.restore_streak, 0),
        "moving_average": _round(average, 2),
        "distance_to_moving_average": _round(deviation),
        "rsi": _round(latest_rsi, 1),
        "next_rsi_tier": next_tier,
        "quick_recovery_days_left": quick_recovery_days_left,
        "quick_recovery_reclaim_price": _round(quick_recovery_price, 2),
        "sell_date": tail.sell_date,
    }


def _actions(
    top_picks: Sequence[dict[str, Any]],
    *,
    exposure: float,
    previous_exposure: float,
) -> list[dict[str, Any]]:
    """Per-ticker instruction implied by the exposure change.

    Model weights are scaled uniformly -- the overlay never picks which holding
    to drop, it only changes how much of the whole portfolio is held.
    """

    if exposure == previous_exposure:
        return []
    rows: list[dict[str, Any]] = []
    for pick in top_picks:
        weight = pick.get("target_weight")
        if weight is None:
            continue
        model_weight = float(weight)
        if model_weight <= 0:
            continue
        before = model_weight * previous_exposure
        after = model_weight * exposure
        if after == 0:
            action = "SELL_ALL"
        elif after < before:
            action = "TRIM"
        else:
            action = "BUY"
        rows.append(
            {
                "ticker": str(pick.get("ticker") or pick.get("Ticker") or "—"),
                "action": action,
                "model_weight": round(model_weight, 4),
                "weight_from": round(before, 4),
                "weight_to": round(after, 4),
            }
        )
    rows.append(
        {
            "ticker": "CASH",
            "action": "RAISE" if exposure < previous_exposure else "DEPLOY",
            "model_weight": None,
            "weight_from": round(1 - previous_exposure, 4),
            "weight_to": round(1 - exposure, 4),
        }
    )
    return rows


def build_macro_alert(
    paths: ProjectPaths,
    top_picks: Sequence[dict[str, Any]],
    *,
    params: DefenseParams = TUNED_DEFENSE,
    download: bool = True,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build `latest_today.json["macro_alert"]`."""

    prices = load_spy_frame(paths, download=download) if frame is None else frame
    state = evaluate_defense(prices, params)
    exposure = state.current_exposure
    today_event = state.event_on(state.as_of)
    previous_exposure = (
        today_event.exposure_from if today_event else exposure
    )

    alert: dict[str, Any] | None = None
    if today_event is not None:
        alert = {
            "kind": today_event.kind,
            "severity": EVENT_SEVERITY.get(today_event.kind, "action"),
            "headline": EVENT_HEADLINES.get(today_event.kind, today_event.kind),
            "date": today_event.date,
            "close": _round(today_event.close, 2),
            "rsi": _round(today_event.rsi, 1),
            "moving_average_deviation": _round(today_event.moving_average_deviation),
            "exposure_from": today_event.exposure_from,
            "exposure_to": today_event.exposure_to,
        }

    history = [
        {
            "date": event.date,
            "kind": event.kind,
            "close": _round(event.close, 2),
            "rsi": _round(event.rsi, 1),
            "moving_average_deviation": _round(event.moving_average_deviation),
            "exposure_from": event.exposure_from,
            "exposure_to": event.exposure_to,
        }
        for event in state.events[-HISTORY_LIMIT:]
    ]

    return {
        "rule": RULE_ID,
        "as_of": state.as_of,
        "phase": state.phase,
        "exposure": exposure,
        "previous_exposure": previous_exposure,
        "spy_close": _round(state.close[-1], 2),
        "alert": alert,
        "actions": _actions(
            top_picks, exposure=exposure, previous_exposure=previous_exposure
        ),
        "watch": _watch(state, params),
        "chart": _chart(state),
        "history": list(reversed(history)),
        "parameters": {
            "moving_average_window": params.moving_average_window,
            "sell_confirm": params.sell_confirm,
            "restore_confirm": params.restore_confirm,
            "rsi_tiers": [
                {"below": params.rsi_high, "exposure": params.rsi_high_exposure},
                {"below": params.rsi_low, "exposure": params.rsi_low_exposure},
            ],
            "quick_recovery_depth": params.quick_recovery_depth,
            "quick_recovery_window": params.quick_recovery_window,
        },
        "notes": (
            "V7.3 모델 포트폴리오의 노출을 줄이는 오버레이입니다. SPY로 갈아타는 "
            "규칙이 아니며, 검증 결과 그 방식은 기각되었습니다. 평가는 매일, "
            "전일 종가 기준으로 이루어지고 당일 실행합니다."
        ),
    }
