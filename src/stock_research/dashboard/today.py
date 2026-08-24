from __future__ import annotations

"""One payload for the mobile app: Top-K rotation plus specialized signals."""

import json
import os
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.dashboard import data_collection, engine, macro_alert
from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard import tsla_signal_ledger
from stock_research.io_utils import read_csv_fallback
from stock_research.paths import ProjectPaths
from stock_research.tsla_integrated import live_v73_v2_signal
from stock_research.tsla_integrated.live_cycle_signal import business_days_after
from stock_research.tsla_integrated.live_v73_v2_signal import LiveV73V2Position

DEFAULT_RESEARCH_START = "2024-01-02"
RESULTS_FOLDER = "mobile_investment_app"
LATEST_FILENAME = "latest_today.json"
_RUN_LOCK = threading.Lock()


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=path.stem + "_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def latest_path(paths: ProjectPaths) -> Path:
    return paths.results / RESULTS_FOLDER / LATEST_FILENAME


def load_latest(paths: ProjectPaths) -> dict[str, Any] | None:
    path = latest_path(paths)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _member_price_path(
    paths: ProjectPaths,
    state: dashboard_state.DashboardState,
    ticker: str,
) -> Path:
    entry = next(
        (item for item in state.tickers if item.ticker.upper() == ticker.upper()), None
    )
    if entry is None:
        raise ValueError(f"{ticker} is not in the dashboard universe")
    return paths.stock_root / entry.price_path


def _load_tsla_position(paths: ProjectPaths) -> tuple[LiveV73V2Position, str, str | None]:
    """Reuses the same persisted account file the prior cycle-strategy
    signal wrote (cash/shares/average_cost reflect what is ACTUALLY held,
    independent of which strategy justified the purchase). The cycle-
    specific fields (cycle_anchor, fired_drawdowns, ...) are simply not
    read -- V7.3+V2 is a plain target-weight rule with no path-dependent
    state of its own."""
    account_path = paths.results / "tsla_cycle_live_signal" / "account_state.json"
    if not account_path.exists():
        return LiveV73V2Position(cash=1.0), "FRESH_PERCENT_ONLY", None
    raw = json.loads(account_path.read_text(encoding="utf-8-sig"))
    position = LiveV73V2Position(
        cash=float(raw.get("cash", 0.0)),
        shares=float(raw.get("shares", 0.0)),
        average_cost=raw.get("average_cost"),
    )
    return position, "PERSISTED_LIVE_ACCOUNT", raw.get("last_buy_date")


def build_tsla_card(
    paths: ProjectPaths,
    state: dashboard_state.DashboardState,
    *,
    maximum_missing_business_days: int = 1,
) -> dict[str, Any]:
    signal = live_v73_v2_signal.compute_today_signal(paths)
    position, account_mode, last_buy_date = _load_tsla_position(paths)
    recommendation = live_v73_v2_signal.recommend_next_session(signal, position)

    missing_days = business_days_after(signal["Date"], date.today())
    fresh = missing_days <= maximum_missing_business_days
    if not fresh and recommendation["OrderSide"] is not None:
        recommendation["UnderlyingActionBeforeFreshnessBlock"] = recommendation["Action"]
        recommendation["Action"] = "NO_ACTION_STALE_DATA"
        recommendation["OrderSide"] = None
        recommendation["OrderEquityFraction"] = 0.0
        recommendation["OrderNotionalAtReferenceClose"] = 0.0
        recommendation["EstimatedSharesAtReferenceClose"] = 0.0

    close = float(signal["Close"])
    market_value = position.shares * close
    equity = position.cash + market_value
    return {
        "ticker": "TSLA",
        "strategy": live_v73_v2_signal.STRATEGY_NAME,
        "account_mode": account_mode,
        "data_fresh": fresh,
        "missing_business_days": missing_days,
        "position": {
            "cash": position.cash,
            "shares": position.shares,
            "average_cost": position.average_cost,
            "market_value": market_value,
            "equity": equity,
            "weight": market_value / equity if equity > 0 else 0.0,
            "unrealized_return": (
                close / position.average_cost - 1.0
                if position.average_cost is not None and position.average_cost > 0
                else None
            ),
            "last_buy_date": last_buy_date,
        },
        "recommendation": recommendation,
    }


def _preserve_published_tsla_card(paths: ProjectPaths, error: Exception) -> dict[str, Any]:
    """Keep the independent TSLA panel from blocking rotation publication.

    The TSLA V7.3+V2 card depends on a research result CSV that deliberately
    is not committed. A clean cloud runner therefore cannot recompute that
    card yet. Preserve the last deployed card, mark it unavailable/stale, and
    block any order while the weekly cross-sectional model continues.
    """

    candidates = [
        latest_path(paths),
        paths.repo_root / "alpha-desk-cloud" / "public" / "data" / LATEST_FILENAME,
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            prior = json.loads(candidate.read_text(encoding="utf-8-sig"))
            card = prior.get("tsla")
            if not isinstance(card, dict) or not isinstance(card.get("recommendation"), dict):
                continue
            # JSON round-trip gives us a deep copy without sharing the imported
            # build-time payload object.
            preserved = json.loads(json.dumps(card, ensure_ascii=False))
            preserved["data_fresh"] = False
            preserved["refresh_error"] = str(error)
            recommendation = preserved["recommendation"]
            if recommendation.get("OrderSide") is not None:
                recommendation["UnderlyingActionBeforeFreshnessBlock"] = recommendation.get("Action")
            recommendation["Action"] = "NO_ACTION_TSLA_DATA_UNAVAILABLE"
            recommendation["OrderSide"] = None
            recommendation["OrderEquityFraction"] = 0.0
            recommendation["OrderNotionalAtReferenceClose"] = 0.0
            recommendation["EstimatedSharesAtReferenceClose"] = 0.0
            return preserved
        except (OSError, json.JSONDecodeError):
            continue
    raise error


def compose_today_payload(
    cross_result: dict[str, Any],
    tsla_card: dict[str, Any],
    *,
    refresh: dict[str, Any] | None,
    macro_alert_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = cross_result["meta"]
    latest_signal_date = str(meta["latest_signal_date"])
    ranking = list(cross_result.get("top15_latest", []))
    top_picks = [
        row
        for row in ranking
        if row.get("target_weight") is not None and float(row["target_weight"]) > 0
    ]
    top_picks.sort(key=lambda row: row.get("rank") or 10_000)
    latest_trades = [
        row for row in cross_result.get("trades", []) if str(row.get("date")) == latest_signal_date
    ]
    specialized_action = tsla_card["recommendation"]["Action"]
    # A macro alert is a same-day action on the whole model portfolio, so it
    # counts once regardless of how many tickers it touches.
    macro_alert_fired = bool(macro_alert_card and macro_alert_card.get("alert"))
    action_count = (
        len(latest_trades)
        + int(
            specialized_action
            not in {
                "HOLD",
                "HOLD_BUY_SPACING",
                "HOLD_MISSING_LAST_BUY_DATE",
                "NO_ACTION_STALE_DATA",
                "NO_ACTION_TSLA_DATA_UNAVAILABLE",
            }
        )
        + int(macro_alert_fired)
    )
    return {
        "generated_on": date.today().isoformat(),
        "research_only_no_broker_order": True,
        "market_as_of": latest_signal_date,
        "top_k": int(meta["top_k"]),
        "universe_count": len(meta["tickers"]),
        "action_count": action_count,
        "refresh": refresh,
        "top_picks": top_picks,
        "rotation_actions": latest_trades,
        "ranking": ranking,
        "tsla": tsla_card,
        "macro_alert": macro_alert_card,
        "backtest_summary": cross_result.get("summary", {}),
        "ticker_summary": cross_result.get("ticker_summary", []),
        "execution_policy": {
            "name": meta.get("execution_policy"),
            "signal_frequency": meta.get("signal_frequency"),
            "weight_reset_frequency": meta.get("weight_reset_frequency"),
            "immediate_membership_changes": meta.get(
                "immediate_membership_changes"
            ),
            "latest_signal_execution_reason": meta.get(
                "latest_signal_execution_reason"
            ),
        },
        "model_notes": {
            "rotation": (
                "Filing V7 Top-K score: Technical + point-in-time SEC factors. "
                "Selection and risk exits are checked weekly; membership changes "
                "execute immediately and unchanged holdings reset to equal weight "
                "on the first signal of each month."
            ),
            "specialized": (
                "TSLA-only V7.3 (frozen Candidate 342) combined with the SPY V2 "
                "tactical-defense exposure overlay -- a target-weight rule, not "
                "the prior cycle-anchored profit-taking rule. Adopted after "
                "head-to-head validation found V7.3+V2 beat the cycle rule in "
                "~69% of 59 monthly-cohort 12-month windows with a consistently "
                "smaller drawdown, and no reliable regime signal existed to "
                "justify switching between the two. Intentionally separate from "
                "the cross-sectional rotation rank."
            ),
            "macro_alert": (
                "SPY tactical defense (V2, tuned) scales the rotation portfolio's "
                "exposure between 0/30/80/100%. It is evaluated daily on the "
                "previous close, unlike the weekly rotation signal, and it never "
                "rotates into SPY."
            ),
        },
    }


def run_today(
    paths: ProjectPaths,
    *,
    top_k: int | None = None,
    refresh_prices: bool = False,
    refresh_filings: bool = False,
    research_start: str = DEFAULT_RESEARCH_START,
) -> dict[str, Any]:
    """Refresh requested data, run shared engines, cache one mobile payload."""

    with _RUN_LOCK:
        state = dashboard_state.load_state(paths)
        if top_k is not None:
            if top_k not in {3, 5}:
                raise ValueError("The mobile app supports top_k 3 or 5.")
            if state.top_k != top_k:
                state = dashboard_state.set_top_k(paths, top_k)

        refresh: dict[str, Any] = {}
        if refresh_prices:
            refresh["prices"] = data_collection.refresh_all_prices(paths, state)
            state = dashboard_state.load_state(paths)
        if refresh_filings:
            artifacts = data_collection.sync_filings_for_dashboard(
                paths, state, refresh_metadata=True
            )
            refresh["filings"] = {
                "status": "ok",
                "point_in_time_features_csv": str(artifacts.point_in_time_features_csv),
            }

        cross_result = engine.run_backtest(
            paths,
            state,
            start=research_start,
            end=date.today().isoformat(),
            top_k=state.top_k,
        )
        try:
            tsla_card = build_tsla_card(paths, state)
            tsla_signal_ledger.append_signal(paths, tsla_card)
        except FileNotFoundError as error:
            tsla_card = _preserve_published_tsla_card(paths, error)
        top_picks = [
            row
            for row in cross_result.get("top15_latest", [])
            if row.get("target_weight") is not None and float(row["target_weight"]) > 0
        ]
        # The macro alert must never take the whole payload down: without it the
        # rotation view is still correct, it just loses the overlay.
        try:
            macro_alert_card = macro_alert.build_macro_alert(
                paths, top_picks, download=True
            )
        except Exception as error:  # noqa: BLE001 - surfaced to the UI, not swallowed
            macro_alert_card = {
                "rule": macro_alert.RULE_ID,
                "status": "UNAVAILABLE",
                "error": str(error),
            }
        payload = compose_today_payload(
            cross_result,
            tsla_card,
            refresh=refresh or None,
            macro_alert_card=macro_alert_card,
        )
        _atomic_json(latest_path(paths), payload)
        return payload
