"""Tests for the dashboard macro-alert card."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_research.dashboard import macro_alert

WARMUP_DAYS = 200
PICKS = [
    {"ticker": "AAPL", "target_weight": 0.2},
    {"ticker": "MSFT", "target_weight": 0.2},
    {"ticker": "NVDA", "target_weight": 0.2},
    {"ticker": "META", "target_weight": 0.2},
    {"ticker": "AVGO", "target_weight": 0.2},
]


def _frame(tail: list[float]) -> pd.DataFrame:
    closes = [100.0] * WARMUP_DAYS + list(tail)
    return pd.DataFrame(
        {"Date": pd.bdate_range("2020-01-01", periods=len(closes)), "Close": closes}
    )


def _card(tail: list[float], picks=PICKS) -> dict:
    return macro_alert.build_macro_alert(
        paths=None, top_picks=picks, download=False, frame=_frame(tail)
    )


def test_quiet_market_produces_no_alert_and_no_actions():
    card = _card([100.0] * 10)
    assert card["alert"] is None
    assert card["actions"] == []
    assert card["phase"] == "INVESTED"
    assert card["exposure"] == 1.0


def test_quiet_card_reports_how_close_the_liquidation_trigger_is():
    watch = _card([100.0] * 8 + [99.0])["watch"]
    assert watch["waiting_for"] == "LIQUIDATE"
    assert watch["days_required"] == 3
    assert watch["days_remaining"] == 3
    assert watch["consecutive_days_below"] == 0


def test_liquidation_day_emits_a_critical_alert_and_sells_every_holding():
    card = _card([99.0, 99.0, 99.0, 99.0])
    alert = card["alert"]
    assert alert is not None
    assert alert["kind"] == "LIQUIDATE"
    assert alert["severity"] == "critical"
    assert alert["exposure_from"] == 1.0
    assert alert["exposure_to"] == 0.0

    equities = [row for row in card["actions"] if row["ticker"] != "CASH"]
    assert len(equities) == len(PICKS)
    assert {row["action"] for row in equities} == {"SELL_ALL"}
    assert all(row["weight_to"] == 0 for row in equities)

    cash = next(row for row in card["actions"] if row["ticker"] == "CASH")
    assert cash["action"] == "RAISE"
    assert cash["weight_to"] == 1.0


def test_defensive_card_explains_the_re_entry_conditions():
    card = _card([99.0, 99.0, 99.0, 99.0])
    watch = card["watch"]
    assert card["phase"] == "DEFENSIVE"
    assert watch["waiting_for"] == "RESTORE"
    assert watch["sell_date"] is not None
    # A shallow sell keeps the quick-recovery window open with a price to beat.
    assert watch["quick_recovery_days_left"] is not None
    assert watch["quick_recovery_reclaim_price"] is not None


def test_scale_in_produces_partial_weights_rather_than_full_reinvestment():
    decline = [100.0 - 1.6 * step for step in range(1, 30)]
    card = macro_alert.build_macro_alert(
        paths=None, top_picks=PICKS, download=False, frame=_frame(decline)
    )
    # Find the day the rule first stepped back in and rebuild the card there.
    scale_in = next(row for row in card["history"] if row["kind"] == "SCALE_IN")
    assert 0 < scale_in["exposure_to"] < 1.0


def test_actions_scale_model_weights_by_exposure():
    rows = macro_alert._actions(PICKS, exposure=0.3, previous_exposure=0.0)
    equities = [row for row in rows if row["ticker"] != "CASH"]
    assert {row["action"] for row in equities} == {"BUY"}
    assert all(row["weight_to"] == pytest.approx(0.06) for row in equities)
    cash = next(row for row in rows if row["ticker"] == "CASH")
    assert cash["action"] == "DEPLOY"
    assert cash["weight_to"] == pytest.approx(0.7)


def test_unchanged_exposure_yields_no_actions():
    assert macro_alert._actions(PICKS, exposure=0.8, previous_exposure=0.8) == []


def test_history_is_newest_first_and_bounded():
    decline = [100.0 - 1.6 * step for step in range(1, 30)]
    card = macro_alert.build_macro_alert(
        paths=None, top_picks=PICKS, download=False, frame=_frame(decline + [200.0] * 8)
    )
    dates = [row["date"] for row in card["history"]]
    assert dates == sorted(dates, reverse=True)
    assert len(card["history"]) <= macro_alert.HISTORY_LIMIT


def test_chart_covers_the_full_history_not_just_a_recent_window():
    card = _card([100.0] * 10)
    assert len(card["chart"]) == WARMUP_DAYS + 10
    assert card["chart"][0]["moving_average"] is None  # still warming up
    latest = card["chart"][-1]
    assert latest["date"] == card["as_of"]
    assert latest["close"] == card["spy_close"]
    assert latest["moving_average"] is not None


def test_chart_matches_watch_on_the_latest_row():
    card = _card([100.0] * 8 + [99.0])
    latest = card["chart"][-1]
    assert latest["moving_average"] == card["watch"]["moving_average"]
    assert latest["rsi"] == card["watch"]["rsi"]


def test_card_records_the_frozen_parameters_it_ran_with():
    parameters = _card([100.0] * 5)["parameters"]
    assert parameters["moving_average_window"] == 150
    assert parameters["sell_confirm"] == 3
    assert parameters["restore_confirm"] == 2
    assert parameters["quick_recovery_depth"] == -0.05
    assert parameters["quick_recovery_window"] == 5
    assert [tier["exposure"] for tier in parameters["rsi_tiers"]] == [0.30, 0.80]
