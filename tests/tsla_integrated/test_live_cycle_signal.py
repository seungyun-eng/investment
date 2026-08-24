from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tesla_v2.features import compute_price_technical_features
from stock_research.tsla_integrated.live_cycle_signal import (
    LiveCycleAccount,
    LiveCyclePolicy,
    apply_filled_order,
    build_live_technical_features,
    recommend_next_session,
)


def _latest(**overrides: float | bool | str) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": "2026-08-07",
        "AdjClose": 100.0,
        "Return60D": -0.20,
        "RSI14": 35.0,
        "DistanceFromSMA200": -0.15,
        "PriceDrawdown63": -0.25,
        "TechnicalReentrySignal": True,
    }
    row.update(overrides)
    return row


def test_fresh_account_recommends_validated_25_percent_initial_position() -> None:
    result = recommend_next_session(
        _latest(),
        LiveCycleAccount(cash=100_000.0),
        LiveCyclePolicy(),
        sessions_since_last_buy=None,
    )
    assert result["Action"] == "BUY_INITIAL_NEXT_OPEN"
    assert result["OrderEquityFraction"] == pytest.approx(0.25)
    assert result["OrderNotionalAtReferenceClose"] == pytest.approx(25_000.0)
    assert result["EstimatedPositionAfterFraction"] == pytest.approx(0.25)


def test_existing_position_scales_35_percent_after_two_new_drawdown_levels() -> None:
    account = LiveCycleAccount(
        cash=75.0,
        shares=0.25,
        average_cost=100.0,
        cycle_id=1,
        cycle_anchor=100.0,
        cycle_start_shares=0.25,
        cycle_buy_notional=25.0,
        cycle_buy_shares=0.25,
        collecting_cycle_buys=True,
        last_buy_date="2026-07-20",
    )
    result = recommend_next_session(
        _latest(), account, LiveCyclePolicy(), sessions_since_last_buy=10
    )
    assert result["Action"] == "BUY_SCALE_NEXT_OPEN"
    assert result["ScaleIn"]["EligibleNewDrawdownLevels"] == [-0.10, -0.20]
    assert result["OrderEquityFraction"] == pytest.approx(0.35)
    assert result["EstimatedPositionAfterFraction"] == pytest.approx(0.60)


def test_scale_in_waits_for_same_ten_session_rule_as_backtest() -> None:
    account = LiveCycleAccount(
        cash=75.0,
        shares=0.25,
        cycle_anchor=100.0,
        cycle_start_shares=0.25,
        last_buy_date="2026-08-03",
    )
    result = recommend_next_session(
        _latest(), account, LiveCyclePolicy(), sessions_since_last_buy=4
    )
    assert result["Action"] == "HOLD_BUY_SPACING"
    assert result["OrderEquityFraction"] == 0.0


def test_existing_position_without_last_buy_date_is_blocked() -> None:
    account = LiveCycleAccount(
        cash=12_000.0,
        shares=26.0,
        average_cost=316.36,
        cycle_id=1,
        cycle_anchor=316.36,
        cycle_start_shares=26.0,
    )
    result = recommend_next_session(
        _latest(), account, LiveCyclePolicy(), sessions_since_last_buy=None
    )
    assert result["Action"] == "HOLD_MISSING_LAST_BUY_DATE"
    assert result["OrderEquityFraction"] == 0.0


def test_profit_instruction_sells_half_of_cycle_start_shares_not_remaining() -> None:
    account = LiveCycleAccount(
        cash=80.0,
        shares=0.60,
        cycle_anchor=100.0,
        cycle_start_shares=1.0,
    )
    result = recommend_next_session(
        _latest(AdjClose=160.0, TechnicalReentrySignal=False, PriceDrawdown63=0.0),
        account,
        LiveCyclePolicy(),
        sessions_since_last_buy=30,
    )
    assert result["ProfitTaking"]["NextTargetPrice"] == pytest.approx(170.0)
    assert result["ProfitTaking"]["ConditionalSellShares"] == pytest.approx(0.50)


def test_live_technical_calculation_matches_shared_tesla_v2_features() -> None:
    dates = pd.bdate_range("2024-01-02", periods=260)
    close = pd.Series([100.0 + index * 0.1 + (index % 9) for index in range(260)])
    prices = pd.DataFrame(
        {
            "Date": dates,
            "AdjClose": close,
            "Close": close,
            "Open": close,
            "Volume": 1_000_000.0,
        }
    )
    spy = pd.DataFrame({"Date": dates, "AdjClose": close * 0.9})
    shared = compute_price_technical_features(prices, spy)
    live = build_live_technical_features(prices)
    for column in ("Return60D", "RSI14", "DistanceFromSMA200"):
        assert live.iloc[-1][column] == pytest.approx(shared.iloc[-1][column])


def test_recorded_fills_preserve_cycle_anchor_and_rearm_after_profit_sale() -> None:
    account = apply_filled_order(
        LiveCycleAccount(cash=100_000.0),
        side="BUY",
        fill_price=100.0,
        fill_shares=250.0,
        fill_date="2026-08-10",
    )
    assert account.cash == pytest.approx(75_000.0)
    assert account.cycle_anchor == pytest.approx(100.0)
    assert account.cycle_start_shares == pytest.approx(250.0)

    account = apply_filled_order(
        account,
        side="SELL",
        fill_price=170.0,
        fill_shares=125.0,
        fill_date="2026-12-01",
    )
    assert account.rearm_pending is True
    assert account.fired_profit_target is True

    account = apply_filled_order(
        account,
        side="BUY",
        fill_price=120.0,
        fill_shares=100.0,
        fill_date="2027-02-01",
        fired_drawdown_levels=[-0.10],
    )
    assert account.cycle_id == 2
    assert account.cycle_anchor == pytest.approx(120.0)
    assert account.cycle_start_shares == pytest.approx(225.0)
    assert account.fired_profit_target is False
    assert account.fired_drawdowns == [-0.10]
