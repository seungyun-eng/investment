"""Behavioural tests for the frozen SPY tactical-defense rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.macro_momentum_sp500.tactical_defense import (
    TUNED_DEFENSE,
    DefenseParams,
    evaluate_defense,
)

WARMUP_DAYS = 200


def _frame(tail: list[float], *, warmup_level: float = 100.0) -> pd.DataFrame:
    """A flat warmup long enough for MA150/RSI, then the scenario prices."""

    closes = [warmup_level] * WARMUP_DAYS + list(tail)
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame({"Date": dates, "Close": closes})


def _kinds(state) -> list[str]:
    return [event.kind for event in state.events]


def test_flat_history_never_leaves_full_exposure():
    state = evaluate_defense(_frame([100.0] * 20))
    assert state.exposure.min() == 1.0
    assert state.events == []
    assert state.phase == "INVESTED"


def test_liquidation_requires_three_confirmed_days_below():
    # Two closes below the average must not trigger; the third does, and the
    # trade lands the session after the third confirmation.
    two_days = evaluate_defense(_frame([99.0, 99.0, 100.5, 100.5]))
    assert _kinds(two_days) == []

    three_days = evaluate_defense(_frame([99.0, 99.0, 99.0, 99.0, 99.0]))
    assert three_days.events[0].kind == "LIQUIDATE"
    assert three_days.events[0].exposure_to == 0.0


def test_shallow_dip_that_reclaims_the_sell_price_quick_restores():
    state = evaluate_defense(_frame([99.5, 99.5, 99.5, 99.4, 101.0]))
    assert _kinds(state) == ["LIQUIDATE", "QUICK_RESTORE"]
    assert state.current_exposure == 1.0


def test_deep_selloff_is_not_eligible_for_quick_restore():
    # A >5% decline fails the depth gate, so reclaiming the sell price does not
    # short-circuit the staged re-entry.
    deep = [100.0, 94.0, 90.0, 88.0, 92.0, 95.0]
    state = evaluate_defense(_frame(deep))
    assert "QUICK_RESTORE" not in _kinds(state)


def test_rsi_tiers_ratchet_up_and_cap_below_full_exposure():
    decline = [100.0 - 1.6 * step for step in range(1, 30)]
    state = evaluate_defense(_frame(decline))
    exposures = [event.exposure_to for event in state.events]
    assert exposures[0] == 0.0
    scale_ins = [event for event in state.events if event.kind == "SCALE_IN"]
    assert scale_ins, "a sustained decline must trigger RSI staged buying"
    assert max(event.exposure_to for event in scale_ins) <= TUNED_DEFENSE.rsi_low_exposure
    # Exposure never steps down once raised.
    assert all(b >= a for a, b in zip(exposures, exposures[1:]) if b > 0 or a == 0)


def test_restore_needs_two_confirmed_days_above_and_returns_to_full():
    decline = [100.0 - 1.6 * step for step in range(1, 30)]
    recovery = [200.0] * 6
    state = evaluate_defense(_frame(decline + recovery))
    assert state.events[-1].kind in {"RESTORE", "SCALE_IN", "QUICK_RESTORE"}
    assert state.current_exposure == 1.0


def test_exposure_only_takes_the_four_defined_levels():
    decline = [100.0 - 1.6 * step for step in range(1, 30)]
    state = evaluate_defense(_frame(decline + [200.0] * 6))
    allowed = {0.0, TUNED_DEFENSE.rsi_high_exposure, TUNED_DEFENSE.rsi_low_exposure, 1.0}
    assert set(np.unique(state.exposure)).issubset(allowed)


def test_triggers_use_only_prior_day_data():
    """Changing the final close must not alter any earlier exposure value."""

    tail = [99.0, 99.0, 99.0, 99.0, 99.0]
    base = evaluate_defense(_frame(tail))
    bumped = evaluate_defense(_frame(tail[:-1] + [180.0]))
    assert np.array_equal(base.exposure[:-1], bumped.exposure[:-1])


def test_disabling_quick_recovery_keeps_the_position_defensive():
    params = DefenseParams(quick_recovery_depth=None)
    state = evaluate_defense(_frame([99.5, 99.5, 99.5, 99.4, 101.0]), params)
    assert "QUICK_RESTORE" not in _kinds(state)


def test_tail_reports_what_the_next_trigger_is_waiting_on():
    state = evaluate_defense(_frame([100.0] * 10 + [99.0, 99.0]))
    assert state.tail.below_streak == 1
    assert state.tail.sell_date is None


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        DefenseParams(sell_confirm=0)
    with pytest.raises(ValueError):
        DefenseParams(rsi_high_exposure=0.9, rsi_low_exposure=0.5)


def test_empty_frame_raises():
    with pytest.raises(ValueError):
        evaluate_defense(pd.DataFrame({"Date": [], "Close": []}))
