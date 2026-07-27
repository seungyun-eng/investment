from __future__ import annotations

import math

import pandas as pd

from .config import FearBuyParams


def _desired_target(
    row: object,
    params: FearBuyParams,
) -> tuple[float, str]:
    percentile = float(row.VixPercentile)
    fear_score = float(row.FearScore)
    drawdown = float(row.Drawdown252)
    if not math.isfinite(percentile) or not math.isfinite(fear_score):
        return params.core_weight, "WARMUP"
    if (
        percentile >= params.panic_vix_percentile
        and fear_score >= params.panic_fear_score
    ) or drawdown <= params.panic_drawdown:
        return 1.0, "PANIC"
    if (
        percentile >= params.fear_vix_percentile
        and fear_score >= params.fear_fear_score
    ) or drawdown <= params.fear_drawdown:
        return min(1.0, params.core_weight + 2 * params.tranche_weight), "FEAR"
    if (
        percentile >= params.mild_vix_percentile
        and fear_score >= params.mild_fear_score
    ) or drawdown <= params.mild_drawdown:
        return min(1.0, params.core_weight + params.tranche_weight), "MILD_FEAR"
    return params.core_weight, "NO_FEAR"


def _allocation_state(target: float, params: FearBuyParams) -> str:
    reserve_used = target - params.core_weight
    if target >= 1.0 - 1e-9:
        return "PANIC_ALLOCATED"
    if reserve_used >= 2 * params.tranche_weight - 1e-9:
        return "FEAR_ALLOCATED"
    if reserve_used >= params.tranche_weight - 1e-9:
        return "MILD_FEAR_ALLOCATED"
    return "CORE_ONLY"


def generate_fear_buy_signals(
    features: pd.DataFrame,
    params: FearBuyParams,
) -> pd.DataFrame:
    """Generate weekly contrarian tranches without using future observations."""

    required = {
        "Date",
        "Close",
        "VixPercentile",
        "FearScore",
        "EuphoriaScore",
        "Drawdown252",
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Fear-buy features are missing: {sorted(missing)}")
    frame = features.copy().sort_values("Date").reset_index(drop=True)
    weeks = frame["Date"].dt.to_period(params.decision_frequency)
    frame["DecisionDay"] = weeks.ne(weeks.shift(-1))

    target = params.core_weight
    tactical_reference_price = float("nan")
    last_buy_index = -1_000_000
    targets: list[float] = []
    desired_targets: list[float] = []
    trigger_levels: list[str] = []
    states: list[str] = []
    reasons: list[str] = []
    reference_prices: list[float] = []
    sessions_since_buy: list[int] = []

    for row in frame.itertuples(index=True):
        desired, trigger = _desired_target(row, params)
        transition_reason = ""
        if bool(row.DecisionDay) and desired > target + 1e-9:
            new_target = min(desired, target + params.tranche_weight)
            old_tactical_weight = max(0.0, target - params.core_weight)
            added_weight = new_target - target
            existing_cost = (
                0.0
                if not math.isfinite(tactical_reference_price)
                else tactical_reference_price * old_tactical_weight
            )
            tactical_reference_price = (
                existing_cost + float(row.Close) * added_weight
            ) / (old_tactical_weight + added_weight)
            target = new_target
            last_buy_index = int(row.Index)
            transition_reason = f"{trigger}_BUY_TRANCHE"
        elif bool(row.DecisionDay) and target > params.core_weight + 1e-9:
            held_long_enough = (
                int(row.Index) - last_buy_index >= params.minimum_hold_sessions
            )
            no_longer_fearful = (
                float(row.FearScore) <= params.trim_max_fear_score
                and float(row.VixPercentile) <= params.trim_max_vix_percentile
            )
            profitable = (
                math.isfinite(tactical_reference_price)
                and float(row.Close)
                >= tactical_reference_price * (1.0 + params.trim_profit_buffer)
            )
            euphoric = float(row.EuphoriaScore) >= params.trim_euphoria_score
            if held_long_enough and no_longer_fearful and profitable and euphoric:
                target = max(params.core_weight, target - params.tranche_weight)
                transition_reason = "EUPHORIA_PROFIT_TRIM"
                if target <= params.core_weight + 1e-9:
                    tactical_reference_price = float("nan")

        targets.append(target)
        desired_targets.append(desired)
        trigger_levels.append(trigger)
        states.append(_allocation_state(target, params))
        reasons.append(transition_reason)
        reference_prices.append(tactical_reference_price)
        sessions_since_buy.append(max(0, int(row.Index) - last_buy_index))

    frame["TargetWeight"] = targets
    frame["DesiredTargetWeight"] = desired_targets
    frame["TriggerLevel"] = trigger_levels
    frame["SignalState"] = states
    frame["TransitionReason"] = reasons
    frame["TacticalReferencePrice"] = reference_prices
    frame["SessionsSinceLastBuy"] = sessions_since_buy
    return frame
