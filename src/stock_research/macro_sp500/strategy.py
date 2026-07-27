from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import MacroSp500Params, MacroSp500Settings


@dataclass
class _StrategyMemory:
    state: str = "NORMAL"
    armed_index: int | None = None
    panic_index: int | None = None
    maximum_stress_level: int = 0
    exit_streak: int = 0


def _stress_level(
    vix_percentile: float,
    drawdown: float,
    params: MacroSp500Params,
    settings: MacroSp500Settings,
) -> int:
    second_quantile = min(
        params.vix_entry_quantile + settings.stage_2_quantile_step,
        0.975,
    )
    vix_level = 0
    if vix_percentile >= settings.stage_3_quantile:
        vix_level = 3
    elif vix_percentile >= second_quantile:
        vix_level = 2
    elif vix_percentile >= params.vix_entry_quantile:
        vix_level = 1

    drawdown_level = 0
    for level, threshold in enumerate(settings.drawdown_levels, start=1):
        if drawdown <= threshold:
            drawdown_level = level
    return max(vix_level, drawdown_level)


def generate_target_weights(
    features: pd.DataFrame,
    params: MacroSp500Params,
    settings: MacroSp500Settings,
) -> pd.DataFrame:
    required = {
        "Date",
        "VixPercentile",
        "Drawdown",
        "WarningScore",
        "FeaturesReady",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Strategy features are missing columns: {sorted(missing)}")
    frame = features.copy().sort_values("Date").reset_index(drop=True)
    memory = _StrategyMemory()

    warning_target = min(1.0, params.core_weight + params.warning_addition)
    stage_targets = tuple(
        max(params.core_weight, target) for target in settings.stage_target_weights
    )

    dates = frame["Date"].to_numpy()
    ready_values = frame["FeaturesReady"].to_numpy(dtype=bool)
    percentile_values = frame["VixPercentile"].to_numpy(dtype=float)
    drawdown_values = frame["Drawdown"].to_numpy(dtype=float)
    warning_values = frame["WarningScore"].to_numpy(dtype=int)
    states: list[str] = []
    stress_levels: list[int] = []
    targets: list[float] = []
    reasons: list[str] = []

    for index in range(len(frame)):
        ready = bool(ready_values[index])
        percentile = percentile_values[index] if ready else float("nan")
        drawdown = drawdown_values[index] if ready else float("nan")
        warning_score = int(warning_values[index])
        level = (
            _stress_level(percentile, drawdown, params, settings)
            if ready
            else 0
        )
        target = params.core_weight
        reason = "WARMUP_CORE" if not ready else "NORMAL_CORE"

        if ready and memory.state == "NORMAL":
            if level:
                memory.state = "PANIC"
                memory.panic_index = index
                memory.maximum_stress_level = level
                memory.exit_streak = 0
                target = stage_targets[level - 1]
                reason = f"PANIC_ENTER_LEVEL_{level}"
            elif warning_score >= params.warning_score_min:
                memory.state = "ARMED"
                memory.armed_index = index
                target = warning_target
                reason = f"WARNING_SCORE_{warning_score}"

        elif ready and memory.state == "ARMED":
            if level:
                memory.state = "PANIC"
                memory.panic_index = index
                memory.maximum_stress_level = level
                memory.exit_streak = 0
                target = stage_targets[level - 1]
                reason = f"PANIC_CONFIRMED_LEVEL_{level}"
            elif (
                memory.armed_index is not None
                and index - memory.armed_index >= settings.panic_confirmation_window
            ):
                memory.state = "NORMAL"
                memory.armed_index = None
                target = params.core_weight
                reason = "WARNING_EXPIRED"
            else:
                target = warning_target
                reason = f"WARNING_ACTIVE_SCORE_{warning_score}"

        elif ready and memory.state == "PANIC":
            memory.maximum_stress_level = max(memory.maximum_stress_level, level)
            target = stage_targets[max(1, memory.maximum_stress_level) - 1]
            holding_days = (
                index - memory.panic_index if memory.panic_index is not None else 0
            )
            exit_eligible = (
                holding_days >= params.minimum_hold_days
                and percentile <= params.exit_vix_quantile
            )
            memory.exit_streak = memory.exit_streak + 1 if exit_eligible else 0
            if memory.exit_streak >= settings.exit_confirmation_days:
                memory.state = "RECOVERY"
                target = params.core_weight
                reason = (
                    f"RECOVERY_EXIT_Q_{percentile:.3f}_"
                    f"HOLD_{holding_days}"
                )
            elif level > 0:
                reason = f"PANIC_ACTIVE_LEVEL_{memory.maximum_stress_level}"
            else:
                reason = (
                    f"PANIC_HOLD_Q_{percentile:.3f}_"
                    f"HOLD_{holding_days}"
                )

        elif ready and memory.state == "RECOVERY":
            memory = _StrategyMemory()
            target = params.core_weight
            reason = "NORMAL_AFTER_RECOVERY"

        states.append(memory.state)
        stress_levels.append(level)
        targets.append(target)
        reasons.append(reason)
    return pd.DataFrame(
        {
            "Date": dates,
            "State": states,
            "StressLevel": stress_levels,
            "TargetWeight": targets,
            "Reason": reasons,
            "WarningScore": warning_values,
            "VixPercentile": percentile_values,
            "Drawdown": drawdown_values,
        }
    )
