from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .config_v2 import (
    DRAWDOWN_PROFILES,
    TARGET_PROFILES,
    MacroSp500V2Params,
    MacroSp500V2Settings,
)


@dataclass(frozen=True)
class CrisisMemory:
    state: str = "NORMAL"
    maximum_stage: int = 0
    holding_days: int = 0
    exit_streak: int = 0


def _drawdown_stage(drawdown: float, profile: str) -> int:
    stage = 0
    for level, threshold in enumerate(DRAWDOWN_PROFILES[profile], start=1):
        if drawdown <= threshold:
            stage = level
    return stage


def generate_v2_target_weights(
    features: pd.DataFrame,
    params: MacroSp500V2Params,
    settings: MacroSp500V2Settings,
    *,
    initial_memory: CrisisMemory | None = None,
) -> tuple[pd.DataFrame, CrisisMemory]:
    required = {
        "Date",
        "Close",
        "VixPercentile",
        "Drawdown",
        "SMA20",
        "SMA200",
        "Rebound20",
        "VixOffPeak20",
        "FeaturesReady",
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"V2 strategy features are missing: {sorted(missing)}")
    frame = features.sort_values("Date").reset_index(drop=True)
    memory = initial_memory or CrisisMemory()
    stage_targets = tuple(
        max(params.core_weight, target)
        for target in TARGET_PROFILES[params.target_profile]
    )

    dates = frame["Date"].to_numpy()
    close_values = frame["Close"].to_numpy(dtype=float)
    percentile_values = frame["VixPercentile"].to_numpy(dtype=float)
    drawdown_values = frame["Drawdown"].to_numpy(dtype=float)
    sma20_values = frame["SMA20"].to_numpy(dtype=float)
    sma200_values = frame["SMA200"].to_numpy(dtype=float)
    rebound_values = frame["Rebound20"].to_numpy(dtype=float)
    vix_off_peak_values = frame["VixOffPeak20"].to_numpy(dtype=float)
    ready_values = frame["FeaturesReady"].to_numpy(dtype=bool)
    states: list[str] = []
    stages: list[int] = []
    targets: list[float] = []
    reasons: list[str] = []
    reversals: list[bool] = []

    for index in range(len(frame)):
        ready = bool(ready_values[index])
        drawdown_stage = (
            _drawdown_stage(drawdown_values[index], params.drawdown_profile)
            if ready
            else 0
        )
        vix_stress = (
            ready and percentile_values[index] >= params.vix_entry_quantile
        )
        reversal = bool(
            ready
            and rebound_values[index] >= params.rebound_threshold
            and (
                close_values[index] >= sma20_values[index]
                or vix_off_peak_values[index] <= -settings.vix_decline_from_peak
            )
        )
        target = params.core_weight
        stage = memory.maximum_stage
        reason = "WARMUP_CORE" if not ready else "NORMAL_CORE"

        if ready and memory.state == "NORMAL":
            should_enter = drawdown_stage >= 2 or (
                drawdown_stage >= 1 and vix_stress
            )
            if should_enter:
                stage = min(drawdown_stage, 2)
                if drawdown_stage >= 3 and reversal:
                    stage = 3
                memory = CrisisMemory(
                    state="CRISIS",
                    maximum_stage=stage,
                    holding_days=0,
                    exit_streak=0,
                )
                target = stage_targets[stage - 1]
                reason = f"CRISIS_ENTER_STAGE_{stage}"

        elif ready and memory.state == "CRISIS":
            stage = memory.maximum_stage
            if drawdown_stage >= 2:
                stage = max(stage, 2)
            if drawdown_stage >= 3 and reversal:
                stage = 3
            holding_days = memory.holding_days + 1
            exit_eligible = (
                holding_days >= params.minimum_hold_days
                and percentile_values[index] <= settings.exit_vix_quantile
                and close_values[index] >= sma200_values[index]
            )
            exit_streak = memory.exit_streak + 1 if exit_eligible else 0
            if exit_streak >= settings.exit_confirmation_days:
                memory = CrisisMemory(state="RECOVERY")
                target = params.core_weight
                reason = f"RECOVERY_EXIT_HOLD_{holding_days}"
                stage = 0
            else:
                memory = replace(
                    memory,
                    maximum_stage=stage,
                    holding_days=holding_days,
                    exit_streak=exit_streak,
                )
                target = stage_targets[max(1, stage) - 1]
                reason = f"CRISIS_ACTIVE_STAGE_{stage}"

        elif ready and memory.state == "RECOVERY":
            memory = CrisisMemory()
            target = params.core_weight
            stage = 0
            reason = "NORMAL_AFTER_RECOVERY"

        states.append(memory.state)
        stages.append(stage)
        targets.append(target)
        reasons.append(reason)
        reversals.append(reversal)

    signals = pd.DataFrame(
        {
            "Date": dates,
            "State": states,
            "StressLevel": stages,
            "TargetWeight": targets,
            "CoreWeight": params.core_weight,
            "RebalanceBand": params.rebalance_band,
            "Reason": reasons,
            "ReversalConfirmed": reversals,
            "VixPercentile": percentile_values,
            "Drawdown": drawdown_values,
        }
    )
    return signals, memory
