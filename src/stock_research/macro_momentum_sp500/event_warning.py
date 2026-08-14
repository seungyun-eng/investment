from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownEvent:
    peak_index: int
    trough_index: int
    recovery_index: int | None
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    drawdown: float


@dataclass(frozen=True)
class WarningRule:
    vulnerability_mode: str
    macro_threshold: float
    breadth_threshold: int
    vulnerability_lookback: int
    trigger_z: float
    momentum_5_threshold: float
    minimum_trigger_components: int
    trigger_confirmation_days: int
    maximum_pre_alert_drawdown: float
    confirmation_window: int = 5
    episode_merge_gap: int = 10
    cooldown_days: int = 42

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AlertEpisode:
    start_index: int
    end_index: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def detect_drawdown_events(
    frame: pd.DataFrame,
    threshold: float = -0.15,
) -> list[DrawdownEvent]:
    """Return disjoint all-time-high-to-recovery drawdown cycles.

    A cycle becomes an event only after it breaches ``threshold``. Its peak is
    the last all-time high before the breach and its trough is the lowest close
    before recovery to that peak. An unrecovered final cycle is retained.
    """

    if not -1.0 < threshold < 0.0:
        raise ValueError("threshold must be between -1 and 0.")
    required = {"Date", "Close"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Drawdown data is missing columns: {sorted(missing)}")

    dates = pd.to_datetime(frame["Date"], errors="coerce").reset_index(drop=True)
    close = pd.to_numeric(frame["Close"], errors="coerce").reset_index(drop=True)
    valid = dates.notna() & close.notna()
    if not valid.all():
        dates = dates[valid].reset_index(drop=True)
        close = close[valid].reset_index(drop=True)
    if close.empty:
        return []

    peak_index = 0
    peak_value = float(close.iloc[0])
    trough_index = 0
    trough_value = peak_value
    in_event = False
    events: list[DrawdownEvent] = []

    def append_event(recovery_index: int | None) -> None:
        events.append(
            DrawdownEvent(
                peak_index=peak_index,
                trough_index=trough_index,
                recovery_index=recovery_index,
                peak_date=dates.iloc[peak_index],
                trough_date=dates.iloc[trough_index],
                recovery_date=(
                    None if recovery_index is None else dates.iloc[recovery_index]
                ),
                drawdown=trough_value / peak_value - 1.0,
            )
        )

    for index in range(1, len(close)):
        value = float(close.iloc[index])
        if not in_event:
            if value >= peak_value:
                peak_index = index
                peak_value = value
            elif value / peak_value - 1.0 <= threshold:
                in_event = True
                trough_index = index
                trough_value = value
            continue

        if value < trough_value:
            trough_index = index
            trough_value = value
        if value >= peak_value:
            append_event(index)
            peak_index = index
            peak_value = value
            trough_index = index
            trough_value = value
            in_event = False

    if in_event:
        append_event(None)
    return events


def warning_rule_candidates() -> Iterable[WarningRule]:
    """Yield a bounded, predeclared grid for conservative two-stage warnings."""

    common = product(
        (21, 42, 63),
        (1.0, 1.5, 2.0),
        (-0.01, -0.02),
        (1, 2, 3),
        (1, 2, 3),
        (-0.01, -0.03, -0.05),
    )
    common_values = list(common)
    for mode in ("macro", "breadth", "both"):
        macro_values = (0.55, 0.60, 0.65) if mode != "breadth" else (0.60,)
        breadth_values = (2, 3) if mode != "macro" else (2,)
        for macro_threshold, breadth_threshold, values in product(
            macro_values,
            breadth_values,
            common_values,
        ):
            yield WarningRule(
                vulnerability_mode=mode,
                macro_threshold=macro_threshold,
                breadth_threshold=breadth_threshold,
                vulnerability_lookback=values[0],
                trigger_z=values[1],
                momentum_5_threshold=values[2],
                minimum_trigger_components=values[3],
                trigger_confirmation_days=values[4],
                maximum_pre_alert_drawdown=values[5],
            )


def _row_max(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    available = [name for name in columns if name in frame]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[available].apply(pd.to_numeric, errors="coerce").max(axis=1)


def build_two_stage_signal(
    features: pd.DataFrame,
    rule: WarningRule,
) -> pd.DataFrame:
    """Build the exact trailing-only signal used by search and simulation."""

    required = {
        "Date",
        "MacroConfirmationScore",
        "EarlyWarningBreadth",
        "Momentum_5",
        "Momentum_21",
        "Drawdown252",
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Warning features are missing columns: {sorted(missing)}")
    if rule.vulnerability_mode not in {"macro", "breadth", "both"}:
        raise ValueError(f"Unknown vulnerability mode: {rule.vulnerability_mode}")

    result = pd.DataFrame(
        {"Date": pd.to_datetime(features["Date"], errors="coerce")},
        index=features.index,
    )
    macro = pd.to_numeric(
        features["MacroConfirmationScore"], errors="coerce"
    ).ge(rule.macro_threshold)
    breadth = pd.to_numeric(
        features["EarlyWarningBreadth"], errors="coerce"
    ).ge(rule.breadth_threshold)
    if rule.vulnerability_mode == "macro":
        vulnerability_event = macro
    elif rule.vulnerability_mode == "breadth":
        vulnerability_event = breadth
    else:
        vulnerability_event = macro & breadth
    result["Vulnerability"] = (
        vulnerability_event.astype(int)
        .rolling(rule.vulnerability_lookback, min_periods=rule.vulnerability_lookback)
        .max()
        .fillna(0)
        .astype(bool)
    )

    result["CreditTriggerZ"] = _row_max(
        features,
        (
            "HYOAS_Change21_Z252",
            "HYYield_Change21_Z252",
            "BAA10Y_Change21_Z252",
            "NFCI_Change21_Z252",
        ),
    )
    result["VolatilityTriggerZ"] = _row_max(
        features,
        ("VIX_Change21_Z252", "VIX_TermStructure_Change21_Z252"),
    )
    momentum_5 = pd.to_numeric(features["Momentum_5"], errors="coerce")
    momentum_21 = pd.to_numeric(features["Momentum_21"], errors="coerce")
    trigger_components = pd.concat(
        [
            result["CreditTriggerZ"].ge(rule.trigger_z),
            result["VolatilityTriggerZ"].ge(rule.trigger_z),
            momentum_5.le(rule.momentum_5_threshold),
            momentum_21.le(rule.momentum_5_threshold * 2.0),
        ],
        axis=1,
    )
    result["TriggerCount"] = trigger_components.sum(axis=1)
    raw_trigger = result["TriggerCount"].ge(rule.minimum_trigger_components)
    result["ConfirmedTrigger"] = (
        raw_trigger.astype(int)
        .rolling(rule.confirmation_window, min_periods=rule.confirmation_window)
        .sum()
        .ge(rule.trigger_confirmation_days)
    )
    result["NearPeak"] = pd.to_numeric(
        features["Drawdown252"], errors="coerce"
    ).ge(rule.maximum_pre_alert_drawdown)
    result["Alert"] = (
        result["Vulnerability"] & result["ConfirmedTrigger"] & result["NearPeak"]
    )
    return result.reset_index(drop=True)


def collapse_alert_episodes(
    signal: pd.DataFrame,
    merge_gap: int,
    cooldown_days: int,
) -> list[AlertEpisode]:
    dates = pd.to_datetime(signal["Date"], errors="coerce").reset_index(drop=True)
    active_indices = np.flatnonzero(signal["Alert"].fillna(False).to_numpy(dtype=bool))
    if not len(active_indices):
        return []

    raw: list[tuple[int, int]] = []
    start = int(active_indices[0])
    end = start
    for value in active_indices[1:]:
        index = int(value)
        if index - end <= merge_gap:
            end = index
        else:
            raw.append((start, end))
            start = index
            end = index
    raw.append((start, end))

    episodes: list[AlertEpisode] = []
    next_allowed = -1
    for start, end in raw:
        if start < next_allowed:
            continue
        episodes.append(
            AlertEpisode(
                start_index=start,
                end_index=end,
                start_date=dates.iloc[start],
                end_date=dates.iloc[end],
            )
        )
        next_allowed = start + cooldown_days
    return episodes


def evaluate_alert_episodes(
    episodes: list[AlertEpisode],
    events: list[DrawdownEvent],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    minimum_lead: int = 7,
    maximum_lead: int = 63,
) -> dict[str, object]:
    """Evaluate one alert onset per episode against pre-peak event windows."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    period_events = [event for event in events if start <= event.peak_date <= end]
    period_episodes = [episode for episode in episodes if start <= episode.start_date <= end]
    event_matches: dict[int, list[AlertEpisode]] = {index: [] for index in range(len(period_events))}
    episode_match: dict[pd.Timestamp, int] = {}
    for episode in period_episodes:
        for event_index, event in enumerate(period_events):
            lead = event.peak_index - episode.start_index
            if minimum_lead <= lead <= maximum_lead:
                event_matches[event_index].append(episode)
                episode_match[episode.start_date] = event_index
                break

    captured = sum(bool(matches) for matches in event_matches.values())
    true_positive_episodes = len(episode_match)
    false_positive_episodes = len(period_episodes) - true_positive_episodes
    precision = (
        true_positive_episodes / len(period_episodes) if period_episodes else None
    )
    recall = captured / len(period_events) if period_events else None
    event_details = []
    for index, event in enumerate(period_events):
        matches = event_matches[index]
        event_details.append(
            {
                "peak_date": event.peak_date.strftime("%Y-%m-%d"),
                "trough_date": event.trough_date.strftime("%Y-%m-%d"),
                "drawdown_percent": event.drawdown * 100.0,
                "captured": bool(matches),
                "alert_dates": [item.start_date.strftime("%Y-%m-%d") for item in matches],
                "lead_sessions": [event.peak_index - item.start_index for item in matches],
            }
        )
    false_positive_dates = [
        episode.start_date.strftime("%Y-%m-%d")
        for episode in period_episodes
        if episode.start_date not in episode_match
    ]
    return {
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "events": len(period_events),
        "captured_events": captured,
        "alert_episodes": len(period_episodes),
        "true_positive_episodes": true_positive_episodes,
        "false_positive_episodes": false_positive_episodes,
        "precision_percent": None if precision is None else precision * 100.0,
        "recall_percent": None if recall is None else recall * 100.0,
        "event_details": event_details,
        "false_positive_dates": false_positive_dates,
    }
