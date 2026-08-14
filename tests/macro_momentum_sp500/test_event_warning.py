from __future__ import annotations

import pandas as pd

from stock_research.macro_momentum_sp500.event_warning import (
    AlertEpisode,
    WarningRule,
    build_two_stage_signal,
    collapse_alert_episodes,
    detect_drawdown_events,
    evaluate_alert_episodes,
)


def test_detect_drawdown_events_uses_peak_trough_and_recovery() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=9),
            "Close": [100, 105, 100, 89, 85, 90, 104, 106, 103],
        }
    )
    events = detect_drawdown_events(frame, threshold=-0.15)
    assert len(events) == 1
    assert events[0].peak_index == 1
    assert events[0].trough_index == 4
    assert events[0].recovery_index == 7
    assert events[0].drawdown == 85 / 105 - 1


def test_two_stage_signal_requires_vulnerability_and_confirmed_trigger() -> None:
    rows = 12
    features = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=rows),
            "MacroConfirmationScore": [0.4, 0.4, 0.7] + [0.4] * (rows - 3),
            "EarlyWarningBreadth": [0] * rows,
            "HYYield_Change21_Z252": [0.0] * 5 + [2.0] * 7,
            "VIX_Change21_Z252": [0.0] * rows,
            "Momentum_5": [0.0] * rows,
            "Momentum_21": [0.0] * rows,
            "Drawdown252": [0.0] * rows,
        }
    )
    rule = WarningRule(
        vulnerability_mode="macro",
        macro_threshold=0.6,
        breadth_threshold=2,
        vulnerability_lookback=5,
        trigger_z=1.5,
        momentum_5_threshold=-0.02,
        minimum_trigger_components=1,
        trigger_confirmation_days=2,
        maximum_pre_alert_drawdown=-0.03,
        confirmation_window=3,
    )
    result = build_two_stage_signal(features, rule)
    assert not result.loc[5, "Alert"]
    assert result.loc[6, "Alert"]
    assert not result.loc[10, "Alert"]


def test_episode_evaluation_counts_event_once_and_strict_false_positives() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=100),
            "Close": [100 + index for index in range(70)]
            + [160, 140, 130, 150, 161]
            + [161 + index for index in range(25)],
        }
    )
    events = detect_drawdown_events(frame, threshold=-0.15)
    assert len(events) == 1
    episodes = [
        AlertEpisode(10, 10, frame.loc[10, "Date"], frame.loc[10, "Date"]),
        AlertEpisode(50, 52, frame.loc[50, "Date"], frame.loc[52, "Date"]),
    ]
    result = evaluate_alert_episodes(
        episodes,
        events,
        frame["Date"].min(),
        frame["Date"].max(),
        minimum_lead=7,
        maximum_lead=30,
    )
    assert result["captured_events"] == 1
    assert result["true_positive_episodes"] == 1
    assert result["false_positive_episodes"] == 1


def test_collapse_alert_episodes_merges_short_gaps_and_applies_cooldown() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    alert = [False] * 20
    for index in (1, 2, 5, 12, 18):
        alert[index] = True
    episodes = collapse_alert_episodes(
        pd.DataFrame({"Date": dates, "Alert": alert}),
        merge_gap=3,
        cooldown_days=10,
    )
    assert [(item.start_index, item.end_index) for item in episodes] == [(1, 5), (12, 12)]
