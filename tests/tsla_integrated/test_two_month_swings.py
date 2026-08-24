from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tsla_integrated.two_month_swings import (
    build_two_month_windows,
    cluster_qualifying_start_dates,
    merge_overlapping_windows,
)


def test_calendar_month_window_uses_first_session_on_or_after_target() -> None:
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-05", "2024-03-04", "2024-03-06", "2024-05-06"]
            ),
            "AdjClose": [100.0, 140.0, 160.0, 200.0],
        }
    )
    windows = build_two_month_windows(prices, start_date="2024-01-01")
    first = windows.iloc[0]
    assert first["TargetDate"] == pd.Timestamp("2024-03-05")
    assert first["EndDate"] == pd.Timestamp("2024-03-06")
    assert first["ReturnPct"] == pytest.approx(60.0)


def test_overlapping_qualifying_windows_merge_into_one_episode() -> None:
    windows = pd.DataFrame(
        {
            "StartDate": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-05-01"]),
            "EndDate": pd.to_datetime(["2024-03-01", "2024-03-15", "2024-07-01"]),
            "StartPrice": [100.0, 90.0, 100.0],
            "EndPrice": [150.0, 180.0, 160.0],
            "ReturnPct": [50.0, 100.0, 60.0],
        }
    )
    episodes = merge_overlapping_windows(windows, direction="up")
    assert len(episodes) == 2
    assert episodes.iloc[0]["QualifyingWindows"] == 2
    assert episodes.iloc[0]["BestReturnPct"] == 100.0


def test_start_date_clusters_keep_separated_bursts_inside_broad_episode() -> None:
    windows = pd.DataFrame(
        {
            "StartDate": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-20"]),
            "EndDate": pd.to_datetime(["2024-03-01", "2024-03-05", "2024-03-20"]),
            "StartPrice": [100.0, 100.0, 100.0],
            "EndPrice": [150.0, 160.0, 170.0],
            "ReturnPct": [50.0, 60.0, 70.0],
        }
    )
    clusters = cluster_qualifying_start_dates(windows, direction="up")
    assert len(clusters) == 2
    assert clusters.iloc[0]["QualifyingWindows"] == 2
