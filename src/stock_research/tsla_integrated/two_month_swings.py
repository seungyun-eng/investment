from __future__ import annotations

import numpy as np
import pandas as pd


def build_two_month_windows(
    prices: pd.DataFrame,
    *,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    months: int = 2,
) -> pd.DataFrame:
    """Build every close-to-close calendar-month forward return window.

    The end observation is the first trading session on or after the exact
    calendar-month anniversary.  Adjusted prices must be supplied by callers.
    """

    if months < 1:
        raise ValueError("months must be positive.")
    required = {"Date", "AdjClose"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    frame = prices.loc[:, ["Date", "AdjClose"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["AdjClose"] = pd.to_numeric(frame["AdjClose"], errors="coerce")
    frame = frame.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
    frame = frame.loc[frame["Date"].ge(start_date)]
    if end_date is not None:
        frame = frame.loc[frame["Date"].le(end_date)]
    frame = frame.reset_index(drop=True)
    dates = frame["Date"].to_numpy(dtype="datetime64[ns]")
    targets = (frame["Date"] + pd.DateOffset(months=months)).to_numpy(
        dtype="datetime64[ns]"
    )
    end_indices = np.searchsorted(dates, targets, side="left")
    valid = end_indices < len(frame)
    end_indices = end_indices[valid]
    result = pd.DataFrame(
        {
            "StartDate": frame.loc[valid, "Date"].to_numpy(),
            "TargetDate": targets[valid],
            "EndDate": frame.iloc[end_indices]["Date"].to_numpy(),
            "StartPrice": frame.loc[valid, "AdjClose"].to_numpy(dtype=float),
            "EndPrice": frame.iloc[end_indices]["AdjClose"].to_numpy(dtype=float),
        }
    )
    result["ReturnPct"] = (
        result["EndPrice"] / result["StartPrice"] - 1.0
    ) * 100.0
    result["CalendarDays"] = (result["EndDate"] - result["StartDate"]).dt.days
    return result


def merge_overlapping_windows(
    qualifying_windows: pd.DataFrame,
    *,
    direction: str,
) -> pd.DataFrame:
    """Merge overlapping qualifying windows into distinct market episodes."""

    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'.")
    if qualifying_windows.empty:
        return pd.DataFrame(
            columns=[
                "EpisodeStart",
                "EpisodeEnd",
                "BestStart",
                "BestEnd",
                "StartPrice",
                "EndPrice",
                "BestReturnPct",
                "QualifyingWindows",
            ]
        )
    windows = qualifying_windows.sort_values("StartDate").reset_index(drop=True)
    groups: list[pd.DataFrame] = []
    start_index = 0
    episode_end = pd.Timestamp(windows.loc[0, "EndDate"])
    for index in range(1, len(windows)):
        start = pd.Timestamp(windows.loc[index, "StartDate"])
        if start > episode_end:
            groups.append(windows.iloc[start_index:index].copy())
            start_index = index
            episode_end = pd.Timestamp(windows.loc[index, "EndDate"])
        else:
            episode_end = max(
                episode_end, pd.Timestamp(windows.loc[index, "EndDate"])
            )
    groups.append(windows.iloc[start_index:].copy())

    rows: list[dict[str, object]] = []
    for episode_id, group in enumerate(groups, start=1):
        best_index = (
            group["ReturnPct"].idxmax()
            if direction == "up"
            else group["ReturnPct"].idxmin()
        )
        best = group.loc[best_index]
        rows.append(
            {
                "EpisodeID": episode_id,
                "EpisodeStart": group["StartDate"].min(),
                "EpisodeEnd": group["EndDate"].max(),
                "BestStart": best["StartDate"],
                "BestEnd": best["EndDate"],
                "StartPrice": best["StartPrice"],
                "EndPrice": best["EndPrice"],
                "BestReturnPct": best["ReturnPct"],
                "QualifyingWindows": len(group),
            }
        )
    return pd.DataFrame(rows)


def cluster_qualifying_start_dates(
    qualifying_windows: pd.DataFrame,
    *,
    direction: str,
    maximum_gap_days: int = 7,
) -> pd.DataFrame:
    """Create a finer inventory by gaps between qualifying start dates."""

    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'.")
    if maximum_gap_days < 1:
        raise ValueError("maximum_gap_days must be positive.")
    if qualifying_windows.empty:
        return merge_overlapping_windows(qualifying_windows, direction=direction)
    windows = qualifying_windows.sort_values("StartDate").reset_index(drop=True)
    group_id = windows["StartDate"].diff().dt.days.gt(maximum_gap_days).cumsum()
    rows: list[dict[str, object]] = []
    for cluster_id, (_, group) in enumerate(windows.groupby(group_id), start=1):
        best_index = (
            group["ReturnPct"].idxmax()
            if direction == "up"
            else group["ReturnPct"].idxmin()
        )
        best = group.loc[best_index]
        rows.append(
            {
                "ClusterID": cluster_id,
                "QualifyingStartFirst": group["StartDate"].min(),
                "QualifyingStartLast": group["StartDate"].max(),
                "LatestWindowEnd": group["EndDate"].max(),
                "BestStart": best["StartDate"],
                "BestEnd": best["EndDate"],
                "StartPrice": best["StartPrice"],
                "EndPrice": best["EndPrice"],
                "BestReturnPct": best["ReturnPct"],
                "QualifyingWindows": len(group),
            }
        )
    return pd.DataFrame(rows)
