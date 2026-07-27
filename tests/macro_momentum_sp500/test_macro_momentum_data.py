from __future__ import annotations

import pandas as pd

from stock_research.macro_momentum_sp500.config import ResearchConfig
from stock_research.macro_momentum_sp500.data import _fred_timing, merge_point_in_time


def test_monthly_series_is_not_visible_before_release_lag() -> None:
    calendar = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-31", "2024-02-14", "2024-02-15", "2024-03-01"])}
    )
    series = pd.DataFrame(
        {
            "ObservationDate": pd.to_datetime(["2024-01-01"]),
            "CPI": [100.0],
        }
    )

    merged = merge_point_in_time(
        calendar,
        series,
        value_column="CPI",
        lag_days=45,
    )

    assert merged.loc[:1, "CPI"].isna().all()
    assert merged.loc[2:, "CPI"].eq(100.0).all()
    assert merged.loc[2, "CPIObservationDate"] == pd.Timestamp("2024-01-01")


def test_asof_merge_never_uses_future_daily_observation() -> None:
    calendar = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])}
    )
    series = pd.DataFrame(
        {
            "ObservationDate": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "VIX": [15.0, 30.0],
        }
    )

    merged = merge_point_in_time(
        calendar,
        series,
        value_column="VIX",
        lag_days=0,
        tolerance_days=7,
    )

    assert merged["VIX"].tolist() == [15.0, 15.0, 30.0]


def test_mixed_fred_file_uses_frequency_specific_lags() -> None:
    config = ResearchConfig(monthly_release_lag_days=45, weekly_release_lag_days=7)

    assert _fred_timing("VIX3M", config) == (0, 10)
    assert _fred_timing("T10Y2Y", config) == (0, 10)
    assert _fred_timing("BAA10Y", config) == (0, 10)
    assert _fred_timing("NFCI", config) == (7, 14)
    assert _fred_timing("GS10", config) == (45, 120)
    assert _fred_timing("TB3MS", config) == (45, 120)
