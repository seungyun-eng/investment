from __future__ import annotations

import pandas as pd

from stock_research.macro_sp500.features import trailing_vix_percentile


def test_trailing_vix_percentile_does_not_look_at_current_or_future_values() -> None:
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=40))
    original = pd.Series([10.0 + index for index in range(40)])
    changed = original.copy()
    changed.iloc[31:] = 1_000.0

    original_percentile = trailing_vix_percentile(
        dates,
        original,
        lookback_years=3,
        minimum_observations=5,
    )
    changed_percentile = trailing_vix_percentile(
        dates,
        changed,
        lookback_years=3,
        minimum_observations=5,
    )

    pd.testing.assert_series_equal(
        original_percentile.iloc[:31],
        changed_percentile.iloc[:31],
    )
    assert original_percentile.iloc[5] == 1.0
