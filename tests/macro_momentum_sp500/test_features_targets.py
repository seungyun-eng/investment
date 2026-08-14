from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.macro_momentum_sp500.config import ResearchConfig
from stock_research.macro_momentum_sp500.features import (
    _trailing_percentile,
    build_features,
)
from stock_research.macro_momentum_sp500.targets import build_targets


def _market_frame(rows: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = pd.Series(np.linspace(100, 160, rows))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.999,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, rows),
            "VIX": 20 + np.sin(np.arange(rows) / 15),
            "CashRate": 5.0,
        }
    )


def test_trailing_percentile_excludes_current_value() -> None:
    values = pd.Series([10.0, 20.0, 30.0, 100.0])
    percentile = _trailing_percentile(values, window=3, minimum=3)
    assert np.isnan(percentile.iloc[2])
    assert percentile.iloc[3] == 1.0


def test_future_changes_do_not_change_historical_features() -> None:
    config = ResearchConfig(distribution_windows=(252,))
    original = _market_frame()
    changed = original.copy()
    changed.loc[350:, "Close"] *= 3
    first = build_features(original, config)
    second = build_features(changed, config)

    pd.testing.assert_series_equal(
        first.loc[:349, "Momentum_63"],
        second.loc[:349, "Momentum_63"],
    )
    pd.testing.assert_series_equal(
        first.loc[:349, "SMA200_Ratio"],
        second.loc[:349, "SMA200_Ratio"],
    )


def test_macro_confirmation_is_bounded_and_uses_only_trailing_data() -> None:
    config = ResearchConfig(distribution_windows=(252,))
    original = _market_frame(500)
    for name, values in {
        "VIX3M": 22 + np.sin(np.arange(500) / 18),
        "NFCI": np.linspace(-0.5, 0.5, 500),
        "HYYield": np.linspace(4.0, 8.0, 500),
        "BAA10Y": np.linspace(1.5, 3.0, 500),
        "Unemployment": np.linspace(3.5, 5.5, 500),
        "T10Y3M": np.linspace(1.5, -0.5, 500),
        "T10Y2Y": np.linspace(1.0, -0.25, 500),
    }.items():
        original[name] = values
    changed = original.copy()
    changed.loc[450:, "NFCI"] = 20.0
    changed.loc[450:, "HYYield"] = 30.0

    first = build_features(original, config)
    second = build_features(changed, config)

    available = first["MacroConfirmationScore"].dropna()
    assert available.between(0, 1).all()
    pd.testing.assert_series_equal(
        first.loc[:449, "MacroConfirmationScore"],
        second.loc[:449, "MacroConfirmationScore"],
    )


def _macro_shock_frames(rows: int = 500, shock_start: int = 450) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A noisy (non-degenerate) baseline plus a variant with a sustained
    acceleration in claims AND high-yield borrowing costs from `shock_start`
    onward -- two distinct domains moving badly at once, unlike a one-off
    level jump which a 21-day change z-score shrugs off within a month."""

    rng = np.random.default_rng(7)
    noise = lambda scale: rng.normal(0, scale, rows)
    original = _market_frame(rows)
    original["InitialJoblessClaims"] = 210_000 + np.cumsum(noise(300))
    original["ContinuingJoblessClaims"] = 1_720_000 + np.cumsum(noise(2000))
    original["Treasury2Y"] = 4.2 + np.cumsum(noise(0.01))
    original["RealYield5Y"] = 1.6 + np.cumsum(noise(0.01))
    original["CoreCPI"] = 300 + np.cumsum(np.abs(noise(0.2)))
    original["CorePCE"] = 120 + np.cumsum(np.abs(noise(0.1)))
    original["HYYield"] = 6.2 + np.cumsum(noise(0.02))
    original["NFCI"] = -0.2 + np.cumsum(noise(0.005))

    changed = original.copy()
    shock_days = np.arange(shock_start, rows) - (shock_start - 1)
    changed.loc[shock_start:, "InitialJoblessClaims"] = (
        original.loc[shock_start - 1, "InitialJoblessClaims"] + shock_days * 800
    )
    changed.loc[shock_start:, "HYYield"] = (
        original.loc[shock_start - 1, "HYYield"] + shock_days * 0.05
    )
    return original, changed


def test_early_warning_breadth_counts_domains_and_uses_only_trailing_data() -> None:
    config = ResearchConfig(distribution_windows=(252,))
    original, changed = _macro_shock_frames()

    first = build_features(original, config)
    second = build_features(changed, config)

    assert first["EarlyWarningBreadth"].dropna().between(0, 5).all()
    pd.testing.assert_series_equal(
        first.loc[:449, "EarlyWarningBreadth"],
        second.loc[:449, "EarlyWarningBreadth"],
    )
    pd.testing.assert_series_equal(
        first.loc[:449, "EarlyWarningPersistence20_Breadth2"],
        second.loc[:449, "EarlyWarningPersistence20_Breadth2"],
    )
    # A sustained (not one-off) acceleration in two domains at once should
    # keep breadth >= 2 for the whole trailing window, unlike the noisy,
    # single-domain-at-a-time baseline.
    assert second.loc[499, "EarlyWarningBreadth"] >= 2
    assert first.loc[499, "EarlyWarningBreadth"] <= 1
    assert second.loc[499, "EarlyWarningPersistence20_Breadth2"] >= 18
    assert second.loc[499, "EarlyWarningPersistence10_Breadth2"] >= 9


def test_early_warning_persistence_requires_a_full_window() -> None:
    config = ResearchConfig(distribution_windows=(252,))
    original, _ = _macro_shock_frames()
    result = build_features(original, config)
    assert pd.isna(result.loc[15, "EarlyWarningPersistence20_Breadth2"])
    assert not pd.isna(result.loc[30, "EarlyWarningPersistence20_Breadth2"])
    assert result["EarlyWarningPersistence20_Breadth2"].dropna().between(0, 20).all()
    assert result["EarlyWarningPersistence10_Breadth2"].dropna().between(0, 10).all()


def test_targets_exclude_current_day_and_end_with_nan() -> None:
    config = ResearchConfig(
        return_horizons=(2,),
        risk_horizons=(2,),
        primary_return_horizon=2,
        primary_risk_horizon=2,
    )
    data = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=5),
            "Close": [100.0, 90.0, 120.0, 80.0, 130.0],
            "CashRate": 0.0,
        }
    )
    targets = build_targets(data, config)

    assert np.isclose(targets.loc[0, "ForwardReturn_2"], 0.20)
    assert np.isclose(targets.loc[0, "FutureMinReturn_2"], -0.10)
    assert targets.loc[0, "DrawdownHit_2_10"] == 1.0
    assert targets.loc[3:, "ForwardReturn_2"].isna().all()
    assert targets.loc[3:, "DrawdownHit_2_10"].isna().all()
    assert targets.loc[0, "TargetEndDate_2"] == data.loc[2, "Date"]


def test_forward_cash_return_uses_future_days_only() -> None:
    config = ResearchConfig(
        return_horizons=(2,),
        risk_horizons=(2,),
        primary_return_horizon=2,
        primary_risk_horizon=2,
    )
    data = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=4),
            "Close": [100.0] * 4,
            "CashRate": [0.0, 10.0, 20.0, 30.0],
        }
    )
    targets = build_targets(data, config)
    expected = np.expm1(np.log1p(0.10) / 252 + np.log1p(0.20) / 252)
    assert targets.loc[0, "ForwardCashReturn_2"] == expected
