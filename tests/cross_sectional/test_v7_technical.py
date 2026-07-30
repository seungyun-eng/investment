from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.cross_sectional.config import (
    ResearchSettings,
    StrategyParams,
)
from stock_research.cross_sectional.signals import score_panel
from stock_research.cross_sectional.v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
    slot5_params,
)


def _params() -> StrategyParams:
    return StrategyParams(
        momentum_weight=0.10,
        trend_weight=0.20,
        growth_weight=0.30,
        quality_weight=0.30,
        risk_control_weight=0.10,
        top_k=3,
        exit_rank=9,
        trend_floor=0.05,
        momentum_floor=0.10,
        loss_aware_exit_enabled=True,
        profit_rotation_exit_rank=9,
        replacement_score_advantage=0.05,
    )


def test_slot5_changes_only_top_k() -> None:
    base = _params()
    updated = slot5_params(base)
    expected = base.as_dict()
    expected["top_k"] = 5
    assert updated.as_dict() == expected


def test_technical_observations_are_causal_and_preserve_v6_scores() -> None:
    dates = pd.date_range("2024-01-02", periods=90, freq="B")
    panel = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "A",
            "Close": np.linspace(100, 150, len(dates)),
            "Volume": np.linspace(1_000_000, 2_000_000, len(dates)),
            "Eligible": True,
            "Trend200": 0.10,
            "Return126": 0.20,
            "Drawdown126": -0.02,
            "MomentumFactor": 0.1,
            "TrendFactor": 0.2,
            "GrowthFactor": 0.3,
            "QualityFactor": 0.4,
            "RiskControlFactor": 0.0,
        }
    )
    before = score_panel(panel, _params())["AlphaScore"]
    observed = add_v7_technical_observations(panel)
    after = score_panel(observed, _params())["AlphaScore"]
    pd.testing.assert_series_equal(before, after)
    latest = observed.iloc[-1]
    assert latest["MACDLineNormalized"] > 0
    assert latest["OBVFlow21"] > 0
    assert latest["RSI14"] == pytest.approx(100.0)
    assert pd.notna(latest["BollingerPercentB"])


def test_variant_replaces_only_trend_factor_with_component_mean() -> None:
    settings = ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        minimum_cross_section_size=1,
    )
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-03", "2025-01-03"]),
            "Ticker": ["A", "B"],
            "Eligible": [True, True],
            "TrendFactor": [0.4, -0.4],
            "MACDLineNormalized": [0.2, -0.2],
            "MACDHistogramNormalized": [0.1, -0.1],
            "OBVFlow21": [0.3, -0.3],
            "OBVFlow63": [0.2, -0.2],
            "RSI14": [75.0, 40.0],
            "BollingerPercentB": [1.0, 0.2],
        }
    )
    factored = add_v7_technical_factors(panel, settings)
    variant = TECHNICAL_VARIANTS[2]
    result = scoring_panel_for_variant(factored, variant)
    expected = result[
        ["MAFactor", "MACDFactor", "OBVFactor"]
    ].mean(axis=1)
    assert result["TrendFactor"].tolist() == pytest.approx(
        expected.tolist()
    )
    assert result["V6TrendFactor"].tolist() == [0.4, -0.4]
