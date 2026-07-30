from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.cross_sectional.portfolio import run_portfolio_backtest
from stock_research.cross_sectional.v7_capital_overlay import (
    OverlayCandidate,
    ResolvedOverlayCandidate,
    build_overlay_targets,
    generate_candidate_grid,
    run_signed_overlay_backtest,
)


def test_signed_engine_matches_long_only_engine_without_carry() -> None:
    dates = pd.date_range("2025-01-02", periods=5, freq="B")
    panel = pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "Ticker": ["A"] * 5 + ["B"] * 5,
            "Open": [100, 101, 102, 103, 104, 50, 49, 48, 47, 46],
            "Close": [101, 102, 103, 104, 105, 49, 48, 47, 46, 45],
        }
    )
    targets = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[2], dates[2]],
            "Ticker": ["A", "B", "A", "B"],
            "TargetWeight": [0.5, 0.5, 1.0, 0.0],
        }
    )
    long_only = run_portfolio_backtest(
        panel,
        targets,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        initial_capital=100_000,
        transaction_cost_bps=10,
    )
    signed = run_signed_overlay_backtest(
        panel,
        targets,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        initial_capital=100_000,
        transaction_cost_bps=10,
        funding_annual_rate=0.06,
        short_borrow_annual_rate=0.03,
    )
    np.testing.assert_allclose(
        signed.daily["Equity"],
        long_only.daily["Equity"],
        rtol=1e-12,
        atol=1e-8,
    )


def test_short_position_profits_when_price_falls_after_execution() -> None:
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    panel = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "A",
            "Open": [100.0, 100.0, 90.0, 80.0],
            "Close": [100.0, 90.0, 80.0, 70.0],
        }
    )
    targets = pd.DataFrame(
        {
            "Date": [dates[0]],
            "Ticker": ["A"],
            "TargetWeight": [-0.5],
        }
    )
    result = run_signed_overlay_backtest(
        panel,
        targets,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        initial_capital=100_000,
        transaction_cost_bps=0,
        funding_annual_rate=0,
        short_borrow_annual_rate=0,
    )
    assert result.summary.final_value == pytest.approx(115_000)
    assert result.summary.roi_percent == pytest.approx(15.0)


def test_overlay_uses_cash_gate_and_strict_short_filter() -> None:
    date = pd.Timestamp("2025-01-03")
    scored = pd.DataFrame(
        {
            "Date": [date] * 4,
            "Ticker": ["L1", "L2", "S1", "S2"],
            "Eligible": [True] * 4,
            "AlphaScore": [0.20, 0.10, -0.20, -0.30],
            "Trend200": [0.2, 0.2, -0.1, 0.1],
            "Return126": [0.3, 0.2, -0.2, -0.2],
        }
    )
    v7_targets = scored.copy()
    v7_targets["ModelSelected"] = [True, True, False, False]
    resolved = ResolvedOverlayCandidate(
        candidate=OverlayCandidate(
            candidate_id=1,
            cash_gate_quantile=0.5,
            strong_gate_quantile=0.9,
            strong_long_gross=1.5,
            short_gross=0.4,
            short_score_max=-0.1,
            short_count=5,
            max_gross=2.0,
        ),
        cash_gate_score=0.16,
        strong_gate_score=0.25,
    )
    targets, exposure, shorts = build_overlay_targets(
        scored,
        v7_targets,
        resolved,
    )
    assert exposure.iloc[0]["Regime"] == "CASH_GATE"
    assert exposure.iloc[0]["LongGross"] == 0
    assert exposure.iloc[0]["ShortGross"] == pytest.approx(0.4)
    assert shorts["Ticker"].tolist() == ["S1"]
    assert targets.loc[targets["Side"].eq("SHORT"), "TargetWeight"].iloc[0] == pytest.approx(-0.4)


def test_candidate_grid_keeps_gross_exposure_at_or_below_cap() -> None:
    candidates = generate_candidate_grid(
        cash_gate_quantiles=[None, 0.5],
        strong_gate_quantiles=[None, 0.9],
        strong_long_gross_values=[1.5, 2.0],
        short_gross_values=[0.0, 0.4],
        short_score_max_values=[-0.05, -0.1],
        short_count=5,
        max_gross=2.0,
    )
    assert candidates[0].strong_long_gross == 1.0
    assert candidates[0].short_gross == 0.0
    assert all(
        candidate.strong_long_gross + candidate.short_gross <= 2.0
        for candidate in candidates
    )
