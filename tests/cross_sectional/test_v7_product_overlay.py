from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.v7_product_overlay import (
    ProductOverlayCandidate,
    build_causal_risk_on,
    generate_product_candidates,
    simulate_product_overlay,
    summarize_benchmark_consistency,
    summarize_calendar_consistency,
    summarize_period,
)


def _candidate(**overrides: object) -> ProductOverlayCandidate:
    values: dict[str, object] = {
        "candidate_id": 1,
        "name": "TEST",
        "product_type": "SYNTHETIC_V7_2X",
        "sma_window": 2,
        "sma_buffer": 0.0,
        "sma_slope_lookback": 0,
        "vix_ceiling": 25.0,
        "base_drawdown_floor": -0.10,
        "risk_on_product_weight": 1.0,
        "risk_off_product_weight": 0.0,
        "risk_off_cash_weight": 0.0,
    }
    values.update(overrides)
    return ProductOverlayCandidate(**values)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-02", periods=4, freq="B"),
            "BaseReturn": [0.0, 0.10, -0.10, 0.02],
            "SPYReturn": [0.0, 0.03, -0.02, 0.01],
            "CashRate": 0.0,
            "SPYClose": [100.0, 101.0, 99.0, 102.0],
            "SPYSMA2": 90.0,
            "BaseDrawdown": 0.0,
            "VIX": 15.0,
        }
    )


def test_candidate_grid_rejects_overfunded_or_more_defensive_leverage() -> None:
    config = {
        "candidate_grid": {
            "product_types": ["SYNTHETIC_V7_2X"],
            "sma_windows": [100],
            "sma_slope_lookbacks": [0],
            "vix_ceilings": [25.0],
            "base_drawdown_floors": [-0.10],
            "risk_on_product_weights": [0.5],
            "risk_off_product_weights": [0.0, 0.5, 0.75],
            "risk_off_cash_weights": [0.0, 0.5, 1.0],
        }
    }
    candidates = generate_product_candidates(config)
    assert len(candidates) == 6
    assert all(
        candidate.risk_off_product_weight
        + candidate.risk_off_cash_weight
        <= 1.0
        for candidate in candidates
    )


def test_risk_regime_uses_prior_close_only() -> None:
    frame = _frame()
    frame["SPYClose"] = [80.0, 100.0, 100.0, 80.0]
    risk_on = build_causal_risk_on(frame, _candidate())
    assert risk_on.tolist() == [False, False, True, True]


def test_daily_reset_v7_product_is_not_twice_cumulative_return() -> None:
    daily = simulate_product_overlay(
        _frame(),
        _candidate(),
        initial_capital=100_000.0,
        annual_expense_ratio=0.0,
        annual_financing_spread=0.0,
        transaction_cost_bps=0.0,
    )
    assert daily["SleeveWeight"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert daily["AdjustedReturn"].tolist() == pytest.approx(
        [0.0, 0.20, -0.20, 0.04]
    )
    assert daily.iloc[-1]["Equity"] == pytest.approx(99_840.0)


def test_risk_off_cash_has_no_negative_cash_or_margin_balance() -> None:
    frame = _frame()
    frame["SPYClose"] = 80.0
    daily = simulate_product_overlay(
        frame,
        _candidate(risk_off_cash_weight=1.0),
        initial_capital=100_000.0,
        annual_expense_ratio=0.0,
        annual_financing_spread=0.0,
        transaction_cost_bps=0.0,
    )
    assert daily["BaseWeight"].tolist() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    assert daily["CashWeight"].tolist() == pytest.approx(
        [0.0, 1.0, 1.0, 1.0]
    )
    assert (daily[["BaseWeight", "SleeveWeight", "CashWeight"]].sum(axis=1) == 1.0).all()
    assert daily["EffectiveExposure"].tolist() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )


def test_summary_roi_uses_final_value_over_initial_capital() -> None:
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "AdjustedReturn": [0.0, 0.10],
            "SleeveWeight": [0.0, 0.0],
            "CashWeight": [0.0, 0.0],
            "EffectiveExposure": [1.0, 1.0],
            "RiskOn": [False, False],
            "TradedFraction": [0.0, 0.0],
            "SwitchCost": [0.0, 0.0],
        }
    )
    summary = summarize_period(
        daily,
        start="2025-01-02",
        end="2025-01-03",
        initial_capital=100_000.0,
    )
    assert summary["FinalValue"] == pytest.approx(110_000.0)
    assert summary["ROI"] == pytest.approx(10.0)


def test_calendar_consistency_penalizes_each_year_below_floor() -> None:
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-12-31",
                    "2021-01-04",
                    "2021-12-31",
                ]
            ),
            "AdjustedReturn": [0.0, 0.10, 0.20, -0.10],
            "Equity": [100_000.0, 110_000.0, 132_000.0, 118_800.0],
        }
    )
    summary = summarize_calendar_consistency(
        daily,
        start="2020-01-02",
        end="2021-12-31",
        target_floor=35.0,
        target_ceiling=45.0,
        rolling_sessions=2,
    )
    assert summary["CalendarWorstReturn"] == pytest.approx(8.0)
    assert summary["CalendarMeanReturn"] == pytest.approx(9.0)
    assert summary["CalendarReturnStd"] == pytest.approx(1.0)
    assert summary["CalendarTargetShortfallRMS"] == pytest.approx(
        ((25.0**2 + 27.0**2) / 2.0) ** 0.5
    )
    assert summary["CalendarYearsAtOrAboveFloor"] == 0
    assert summary["CalendarFloorMetEveryYear"] is False
    assert summary["RollingWorstReturn"] == pytest.approx(8.0)


def test_calendar_consistency_rejects_inverted_target_band() -> None:
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"]),
            "AdjustedReturn": [0.0],
            "Equity": [100_000.0],
        }
    )
    with pytest.raises(ValueError, match="target_ceiling"):
        summarize_calendar_consistency(
            daily,
            start="2020-01-02",
            end="2020-12-31",
            target_floor=45.0,
            target_ceiling=35.0,
        )


def test_benchmark_consistency_requires_each_year_to_outperform() -> None:
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-12-31",
                    "2021-01-04",
                    "2021-12-31",
                ]
            ),
            "AdjustedReturn": [0.0, 0.10, 0.20, -0.10],
            "Equity": [100_000.0, 110_000.0, 132_000.0, 118_800.0],
        }
    )
    summary = summarize_benchmark_consistency(
        daily,
        pd.Series([0.0, 0.05, 0.10, -0.05]),
        start="2020-01-02",
        end="2021-12-31",
        rolling_sessions=2,
    )
    assert summary["CalendarReturn2020"] == pytest.approx(10.0)
    assert summary["BenchmarkReturn2020"] == pytest.approx(5.0)
    assert summary["CalendarAlpha2020"] == pytest.approx(5.0)
    assert summary["CalendarAlpha2021"] == pytest.approx(3.5)
    assert summary["BenchmarkWorstAlpha"] == pytest.approx(3.5)
    assert summary["BenchmarkMaxDrawdown"] == pytest.approx(-5.0)
    assert summary["BenchmarkYearsOutperformed"] == 2
    assert summary["BenchmarkOutperformedEveryYear"] is True
    assert summary["BenchmarkRollingWorstAlpha"] == pytest.approx(3.5)


def test_benchmark_consistency_rejects_mismatched_rows() -> None:
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"]),
            "AdjustedReturn": [0.0],
            "Equity": [100_000.0],
        }
    )
    with pytest.raises(ValueError, match="equal length"):
        summarize_benchmark_consistency(
            daily,
            pd.Series([0.0, 0.01]),
            start="2020-01-02",
            end="2020-12-31",
        )
