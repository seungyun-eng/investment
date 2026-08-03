from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.cross_sectional.v7_risk_optimization import (
    RiskCandidate,
    add_causal_asset_volatility,
    assert_baseline_parity,
    build_risk_adjusted_targets,
    constrained_composition,
    generate_risk_candidates,
    validate_membership_history,
)


def _candidate(**overrides: object) -> RiskCandidate:
    values: dict[str, object] = {
        "candidate_id": 1,
        "name": "TEST",
        "allocation_profile": "TEST",
        "risk_profile": "TEST",
        "score_strength": 0.5,
        "inverse_volatility_strength": 0.5,
        "apply_concentration_caps": True,
        "fixed_long_gross": None,
        "target_portfolio_volatility": None,
        "minimum_long_gross": 0.0,
        "max_long_gross": 1.0,
        "risk_off_gross_cap": 0.75,
    }
    values.update(overrides)
    return RiskCandidate(**values)


def test_asset_volatility_is_causal() -> None:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    panel = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "A",
            "Close": [100.0, 101.0, 99.0, 102.0, 105.0, 50.0],
        }
    )
    before = add_causal_asset_volatility(
        panel.iloc[:-1],
        window=4,
        minimum_observations=2,
    )
    after = add_causal_asset_volatility(
        panel,
        window=4,
        minimum_observations=2,
    )
    pd.testing.assert_series_equal(
        before["AssetVolatility"],
        after.iloc[:-1]["AssetVolatility"],
        check_names=False,
    )


def test_constrained_composition_respects_ticker_and_sector_caps() -> None:
    raw = pd.Series(
        [0.7, 0.1, 0.1, 0.05, 0.05],
        index=["A", "B", "C", "D", "E"],
    )
    sectors = pd.Series(
        ["TECH", "TECH", "FIN", "ENERGY", "HEALTH"],
        index=raw.index,
    )
    weights, ticker_cap, sector_cap = constrained_composition(
        raw,
        sectors,
        max_ticker_weight=0.30,
        max_sector_weight=0.40,
    )
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= ticker_cap + 1e-10
    assert weights.groupby(sectors).sum().max() <= sector_cap + 1e-10


def test_constrained_composition_relaxes_infeasible_sector_cap() -> None:
    raw = pd.Series(
        [0.40, 0.25, 0.15, 0.10, 0.10],
        index=["A", "B", "C", "D", "E"],
    )
    sectors = pd.Series(
        ["TECH", "TECH", "TECH", "TECH", "FIN"],
        index=raw.index,
    )
    weights, ticker_cap, sector_cap = constrained_composition(
        raw,
        sectors,
        max_ticker_weight=0.30,
        max_sector_weight=0.40,
    )
    assert weights.sum() == pytest.approx(1.0)
    assert ticker_cap == pytest.approx(0.30)
    assert sector_cap == pytest.approx(0.70)
    assert weights.max() <= ticker_cap + 1e-10
    assert weights.groupby(sectors).sum().max() <= sector_cap + 1e-10


def test_score_and_inverse_volatility_tilt_selected_weights() -> None:
    date = pd.Timestamp("2025-03-07")
    target_rows = []
    for rank, (ticker, score, volatility) in enumerate(
        [
            ("A", 0.50, 0.15),
            ("B", 0.40, 0.20),
            ("C", 0.30, 0.25),
            ("D", 0.20, 0.30),
            ("E", 0.10, 0.35),
        ],
        start=1,
    ):
        target_rows.append(
            {
                "Date": date,
                "Ticker": ticker,
                "ModelSelected": True,
                "Rank": float(rank),
                "AlphaScore": score,
                "AssetVolatility": volatility,
            }
        )
    targets = pd.DataFrame(target_rows)
    history_dates = pd.date_range("2024-12-02", date, freq="B")
    returns = pd.DataFrame(
        {
            ticker: np.linspace(0.001, 0.002, len(history_dates))
            * (index + 1)
            for index, ticker in enumerate(["A", "B", "C", "D", "E"])
        },
        index=history_dates,
    )
    adjusted, diagnostics = build_risk_adjusted_targets(
        targets,
        returns,
        pd.Series({date: True}),
        _candidate(),
        sector_map={
            "A": "TECH",
            "B": "FIN",
            "C": "ENERGY",
            "D": "HEALTH",
            "E": "INDUSTRIALS",
        },
        volatility_window=20,
        minimum_volatility_observations=10,
        covariance_shrinkage=0.5,
        asset_volatility_floor=0.10,
        max_ticker_weight=0.30,
        max_sector_weight=0.40,
    )
    weights = adjusted.set_index("Ticker")["TargetWeight"]
    assert weights.sum() == pytest.approx(1.0)
    assert weights["A"] > weights["E"]
    assert diagnostics.iloc[0]["LongGross"] == pytest.approx(1.0)


def test_risk_off_cap_limits_target_gross() -> None:
    date = pd.Timestamp("2025-03-07")
    targets = pd.DataFrame(
        {
            "Date": [date] * 5,
            "Ticker": list("ABCDE"),
            "ModelSelected": True,
            "Rank": range(1, 6),
            "AlphaScore": np.linspace(0.5, 0.1, 5),
            "AssetVolatility": 0.20,
        }
    )
    history_dates = pd.date_range("2024-12-02", date, freq="B")
    returns = pd.DataFrame(
        {
            ticker: np.sin(np.arange(len(history_dates)) + index) / 100
            for index, ticker in enumerate(list("ABCDE"))
        },
        index=history_dates,
    )
    adjusted, diagnostics = build_risk_adjusted_targets(
        targets,
        returns,
        pd.Series({date: False}),
        _candidate(
            target_portfolio_volatility=0.30,
            max_long_gross=1.25,
            risk_off_gross_cap=0.75,
        ),
        sector_map={},
        volatility_window=20,
        minimum_volatility_observations=10,
        covariance_shrinkage=0.5,
        asset_volatility_floor=0.10,
        max_ticker_weight=0.30,
        max_sector_weight=0.40,
    )
    assert adjusted["TargetWeight"].sum() <= 0.75 + 1e-10
    assert diagnostics.iloc[0]["LongGross"] <= 0.75 + 1e-10


def test_fixed_long_gross_is_applied() -> None:
    date = pd.Timestamp("2025-03-07")
    targets = pd.DataFrame(
        {
            "Date": [date] * 5,
            "Ticker": list("ABCDE"),
            "ModelSelected": True,
            "Rank": range(1, 6),
            "AlphaScore": np.linspace(0.5, 0.1, 5),
            "AssetVolatility": 0.20,
        }
    )
    history_dates = pd.date_range("2024-12-02", date, freq="B")
    returns = pd.DataFrame(
        {ticker: 0.001 for ticker in list("ABCDE")},
        index=history_dates,
    )
    adjusted, diagnostics = build_risk_adjusted_targets(
        targets,
        returns,
        pd.Series({date: True}),
        _candidate(
            fixed_long_gross=1.55,
            max_long_gross=1.55,
            risk_off_gross_cap=1.55,
        ),
        sector_map={},
        volatility_window=20,
        minimum_volatility_observations=10,
        covariance_shrinkage=0.5,
        asset_volatility_floor=0.10,
        max_ticker_weight=0.30,
        max_sector_weight=0.40,
    )
    assert adjusted["TargetWeight"].sum() == pytest.approx(1.55)
    assert diagnostics.iloc[0]["LongGross"] == pytest.approx(1.55)


def test_membership_history_rejects_annual_snapshots() -> None:
    annual = pd.DataFrame(
        {
            "AsOfDate": [f"{year}-01-01" for year in range(2019, 2027)],
            "Ticker": "A",
        }
    )
    with pytest.raises(ValueError, match="sp500_membership_changes.csv"):
        validate_membership_history(
            annual,
            minimum_snapshot_count=50,
        )


def test_baseline_parity_rejects_metric_mismatch() -> None:
    class Summary:
        start_date = pd.Timestamp("2020-01-02")
        end_date = pd.Timestamp("2026-07-29")
        roi_percent = 300.0
        cagr_percent = 24.0
        max_drawdown_percent = -36.0
        sharpe_ratio = 1.0

    class Result:
        summary = Summary()

    with pytest.raises(RuntimeError, match="baseline parity failed"):
        assert_baseline_parity(
            Result(),  # type: ignore[arg-type]
            {
                "start_date": "2020-01-02",
                "end_date": "2026-07-29",
                "roi": 344.0,
                "cagr": 25.0,
                "max_drawdown": -36.0,
                "sharpe": 1.0,
            },
        )


def test_candidate_grid_keeps_baseline_first() -> None:
    config = {
        "allocation_profiles": [
            {
                "name": "EQUAL_UNCAPPED",
                "score_strength": 0.0,
                "inverse_volatility_strength": 0.0,
                "apply_concentration_caps": False,
            },
            {
                "name": "TILT",
                "score_strength": 0.5,
                "inverse_volatility_strength": 0.5,
                "apply_concentration_caps": True,
            },
        ],
        "risk_profiles": [
            {
                "name": "UNLEVERED",
                "target_portfolio_volatility": None,
                "max_long_gross": 1.0,
                "risk_off_gross_cap": 1.0,
            },
            {
                "name": "VOL",
                "target_portfolio_volatility": 0.25,
                "max_long_gross": 1.25,
                "risk_off_gross_cap": 0.75,
            },
        ],
    }
    candidates = generate_risk_candidates(config)
    assert len(candidates) == 4
    assert candidates[0].candidate_id == 0
    assert candidates[0].name == "EQUAL_UNCAPPED__UNLEVERED"
