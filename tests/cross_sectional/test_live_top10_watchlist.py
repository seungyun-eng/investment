from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.cross_sectional.live_top10_watchlist import (
    allocation_with_cash,
    apply_full_cash_market_gate,
    apply_graduated_exposure,
    build_top_n_plus_watchlist_membership,
    compute_graduated_exposure,
    new_account_allocation,
    watchlist_entries,
)


def test_watchlist_enters_on_causal_date_and_does_not_duplicate_top_name() -> None:
    top = pd.DataFrame(
        {
            "AsOfDate": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "DataSymbol": ["A", "B"],
            "Rank": [1, 2],
            "Selected": [True, True],
        }
    )
    watchlist = watchlist_entries(
        [
            {"ticker": "B", "eligible_from": "2020-01-01"},
            {"ticker": "IPO", "eligible_from": "2020-09-18"},
        ]
    )
    result = build_top_n_plus_watchlist_membership(
        top, watchlist, end_date="2020-12-31"
    )
    january = result.loc[result["AsOfDate"].eq(pd.Timestamp("2020-01-01"))]
    september = result.loc[
        result["AsOfDate"].eq(pd.Timestamp("2020-09-18"))
    ]
    assert january["DataSymbol"].tolist() == ["A", "B"]
    assert september["DataSymbol"].tolist() == ["A", "B", "IPO"]
    assert september.set_index("DataSymbol").loc["IPO", "MembershipBucket"] == "WATCHLIST"


def test_market_gate_turns_entire_universe_into_cash() -> None:
    dates = pd.date_range("2020-01-01", periods=220, freq="B")
    spy = pd.DataFrame(
        {"Date": dates, "Close": list(range(300, 500)) + list(range(20, 0, -1))}
    )
    panel = pd.DataFrame(
        {
            "Date": [dates[-1], dates[-1]],
            "Ticker": ["A", "B"],
            "Eligible": [True, True],
            "UniverseMember": [True, True],
        }
    )
    gated = apply_full_cash_market_gate(panel, spy)
    assert not gated["MarketRiskOn"].any()
    assert not gated["Eligible"].any()
    assert not gated["UniverseMember"].any()


def test_graduated_exposure_ramps_between_thresholds() -> None:
    dates = pd.date_range("2020-01-01", periods=220, freq="B")
    closes = [100.0] * 200 + list(np.linspace(100.0, 94.0, 20))
    spy = pd.DataFrame({"Date": dates, "Close": closes})
    exposure = compute_graduated_exposure(spy, slow_sessions=200)
    last = exposure.iloc[-1]
    assert -0.10 < last["SPYTrendRegime"] < 0.03
    assert 0.0 < last["ExposureScale"] < 1.0


def test_graduated_exposure_clips_to_zero_below_floor() -> None:
    dates = pd.date_range("2020-01-01", periods=220, freq="B")
    closes = [100.0] * 200 + list(np.linspace(100.0, 50.0, 20))
    spy = pd.DataFrame({"Date": dates, "Close": closes})
    exposure = compute_graduated_exposure(spy, slow_sessions=200)
    assert exposure.iloc[-1]["ExposureScale"] == pytest.approx(0.0)


def test_graduated_exposure_defaults_to_full_before_history_builds() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    spy = pd.DataFrame({"Date": dates, "Close": [100.0, 101.0, 99.0, 102.0, 98.0]})
    exposure = compute_graduated_exposure(spy, slow_sessions=200)
    assert (exposure["ExposureScale"] == 1.0).all()


def test_apply_graduated_exposure_scales_weight_and_frees_cash() -> None:
    targets = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-31", "2026-07-31"]),
            "Ticker": ["A", "B"],
            "TargetWeight": [0.5, 0.5],
        }
    )
    exposure = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-31"]),
            "SPYTrendRegime": [-0.05],
            "ExposureScale": [0.4],
        }
    )
    scaled = apply_graduated_exposure(targets, exposure).set_index("Ticker")
    assert scaled.loc["A", "TargetWeight"] == pytest.approx(0.2)
    assert scaled.loc["B", "TargetWeight"] == pytest.approx(0.2)


def test_allocation_always_shows_cash_remainder() -> None:
    targets = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-31", "2026-07-31"]),
            "Ticker": ["A", "B"],
            "TargetWeight": [0.3, 0.2],
        }
    )
    result = allocation_with_cash(targets).set_index("Ticker")
    assert result.loc["CASH", "TargetWeight"] == pytest.approx(0.5)


def test_new_account_does_not_inherit_unqualified_historical_hold() -> None:
    scored = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-31"] * 3),
            "Ticker": ["NEW", "SECOND", "OLD_HOLD"],
            "Qualified": [True, True, False],
            "Rank": [1.0, 2.0, pd.NA],
            "MarketRiskOn": [True, True, True],
        }
    )
    result = new_account_allocation(scored, top_k=3).set_index("Ticker")
    assert "OLD_HOLD" not in result.index
    assert result.loc["NEW", "TargetWeight"] == pytest.approx(1 / 3)
    assert result.loc["CASH", "TargetWeight"] == pytest.approx(1 / 3)
