from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.config import ResearchSettings
from stock_research.cross_sectional.pit_validation import (
    apply_delisting_return_policy,
    apply_membership_to_panel,
    build_annual_liquidity_membership,
    causal_signal_day_panel,
)
from stock_research.cross_sectional.portfolio import (
    run_portfolio_backtest,
)


def _settings() -> ResearchSettings:
    return ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        minimum_price_history_sessions=1,
        minimum_cross_section_size=1,
    )


def _factor_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-31", "2025-02-03"])
    rows = pd.MultiIndex.from_product(
        [dates, ["A", "B"]],
        names=["Date", "Ticker"],
    ).to_frame(index=False)
    rows["Open"] = 10.0
    rows["Close"] = 10.0
    rows["Volume"] = 1_000_000.0
    rows["Shares"] = 1_000_000_000.0
    rows["Eligible"] = True
    rows["Return21"] = 0.1
    rows["Return63"] = 0.1
    rows["Return126"] = 0.1
    rows["Trend50"] = 0.1
    rows["Trend200"] = 0.1
    rows["Volatility63"] = 0.2
    rows["Drawdown126"] = -0.1
    for column in (
        "RevenueGrowthYoY",
        "EpsGrowthYoY",
        "OperatingMarginChangeYoY",
        "OperatingMargin",
        "FreeCashFlowMargin",
        "ReturnOnInvestment",
        "NetCashToAssets",
        "EpsTtmGrowthYoY",
        "EpsTtmGrowthAcceleration",
        "DcfPriceGrowthYoY",
        "EbitdaTtmGrowthYoY",
        "EbitdaTtmGrowthAcceleration",
        "DcfUpside",
        "GrowthAdjustedPe",
        "GrowthAdjustedEvEbitda",
    ):
        rows[column] = 0.1
    return rows


def test_causal_signal_date_uses_calendar_not_future_coverage() -> None:
    thursday = pd.Timestamp("2025-01-09")
    friday = pd.Timestamp("2025-01-10")
    panel = pd.DataFrame(
        {
            "Date": [thursday, thursday, friday],
            "Ticker": ["A", "B", "A"],
        }
    )
    result = causal_signal_day_panel(
        panel,
        start="2025-01-06",
        end="2025-01-10",
        reference_market_dates=pd.date_range(
            "2025-01-06",
            "2025-01-10",
            freq="B",
        ),
    )
    assert result["Date"].nunique() == 1
    assert pd.Timestamp(result["Date"].iloc[0]) == friday


def test_causal_signal_date_excludes_incomplete_final_week() -> None:
    dates = pd.date_range("2025-01-06", "2025-01-14", freq="B")
    panel = pd.DataFrame({"Date": dates, "Ticker": "A"})
    result = causal_signal_day_panel(
        panel,
        start="2025-01-06",
        end="2025-01-14",
        reference_market_dates=dates,
    )
    assert result["Date"].nunique() == 1
    assert pd.Timestamp(result["Date"].iloc[0]) == pd.Timestamp(
        "2025-01-10"
    )


def test_causal_signal_date_accepts_holiday_shortened_week() -> None:
    dates = pd.to_datetime(
        [
            "2025-06-30",
            "2025-07-01",
            "2025-07-02",
            "2025-07-03",
            "2025-07-07",
        ]
    )
    panel = pd.DataFrame({"Date": dates, "Ticker": "A"})
    result = causal_signal_day_panel(
        panel,
        start="2025-06-30",
        end="2025-07-03",
        reference_market_dates=dates,
    )
    assert result["Date"].nunique() == 1
    assert pd.Timestamp(result["Date"].iloc[0]) == pd.Timestamp(
        "2025-07-03"
    )


def test_membership_changes_only_on_known_snapshot_date() -> None:
    membership = pd.DataFrame(
        {
            "AsOfDate": pd.to_datetime(
                ["2025-01-01", "2025-02-01"]
            ),
            "Ticker": ["A", "B"],
            "Rank": [1, 1],
        }
    )
    result = apply_membership_to_panel(
        _factor_panel(),
        membership,
        _settings(),
    )
    january = result.loc[result["Date"].eq("2025-01-31")]
    february = result.loc[result["Date"].eq("2025-02-03")]
    assert january.set_index("Ticker")["UniverseMember"].to_dict() == {
        "A": True,
        "B": False,
    }
    assert february.set_index("Ticker")["UniverseMember"].to_dict() == {
        "A": False,
        "B": True,
    }


def test_bankruptcy_forces_total_loss_and_missing_policy_is_explicit() -> None:
    events = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "EffectiveDate": ["2025-01-06", "2025-01-06"],
            "DelistingReturn": [None, None],
            "DelistingCategory": ["BANKRUPTCY", "OTHER"],
            "Exchange": ["NASDAQ", "NYSE"],
        }
    )
    resolved = apply_delisting_return_policy(
        events,
        missing_return_policy="EXCHANGE_HAIRCUT",
    ).set_index("Ticker")
    assert resolved.loc["A", "DelistingReturn"] == pytest.approx(-1.0)
    assert resolved.loc["B", "DelistingReturn"] == pytest.approx(-0.30)
    assert not bool(resolved.loc["A", "ReturnWasImputed"])
    assert bool(resolved.loc["B", "ReturnWasImputed"])


def test_delisting_return_is_settled_in_cash_instead_of_ffill() -> None:
    dates = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    )
    panel = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "A",
            "Open": 10.0,
            "Close": 10.0,
        }
    )
    targets = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-02")],
            "Ticker": ["A"],
            "TargetWeight": [1.0],
        }
    )
    events = pd.DataFrame(
        {
            "Ticker": ["A"],
            "EffectiveDate": [pd.Timestamp("2025-01-06")],
            "DelistingReturn": [-0.5],
        }
    )
    result = run_portfolio_backtest(
        panel,
        targets,
        start="2025-01-02",
        end="2025-01-07",
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        record_attribution=True,
        delisting_events=events,
    )
    assert result.summary.final_value == pytest.approx(50.0)
    assert result.daily.iloc[-1]["Cash"] == pytest.approx(50.0)
    assert result.daily.iloc[-1]["SelectedCount"] == 0
    assert result.delistings is not None
    assert result.delistings.iloc[0]["DelistingPnL"] == pytest.approx(-50.0)
    assert result.attribution is not None
    assert result.attribution["NetPnL"].sum() == pytest.approx(-50.0)


def test_local_membership_converts_macrotrends_million_shares() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-12-27", "2024-12-30"]),
            "Ticker": ["WDC", "WDC"],
            "Close": [10.0, 10.0],
            "Volume": [2_000_000.0, 2_000_000.0],
            "Shares": [100.0, 100.0],
        }
    )
    membership = build_annual_liquidity_membership(
        panel,
        [pd.Timestamp("2025-01-01")],
        target_size=200,
        lookback_sessions=2,
        minimum_sessions=2,
        minimum_market_cap=1_000_000_000.0,
        minimum_dollar_volume=10_000_000.0,
        minimum_ipo_age_years=0,
    )
    assert membership.iloc[0]["Ticker"] == "WDC"
    assert membership.iloc[0]["MarketCap"] == pytest.approx(
        1_000_000_000.0
    )


def test_local_membership_requires_market_cap_and_two_year_history() -> None:
    dates = pd.to_datetime(["2022-12-30", "2024-12-30"])
    panel = pd.DataFrame(
        {
            "Date": [dates[0], dates[1], dates[0], dates[1]],
            "Ticker": ["OLD", "OLD", "NEW", "NEW"],
            "Close": 10.0,
            "Volume": 2_000_000.0,
            "Shares": [100.0, 100.0, None, None],
        }
    )
    membership = build_annual_liquidity_membership(
        panel,
        [pd.Timestamp("2025-01-01")],
        lookback_sessions=2,
        minimum_sessions=2,
    )
    assert membership["Ticker"].tolist() == ["OLD"]
