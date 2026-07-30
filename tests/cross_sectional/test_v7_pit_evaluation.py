from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stock_research.cross_sectional.config import ResearchSettings
from stock_research.cross_sectional.v7_pit_evaluation import (
    build_financial_lag_audit,
    build_membership_coverage,
    load_raw_company_prices,
    load_ready_tickers,
    normalize_change_membership,
)


def test_load_ready_tickers_enforces_expected_count(tmp_path: Path) -> None:
    status = tmp_path / "status.csv"
    pd.DataFrame(
        {
            "Ticker": ["A", "B", "C"],
            "V6Ready": [True, "false", "YES"],
        }
    ).to_csv(status, index=False)
    assert load_ready_tickers(status, expected_count=2) == {"A", "C"}
    with pytest.raises(ValueError, match="Expected 3"):
        load_ready_tickers(status, expected_count=3)


def test_membership_uses_data_symbol_for_historical_alias() -> None:
    source = pd.DataFrame(
        {
            "AsOfDate": ["2020-01-01", "2020-01-01"],
            "Ticker": ["FB", "AAPL"],
            "DataSymbol": ["META", "AAPL"],
            "Rank": [2, 1],
            "Selected": [True, True],
        }
    )
    result = normalize_change_membership(source)
    assert result["Ticker"].tolist() == ["AAPL", "META"]
    assert result.set_index("Ticker").loc["META", "HistoricalTicker"] == "FB"


def test_raw_price_loader_parses_volume_suffix_and_merges_fragments(
    tmp_path: Path,
) -> None:
    company = tmp_path / "Example"
    company.mkdir()
    pd.DataFrame(
        {
            "Date": ["1/3/2020", "1/2/2020"],
            "Price": ["11.00", "10.00"],
            "Open": ["10.50", "9.50"],
            "High": ["11.20", "10.20"],
            "Low": ["10.20", "9.20"],
            "Vol.": ["1.5M", "900K"],
            "Change %": ["1%", "2%"],
        }
    ).to_csv(company / "part.csv", index=False)
    prices, kind, _ = load_raw_company_prices(company)
    assert kind == "RAW_NORMALIZED"
    assert prices["Date"].tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    assert prices["Volume"].tolist() == pytest.approx(
        [900_000.0, 1_500_000.0]
    )


def test_financial_audit_detects_lag_and_stale_signal_violations() -> None:
    settings = ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        financial_release_lag_days=45,
        max_financial_age_days=180,
    )
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-03-01", "2020-10-01"]),
            "Ticker": ["A", "A"],
            "FinancialPeriodEnd": pd.to_datetime(
                ["2019-12-31", "2019-12-31"]
            ),
            "FinancialAvailableDate": pd.to_datetime(
                ["2020-02-14", "2020-02-14"]
            ),
            "FinancialAgeDays": [16, 230],
        }
    )
    from stock_research.cross_sectional.features import (
        FINANCIAL_SIGNAL_COLUMNS,
    )

    for column in FINANCIAL_SIGNAL_COLUMNS:
        frame[column] = 1.0
    audit = build_financial_lag_audit(
        frame,
        settings,
        {"TRAIN_2020_2024": ("2020-01-01", "2024-12-31")},
    ).iloc[0]
    assert audit["FinancialLagDaysMedian"] == pytest.approx(45)
    assert audit["StaleRows"] == 1
    assert audit["StaleSignalNonNullCells"] == len(
        FINANCIAL_SIGNAL_COLUMNS
    )


def test_membership_coverage_accepts_recent_prior_market_session() -> None:
    membership = pd.DataFrame(
        {
            "AsOfDate": pd.to_datetime(["2026-07-30"]),
            "Ticker": ["A"],
        }
    )
    audit = pd.DataFrame(
        {
            "Ticker": ["A"],
            "BuildStatus": ["INCLUDED"],
            "PriceStart": ["2019-01-02"],
            "PriceEnd": ["2026-07-29"],
        }
    )
    coverage = build_membership_coverage(
        membership,
        {"A"},
        audit,
    ).iloc[0]
    assert coverage["PriceCoveredOnDate"] == 1
    assert coverage["MaximumPriceGapDays"] == 7
