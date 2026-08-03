from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stock_research.cross_sectional.sp500_universe import (
    HISTORICAL_COMPANY_NAMES,
    Sp500UniverseSettings,
    build_sp500_change_membership,
    build_sp500_membership,
    build_sp500_union,
    data_ticker,
    normalize_sp500_ticker,
    parse_wikipedia_sp500_tables,
)


def test_parse_wikipedia_tables_normalizes_current_and_changes() -> None:
    current = pd.DataFrame(
        {
            "Symbol": ["BRK.B", "META"],
            "Security": ["Berkshire Hathaway", "Meta Platforms"],
            "GICS Sector": ["Financials", "Communication Services"],
            "GICS Sub-Industry": ["Multi-Sector Holdings", "Media"],
            "CIK": ["1067983", "1326801"],
        }
    )
    changes = pd.DataFrame(
        {
            (
                "Effective Date",
                "Effective Date",
            ): ["June 30, 2026"],
            ("Added", "Ticker"): ["XYZ"],
            ("Added", "Security"): ["Example Added"],
            ("Removed", "Ticker"): ["ABC"],
            ("Removed", "Security"): ["Example Removed"],
            ("Reason", "Reason"): ["Market capitalization change."],
        }
    )

    parsed_current, parsed_changes = parse_wikipedia_sp500_tables(
        [current, changes]
    )

    assert parsed_current["Ticker"].tolist() == ["BRK-B", "META"]
    assert parsed_changes.iloc[0]["AddedTicker"] == "XYZ"
    assert parsed_changes.iloc[0]["RemovedTicker"] == "ABC"
    assert parsed_changes.iloc[0]["EffectiveDate"] == pd.Timestamp(
        "2026-06-30"
    )


def test_membership_uses_history_then_reverses_live_changes() -> None:
    history = pd.DataFrame(
        {
            "date": ["2024-12-31", "2025-08-23"],
            "tickers": ["A,B", "A,B,C"],
        }
    )
    current = pd.DataFrame(
        {
            "Ticker": ["A", "D", "E"],
            "Company": ["Alpha", "Delta", "Echo"],
        }
    )
    changes = pd.DataFrame(
        {
            "EffectiveDate": pd.to_datetime(
                ["2025-10-01", "2026-06-01"]
            ),
            "AddedTicker": ["D", "E"],
            "AddedCompany": ["Delta", "Echo"],
            "RemovedTicker": ["B", "C"],
            "RemovedCompany": ["Beta", "Charlie"],
            "Reason": ["", ""],
        }
    )
    settings = Sp500UniverseSettings(snapshot_years=(2025, 2026))

    result = build_sp500_membership(
        history,
        current,
        changes,
        settings,
        fetched_at=datetime(2026, 7, 1, tzinfo=UTC),
    )

    y2025 = set(result.loc[result["AsOfDate"].eq("2025-01-01"), "Ticker"])
    y2026 = set(result.loc[result["AsOfDate"].eq("2026-01-01"), "Ticker"])
    assert y2025 == {"A", "B"}
    assert y2026 == {"A", "C", "D"}
    assert set(result.loc[result["AsOfDate"].eq("2026-01-01"), "MembershipSource"]) == {
        "WIKIPEDIA_CURRENT_REVERSED_CHANGES"
    }


def test_union_is_unique_and_preserves_membership_years() -> None:
    membership = pd.DataFrame(
        [
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "META",
                "DataSymbol": "META",
                "HistoricalTickers": "FB",
                "Company": "Meta Platforms",
            },
            {
                "AsOfDate": "2020-01-01",
                "Ticker": "META",
                "DataSymbol": "META",
                "HistoricalTickers": "FB",
                "Company": "Meta Platforms",
            },
            {
                "AsOfDate": "2020-01-01",
                "Ticker": "ABC",
                "DataSymbol": "ABC",
                "HistoricalTickers": "ABC",
                "Company": "Example",
            },
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "APC",
                "DataSymbol": "APC",
                "HistoricalTickers": "APC",
                "Company": "Anadarko Petroleum",
            },
        ]
    )

    result = build_sp500_union(membership)

    assert result["DataSymbol"].is_unique
    meta = result.set_index("DataSymbol").loc["META"]
    assert meta["MembershipYears"] == "2019,2020"
    assert meta["SnapshotCount"] == 2
    assert data_ticker("FB") == "META"
    assert data_ticker("FI") == "FISV"
    assert data_ticker("GPS") == "GAP"
    assert data_ticker("PKI") == "RVTY"
    assert (
        result.set_index("DataSymbol").loc["APC", "CrawlBlockReason"]
        == "HISTORICAL_SYMBOL_REUSED_BY_DIFFERENT_SECURITY"
    )
    assert HISTORICAL_COMPANY_NAMES["UTX"] == "United Technologies"


def test_change_membership_keeps_intra_year_constituents() -> None:
    history = pd.DataFrame(
        {
            "date": [
                "2018-12-31",
                "2019-06-01",
                "2019-09-01",
                "2020-01-01",
            ],
            "tickers": [
                "A,B",
                "A,B,C",
                "A,B",
                "A,B",
            ],
        }
    )
    current = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "Company": ["Alpha", "Beta"],
        }
    )
    changes = pd.DataFrame(
        columns=[
            "EffectiveDate",
            "AddedTicker",
            "AddedCompany",
            "RemovedTicker",
            "RemovedCompany",
        ]
    )
    settings = Sp500UniverseSettings(snapshot_years=(2019,))

    membership = build_sp500_change_membership(
        history,
        current,
        changes,
        settings,
        fetched_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    annual = membership.loc[
        membership["AsOfDate"].isin(["2019-01-01"])
    ].copy()
    union = build_sp500_union(
        membership,
        annual_membership=annual,
    )

    assert set(union["DataSymbol"]) == {"A", "B", "C"}
    assert union.set_index("DataSymbol").loc["C", "SnapshotCount"] == 0
    assert normalize_sp500_ticker("RVTY (previously PKI)") == "RVTY"
