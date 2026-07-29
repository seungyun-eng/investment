from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stock_research.cross_sectional.automatic_universe import (
    AutomaticUniverseSettings,
    build_automatic_universe,
)

NASDAQ_TEXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAA|Alpha Inc. - Common Stock|Q|N|N|100|N|N
BBB|Beta Growth ETF|G|N|N|100|Y|N
CCC|Capital III Acquisition Corp. - Class A Ordinary Shares|G|N|N|100|N|N
DDD|Delta Inc. - Common Stock|S|N|D|100|N|N
File Creation Time: 0728202618:00|||||||
"""

OTHER_TEXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
EEE|Epsilon Corporation Common Stock|N|EEE|N|100|N|EEE
FFF|Foxtrot Corporation Warrant|A|FFF|N|100|N|FFF
GGG|Alpha Inc. Class B Common Stock|N|GGG|N|100|N|GGG
File Creation Time: 0728202618:00|||||||
"""


def _screener_payload() -> dict[str, object]:
    rows = [
        {
            "symbol": "AAA",
            "name": "Alpha",
            "lastsale": "$20.00",
            "volume": "2000000",
            "marketCap": "5000000000",
            "country": "United States",
            "ipoyear": "2010",
            "industry": "Software",
            "sector": "Technology",
        },
        {
            "symbol": "BBB",
            "name": "Beta",
            "lastsale": "$100.00",
            "volume": "3000000",
            "marketCap": "10000000000",
            "country": "United States",
            "ipoyear": "2010",
            "industry": "Investment Managers",
            "sector": "Finance",
        },
        {
            "symbol": "CCC",
            "name": "Capital III",
            "lastsale": "$10.00",
            "volume": "2000000",
            "marketCap": "2000000000",
            "country": "United States",
            "ipoyear": "2025",
            "industry": "Blank Checks",
            "sector": "Finance",
        },
        {
            "symbol": "DDD",
            "name": "Delta",
            "lastsale": "$50.00",
            "volume": "2000000",
            "marketCap": "5000000000",
            "country": "United States",
            "ipoyear": "2010",
            "industry": "Software",
            "sector": "Technology",
        },
        {
            "symbol": "EEE",
            "name": "Epsilon",
            "lastsale": "$30.00",
            "volume": "1000000",
            "marketCap": "4000000000",
            "country": "United States",
            "ipoyear": "2000",
            "industry": "Industrial Machinery",
            "sector": "Industrials",
        },
        {
            "symbol": "FFF",
            "name": "Foxtrot",
            "lastsale": "$8.00",
            "volume": "5000000",
            "marketCap": "2000000000",
            "country": "United States",
            "ipoyear": "2010",
            "industry": "Industrial Machinery",
            "sector": "Industrials",
        },
        {
            "symbol": "GGG",
            "name": "Alpha Class B",
            "lastsale": "$20.00",
            "volume": "1000000",
            "marketCap": "5000000000",
            "country": "United States",
            "ipoyear": "2010",
            "industry": "Software",
            "sector": "Technology",
        },
    ]
    return {"data": {"rows": rows}}


def test_automatic_universe_filters_security_types_and_ranks_liquidity() -> None:
    settings = AutomaticUniverseSettings(
        target_size=2,
        minimum_price=5,
        minimum_market_cap=1_000_000_000,
        minimum_dollar_volume=10_000_000,
        minimum_ipo_age_years=2,
    )
    selected, audit = build_automatic_universe(
        NASDAQ_TEXT,
        OTHER_TEXT,
        _screener_payload(),
        settings,
        as_of=datetime(2026, 7, 28, tzinfo=UTC),
    )

    assert selected["DataSymbol"].tolist() == ["AAA", "EEE"]
    assert selected["LiquidityRank"].astype(int).tolist() == [1, 2]
    assert set(audit.loc[audit["Selected"], "Status"]) == {"SELECTED"}
    reasons = audit.set_index("DataSymbol")["ExclusionReasons"]
    assert "ETF" in reasons["BBB"]
    assert "SPAC_OR_BLANK_CHECK" in reasons["CCC"]
    assert "ABNORMAL_FINANCIAL_STATUS" in reasons["DDD"]
    assert "EXCLUDED_SECURITY_TYPE" in reasons["FFF"]
    assert "DUPLICATE_ISSUER_SHARE_CLASS" in reasons["GGG"]


def test_unknown_ipo_year_is_audited_but_not_excluded() -> None:
    payload = _screener_payload()
    rows = payload["data"]["rows"]
    assert isinstance(rows, list)
    rows[0]["ipoyear"] = ""
    settings = AutomaticUniverseSettings(target_size=1)

    selected, _ = build_automatic_universe(
        NASDAQ_TEXT,
        OTHER_TEXT,
        payload,
        settings,
        as_of=datetime(2026, 7, 28, tzinfo=UTC),
    )

    assert selected.iloc[0]["DataSymbol"] == "AAA"
    assert pd.isna(selected.iloc[0]["IPOAgeYears"])
