from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.pit_universe_builder import (
    PitUniverseSettings,
    _candidate_observation,
    build_hybrid_snapshot,
    compare_direct_and_proxy,
    contemporaneous_ticker,
    historical_members_as_of,
    parse_tradefomo_ranking,
    rank_sp500_proxy,
)


def test_parse_tradefomo_ranking_keeps_published_rank_and_cap() -> None:
    html = """
    <a href="/stock-analysis-and-signals/MSFT">
      1 MSFT Name MICROSOFT CORP Industry Software - Infrastructure
      Market Cap $785.50B
    </a>
    <a href="/stock-analysis-and-signals/AMZN">
      2 AMZN Name AMAZON COM INC Industry Internet Retail
      Market Cap $1.20T
    </a>
    <a href="/stock-analysis-and-signals/BRK.B">
      3 BRK.B Name BERKSHIRE HATHAWAY INC Industry Market Cap $498.83B
    </a>
    """
    result = parse_tradefomo_ranking(html, 2019)

    assert result["Ticker"].tolist() == ["MSFT", "AMZN", "BRK.B"]
    assert result["Rank"].tolist() == [1, 2, 3]
    assert result["MarketCap"].tolist() == pytest.approx(
        [785.50e9, 1.20e12, 498.83e9]
    )


def test_historical_members_uses_latest_snapshot_not_future_row() -> None:
    history = pd.DataFrame(
        {
            "date": ["2018-12-24", "2019-01-03"],
            "tickers": ["AAPL,MSFT", "AAPL,MSFT,XYZ"],
        }
    )

    source_date, tickers = historical_members_as_of(
        history,
        "2019-01-01",
    )

    assert source_date == pd.Timestamp("2018-12-24")
    assert tickers == ["AAPL", "MSFT"]


def test_known_current_ticker_is_mapped_back_at_snapshot_date() -> None:
    assert contemporaneous_ticker("META", "2019-01-01") == "FB"
    assert contemporaneous_ticker("META", "2023-01-01") == "META"


def test_candidate_observation_restores_raw_close_after_future_split() -> None:
    settings = PitUniverseSettings(
        snapshot_years=(2019,),
        maximum_price_age_days=10,
        maximum_shares_age_days=550,
    )
    price = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2018-12-31", tz="UTC")],
            "SplitAdjustedClose": [40.0],
        }
    )
    shares = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2018-12-30", tz="UTC")],
            "SharesOutstanding": [5e9],
        }
    )
    splits = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-08-31", tz="UTC")],
            "Factor": [4.0],
        }
    )

    row = _candidate_observation(
        ticker="AAPL",
        company="Apple Inc.",
        as_of=pd.Timestamp("2019-01-01", tz="UTC"),
        price_frame=price,
        shares=shares,
        splits=splits,
        settings=settings,
    )

    assert row["RawClose"] == pytest.approx(160.0)
    assert row["MarketCap"] == pytest.approx(800e9)
    assert row["EligibleForRanking"] is True


def test_proxy_deduplicates_share_classes_by_issuer() -> None:
    candidates = pd.DataFrame(
        [
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "GOOG",
                "Company": "Alphabet Inc.",
                "MarketCap": 700e9,
                "PriceDataAvailable": True,
                "SharesDataAvailable": True,
                "EligibleForRanking": True,
            },
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "GOOGL",
                "Company": "Alphabet Inc.",
                "MarketCap": 699e9,
                "PriceDataAvailable": True,
                "SharesDataAvailable": True,
                "EligibleForRanking": True,
            },
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "MSFT",
                "Company": "Microsoft Corp.",
                "MarketCap": 600e9,
                "PriceDataAvailable": True,
                "SharesDataAvailable": True,
                "EligibleForRanking": True,
            },
        ]
    )

    result = rank_sp500_proxy(candidates, target_size=100)

    assert result["Ticker"].tolist() == ["GOOG", "MSFT"]
    assert result["Rank"].tolist() == [1, 2]


def test_hybrid_labels_proxy_fill_as_not_actual_rank() -> None:
    direct = pd.DataFrame(
        [
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "AAA",
                "Company": "Alpha Inc.",
                "MarketCap": 100.0,
                "Rank": 1,
                "RankSource": "TRADEFOMO_DIRECT_PUBLISHED",
            }
        ]
    )
    proxy = pd.DataFrame(
        [
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "AAA",
                "Company": "Alpha Inc.",
                "MarketCap": 100.0,
                "Rank": 1,
            },
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "BBB",
                "Company": "Beta Inc.",
                "MarketCap": 90.0,
                "Rank": 2,
            },
        ]
    )

    result = build_hybrid_snapshot(direct, proxy, target_size=2)

    assert result["Ticker"].tolist() == ["AAA", "BBB"]
    assert result.iloc[1]["RankSource"] == (
        "SP500_PROXY_FILL_NOT_ACTUAL_WHOLE_MARKET_RANK"
    )


def test_source_comparison_normalizes_dot_and_hyphen_share_class() -> None:
    direct = pd.DataFrame(
        [
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "BRK.B",
            }
        ]
    )
    proxy = pd.DataFrame(
        [
            {
                "AsOfDate": "2019-01-01",
                "Ticker": "BRK-B",
            }
        ]
    )

    result = compare_direct_and_proxy(direct, proxy)

    assert result.iloc[0]["DirectCanonicalTickerOverlapCount"] == 1
    assert result.iloc[0]["DirectOnlyTickers"] == ""
