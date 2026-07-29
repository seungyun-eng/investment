from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.automatic_backfill import (
    AutomaticBackfillSettings,
    _financial_status,
    _price_status,
    build_automatic_ticker_configs,
)
from stock_research.io_utils import atomic_to_excel
from stock_research.market_data import (
    _download_yahoo_chart,
    update_one_price,
)
from stock_research.paths import load_paths
from stock_research.tickers import TickerConfig, load_tickers


def test_build_automatic_ticker_configs_preserves_base_and_generates_names(
    tmp_path: Path,
) -> None:
    universe = pd.DataFrame(
        [
            {
                "DataSymbol": "AAPL",
                "LiquidityRank": 1,
                "ScreenerName": "Apple Inc. Common Stock",
            },
            {
                "DataSymbol": "MU",
                "LiquidityRank": 2,
                "ScreenerName": "Micron Technology Inc. Common Stock",
            },
        ]
    )
    base = {
        "AAPL": TickerConfig("AAPL", "apple", 9),
    }

    result = build_automatic_ticker_configs(universe, base)

    assert result["AAPL"] is base["AAPL"]
    assert result["MU"].company_slug == "micron-technology"
    assert result["MU"].display_name == "Micron Technology"


def test_generated_config_round_trips_optional_display_name(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tickers.json"
    config_path.write_text(
        """
        {
          "MU": {
            "company_slug": "micron-technology",
            "fiscal_year_end_month": 8,
            "display_name": "Micron Technology"
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_tickers(config_path)["MU"]

    assert config.display_name == "Micron Technology"
    assert config.fiscal_year_end_month == 8


def test_price_and_financial_validation_use_v6_input_shapes(
    tmp_path: Path,
) -> None:
    paths = load_paths(tmp_path)
    config = TickerConfig(
        "TEST",
        "test-company",
        12,
        display_name_override="Test Company",
    )
    dates = pd.bdate_range("2024-01-01", periods=300)
    price = pd.DataFrame(
        {
            "날짜": dates,
            "종가": range(300),
            "시가": range(300),
            "고가": range(300),
            "저가": range(300),
            "거래량": [1_000_000] * 300,
        }
    )
    price.to_csv(
        paths.processed / "Test Company_지표포함.csv",
        index=False,
        encoding="utf-8-sig",
    )
    periods = pd.date_range("2022-03-31", periods=8, freq="QE")
    def statement(metric: str) -> pd.DataFrame:
        return pd.DataFrame(
            {period.strftime("%Y-%m-%d"): [1.0] for period in periods},
            index=[metric],
        )

    financial_path = paths.financial_raw / "TEST_financials_Q.xlsx"
    atomic_to_excel(
        {
            "Income Statement": statement("Revenue"),
            "Balance Sheet": statement("Total Assets"),
            "Cash Flow Statement": statement("Operating Cash Flow"),
            "Key Financial Ratios": statement("Return On Equity"),
        },
        financial_path,
    )
    settings = AutomaticBackfillSettings(
        minimum_price_rows=252,
        maximum_price_age_days=5,
        minimum_financial_periods=8,
    )

    price_result = _price_status(
        paths,
        config,
        settings,
        as_of=datetime(2025, 2, 24, tzinfo=UTC),
    )
    financial_result = _financial_status(paths, config, settings)

    assert price_result["Valid"]
    assert price_result["Rows"] == 300
    assert financial_result["Valid"]
    assert financial_result["Periods"] == 8


def test_yahoo_chart_transport_normalizes_ohlcv() -> None:
    class FakeResponse:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "timestamp": [1704067200, 1704153600],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [10.0, 11.0],
                                        "high": [12.0, 13.0],
                                        "low": [9.0, 10.0],
                                        "close": [11.0, 12.0],
                                        "volume": [1000, 2000],
                                    }
                                ]
                            },
                        }
                    ],
                }
            }

    class FakeSession:
        @staticmethod
        def get(
            url: str,
            params: dict[str, object],
            headers: dict[str, str],
            timeout: float,
        ) -> FakeResponse:
            assert url.endswith("/TEST")
            assert params["interval"] == "1d"
            assert headers["User-Agent"]
            assert timeout == 12.0
            return FakeResponse()

    result = _download_yahoo_chart(
        "TEST",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        timeout_seconds=12.0,
        session=FakeSession(),
    )

    assert list(result.columns) == [
        "Date",
        "Price",
        "Open",
        "High",
        "Low",
        "Volume",
    ]
    assert result["Price"].tolist() == [11.0, 12.0]
    assert (result["Date"].dt.hour == 0).all()


def test_price_refresh_redownloads_internal_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = TickerConfig("TEST", "test-company", 12)
    company_dir = tmp_path / config.display_name
    company_dir.mkdir()
    existing = pd.DataFrame(
        {
            "Date": ["01/02/2024", "01/04/2024"],
            "Price": [10.0, 12.0],
            "Open": [10.0, 12.0],
            "High": [10.0, 12.0],
            "Low": [10.0, 12.0],
            "Vol.": ["1.00K", "1.00K"],
            "Change %": ["", "20.00%"],
        }
    )
    existing.to_csv(
        company_dir / "2024.01.02_2024.01.04 Test Company Historical Data.csv",
        index=False,
    )
    captured: dict[str, datetime] = {}

    def fake_download(
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        transport: str,
        timeout_seconds: float,
    ) -> pd.DataFrame:
        captured["start"] = start
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "Price": [10.0, 11.0, 12.0],
                "Open": [10.0, 11.0, 12.0],
                "High": [10.0, 11.0, 12.0],
                "Low": [10.0, 11.0, 12.0],
                "Volume": [1000, 1000, 1000],
            }
        )

    monkeypatch.setattr(
        "stock_research.market_data._download",
        fake_download,
    )
    refresh_start = datetime.fromisoformat("2024-01-01")

    output = update_one_price(
        config,
        tmp_path,
        refresh_start=refresh_start,
        transport="yahoo_chart",
    )

    assert output is not None
    assert captured["start"] == refresh_start
    refreshed = pd.read_csv(output)
    assert len(refreshed) == 3
    assert "01/03/2024" in set(refreshed["Date"])
