from __future__ import annotations

import pandas as pd
import pytest

from stock_research.financial_crawler import (
    _latest_statement_date,
    _merge_statement_history,
    _normalise_statement,
    fetch_statement_http,
)
from stock_research.tickers import TickerConfig


def test_normalise_statement_formats_excel_date_columns() -> None:
    statement = pd.DataFrame(
        {"2025-12-31": [1.0], pd.Timestamp("2026-03-31"): [2.0]},
        index=["Revenue"],
    )

    result = _normalise_statement(statement)

    assert list(result.columns) == ["2025-12-31", "2026-03-31"]
    assert _latest_statement_date(result) == pd.Timestamp("2026-03-31")


def test_merge_statement_history_prefers_fresh_and_keeps_old_quarters() -> None:
    existing = pd.DataFrame(
        {
            pd.Timestamp("2025-06-30"): [10.0],
            pd.Timestamp("2025-03-31"): [9.0],
        },
        index=["Revenue"],
    )
    fresh = pd.DataFrame(
        {"2025-09-30": [12.0], "2025-06-30": [11.0]},
        index=["Revenue"],
    )

    result = _merge_statement_history(fresh, existing)

    assert list(result.columns) == ["2025-09-30", "2025-06-30", "2025-03-31"]
    assert result.loc["Revenue", "2025-06-30"] == 11.0
    assert result.loc["Revenue", "2025-03-31"] == 9.0


def test_merge_statement_history_rejects_regression() -> None:
    existing = pd.DataFrame({"2025-09-30": [10.0]}, index=["Revenue"])
    fresh = pd.DataFrame({"2025-06-30": [9.0]}, index=["Revenue"])

    with pytest.raises(ValueError, match="older"):
        _merge_statement_history(fresh, existing)


def test_fetch_statement_http_uses_macrotrends_payload() -> None:
    payload = """
    <script>
    var originalData = [
      {"field_name": "<span>Revenue</span>", "2026-03-31": "12.5"}
    ];
    </script>
    """

    class FakeResponse:
        url = "https://www.macrotrends.net/stocks/charts/TEST/test/income-statement"
        text = payload

        @staticmethod
        def raise_for_status() -> None:
            return None

    class FakeSession:
        @staticmethod
        def get(url: str, headers: dict[str, str], timeout: float):
            assert "/TEST/test/income-statement" in url
            assert "User-Agent" in headers
            assert timeout == 12.0
            return FakeResponse()

    config = TickerConfig("TEST", "test", 12)
    result = fetch_statement_http(
        FakeSession(),
        config,
        "income-statement",
        timeout_seconds=12.0,
    )

    assert result.loc["Revenue", "2026-03-31"] == "12.5"
