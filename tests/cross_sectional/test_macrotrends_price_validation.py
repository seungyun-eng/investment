from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.macrotrends_price_validation import (
    PriceProbeTarget,
    extract_macrotrends_daily,
    probe_macrotrends_price_history,
)


def test_extract_macrotrends_daily_normalizes_ohlcv_and_volume_units() -> None:
    html = """
    <script>
    var dataDaily = [
      {"d":"2019-01-02","o":"10","h":"12","l":"9","c":"11","v":"6.022"},
      {"d":"2019-01-03","o":"11","h":"13","l":"10","c":"12","v":"4.420"}
    ];
    </script>
    """

    result = extract_macrotrends_daily(html)

    assert list(result.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert result.loc[0, "Date"] == pd.Timestamp("2019-01-02")
    assert result.loc[0, "Close"] == 11.0
    assert result.loc[0, "Volume"] == 6_022_000.0


def test_extract_macrotrends_daily_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="dataDaily"):
        extract_macrotrends_daily("<html></html>")


def test_probe_records_page_404_and_chart_500_without_inventing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)

    class FakeResponse:
        def __init__(self, status_code: int, url: str, text: str = "") -> None:
            self.status_code = status_code
            self.url = url
            self.text = text

    def fake_get(
        url: str,
        params: dict[str, object] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        assert headers["User-Agent"]
        assert timeout == 12.0
        if params:
            assert params == {"t": "CELG", "yb": 15}
            return FakeResponse(500, url)
        return FakeResponse(404, "https://www.macrotrends.net/stocks/charts/CELG//")

    result, daily = probe_macrotrends_price_history(
        PriceProbeTarget("CELG", "celgene", "Celgene"),
        timeout_seconds=12.0,
        retries=0,
        request_get=fake_get,
    )

    assert daily.empty
    assert result.price_page_not_found
    assert result.chart_status == 500
    assert not result.price_data_available
    assert result.failure_reason == "PRICE_PAGE_404; CHART_HTTP_500"
