from __future__ import annotations

from pathlib import Path

from stock_research.macro_sp500.data import (
    load_macro_sp500_data,
    load_sp500_proxy,
)


def test_legacy_proxy_and_vix_are_merged_without_forward_fill(tmp_path: Path) -> None:
    price_path = tmp_path / "S&P500 Historical Data.csv"
    vix_path = tmp_path / "VIX.csv"
    price_path.write_text(
        "Date,Price,Open,Vol.\n"
        "2024-01-02,470.00,469.00,1000000\n"
        "2024-01-03,471.00,470.00,1100000\n"
        "2024-01-04,472.00,471.00,1200000\n",
        encoding="utf-8",
    )
    vix_path.write_text(
        "Date,Value\n"
        "2024-01-02,13.2\n"
        "2024-01-04,14.1\n",
        encoding="utf-8",
    )

    price = load_sp500_proxy(price_path)
    merged = load_macro_sp500_data(
        tmp_path,
        price_file=price_path,
        vix_file=vix_path,
    )

    assert price.attrs["dividend_adjusted"] is False
    assert merged["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-04",
    ]
    assert merged["VIX"].tolist() == [13.2, 14.1]
    assert merged.attrs["merged_rows"] == 2
