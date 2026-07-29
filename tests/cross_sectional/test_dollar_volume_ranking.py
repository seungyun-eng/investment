from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_research.cross_sectional.dollar_volume_ranking import (
    parse_abbreviated_volume,
    parse_historical_price_file,
    scan_backtest_folder,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("9.84M", 9_840_000.0),
        ("12.5K", 12_500.0),
        ("1.25B", 1_250_000_000.0),
        ("1,234", 1_234.0),
        (" 2.0m ", 2_000_000.0),
    ],
)
def test_abbreviated_volume_supports_k_m_b(
    raw: str,
    expected: float,
) -> None:
    assert parse_abbreviated_volume(raw) == expected


@pytest.mark.parametrize("raw", ["", "-", "N/A", None, "bad"])
def test_abbreviated_volume_returns_nan_for_missing_or_invalid(
    raw: object,
) -> None:
    assert np.isnan(parse_abbreviated_volume(raw))


def test_scan_and_parse_wdc_style_file(tmp_path: Path) -> None:
    backtest = tmp_path / "Back Test"
    wdc_dir = backtest / "Western Digital"
    wdc_dir.mkdir(parents=True)
    empty_dir = backtest / "Empty"
    empty_dir.mkdir()
    path = wdc_dir / "Western Digital Historical Data.csv"
    pd.DataFrame(
        [
            {
                "Date": "07/28/2026",
                "Price": "463.510009765625",
                "Open": "466.09",
                "High": "467.26",
                "Low": "422.11",
                "Vol.": "9.84M",
                "Change %": "-6.91%",
            },
            {
                "Date": "bad-date",
                "Price": "100",
                "Open": "100",
                "High": "100",
                "Low": "100",
                "Vol.": "1M",
                "Change %": "0%",
            },
        ]
    ).to_csv(path, index=False)

    inventory, issues = scan_backtest_folder(backtest)
    parsed = parse_historical_price_file(
        inventory.iloc[0]["FilePath"],
        ticker="WDC",
        company="Western Digital",
    )

    assert inventory["Company"].tolist() == ["Western Digital"]
    assert issues["Issue"].tolist() == ["EMPTY_COMPANY_FOLDER"]
    assert len(parsed.data) == 1
    row = parsed.data.iloc[0]
    assert row["Date"] == pd.Timestamp("2026-07-28")
    assert row["DollarVolume"] == pytest.approx(
        463.510009765625 * 9_840_000
    )
    assert parsed.audit.iloc[0]["InvalidDateRows"] == 1


def test_parse_file_requires_date_price_and_volume(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"Date": "07/28/2026", "Price": "1"}]).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="Vol"):
        parse_historical_price_file(
            path,
            ticker="BAD",
            company="Bad",
        )
