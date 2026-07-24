import numpy as np
import pandas as pd

from stock_research.indicators import add_indicators, normalize_price_columns, parse_number


def test_parse_number_suffixes():
    assert parse_number("1.5K") == 1500
    assert parse_number("2M") == 2_000_000
    assert parse_number("1.2B") == 1_200_000_000
    assert parse_number("3.5%") == 3.5


def test_normalize_and_indicators():
    dates = pd.date_range("2024-01-01", periods=260)
    raw = pd.DataFrame({
        "Date": dates.strftime("%m/%d/%Y"),
        "Price": np.linspace(100, 200, 260),
        "Open": np.linspace(99, 199, 260),
        "High": np.linspace(101, 201, 260),
        "Low": np.linspace(98, 198, 260),
        "Vol.": ["1.00M"] * 260,
        "Change %": ["0.10%"] * 260,
    })
    normalized = normalize_price_columns(raw)
    result = add_indicators(normalized)
    for column in (
        "RSI (14일)", "볼린저밴드 상단", "볼린저밴드 하단",
        "MACD", "MACD 시그널", "SMA 200일", "OBV", "OBV_SIG9",
    ):
        assert column in result.columns
    assert len(result) == len(raw)
