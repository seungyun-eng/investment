from __future__ import annotations

import pandas as pd
import pytest

from stock_research.macro_fear_buy_sp500.leverage import (
    daily_reset_instrument,
)


def test_daily_reset_two_x_and_inverse_move_in_opposite_directions() -> None:
    signals = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Open": [100.0, 110.0],
            "Close": [110.0, 99.0],
            "CashRate": [0.0, 0.0],
            "TargetWeight": [1.0, 1.0],
        }
    )
    long = daily_reset_instrument(
        signals,
        multiple=2.0,
        annual_expense_ratio=0.0,
        annual_financing_spread=0.0,
    )
    inverse = daily_reset_instrument(
        signals,
        multiple=-2.0,
        annual_expense_ratio=0.0,
        annual_financing_spread=0.0,
    )
    assert long.loc[0, "Close"] == pytest.approx(120.0)
    assert inverse.loc[0, "Close"] == pytest.approx(80.0)
    assert long.loc[1, "Open"] == pytest.approx(120.0)
    assert inverse.loc[1, "Open"] == pytest.approx(80.0)
    assert long.loc[1, "Close"] == pytest.approx(96.0)
    assert inverse.loc[1, "Close"] == pytest.approx(96.0)


def test_daily_reset_preserves_signal_columns() -> None:
    signals = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02"]),
            "Open": [100.0],
            "Close": [101.0],
            "CashRate": [4.0],
            "TargetWeight": [0.8],
            "FearScore": [0.7],
        }
    )
    transformed = daily_reset_instrument(signals, multiple=2.0)
    assert transformed.loc[0, "TargetWeight"] == pytest.approx(0.8)
    assert transformed.loc[0, "FearScore"] == pytest.approx(0.7)
    assert transformed.loc[0, "UnderlyingClose"] == pytest.approx(101.0)
