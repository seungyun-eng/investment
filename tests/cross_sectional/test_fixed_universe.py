from __future__ import annotations

import pandas as pd

from stock_research.cross_sectional.fixed_universe import (
    build_staggered_equal_slot_equity,
)


def test_staggered_equal_slot_reserves_cash_until_listing() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "Ticker": ["A", "A", "IPO"],
            "Open": [10.0, 11.0, 20.0],
            "Close": [10.0, 12.0, 22.0],
        }
    )
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])

    curve, audit = build_staggered_equal_slot_equity(
        panel,
        reference_dates=pd.DatetimeIndex(dates),
        start="2020-01-02",
        end="2020-01-03",
        initial_capital=100.0,
        transaction_cost_bps=0.0,
    )

    assert curve["Equity"].tolist() == [100.0, 115.0]
    ipo = audit.loc[audit["Ticker"].eq("IPO")].iloc[0]
    assert ipo["BuyDate"] == pd.Timestamp("2020-01-03")
    assert ipo["ReservedSlot"] == 50.0
