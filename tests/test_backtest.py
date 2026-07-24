import pandas as pd

from stock_research.backtest import run_long_only


def test_net_roi_and_single_position():
    frame = pd.DataFrame({
        "날짜": pd.date_range("2024-01-01", periods=4),
        "종가": [10.0, 11.0, 12.0, 20.0],
        "buy": [True, True, False, False],
        "sell": [False, False, False, True],
    })

    result = run_long_only(
        frame,
        None,
        lambda row, _: bool(row["buy"]),
        lambda row, _: bool(row["sell"]),
        initial_cash=100.0,
    )
    assert result.summary.buys == 1
    assert result.summary.roi_percent == 100.0
    assert list(result.trades["액션"]) == ["BUY", "SELL"]
