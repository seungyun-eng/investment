from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tsla_integrated.cycle_profit import (
    CycleProfitParams,
    run_cycle_profit_backtest,
)


def _signals(
    prices: list[float],
    *,
    bearish: list[bool] | None = None,
    drawdowns: list[float] | None = None,
) -> pd.DataFrame:
    periods = len(prices)
    bear = bearish or [False] * periods
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=periods, freq="B"),
            "Open": prices,
            "Close": prices,
            "CompositeScore": [0.0 if value else 1.0 for value in bear],
            "DownsideProbability21": [0.0] * periods,
            "PriceDrawdown63": drawdowns or [0.0] * periods,
            "BearRegime": bear,
        }
    )


def test_actual_reentry_starts_new_cycle_above_legacy_average_cost() -> None:
    params = CycleProfitParams(
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(1.0,),
        profit_targets=(0.50,),
        cycle_sell_fractions=(0.50,),
        minimum_sessions_between_buys=0,
    )
    result = run_cycle_profit_backtest(
        _signals(
            [100.0, 150.0, 300.0, 300.0, 450.0],
            bearish=[False, False, True, False, False],
            drawdowns=[0.0, 0.0, -0.20, 0.0, 0.0],
        ),
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == [
        "BUY_INITIAL",
        "SELL_CYCLE_TP_50",
        "BUY_CYCLE_DRAWDOWN_10",
        "SELL_CYCLE_TP_50",
    ]
    second_cycle_buy = result.trades.iloc[2]
    assert second_cycle_buy["AverageCost"] < 300.0
    assert second_cycle_buy["CycleAnchor"] == pytest.approx(300.0)
    assert second_cycle_buy["CycleId"] == 2


def test_hybrid_ladder_sells_fixed_fractions_of_cycle_start_shares() -> None:
    params = CycleProfitParams(
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.25,),
        profit_targets=(0.50, 0.80),
        cycle_sell_fractions=(0.25, 0.25),
    )
    result = run_cycle_profit_backtest(
        _signals([100.0, 150.0, 180.0]),
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == [
        "BUY_INITIAL",
        "SELL_CYCLE_TP_50",
        "SELL_CYCLE_TP_80",
    ]
    assert result.trades["SoldShares"].tolist() == pytest.approx([0.0, 0.25, 0.25])
    assert result.trades.iloc[-1]["Shares"] == pytest.approx(0.50)


def test_profit_target_cannot_repeat_without_an_actual_reentry_buy() -> None:
    params = CycleProfitParams(
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.25,),
        profit_targets=(0.50,),
        cycle_sell_fractions=(0.50,),
    )
    result = run_cycle_profit_backtest(
        _signals([100.0, 150.0, 200.0, 300.0]),
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY_INITIAL", "SELL_CYCLE_TP_50"]
    assert result.summary.completed_trades == 1


def test_cycle_sell_fractions_cannot_exceed_cycle_shares() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        CycleProfitParams(
            profit_targets=(0.50, 0.80),
            cycle_sell_fractions=(0.60, 0.50),
        )
