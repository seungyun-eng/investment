from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tsla_integrated.score_accumulation import (
    ScoreAccumulationParams,
    run_score_accumulation_backtest,
)


def _signals(
    opens: list[float],
    closes: list[float],
    *,
    scores: list[float] | None = None,
    downside: list[float] | None = None,
    drawdowns: list[float] | None = None,
) -> pd.DataFrame:
    periods = len(opens)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=periods, freq="B"),
            "Open": opens,
            "Close": closes,
            "CompositeScore": scores or [0.6] * periods,
            "DownsideProbability21": downside or [0.2] * periods,
            "PriceDrawdown63": drawdowns or [0.0] * periods,
        }
    )


def test_bearish_drawdown_buys_at_next_open() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=0.25,
        bearish_score_max=0.45,
        downside_probability_min=0.55,
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.25,),
        profit_targets=(),
        profit_sell_fractions=(),
        minimum_sessions_between_buys=0,
    )
    signals = _signals(
        [100.0, 90.0, 80.0],
        [100.0, 90.0, 80.0],
        scores=[0.6, 0.4, 0.4],
        downside=[0.2, 0.8, 0.8],
        drawdowns=[0.0, -0.15, -0.20],
    )
    result = run_score_accumulation_backtest(
        signals,
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY_INITIAL", "BUY_DRAWDOWN_10"]
    assert result.trades["Open"].tolist() == [100.0, 80.0]


def test_half_position_is_sold_at_fifty_percent_profit() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=1.0,
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.25,),
        profit_targets=(0.50,),
        profit_sell_fractions=(0.50,),
    )
    signals = _signals(
        [100.0, 149.0, 150.0, 160.0],
        [100.0, 149.0, 150.0, 160.0],
    )
    result = run_score_accumulation_backtest(
        signals,
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY_INITIAL", "SELL_TP_50"]
    assert result.trades.iloc[-1]["Open"] == 150.0
    assert result.trades.iloc[-1]["Shares"] == pytest.approx(0.5)
    assert result.summary.final_value == pytest.approx(155.0)
    assert result.summary.roi_percent == pytest.approx(55.0)


def test_scale_in_requires_both_score_and_downside_confirmation() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=0.25,
        bearish_score_max=0.45,
        downside_probability_min=0.70,
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.25,),
        profit_targets=(),
        profit_sell_fractions=(),
        minimum_sessions_between_buys=0,
    )
    signals = _signals(
        [100.0, 90.0, 80.0],
        [100.0, 90.0, 80.0],
        scores=[0.6, 0.4, 0.4],
        downside=[0.2, 0.6, 0.6],
        drawdowns=[0.0, -0.15, -0.20],
    )
    result = run_score_accumulation_backtest(
        signals,
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY_INITIAL"]


def test_scale_in_can_be_restricted_to_bear_regime() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=0.25,
        bearish_score_max=1.0,
        downside_probability_min=0.0,
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.25,),
        profit_targets=(),
        profit_sell_fractions=(),
        minimum_sessions_between_buys=0,
        require_bear_regime_for_scale_in=True,
    )
    signals = _signals(
        [100.0, 90.0, 80.0, 70.0],
        [100.0, 90.0, 80.0, 70.0],
        drawdowns=[0.0, -0.15, -0.20, -0.25],
    )
    signals["BearRegime"] = [False, False, True, True]
    result = run_score_accumulation_backtest(
        signals,
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY_INITIAL", "BUY_DRAWDOWN_10"]
    assert result.trades.iloc[-1]["Open"] == 70.0


def test_cash_investor_enters_after_bear_end_signal() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=0.0,
        bearish_score_max=1.0,
        downside_probability_min=0.0,
        drawdown_levels=(-0.10, -0.20, -0.30),
        buy_equity_fractions=(0.15, 0.20, 0.30),
        profit_targets=(0.50,),
        profit_sell_fractions=(0.50,),
        bear_end_entry_fraction=1.0,
        require_bear_regime_for_scale_in=True,
    )
    signals = _signals(
        [100.0, 80.0, 90.0, 135.0],
        [100.0, 80.0, 90.0, 135.0],
        drawdowns=[0.0, -0.20, -0.10, 0.0],
    )
    signals["BearRegime"] = [False, True, True, False]
    signals["BearEndBuySignal"] = [False, True, False, False]
    result = run_score_accumulation_backtest(
        signals,
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY_BEAR_END", "SELL_TP_50"]
    assert result.trades["Open"].tolist() == [90.0, 135.0]
    assert result.summary.roi_percent == pytest.approx(50.0)


def test_bear_regime_ladder_resets_after_price_and_regime_recover() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=0.25,
        bearish_score_max=1.0,
        downside_probability_min=0.0,
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.10,),
        profit_targets=(),
        profit_sell_fractions=(),
        minimum_sessions_between_buys=0,
        require_bear_regime_for_scale_in=True,
    )
    signals = _signals(
        [100.0, 90.0, 100.0, 90.0, 80.0],
        [100.0, 90.0, 100.0, 90.0, 80.0],
        drawdowns=[0.0, -0.15, -0.02, -0.15, -0.20],
    )
    signals["BearRegime"] = [False, True, False, True, True]
    result = run_score_accumulation_backtest(
        signals,
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == [
        "BUY_INITIAL",
        "BUY_DRAWDOWN_10",
        "BUY_DRAWDOWN_10",
    ]


def test_high_priced_dip_addition_does_not_restart_fifty_percent_sale() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=1.0,
        bearish_score_max=1.0,
        downside_probability_min=0.0,
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.10,),
        profit_targets=(0.50,),
        profit_sell_fractions=(0.50,),
        minimum_sessions_between_buys=0,
        require_bear_regime_for_scale_in=True,
    )
    signals = _signals(
        [100.0, 150.0, 300.0, 301.0, 302.0],
        [100.0, 150.0, 300.0, 301.0, 302.0],
        drawdowns=[0.0, 0.0, -0.20, -0.20, -0.20],
    )
    signals["BearRegime"] = [False, False, True, True, True]
    result = run_score_accumulation_backtest(
        signals,
        params,
        initial_capital=100.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == [
        "BUY_INITIAL",
        "SELL_TP_50",
        "BUY_DRAWDOWN_10",
    ]


def test_roi_uses_total_injected_capital() -> None:
    params = ScoreAccumulationParams(
        initial_allocation=1.0,
        drawdown_levels=(-0.10,),
        buy_equity_fractions=(0.25,),
        profit_targets=(),
        profit_sell_fractions=(),
    )
    result = run_score_accumulation_backtest(
        _signals([100.0, 120.0], [100.0, 120.0]),
        params,
        initial_capital=200.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.summary.total_injected == 200.0
    assert result.summary.roi_percent == pytest.approx(
        (result.summary.final_value / result.summary.total_injected - 1) * 100
    )
