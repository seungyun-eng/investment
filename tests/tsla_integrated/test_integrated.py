from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tsla_integrated.config import (
    IntegratedParams,
    IntegratedSettings,
)
from stock_research.tsla_integrated.downside import (
    DOWNSIDE_FEATURES,
    add_strict_oos_downside_probability,
)
from stock_research.tsla_integrated.features import build_integrated_features
from stock_research.tsla_integrated.optimization import (
    DEFAULT_FOLDS,
    _alpha_robustness_tiers,
    buy_and_hold_params,
)
from stock_research.tsla_integrated.portfolio import run_integrated_backtest
from stock_research.tsla_integrated.strategy import (
    generate_consensus_signals,
    generate_integrated_signals,
)


def _prices(periods: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=periods, freq="B")
    close = pd.Series(range(100, 100 + periods), dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close.values,
            "High": close.values + 1,
            "Low": close.values - 1,
            "Close": close.values,
            "Volume": 1_000,
        }
    )


def _financials() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2019-03-31",
            "2019-06-30",
            "2019-09-30",
            "2019-12-31",
            "2020-03-31",
        ]
    )
    return pd.DataFrame(
        {
            "Date": dates,
            "Revenue": [100, 110, 120, 130, 200],
            "Gross Profit": [20, 22, 24, 26, 50],
            "Operating Income": [10, 11, 12, 13, 30],
            "Net Income": [8, 9, 10, 11, 25],
            "Cash On Hand": [20, 21, 22, 23, 40],
            "Total Liabilities": [10, 10, 10, 10, 10],
            "Cash Flow From Operating Activities": [15, 16, 17, 18, 35],
            "Capital Expenditures": [-5, -5, -5, -5, -7],
        }
    )


def _macro(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": prices["Date"],
            "VIX": 20.0,
            "MacroConfirmationScore": 0.3,
            "RiskProbability_63": 0.2,
            "RiskProbability_126": 0.2,
        }
    )


def _filing_features(ticker: str = "TSLA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": [ticker],
            "AvailableDate": ["2020-01-10"],
            "RevenueGrowthYoYFiled": [0.9],
            "OperatingMargin": [0.9],
            "FreeCashFlowMargin": [0.9],
            "GoingConcernFlag": [False],
            "MaterialWeaknessFlag": [False],
            "RestatementFlag": [False],
        }
    )


def test_filing_features_override_macrotrends_margin() -> None:
    prices = _prices()
    baseline = build_integrated_features(prices, _financials(), _macro(prices))
    augmented = build_integrated_features(
        prices,
        _financials(),
        _macro(prices),
        filing_features=_filing_features(),
        ticker="TSLA",
    )
    after = augmented[augmented["Date"] >= "2020-01-13"]
    baseline_after = baseline[baseline["Date"] >= "2020-01-13"]
    assert (after["OperatingMargin"] == 0.9).all()
    assert not (baseline_after["OperatingMargin"] == 0.9).all()


def test_filing_critical_flag_blocks_entries_and_forces_exit() -> None:
    params = IntegratedParams()
    flagged = _reentry_probe_row(
        DownsideProbability21=0.20,
        VixPercentile=0.2,
        MacroConfirmationScore=0.2,
        ModelRisk=0.2,
    )
    flagged["FilingCriticalFlag"] = True
    signals = generate_integrated_signals(flagged, params)
    assert not bool(signals.loc[0, "BuySignal"])
    assert not bool(signals.loc[0, "RecoveryBuySignal"])
    assert bool(signals.loc[0, "SellSignal"])


def test_short_signal_ignores_strong_financials_when_technicals_are_bearish() -> None:
    # Regression test: TSLA's revenue/margins kept growing through the 2022
    # crash, which pinned CompositeScore above 0.5 all year (see the
    # walk-forward diagnosis) so no sampled short_threshold could ever fire.
    # ShortSignal must key off TacticalScore (technical+macro only), not the
    # fundamentals-anchored CompositeScore.
    params = IntegratedParams()
    row = _reentry_probe_row(
        Trend50=-0.20,
        RevenueGrowthYoY=0.5,
        OperatingMargin=0.3,
        FreeCashFlowMargin=0.3,
        VixPercentile=0.9,
        MacroConfirmationScore=0.9,
        ModelRisk=0.9,
        DownsideProbability21=0.9,
    )
    signals = generate_integrated_signals(row, params)
    assert signals.loc[0, "FinancialScore"] > 0.8
    assert signals.loc[0, "CompositeScore"] > params.short_threshold
    assert signals.loc[0, "TacticalScore"] <= params.short_threshold
    assert bool(signals.loc[0, "ShortSignal"])


def test_hy_spread_stress_lowers_macro_score() -> None:
    params = IntegratedParams()
    calm = _reentry_probe_row(HYSpreadPercentile=0.0, YieldCurveInverted=0.0)
    stressed = _reentry_probe_row(HYSpreadPercentile=1.0, YieldCurveInverted=1.0)
    calm_signals = generate_integrated_signals(calm, params)
    stressed_signals = generate_integrated_signals(stressed, params)
    assert (
        stressed_signals.loc[0, "MacroScore"] < calm_signals.loc[0, "MacroScore"]
    )


def test_financials_are_not_visible_before_release_lag() -> None:
    prices = _prices()
    features = build_integrated_features(
        prices, _financials(), _macro(prices), financial_release_lag_days=45
    )
    before = features[features["Date"] < "2020-05-15"]
    after = features[features["Date"] >= "2020-05-15"]
    assert (before["FinancialPeriodEnd"] <= pd.Timestamp("2019-12-31")).all()
    assert after.iloc[0]["FinancialPeriodEnd"] == pd.Timestamp("2020-03-31")


def test_backtest_executes_prior_signal_at_next_open() -> None:
    params = IntegratedParams(minimum_hold_sessions=1)
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "Open": [100.0, 110.0, 120.0],
            "Close": [105.0, 115.0, 125.0],
            "CompositeScore": [0.8, 0.2, 0.2],
            "BuySignal": [True, False, False],
            "SellSignal": [False, True, True],
        }
    )
    result = run_integrated_backtest(
        signals,
        params,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY", "SELL"]
    assert result.trades["Open"].tolist() == [110.0, 120.0]
    assert result.summary.roi_percent == pytest.approx((120 / 110 - 1) * 100)


def test_signal_function_is_deterministic() -> None:
    prices = _prices()
    features = build_integrated_features(prices, _financials(), _macro(prices))
    params = IntegratedParams()
    first = generate_integrated_signals(features, params)
    second = generate_integrated_signals(features, params)
    pd.testing.assert_series_equal(first["CompositeScore"], second["CompositeScore"])


def _reentry_probe_row(**overrides: float) -> pd.DataFrame:
    base = {
        "Date": pd.to_datetime(["2026-04-16"]),
        "RSI14": [50.0],
        "Trend50": [0.02],
        "MACD": [0.5],
        "MACDSignal": [1.0],
        "RevenueGrowthYoY": [0.1],
        "OperatingMargin": [0.1],
        "FreeCashFlowMargin": [0.1],
        "VixPercentile": [0.5],
        "MacroConfirmationScore": [0.5],
        "ModelRisk": [0.5],
        "DownsideProbability21": [0.5],
        "Return21": [0.05],
    }
    base.update({key: [value] for key, value in overrides.items()})
    return pd.DataFrame(base)


def test_recovery_buy_respects_reentry_downside_probability_gate() -> None:
    # Regression test: the 2026-04-16 TSLA holdout re-entry happened at
    # DownsideProbability21=0.481, above buy_downside_probability_max
    # (0.40), because recovery_buy checked trend only. It must now also be
    # blocked by reentry_downside_probability_max like a primary entry.
    params = IntegratedParams()
    blocked = generate_integrated_signals(
        _reentry_probe_row(DownsideProbability21=0.60), params
    )
    assert not bool(blocked.loc[0, "RecoveryBuySignal"])
    assert not bool(blocked.loc[0, "BuySignal"])


def test_recovery_buy_allows_reentry_within_risk_gates() -> None:
    params = IntegratedParams()
    allowed = generate_integrated_signals(
        _reentry_probe_row(
            DownsideProbability21=0.20,
            VixPercentile=0.2,
            MacroConfirmationScore=0.2,
            ModelRisk=0.2,
        ),
        params,
    )
    assert bool(allowed.loc[0, "RecoveryBuySignal"])
    assert bool(allowed.loc[0, "BuySignal"])


def test_consensus_requires_configured_entry_agreement() -> None:
    prices = _prices()
    features = build_integrated_features(prices, _financials(), _macro(prices))
    bullish = IntegratedParams(
        short_threshold=0.0,
        sell_threshold=0.01,
        buy_threshold=0.02,
        buy_macro_score_min=0.0,
        short_downside_probability_min=1.0,
        buy_downside_probability_max=0.99,
    )
    blocked = IntegratedParams(
        buy_downside_probability_max=0.0,
        reentry_downside_probability_max=0.0,
        trend_entry_threshold=1.0,
    )
    features["DownsideProbability21"] = 0.5
    members = [bullish, bullish, blocked]
    signals = generate_consensus_signals(
        features,
        members,
        entry_consensus=0.70,
    )
    assert (signals["BuyVote"] <= 2 / 3).all()
    assert not signals["BuySignal"].any()


def test_default_research_split_is_2019_2025_then_2026() -> None:
    settings = IntegratedSettings()
    assert settings.development_start == "2019-01-01"
    assert settings.development_end == "2025-12-31"
    assert settings.holdout_start == "2026-01-01"


def test_alpha_robustness_prioritizes_recent_repeatability() -> None:
    candidates = pd.DataFrame(
        {
            "Fold_2019_2021ExcessROI(%)": [1.0, -1.0, -1.0, 1.0],
            "Fold_2022_2023ExcessROI(%)": [1.0, 1.0, -1.0, -1.0],
            "Fold_2024_2025ExcessROI(%)": [1.0, 1.0, 1.0, -1.0],
        }
    )
    positive_counts, tiers = _alpha_robustness_tiers(
        candidates,
        DEFAULT_FOLDS,
    )
    assert positive_counts.tolist() == [3, 2, 1, 1]
    assert tiers.tolist() == [3, 2, 1, 0]


def test_buy_and_hold_benchmark_never_sells() -> None:
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "Open": [100.0, 50.0, 25.0],
            "Close": [100.0, 50.0, 25.0],
            "CompositeScore": [1.0, 1.0, 1.0],
            "BuySignal": [True, True, True],
            "SellSignal": [False, False, False],
        }
    )
    result = run_integrated_backtest(
        signals,
        buy_and_hold_params(),
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY"]
    assert result.summary.completed_trades == 0


def test_initial_long_enters_when_no_tactical_signal_exists() -> None:
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "Open": [100.0, 110.0, 120.0],
            "Close": [100.0, 115.0, 125.0],
            "CompositeScore": [0.5, 0.5, 0.5],
            "BuySignal": [False, False, False],
            "SellSignal": [False, False, False],
        }
    )
    result = run_integrated_backtest(
        signals,
        IntegratedParams(),
        transaction_cost_bps=0,
        slippage_bps=0,
        initial_long=True,
    )
    assert result.trades["Action"].tolist() == ["BUY"]
    assert result.trades.iloc[0]["Open"] == 110.0
    assert result.summary.roi_percent == pytest.approx((125 / 110 - 1) * 100)


def test_short_signal_executes_next_open_and_profits_from_decline() -> None:
    params = IntegratedParams(
        minimum_hold_sessions=1,
        short_stop_loss=0.50,
    )
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=4, freq="B"),
            "Open": [100.0, 100.0, 80.0, 70.0],
            "Close": [100.0, 90.0, 75.0, 70.0],
            "CompositeScore": [0.1, 0.1, 0.6, 0.6],
            "BuySignal": [False, False, False, False],
            "SellSignal": [True, True, False, False],
            "ShortSignal": [True, True, False, False],
            "CoverSignal": [False, False, True, True],
        }
    )
    result = run_integrated_backtest(
        signals,
        params,
        transaction_cost_bps=0,
        slippage_bps=0,
        annual_short_borrow_bps=0,
    )
    assert result.trades["Action"].tolist() == ["SHORT", "COVER"]
    assert result.trades["Open"].tolist() == [100.0, 70.0]
    assert result.summary.roi_percent == pytest.approx(30.0)


def test_short_leverage_scales_pnl_on_notional_exposure() -> None:
    params = IntegratedParams(
        minimum_hold_sessions=1,
        short_stop_loss=0.50,
        short_leverage=2.0,
    )
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=4, freq="B"),
            "Open": [100.0, 100.0, 80.0, 70.0],
            "Close": [100.0, 90.0, 75.0, 70.0],
            "CompositeScore": [0.1, 0.1, 0.6, 0.6],
            "BuySignal": [False, False, False, False],
            "SellSignal": [True, True, False, False],
            "ShortSignal": [True, True, False, False],
            "CoverSignal": [False, False, True, True],
        }
    )
    result = run_integrated_backtest(
        signals,
        params,
        transaction_cost_bps=0,
        slippage_bps=0,
        annual_short_borrow_bps=0,
    )
    # 1x leverage on the same price path returns +30% (see the unleveraged
    # short test above); 2x notional exposure should roughly double that.
    assert result.summary.roi_percent == pytest.approx(60.0)


def test_long_trailing_stop_uses_only_prior_peak() -> None:
    params = IntegratedParams(
        stop_loss=0.90,
        trailing_stop=0.10,
        minimum_hold_sessions=100,
    )
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "Open": [100.0, 100.0, 100.0],
            "Close": [100.0, 120.0, 105.0],
            "CompositeScore": [0.8, 0.8, 0.8],
            "BuySignal": [True, False, False],
            "SellSignal": [False, False, False],
        }
    )
    result = run_integrated_backtest(
        signals,
        params,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY", "SELL"]
    assert result.trades.iloc[-1]["Open"] == 100.0


def test_downside_probabilities_do_not_change_from_later_prices() -> None:
    periods = 800
    dates = pd.date_range("2018-01-02", periods=periods, freq="B")
    base = pd.DataFrame(
        {
            "Date": dates,
            "Close": 100 + pd.Series(range(periods)) * 0.1,
        }
    )
    for offset, column in enumerate(DOWNSIDE_FEATURES):
        base[column] = (
            pd.Series(range(periods), dtype=float).mod(17 + offset) / 20
        )
    base.loc[::37, "Close"] *= 0.80
    changed = base.copy()
    changed.loc[changed.index >= 720, "Close"] *= 0.25
    first = add_strict_oos_downside_probability(
        base, minimum_training_rows=200
    )
    second = add_strict_oos_downside_probability(
        changed, minimum_training_rows=200
    )
    pd.testing.assert_series_equal(
        first.loc[:650, "DownsideProbability21"],
        second.loc[:650, "DownsideProbability21"],
    )
    pd.testing.assert_series_equal(
        first["DownsideProbability21"].rename(
            "TslaDownsideProbability21"
        ),
        first["TslaDownsideProbability21"],
    )
