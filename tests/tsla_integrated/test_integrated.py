from __future__ import annotations

import numpy as np
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
    INTEGER_PARAM_NAMES,
    _alpha_robustness_tiers,
    buy_and_hold_params,
    params_from_candidate_row,
    sample_params,
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
            "EBITDA": [12, 13, 15, 16, 36],
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


def test_ebitda_margin_and_growth_are_computed_point_in_time() -> None:
    # EBITDA is already crawled into the source workbook (see
    # _financial_features in features.py) but was never turned into a
    # feature. Confirm the margin (EBITDA/Revenue) and YoY growth compute
    # correctly and only become visible after the release lag, same as the
    # other Macrotrends-derived ratios.
    prices = _prices()
    features = build_integrated_features(
        prices, _financials(), _macro(prices), financial_release_lag_days=45
    )
    after = features[features["Date"] >= "2020-05-15"]
    first_row = after.iloc[0]
    assert first_row["EBITDAMargin"] == pytest.approx(36 / 200)
    assert first_row["EBITDAGrowthYoY"] == pytest.approx(36 / 12 - 1)


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
        EBITDAGrowthYoY=0.5,
        EBITDAMargin=0.3,
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


def test_market_bear_confirms_short_even_when_tactical_score_is_borderline() -> None:
    # A confirmed SPY-wide bear regime (fixed 200/50-session rule, not fit
    # to TSLA's own history) should be enough to justify a short on its own,
    # since TSLA's training window may never have lived through a real bear
    # market to calibrate short_threshold/short_macro_score_max against.
    params = IntegratedParams(short_threshold=0.05, short_macro_score_max=0.05)
    row = _reentry_probe_row(
        Trend50=-0.20,
        DownsideProbability21=0.9,
        MarketExposureScale=0.05,
    )
    signals = generate_integrated_signals(row, params)
    assert signals.loc[0, "TacticalScore"] > params.short_threshold
    assert bool(signals.loc[0, "MarketBearConfirmed"])
    assert bool(signals.loc[0, "ShortSignal"])


def test_market_bear_not_confirmed_when_exposure_scale_is_high() -> None:
    params = IntegratedParams(short_threshold=0.05, short_macro_score_max=0.05)
    row = _reentry_probe_row(
        Trend50=-0.20,
        DownsideProbability21=0.9,
        MarketExposureScale=0.95,
    )
    signals = generate_integrated_signals(row, params)
    assert not bool(signals.loc[0, "MarketBearConfirmed"])
    assert not bool(signals.loc[0, "ShortSignal"])


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


def test_seasonal_rebound_buys_in_january_after_a_brutal_prior_year() -> None:
    # Regression test for the 2023-01 walk-forward: TSLA bottomed 2023-01-06
    # at $113 after a -65% 2022, but Trend50 (and every sampled
    # trend_entry_window) didn't turn positive until 2023-01-26 at $160, and
    # the ordinary reentry downside-probability gate (<= 0.55) was also
    # still shut. Trend10 turned positive just 2 sessions off the bottom
    # (2023-01-09 at $120) -- a January date following a >=20%-down prior
    # calendar year should ride that fast confirmation in at a looser
    # downside probability than the ordinary reentry gate permits.
    params = IntegratedParams()
    row = _reentry_probe_row(
        Date=pd.Timestamp("2023-01-10"),
        Trend10=0.05,
        Trend50=-0.20,
        DownsideProbability21=0.65,
        PriorCalendarYearReturn=-0.65,
    )
    signals = generate_integrated_signals(row, params)
    assert 0.65 > params.reentry_downside_probability_max
    assert bool(signals.loc[0, "JanuaryReboundWindow"])
    assert bool(signals.loc[0, "SeasonalReboundBuySignal"])
    assert bool(signals.loc[0, "BuySignal"])


def test_seasonal_rebound_does_not_fire_outside_january() -> None:
    params = IntegratedParams()
    row = _reentry_probe_row(
        Date=pd.Timestamp("2023-02-10"),
        Trend10=0.05,
        DownsideProbability21=0.65,
        PriorCalendarYearReturn=-0.65,
    )
    signals = generate_integrated_signals(row, params)
    assert not bool(signals.loc[0, "JanuaryReboundWindow"])
    assert not bool(signals.loc[0, "SeasonalReboundBuySignal"])
    assert not bool(signals.loc[0, "BuySignal"])


def test_seasonal_rebound_does_not_fire_after_a_mild_prior_year() -> None:
    params = IntegratedParams()
    row = _reentry_probe_row(
        Date=pd.Timestamp("2023-01-10"),
        Trend10=0.05,
        DownsideProbability21=0.65,
        PriorCalendarYearReturn=-0.05,
    )
    signals = generate_integrated_signals(row, params)
    assert not bool(signals.loc[0, "JanuaryReboundWindow"])
    assert not bool(signals.loc[0, "BuySignal"])


def _bear_regime_probe_frame(downside_values: list[float]) -> pd.DataFrame:
    n = len(downside_values)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2022-08-01", periods=n, freq="B"),
            "RSI14": [80.0] * n,
            "Trend50": [-0.20] * n,
            "Trend200": [-0.25] * n,
            "Trend10": [-0.05] * n,
            "MACD": [-0.5] * n,
            "MACDSignal": [0.0] * n,
            "RevenueGrowthYoY": [0.1] * n,
            "OperatingMargin": [0.1] * n,
            "FreeCashFlowMargin": [0.1] * n,
            "VixPercentile": [0.9] * n,
            "MacroConfirmationScore": [0.9] * n,
            "ModelRisk": [0.9] * n,
            "DownsideProbability21": downside_values,
            "Return21": [-0.10] * n,
        }
    )


def test_contrarian_buy_fires_after_bear_regime_washout_stabilizes() -> None:
    # A confirmed bear regime (Trend200 <= -0.15) plus a washed-out
    # TacticalScore should be allowed to buy once DownsideProbability21 has
    # come off its rolling peak (stabilizing), even though the ordinary
    # trend-following gates (bullish_trend requires Trend50 >= 0) stay shut
    # the whole time -- see the walk-forward ablation at
    # BEAR_REGIME_TREND_MAX for why an always-off momentum-only engine
    # missed the 2022-2023 recovery.
    params = IntegratedParams()
    downside = [0.3 + 0.6 * i / 14 for i in range(15)] + [0.85, 0.75, 0.65, 0.55, 0.50]
    row = _bear_regime_probe_frame(downside)
    signals = generate_integrated_signals(row, params)
    last = signals.iloc[-1]
    assert last["TacticalScore"] <= params.contrarian_buy_tactical_max
    assert bool(last["BearRegime"])
    assert bool(last["ContrarianBuySignal"])
    assert bool(last["BuySignal"])


def test_contrarian_buy_does_not_fire_outside_any_washout_regime() -> None:
    params = IntegratedParams()
    downside = [0.3 + 0.6 * i / 14 for i in range(15)] + [0.85, 0.75, 0.65, 0.55, 0.50]
    row = _bear_regime_probe_frame(downside)
    row["Trend200"] = 0.05
    row["Trend50"] = 0.05
    signals = generate_integrated_signals(row, params)
    last = signals.iloc[-1]
    assert not bool(last["BearRegime"])
    assert not bool(last["FastWashout"])
    assert not bool(last["ContrarianBuySignal"])


def test_contrarian_buy_fires_via_fast_washout_without_deep_bear_regime() -> None:
    # Oracle audit regression: a perfect-foresight walk of the 10 largest
    # TSLA swings found only 4/10 of the best long entries had a confirmed
    # Trend200 bear regime -- the rest were sharper pullbacks visible on
    # Trend50 that never dragged the much slower 200-session average down
    # far enough. FastWashout (Trend50 <= -15%) must independently qualify
    # a contrarian buy even when Trend200 alone would not.
    params = IntegratedParams()
    downside = [0.3 + 0.6 * i / 14 for i in range(15)] + [0.85, 0.75, 0.65, 0.55, 0.50]
    row = _bear_regime_probe_frame(downside)
    row["Trend200"] = 0.05
    signals = generate_integrated_signals(row, params)
    last = signals.iloc[-1]
    assert not bool(last["BearRegime"])
    assert bool(last["FastWashout"])
    assert bool(last["ContrarianBuySignal"])


def test_contrarian_buy_fires_via_rsi_oversold_when_macro_is_calm() -> None:
    # Oracle audit regression: 9/10 of the best long entries had
    # MarketExposureScale == 1.0 -- the broad market/macro backdrop was
    # calm, so TacticalScore (which blends in MacroScore) often never drops
    # low enough to confirm a TSLA-specific washout. A directly oversold
    # RSI14 must independently qualify a contrarian buy even when
    # TacticalScore stays elevated because macro conditions are fine.
    params = IntegratedParams()
    downside = [0.3 + 0.6 * i / 14 for i in range(15)] + [0.85, 0.75, 0.65, 0.55, 0.50]
    row = _bear_regime_probe_frame(downside)
    row["RSI14"] = 25.0
    row["VixPercentile"] = 0.1
    row["MacroConfirmationScore"] = 0.1
    row["ModelRisk"] = 0.1
    signals = generate_integrated_signals(row, params)
    last = signals.iloc[-1]
    assert bool(last["BearRegime"])
    assert last["TacticalScore"] > params.contrarian_buy_tactical_max
    assert bool(last["ContrarianBuySignal"])


def test_contrarian_buy_does_not_fire_while_downside_probability_still_rising() -> None:
    params = IntegratedParams()
    downside = [0.3 + 0.6 * i / 19 for i in range(20)]
    row = _bear_regime_probe_frame(downside)
    signals = generate_integrated_signals(row, params)
    last = signals.iloc[-1]
    assert bool(last["BearRegime"])
    assert not bool(last["ContrarianBuySignal"])


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


def test_params_from_candidate_row_round_trips_every_integer_field() -> None:
    # Regression test: a candidates DataFrame (as produced by
    # optimize_on_development, and consumed via generate_consensus_signals
    # for the consensus holdout path) stores every field as float64. Any
    # dataclass field that must stay an int (e.g. contrarian_lookback_sessions
    # feeding a pandas .rolling(min_periods=...) call) has to be listed in
    # INTEGER_PARAM_NAMES or the round-trip silently hands back a float and
    # breaks downstream, as generate_consensus_signals did when
    # contrarian_lookback_sessions was added but not registered here.
    params = sample_params(np.random.default_rng(0))
    row = pd.Series(params.as_dict(), dtype="float64")
    restored = params_from_candidate_row(row)
    for name in INTEGER_PARAM_NAMES:
        assert isinstance(getattr(restored, name), int), name
    assert restored == params


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


def test_contrarian_leverage_scales_pnl_on_notional_exposure() -> None:
    # contrarian_leverage mirrors short_leverage but for the specific,
    # oracle-validated entry path (oversold + regime-confirmed washout):
    # margin borrowed against the same capital, applied only when the prior
    # day's ContrarianBuySignal (not just any BuySignal) triggered the entry.
    params = IntegratedParams(
        minimum_hold_sessions=1,
        stop_loss=0.50,
        trailing_stop=0.50,
        contrarian_leverage=2.0,
    )
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=4, freq="B"),
            "Open": [100.0, 100.0, 120.0, 130.0],
            "Close": [100.0, 110.0, 125.0, 130.0],
            "CompositeScore": [0.6, 0.6, 0.1, 0.1],
            "BuySignal": [True, True, False, False],
            "ContrarianBuySignal": [True, True, False, False],
            "SellSignal": [False, False, True, True],
        }
    )
    result = run_integrated_backtest(
        signals,
        params,
        transaction_cost_bps=0,
        slippage_bps=0,
        annual_short_borrow_bps=0,
    )
    assert result.trades["Action"].tolist() == ["BUY", "SELL"]
    assert result.trades["Open"].tolist() == [100.0, 130.0]
    assert result.summary.roi_percent == pytest.approx(60.0)


def test_contrarian_leverage_does_not_apply_to_non_contrarian_buys() -> None:
    params = IntegratedParams(
        minimum_hold_sessions=1,
        stop_loss=0.50,
        trailing_stop=0.50,
        contrarian_leverage=2.0,
    )
    signals = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=4, freq="B"),
            "Open": [100.0, 100.0, 120.0, 130.0],
            "Close": [100.0, 110.0, 125.0, 130.0],
            "CompositeScore": [0.6, 0.6, 0.1, 0.1],
            "BuySignal": [True, True, False, False],
            "ContrarianBuySignal": [False, False, False, False],
            "SellSignal": [False, False, True, True],
        }
    )
    result = run_integrated_backtest(
        signals,
        params,
        transaction_cost_bps=0,
        slippage_bps=0,
        annual_short_borrow_bps=0,
    )
    assert result.summary.roi_percent == pytest.approx(30.0)


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
