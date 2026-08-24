from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import IntegratedParams

# A fixed (not fit-to-TSLA) -20% prior-calendar-year-return trigger for the
# December-tax-loss-selling / January-rebound window: the 2019-2025
# development window contains exactly one such year (2022), far too few
# examples to responsibly calibrate this threshold via the candidate search
# -- same reasoning as the fixed SPY 200/50 trend rule in features.py.
JANUARY_REBOUND_PRIOR_YEAR_RETURN_MAX = -0.20

# A fixed (not fit-to-TSLA) -15% Trend200 cutoff for a confirmed bear/
# correction regime -- price 15% below its own 200-session average is a
# commonly used technical threshold, not searched over, because the
# 2019-2025 development window contains only a couple of episodes this deep
# (2022 and the 2020 crash), too few to responsibly calibrate an exact cutoff
# via the candidate search. A regime-switch walk-forward ablation (bull
# years use momentum unchanged; a contrarian "buy the washout" overlay only
# activates once BearRegime is confirmed) improved Fold_2022_2023 ROI from
# 54% to 70-100%+ and Fold_2024_2025 from 78% to 100-160%, without damaging
# the momentum engine's performance in the 2019-2021 bull fold (1360% vs.
# 1340-1500%) -- unlike an always-on contrarian strategy, which gutted the
# 2019-2021 bull fold to 130-860% by fading genuine trend continuation.
BEAR_REGIME_TREND_MAX = -0.15

# A fixed (not fit-to-TSLA) -15% Trend50 cutoff, same reasoning as
# BEAR_REGIME_TREND_MAX but on a faster window: a perfect-foresight audit of
# the 10 largest TSLA swings (2019-2026) found the best long entries were
# overwhelmingly RSI-oversold TSLA-specific washouts, not necessarily deep,
# sustained bears -- only 4/10 had Trend200 <= BEAR_REGIME_TREND_MAX. Trend50
# catches the other, sharper pullbacks that never drag the much slower
# 200-session average down far enough to confirm.
FAST_WASHOUT_TREND_MAX = -0.15


def _clip(value: pd.Series) -> pd.Series:
    return value.clip(0.0, 1.0).fillna(0.5)


def generate_integrated_signals(
    features: pd.DataFrame,
    params: IntegratedParams,
) -> pd.DataFrame:
    """Canonical signal function shared by optimization and simulation."""

    frame = features.copy().sort_values("Date").reset_index(drop=True)
    rsi_value = pd.to_numeric(frame["RSI14"], errors="coerce")
    rsi_score = _clip(
        (params.rsi_overbought - rsi_value)
        / max(params.rsi_overbought - params.rsi_oversold, 1.0)
    )
    trend_score = (
        0.6 * (pd.to_numeric(frame["Trend50"], errors="coerce") > 0).astype(float)
        + 0.4
        * (pd.to_numeric(frame["MACD"], errors="coerce")
           > pd.to_numeric(frame["MACDSignal"], errors="coerce")).astype(float)
    )
    frame["TechnicalScore"] = _clip(0.55 * trend_score + 0.45 * rsi_score)

    revenue_score = _clip(
        (pd.to_numeric(frame["RevenueGrowthYoY"], errors="coerce") + 0.20) / 0.60
    )
    margin_score = _clip(
        (pd.to_numeric(frame["OperatingMargin"], errors="coerce") + 0.10) / 0.30
    )
    cash_score = _clip(
        (pd.to_numeric(frame["FreeCashFlowMargin"], errors="coerce") + 0.15) / 0.35
    )
    # EBITDA was already crawled into the source workbook but never fed into
    # FinancialScore -- growth (EBITDA YoY) and quality (EBITDA margin) of
    # earnings before financing/accounting choices, which revenue growth and
    # a GAAP operating margin alone don't capture.
    ebitda_growth_score = _clip(
        (
            pd.to_numeric(
                frame.get("EBITDAGrowthYoY", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            + 0.20
        )
        / 0.60
    )
    ebitda_margin_score = _clip(
        (
            pd.to_numeric(
                frame.get("EBITDAMargin", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            + 0.05
        )
        / 0.35
    )
    frame["FinancialScore"] = _clip(
        0.25 * revenue_score
        + 0.15 * margin_score
        + 0.15 * cash_score
        + 0.25 * ebitda_growth_score
        + 0.20 * ebitda_margin_score
    )

    # Credit-spread and yield-curve stress default to neutral (0.5) when the
    # extra macro series was not supplied, so this stays a no-op for callers
    # that only pass the original VIX/ModelRisk-based macro inputs.
    hy_spread_stress = pd.to_numeric(
        frame.get("HYSpreadPercentile", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(0.5)
    yield_curve_stress = pd.to_numeric(
        frame.get("YieldCurveInverted", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(0.5)
    macro_stress = _clip(
        0.30 * pd.to_numeric(frame["VixPercentile"], errors="coerce")
        + 0.20 * pd.to_numeric(frame["MacroConfirmationScore"], errors="coerce")
        + 0.20 * pd.to_numeric(frame["ModelRisk"], errors="coerce")
        + 0.20 * hy_spread_stress
        + 0.10 * yield_curve_stress
    )
    frame["MacroScore"] = 1.0 - macro_stress
    frame["CompositeScore"] = (
        params.technical_weight * frame["TechnicalScore"]
        + params.financial_weight * frame["FinancialScore"]
        + params.macro_weight * frame["MacroScore"]
    ).clip(0.0, 1.0)
    # Short/cover decisions use technical+macro only, excluding
    # FinancialScore. TSLA's revenue/margins kept growing through the 2022
    # crash even as the stock fell ~65%, which kept CompositeScore pinned
    # above ~0.5 all year -- above every short_threshold the search could
    # sample -- so a fundamentals-anchored score can never recognize a
    # price-driven selloff worth shorting.
    tactical_weight_total = max(
        params.technical_weight + params.macro_weight, 1e-9
    )
    frame["TacticalScore"] = (
        (
            params.technical_weight * frame["TechnicalScore"]
            + params.macro_weight * frame["MacroScore"]
        )
        / tactical_weight_total
    ).clip(0.0, 1.0)
    downside_column = (
        "DownsideProbability21"
        if "DownsideProbability21" in frame
        else "TslaDownsideProbability21"
    )
    downside_probability = pd.to_numeric(
        frame.get(downside_column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    entry_trend = pd.to_numeric(
        frame[f"Trend{params.trend_entry_window}"],
        errors="coerce",
    )
    exit_trend = pd.to_numeric(
        frame[f"Trend{params.trend_exit_window}"],
        errors="coerce",
    )
    bullish_trend = (
        (entry_trend >= params.trend_entry_threshold)
        & (
            pd.to_numeric(frame["MACD"], errors="coerce")
            > pd.to_numeric(frame["MACDSignal"], errors="coerce")
        )
    )
    primary_buy = (
        (frame["CompositeScore"] >= params.buy_threshold)
        & bullish_trend
        & (frame["MacroScore"] >= params.buy_macro_score_min)
        & (downside_probability <= params.buy_downside_probability_max)
    )
    return21 = pd.to_numeric(frame.get("Return21"), errors="coerce")
    # Re-entry after a stop/exit must still clear the same kind of risk
    # checks as a primary entry, just at the (looser) reentry_* thresholds
    # instead of trend alone -- trend-only reentry bought back into TSLA in
    # 2026-04 at a downside probability (0.48) that the primary entry gate
    # would have rejected (buy_downside_probability_max=0.40), then rode the
    # subsequent decline for the rest of the holdout.
    recovery_buy = (
        (entry_trend >= params.trend_entry_threshold)
        & (frame["MacroScore"] >= params.reentry_macro_score_min)
        & (downside_probability <= params.reentry_downside_probability_max)
        & (return21 >= params.reentry_return21_min)
    )
    critical_flag = frame.get(
        "FilingCriticalFlag", pd.Series(False, index=frame.index)
    ).fillna(False)
    frame["FilingCriticalFlag"] = critical_flag
    prior_year_return = pd.to_numeric(
        frame.get("PriorCalendarYearReturn", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    frame["JanuaryReboundWindow"] = pd.to_datetime(
        frame["Date"], errors="coerce"
    ).dt.month.eq(1) & (prior_year_return <= JANUARY_REBOUND_PRIOR_YEAR_RETURN_MAX)
    # After a year severe enough to trigger heavy tax-loss selling, both the
    # downside-probability gate AND the slow trend_entry_window filter stay
    # shut through most of the January rebound: a 30-50 session trend
    # average doesn't turn positive until the bounce is already well
    # underway (see the 2023-01 walk-forward -- TSLA bottomed 2023-01-06 at
    # $113 but Trend50 didn't turn positive until 2023-01-26 at $160,
    # missing ~40% of the move; the sampled trend_entry_window only makes
    # this worse). Trend10 turned positive just 2 sessions off the bottom
    # (2023-01-09 at $120), so the seasonal path uses that fixed, fast
    # window instead of params.trend_entry_window -- capitulation reversals
    # need a quick confirmation, not a slow moving average.
    seasonal_trend_confirmed = pd.to_numeric(
        frame.get("Trend10", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ) > 0
    seasonal_recovery_buy = (
        frame["JanuaryReboundWindow"]
        & seasonal_trend_confirmed
        & (frame["MacroScore"] >= params.reentry_macro_score_min)
        & (downside_probability <= params.january_rebound_downside_probability_max)
    )
    # Regime-gated contrarian overlay: momentum (buy strength) works well in
    # a genuine bull trend but actively fights mean-reversion during a
    # confirmed bear/correction regime, where fading strength and buying
    # capitulation washouts has the edge instead (see the walk-forward
    # ablation referenced at BEAR_REGIME_TREND_MAX). Only activates once
    # TSLA's own price is >=15% below its 200-session average, so it stays
    # a no-op through ordinary bull-market pullbacks.
    trend200 = pd.to_numeric(
        frame.get("Trend200", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    trend50_for_washout = pd.to_numeric(
        frame.get("Trend50", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    frame["BearRegime"] = trend200 <= BEAR_REGIME_TREND_MAX
    frame["FastWashout"] = trend50_for_washout <= FAST_WASHOUT_TREND_MAX
    downside_off_peak = downside_probability < downside_probability.rolling(
        params.contrarian_lookback_sessions,
        min_periods=params.contrarian_lookback_sessions,
    ).max()
    # TacticalScore blends in MacroScore, which the oracle audit found was
    # often *not* stressed at the best entries (9/10 had MarketExposureScale
    # == 1.0, i.e. the broad market was fine -- these were TSLA-idiosyncratic
    # washouts, not macro-driven selloffs). Requiring TacticalScore alone can
    # therefore miss a washout that's real on TSLA's own RSI but diluted by
    # a calm macro backdrop, so RSI-oversold is accepted as an independent,
    # TSLA-specific alternative confirmation.
    rsi_oversold_confirmed = (
        pd.to_numeric(frame["RSI14"], errors="coerce")
        <= params.contrarian_rsi_oversold_max
    )
    contrarian_buy = (
        (frame["BearRegime"] | frame["FastWashout"])
        & (
            (frame["TacticalScore"] <= params.contrarian_buy_tactical_max)
            | rsi_oversold_confirmed
        )
        & downside_off_peak
        & (frame["FinancialScore"] >= params.contrarian_buy_financial_score_min)
    )
    frame["PrimaryBuySignal"] = primary_buy & ~critical_flag
    frame["RecoveryBuySignal"] = recovery_buy & ~critical_flag
    frame["SeasonalReboundBuySignal"] = seasonal_recovery_buy & ~critical_flag
    frame["ContrarianBuySignal"] = contrarian_buy & ~critical_flag
    # A cash investor cannot average down an existing position.  This stricter
    # contrarian path waits for the fast trend to turn up after a confirmed
    # bear/washout setup, then executes at the following session's open.
    frame["BearEndBuySignal"] = (
        frame["ContrarianBuySignal"] & seasonal_trend_confirmed
    )
    frame["BuySignal"] = (
        frame["PrimaryBuySignal"]
        | frame["RecoveryBuySignal"]
        | frame["SeasonalReboundBuySignal"]
        | frame["ContrarianBuySignal"]
    )
    bearish_trend = (
        (exit_trend <= params.trend_exit_threshold)
        & (
            pd.to_numeric(frame["MACD"], errors="coerce")
            < pd.to_numeric(frame["MACDSignal"], errors="coerce")
        )
    )
    risk_exit = (
        (frame["CompositeScore"] <= params.sell_threshold)
        & bearish_trend
        & (
            downside_probability
            >= params.sell_downside_probability_min
        )
    )
    frame["SellSignal"] = (
        (exit_trend <= params.trend_exit_threshold) | risk_exit | critical_flag
    )
    market_exposure = pd.to_numeric(
        frame.get("MarketExposureScale", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(1.0)
    frame["MarketBearConfirmed"] = market_exposure <= params.market_short_exposure_max
    tactical_short_ready = (
        (frame["TacticalScore"] <= params.short_threshold)
        & (frame["MacroScore"] <= params.short_macro_score_max)
    )
    frame["ShortSignal"] = (
        bearish_trend
        & (downside_probability >= params.short_downside_probability_min)
        & (tactical_short_ready | frame["MarketBearConfirmed"])
    )
    frame["CoverSignal"] = (
        (frame["TacticalScore"] >= params.cover_threshold)
        | ~bearish_trend
        | (downside_probability <= params.cover_downside_probability_max)
    )
    frame["SignalAvailable"] = np.isfinite(frame["CompositeScore"])
    return frame


def generate_consensus_signals(
    features: pd.DataFrame,
    members: Sequence[IntegratedParams],
    *,
    entry_consensus: float = 0.70,
    exit_consensus: float = 0.50,
) -> pd.DataFrame:
    """Require broad agreement among development-selected signal members."""

    if not members:
        raise ValueError("Consensus signals require at least one member.")
    if not 0.5 <= entry_consensus <= 1:
        raise ValueError("entry_consensus must be in [0.5, 1].")
    if not 0.5 <= exit_consensus <= 1:
        raise ValueError("exit_consensus must be in [0.5, 1].")
    member_signals = [
        generate_integrated_signals(features, params)
        for params in members
    ]
    frame = member_signals[0].copy()
    frame["CompositeScore"] = np.mean(
        [signals["CompositeScore"].to_numpy() for signals in member_signals],
        axis=0,
    )
    vote_columns = {
        "BuyVote": "BuySignal",
        "SellVote": "SellSignal",
        "ShortVote": "ShortSignal",
        "CoverVote": "CoverSignal",
    }
    for vote_column, signal_column in vote_columns.items():
        frame[vote_column] = np.mean(
            [
                signals[signal_column].fillna(False).to_numpy(dtype=bool)
                for signals in member_signals
            ],
            axis=0,
        )
    frame["BuySignal"] = frame["BuyVote"] >= entry_consensus
    frame["ShortSignal"] = frame["ShortVote"] >= entry_consensus
    frame["SellSignal"] = frame["SellVote"] >= exit_consensus
    frame["CoverSignal"] = frame["CoverVote"] >= exit_consensus
    frame["SignalAvailable"] = np.isfinite(frame["CompositeScore"])
    return frame
