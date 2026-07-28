from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import IntegratedParams


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
    frame["FinancialScore"] = _clip(
        0.40 * revenue_score + 0.35 * margin_score + 0.25 * cash_score
    )

    macro_stress = _clip(
        0.40 * pd.to_numeric(frame["VixPercentile"], errors="coerce")
        + 0.30 * pd.to_numeric(frame["MacroConfirmationScore"], errors="coerce")
        + 0.30 * pd.to_numeric(frame["ModelRisk"], errors="coerce")
    )
    frame["MacroScore"] = 1.0 - macro_stress
    frame["CompositeScore"] = (
        params.technical_weight * frame["TechnicalScore"]
        + params.financial_weight * frame["FinancialScore"]
        + params.macro_weight * frame["MacroScore"]
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
    recovery_buy = (
        entry_trend >= params.trend_entry_threshold
    )
    frame["PrimaryBuySignal"] = primary_buy
    frame["RecoveryBuySignal"] = recovery_buy
    frame["BuySignal"] = primary_buy | recovery_buy
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
        exit_trend <= params.trend_exit_threshold
    ) | risk_exit
    frame["ShortSignal"] = (
        (frame["CompositeScore"] <= params.short_threshold)
        & bearish_trend
        & (frame["MacroScore"] <= params.short_macro_score_max)
        & (downside_probability >= params.short_downside_probability_min)
    )
    frame["CoverSignal"] = (
        (frame["CompositeScore"] >= params.cover_threshold)
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
