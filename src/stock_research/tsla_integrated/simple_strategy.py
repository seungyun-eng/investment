from __future__ import annotations

"""Evidence-based simplified TSLA strategy: built ONLY from the two signals
that run_signal_validity_check.py found to have a statistically real,
direction-consistent relationship with TSLA's own forward returns
(Spearman IC over 2019-2026, checked for sign stability year by year):

  - MACD-minus-signal spread: IC=+0.113 (p<0.001), 62.5% of years agreed
  - ModelRisk: IC=-0.279 (p<0.001), 62.5% of years agreed -- the single
    strongest signal found

Every other input the original CompositeScore (strategy.py) used --
FinancialScore and its 5 fundamental sub-components, VixPercentile,
MacroConfirmationScore, DownsideProbability21 -- was either statistically
indistinguishable from noise or significant in the OPPOSITE direction from
what that formula assumed. None of them appear here. This is not an
oversight; it is the point.

The blend weight between the two signals (`macd_weight`) is fixed at 0.5,
not searched by the optimizer: with only two validated ingredients, tuning
their relative weight on the same data that validated them would
reintroduce exactly the over-parameterization risk this whole exercise was
trying to get away from. Everything that IS searched (entry/exit thresholds,
stop-loss/trailing-stop, short leverage, minimum hold) governs execution
discipline, not which information the model trusts.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import IntegratedParams

DEFAULT_MACD_WEIGHT = 0.5
DEFAULT_NORMALIZATION_WINDOW = 252


def generate_simple_signals(
    features: pd.DataFrame,
    params: IntegratedParams,
    *,
    macd_weight: float = DEFAULT_MACD_WEIGHT,
    normalization_window: int = DEFAULT_NORMALIZATION_WINDOW,
) -> pd.DataFrame:
    frame = features.copy().sort_values("Date").reset_index(drop=True)
    macd_spread = pd.to_numeric(frame["MACD"], errors="coerce") - pd.to_numeric(
        frame["MACDSignal"], errors="coerce"
    )
    model_risk = pd.to_numeric(frame["ModelRisk"], errors="coerce")
    # Rolling (not expanding) percentile rank: causal, uses only trailing
    # data, and puts both signals on a comparable [0, 1] scale despite very
    # different raw units/ranges.
    macd_rank = macd_spread.rolling(normalization_window, min_periods=normalization_window).rank(pct=True)
    risk_rank = model_risk.rolling(normalization_window, min_periods=normalization_window).rank(pct=True)
    frame["MacdRank"] = macd_rank
    frame["ModelRiskRank"] = risk_rank
    frame["CompositeScore"] = (
        macd_weight * macd_rank + (1.0 - macd_weight) * (1.0 - risk_rank)
    ).clip(0.0, 1.0)
    frame["BuySignal"] = frame["CompositeScore"] >= params.buy_threshold
    frame["SellSignal"] = frame["CompositeScore"] <= params.sell_threshold
    frame["ShortSignal"] = frame["CompositeScore"] <= params.short_threshold
    frame["CoverSignal"] = frame["CompositeScore"] >= params.cover_threshold
    frame["SignalAvailable"] = np.isfinite(frame["CompositeScore"])
    return frame


def generate_simple_consensus_signals(
    features: pd.DataFrame,
    members: Sequence[IntegratedParams],
    *,
    entry_consensus: float = 0.70,
    exit_consensus: float = 0.50,
    macd_weight: float = DEFAULT_MACD_WEIGHT,
    normalization_window: int = DEFAULT_NORMALIZATION_WINDOW,
) -> pd.DataFrame:
    """Mirrors strategy.generate_consensus_signals's voting mechanics
    exactly, over generate_simple_signals members instead."""

    if not members:
        raise ValueError("Consensus signals require at least one member.")
    if not 0.5 <= entry_consensus <= 1:
        raise ValueError("entry_consensus must be in [0.5, 1].")
    if not 0.5 <= exit_consensus <= 1:
        raise ValueError("exit_consensus must be in [0.5, 1].")
    member_signals = [
        generate_simple_signals(
            features, params, macd_weight=macd_weight, normalization_window=normalization_window
        )
        for params in members
    ]
    frame = member_signals[0].copy()
    frame["CompositeScore"] = np.mean(
        [signals["CompositeScore"].to_numpy() for signals in member_signals], axis=0
    )
    vote_columns = {
        "BuyVote": "BuySignal",
        "SellVote": "SellSignal",
        "ShortVote": "ShortSignal",
        "CoverVote": "CoverSignal",
    }
    for vote_column, signal_column in vote_columns.items():
        frame[vote_column] = np.mean(
            [signals[signal_column].fillna(False).to_numpy(dtype=bool) for signals in member_signals],
            axis=0,
        )
    frame["BuySignal"] = frame["BuyVote"] >= entry_consensus
    frame["ShortSignal"] = frame["ShortVote"] >= entry_consensus
    frame["SellSignal"] = frame["SellVote"] >= exit_consensus
    frame["CoverSignal"] = frame["CoverVote"] >= exit_consensus
    frame["SignalAvailable"] = np.isfinite(frame["CompositeScore"])
    return frame


def sample_simple_params(rng: np.random.Generator) -> IntegratedParams:
    """Ten tunable dimensions (thresholds + execution discipline) instead of
    the ~30 in the original sample_params -- fewer knobs, less room to
    curve-fit to this one stock's history."""

    sell = float(rng.uniform(0.35, 0.55))
    short = float(rng.uniform(0.15, min(0.40, sell - 0.02)))
    buy = float(rng.uniform(max(0.55, sell + 0.05), 0.80))
    cover = float(rng.uniform(max(short + 0.05, 0.35), 0.65))
    return IntegratedParams(
        buy_threshold=buy,
        sell_threshold=sell,
        short_threshold=short,
        cover_threshold=cover,
        stop_loss=float(rng.uniform(0.12, 0.35)),
        trailing_stop=float(rng.uniform(0.10, 0.35)),
        short_stop_loss=float(rng.uniform(0.08, 0.25)),
        short_trailing_stop=float(rng.uniform(0.08, 0.25)),
        short_leverage=float(rng.uniform(1.0, 2.0)),
        minimum_hold_sessions=int(rng.choice([5, 10, 21, 42])),
    )
