from __future__ import annotations

"""Causal, interpretable TSLA two-month pattern-regime classifier.

The classifier deliberately emits ``NEUTRAL_TRANSITION`` when evidence is
weak or conflicting.  Its five named regimes describe market state; they are
not direct buy/sell instructions and do not use forward returns as inputs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


PATTERN_COLUMNS = (
    "MOMENTUM_CONTINUATION",
    "WASHOUT_REVERSAL",
    "CATALYST_RERATING",
    "BLOWOFF_RISK",
    "BEAR_CONTINUATION_RISK",
)
NEUTRAL_PATTERN = "NEUTRAL_TRANSITION"


@dataclass(frozen=True)
class PatternRegimeConfig:
    minimum_score: float = 0.58
    minimum_margin: float = 0.06
    confirmation_sessions: int = 3

    def validate(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be in [0, 1].")
        if not 0.0 <= self.minimum_margin <= 1.0:
            raise ValueError("minimum_margin must be in [0, 1].")
        if self.confirmation_sessions < 1:
            raise ValueError("confirmation_sessions must be positive.")


REQUIRED_COLUMNS = {
    "Date",
    "Return5D",
    "Return20D",
    "Return60D",
    "RealizedVol21",
    "VolumeSurprise21",
    "MACDSpread",
    "RSI14",
    "DistanceFromSMA50",
    "DistanceFromSMA200",
    "MarginTrend",
    "GrowthComposite",
    "ShareGrowthYoYFiled",
    "Treasury10Y",
    "YieldCurve10Y2Y",
    "VIX_Level",
}


def _sigmoid(value: pd.Series, center: float, scale: float) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    argument = ((numeric - center) / scale).clip(-40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-argument))).fillna(0.5)


def _minimum(*values: pd.Series) -> pd.Series:
    return pd.concat(values, axis=1).min(axis=1)


def _maximum(*values: pd.Series) -> pd.Series:
    return pd.concat(values, axis=1).max(axis=1)


def calculate_pattern_scores(
    features: pd.DataFrame,
    news_proxy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate five fixed, causal regime-membership scores in [0, 1]."""

    missing = REQUIRED_COLUMNS.difference(features.columns)
    if missing:
        raise ValueError(f"Missing pattern feature columns: {sorted(missing)}")
    frame = features.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if news_proxy is not None:
        required_news = {"Date", "NewsScore", "NewsSignalActive"}
        if news_missing := required_news.difference(news_proxy.columns):
            raise ValueError(f"Missing news columns: {sorted(news_missing)}")
        news = news_proxy.copy()
        news["Date"] = pd.to_datetime(news["Date"], errors="coerce")
        keep = [
            column
            for column in (
                "Date",
                "NewsScore",
                "NewsSignalActive",
                "MarketAdjustedReaction",
                "VolumeSurprise20D",
                "EventClusters",
            )
            if column in news.columns
        ]
        frame = frame.merge(news[keep], how="left", on="Date", validate="one_to_one")
    if "NewsScore" not in frame:
        frame["NewsScore"] = 0.5
    if "NewsSignalActive" not in frame:
        frame["NewsSignalActive"] = False
    frame["NewsScore"] = pd.to_numeric(frame["NewsScore"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    frame["NewsSignalActive"] = frame["NewsSignalActive"].fillna(False).astype(bool)

    r5 = pd.to_numeric(frame["Return5D"], errors="coerce")
    r20 = pd.to_numeric(frame["Return20D"], errors="coerce")
    r60 = pd.to_numeric(frame["Return60D"], errors="coerce")
    rsi = pd.to_numeric(frame["RSI14"], errors="coerce")
    d50 = pd.to_numeric(frame["DistanceFromSMA50"], errors="coerce")
    d200 = pd.to_numeric(frame["DistanceFromSMA200"], errors="coerce")
    macd = pd.to_numeric(frame["MACDSpread"], errors="coerce")
    vol = pd.to_numeric(frame["RealizedVol21"], errors="coerce")
    volume = pd.to_numeric(frame["VolumeSurprise21"], errors="coerce")
    vix = pd.to_numeric(frame["VIX_Level"], errors="coerce")
    treasury = pd.to_numeric(frame["Treasury10Y"], errors="coerce")
    curve = pd.to_numeric(frame["YieldCurve10Y2Y"], errors="coerce")

    frame["RSIChange5"] = rsi - rsi.shift(5)
    frame["MACDChange5"] = macd - macd.shift(5)
    positive_news = ((frame["NewsScore"] - 0.5) * 2.0).clip(0.0, 1.0)
    negative_news = ((0.5 - frame["NewsScore"]) * 2.0).clip(0.0, 1.0)

    fundamental_health = (
        0.40 * _sigmoid(frame["MarginTrend"], 0.0, 0.04)
        + 0.35 * _sigmoid(frame["GrowthComposite"], 0.0, 0.40)
        + 0.25 * _sigmoid(-frame["ShareGrowthYoYFiled"], -0.06, 0.04)
    )
    frame["FundamentalHealth"] = fundamental_health.clip(0.0, 1.0)

    healthy_rsi = _minimum(_sigmoid(rsi, 42.0, 7.0), _sigmoid(78.0 - rsi, 0.0, 7.0))
    trend_strength = (
        0.30 * _sigmoid(r60, 0.08, 0.18)
        + 0.25 * _sigmoid(d200, 0.0, 0.15)
        + 0.15 * _sigmoid(r20, 0.0, 0.10)
        + 0.10 * _sigmoid(macd, 0.0, 0.010)
        + 0.10 * healthy_rsi
        + 0.05 * positive_news
        + 0.05 * fundamental_health
    )

    blowoff = (
        0.38 * _sigmoid(r60, 0.75, 0.18)
        + 0.32 * _sigmoid(d200, 0.50, 0.18)
        + 0.18 * _sigmoid(rsi, 75.0, 8.0)
        + 0.07 * _sigmoid(volume, 0.35, 0.30)
        + 0.05 * _sigmoid(vol, 0.85, 0.25)
    ).clip(0.0, 1.0)
    momentum = (trend_strength * (1.0 - 0.45 * blowoff)).clip(0.0, 1.0)

    washout_setup = (
        0.28 * _sigmoid(-r60, 0.18, 0.12)
        + 0.24 * _sigmoid(-d200, 0.08, 0.12)
        + 0.25 * _sigmoid(38.0 - rsi, 0.0, 7.0)
        + 0.13 * _sigmoid(vol, 0.70, 0.20)
        + 0.10 * fundamental_health
    )
    reversal_confirmation = (
        0.30 * _sigmoid(r5, 0.0, 0.05)
        + 0.25 * _sigmoid(frame["RSIChange5"], 1.0, 5.0)
        + 0.25 * _sigmoid(frame["MACDChange5"], 0.0, 0.005)
        + 0.20 * positive_news
    )
    washout = np.sqrt((washout_setup * reversal_confirmation).clip(0.0, 1.0))

    neutral_base = (
        1.0 - (r60.abs() / 0.25).clip(0.0, 1.0)
    ).fillna(0.0)
    catalyst = (
        positive_news
        * (
            0.62
            + 0.13 * neutral_base
            + 0.10 * _sigmoid(volume, 0.0, 0.25)
            + 0.15 * fundamental_health
        )
    ).clip(0.0, 1.0)

    macro_stress = (
        0.50 * _sigmoid(vix, 23.0, 5.0)
        + 0.30 * _sigmoid(treasury, 3.25, 0.65)
        + 0.20 * _sigmoid(-curve, 0.15, 0.35)
    ).clip(0.0, 1.0)
    frame["MacroStress"] = macro_stress
    downtrend = (
        0.24 * _sigmoid(-r60, 0.10, 0.15)
        + 0.20 * _sigmoid(-d200, 0.03, 0.12)
        + 0.16 * _sigmoid(-r20, 0.03, 0.08)
        + 0.14 * _sigmoid(45.0 - rsi, 0.0, 7.0)
        + 0.12 * _sigmoid(-macd, 0.0, 0.010)
        + 0.08 * negative_news
        + 0.06 * (1.0 - fundamental_health)
    )
    failed_rebound_risk = (
        0.38 * macro_stress
        + 0.19 * _sigmoid(r60, 0.15, 0.15)
        + 0.13 * _sigmoid(d50, 0.0, 0.10)
        + 0.15 * (1.0 - fundamental_health)
        + 0.15 * negative_news
    )
    bear_continuation = _maximum(
        0.78 * downtrend + 0.22 * macro_stress,
        failed_rebound_risk,
    ).clip(0.0, 1.0)

    frame["MOMENTUM_CONTINUATION"] = momentum
    frame["WASHOUT_REVERSAL"] = washout
    frame["CATALYST_RERATING"] = catalyst
    frame["BLOWOFF_RISK"] = blowoff
    frame["BEAR_CONTINUATION_RISK"] = bear_continuation
    return frame


def _confirmed_states(candidates: pd.Series, sessions: int) -> pd.Series:
    stable = NEUTRAL_PATTERN
    pending = NEUTRAL_PATTERN
    pending_count = 0
    output: list[str] = []
    for candidate in candidates.astype(str):
        if candidate == stable:
            pending = candidate
            pending_count = 0
        elif candidate == pending:
            pending_count += 1
        else:
            pending = candidate
            pending_count = 1
        if candidate != stable and pending_count >= sessions:
            stable = candidate
            pending_count = 0
        output.append(stable)
    return pd.Series(output, index=candidates.index, dtype="object")


def classify_pattern_regimes(
    scores: pd.DataFrame,
    config: PatternRegimeConfig | None = None,
) -> pd.DataFrame:
    """Add raw and hysteresis-confirmed pattern labels to score rows."""

    settings = config or PatternRegimeConfig()
    settings.validate()
    missing = set(PATTERN_COLUMNS).difference(scores.columns)
    if missing:
        raise ValueError(f"Missing pattern score columns: {sorted(missing)}")
    frame = scores.copy().sort_values("Date").reset_index(drop=True)
    matrix = frame.loc[:, PATTERN_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ordered = np.sort(matrix.to_numpy(dtype=float), axis=1)
    frame["TopScore"] = ordered[:, -1]
    frame["ScoreMargin"] = ordered[:, -1] - ordered[:, -2]
    frame["TopPattern"] = matrix.idxmax(axis=1)
    decisive = frame["TopScore"].ge(settings.minimum_score) & frame["ScoreMargin"].ge(settings.minimum_margin)
    frame["RawPattern"] = frame["TopPattern"].where(decisive, NEUTRAL_PATTERN)
    frame["StablePattern"] = _confirmed_states(frame["RawPattern"], settings.confirmation_sessions)
    frame["PatternConfidence"] = (
        0.65 * frame["TopScore"] + 0.35 * (frame["ScoreMargin"] / 0.25).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return frame


def build_pattern_regime_model(
    features: pd.DataFrame,
    news_proxy: pd.DataFrame | None = None,
    config: PatternRegimeConfig | None = None,
) -> pd.DataFrame:
    """Run the shared score and state functions used by reports/simulations."""

    return classify_pattern_regimes(
        calculate_pattern_scores(features, news_proxy), config=config
    )


def summarize_forward_outcomes(classified: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized exact-two-month outcomes without fitting the model."""

    required = {"StablePattern", "ForwardReturnPct"}
    if missing := required.difference(classified.columns):
        raise ValueError(f"Missing outcome columns: {sorted(missing)}")
    valid = classified.dropna(subset=["ForwardReturnPct"]).copy()
    valid["Up50"] = valid["ForwardReturnPct"].ge(50.0)
    valid["Down40"] = valid["ForwardReturnPct"].le(-40.0)
    return (
        valid.groupby("StablePattern", as_index=False)
        .agg(
            Observations=("ForwardReturnPct", "size"),
            MeanForwardReturnPct=("ForwardReturnPct", "mean"),
            MedianForwardReturnPct=("ForwardReturnPct", "median"),
            Up50Count=("Up50", "sum"),
            Up50Rate=("Up50", "mean"),
            Down40Count=("Down40", "sum"),
            Down40Rate=("Down40", "mean"),
        )
        .sort_values("StablePattern")
        .reset_index(drop=True)
    )
