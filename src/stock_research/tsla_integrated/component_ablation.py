from __future__ import annotations

"""Cumulative TSLA information-component ablation helpers."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


COMPONENT_COLUMNS = {
    "technical": "TechnicalScore",
    "sec": "FinancialScore",
    "macro": "MacroScore",
}


def generate_component_ablation_signals(
    scored_features: pd.DataFrame,
    *,
    components: Iterable[str],
    component_weights: dict[str, float],
    buy_threshold: float,
    sell_threshold: float,
    news_score: pd.Series | None = None,
    news_max_adjustment: float = 0.10,
) -> pd.DataFrame:
    """Build long/cash signals while changing only the included information.

    Technical/SEC/macro weights are renormalized over the included set.
    A neutral news score of 0.5 has exactly zero effect; news can move the
    cumulative score up/down by at most ``news_max_adjustment``.
    """

    included = tuple(components)
    invalid = set(included).difference(COMPONENT_COLUMNS)
    if invalid:
        raise ValueError(f"Unknown components: {sorted(invalid)}")
    if not included:
        raise ValueError("At least one component is required.")
    if not 0 <= sell_threshold < buy_threshold <= 1:
        raise ValueError("Require 0 <= sell threshold < buy threshold <= 1.")
    if not 0 <= news_max_adjustment <= 0.5:
        raise ValueError("news_max_adjustment must be in [0, 0.5].")

    frame = scored_features.copy().sort_values("Date").reset_index(drop=True)
    weights = {name: float(component_weights[name]) for name in included}
    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Included component weights must be non-negative and nonzero.")
    total_weight = sum(weights.values())
    composite = pd.Series(0.0, index=frame.index)
    for name, weight in weights.items():
        column = COMPONENT_COLUMNS[name]
        composite += (
            pd.to_numeric(frame[column], errors="coerce") * weight / total_weight
        )

    if news_score is not None:
        aligned_news = pd.to_numeric(news_score, errors="coerce").reindex(
            frame.index
        ).fillna(0.5).clip(0.0, 1.0)
        frame["NewsScore"] = aligned_news
        composite += news_max_adjustment * (aligned_news - 0.5) * 2.0
    else:
        frame["NewsScore"] = 0.5

    frame["CompositeScore"] = composite.clip(0.0, 1.0)
    frame["BuySignal"] = frame["CompositeScore"] >= buy_threshold
    frame["SellSignal"] = frame["CompositeScore"] <= sell_threshold
    if "sec" in included:
        critical = frame.get(
            "FilingCriticalFlag", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        frame["BuySignal"] &= ~critical
        frame["SellSignal"] |= critical
    frame["ShortSignal"] = False
    frame["CoverSignal"] = True
    frame["SignalAvailable"] = np.isfinite(frame["CompositeScore"])
    return frame


def build_structured_news_reaction_proxy(
    events: pd.DataFrame,
    reactions: pd.DataFrame,
    trading_dates: pd.Series,
    *,
    persistence_sessions: int = 5,
    minimum_absolute_reaction: float = 0.02,
) -> pd.DataFrame:
    """Create a causal daily news proxy from stored material-event reactions.

    Only medium/high SEC events and NHTSA recalls are retained.  The event-day
    close reaction becomes actionable at the following open through the
    portfolio engine's one-session signal lag.  Complaints are excluded.
    """

    if persistence_sessions < 1:
        raise ValueError("persistence_sessions must be positive.")
    required_events = {
        "EventClusterID",
        "SourceType",
        "EventSubtype",
        "Materiality",
    }
    required_reactions = {
        "EventClusterID",
        "AnchorDate",
        "MarketAdjustedReaction",
        "VolumeSurprise20D",
    }
    if missing := required_events.difference(events.columns):
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    if missing := required_reactions.difference(reactions.columns):
        raise ValueError(f"Missing reaction columns: {sorted(missing)}")

    event_subset = events.copy()
    eligible = (
        (
            event_subset["SourceType"].eq("SEC_EDGAR")
            & event_subset["Materiality"].isin(["HIGH", "MEDIUM"])
        )
        | (
            event_subset["SourceType"].eq("NHTSA")
            & event_subset["EventSubtype"].eq("Recall")
        )
    )
    event_subset = event_subset.loc[eligible].drop_duplicates("EventClusterID")
    merged = event_subset.merge(
        reactions,
        how="inner",
        on="EventClusterID",
        validate="one_to_many",
    )
    merged["AnchorDate"] = pd.to_datetime(
        merged["AnchorDate"], errors="coerce"
    ).dt.tz_localize(None)
    merged["MarketAdjustedReaction"] = pd.to_numeric(
        merged["MarketAdjustedReaction"], errors="coerce"
    )
    merged["VolumeSurprise20D"] = pd.to_numeric(
        merged["VolumeSurprise20D"], errors="coerce"
    )
    merged = merged.dropna(
        subset=["AnchorDate", "MarketAdjustedReaction", "VolumeSurprise20D"]
    )
    daily_events = (
        merged.groupby("AnchorDate", as_index=False)
        .agg(
            MarketAdjustedReaction=("MarketAdjustedReaction", "median"),
            VolumeSurprise20D=("VolumeSurprise20D", "median"),
            EventClusters=("EventClusterID", "nunique"),
        )
        .sort_values("AnchorDate")
    )
    impactful = (
        daily_events["MarketAdjustedReaction"].abs()
        >= minimum_absolute_reaction
    ) & daily_events["VolumeSurprise20D"].ge(0.0)
    daily_events["EventNewsScore"] = 0.5
    daily_events.loc[
        impactful & daily_events["MarketAdjustedReaction"].gt(0),
        "EventNewsScore",
    ] = 1.0
    daily_events.loc[
        impactful & daily_events["MarketAdjustedReaction"].lt(0),
        "EventNewsScore",
    ] = 0.0

    calendar = pd.DataFrame(
        {"Date": pd.to_datetime(trading_dates, errors="coerce")}
    ).dropna().drop_duplicates().sort_values("Date").reset_index(drop=True)
    calendar = calendar.merge(
        daily_events.rename(columns={"AnchorDate": "Date"}),
        how="left",
        on="Date",
        validate="one_to_one",
    )
    score = pd.Series(0.5, index=calendar.index, dtype=float)
    remaining = 0
    active_score = 0.5
    for index, event_score in enumerate(calendar["EventNewsScore"]):
        if pd.notna(event_score) and float(event_score) != 0.5:
            active_score = float(event_score)
            remaining = persistence_sessions
        if remaining > 0:
            score.iloc[index] = active_score
            remaining -= 1
    calendar["NewsScore"] = score
    calendar["NewsSignalActive"] = calendar["NewsScore"].ne(0.5)
    return calendar[
        [
            "Date",
            "NewsScore",
            "NewsSignalActive",
            "MarketAdjustedReaction",
            "VolumeSurprise20D",
            "EventClusters",
        ]
    ]
