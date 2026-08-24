from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tsla_integrated.component_ablation import (
    build_structured_news_reaction_proxy,
    generate_component_ablation_signals,
)


def test_component_weights_are_renormalized_and_news_neutral_is_noop() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=2, freq="B"),
            "TechnicalScore": [0.8, 0.2],
            "FinancialScore": [0.4, 0.6],
            "MacroScore": [0.5, 0.5],
        }
    )
    weights = {"technical": 0.25, "sec": 0.50, "macro": 0.25}
    signals = generate_component_ablation_signals(
        frame,
        components=("technical", "sec"),
        component_weights=weights,
        buy_threshold=0.62,
        sell_threshold=0.42,
        news_score=pd.Series([0.5, 0.5]),
    )
    assert signals.loc[0, "CompositeScore"] == pytest.approx(
        0.8 / 3 + 0.4 * 2 / 3
    )
    assert signals.loc[1, "CompositeScore"] == pytest.approx(
        0.2 / 3 + 0.6 * 2 / 3
    )


def test_news_proxy_excludes_complaints_and_persists() -> None:
    events = pd.DataFrame(
        {
            "EventClusterID": ["sec", "complaint", "recall"],
            "SourceType": ["SEC_EDGAR", "NHTSA", "NHTSA"],
            "EventSubtype": ["Earnings", "Complaint", "Recall"],
            "Materiality": ["HIGH", "MEDIUM", "HIGH"],
        }
    )
    reactions = pd.DataFrame(
        {
            "EventClusterID": ["sec", "complaint", "recall"],
            "AnchorDate": ["2024-01-02", "2024-01-03", "2024-01-08"],
            "MarketAdjustedReaction": [0.03, -0.08, -0.04],
            "VolumeSurprise20D": [0.2, 1.0, 0.1],
        }
    )
    proxy = build_structured_news_reaction_proxy(
        events,
        reactions,
        pd.Series(pd.date_range("2024-01-02", periods=7, freq="B")),
        persistence_sessions=2,
    )
    assert proxy["NewsScore"].tolist() == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.5]
