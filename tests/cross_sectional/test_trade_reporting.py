from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.trade_reporting import (
    _latest_complete_date,
    add_signal_strengths,
    attach_execution_prices,
    attach_position_outcomes,
    build_position_ledger,
)


def test_latest_complete_date_prefers_full_cross_section() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-07-27",
                    "2026-07-27",
                    "2026-07-28",
                ]
            ),
            "Ticker": ["A", "B", "A"],
        }
    )

    assert _latest_complete_date(frame) == pd.Timestamp("2026-07-27")


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-03", "2025-01-10"]),
            "Ticker": ["A", "A"],
            "Company": ["Alpha", "Alpha"],
            "TradeAction": ["BUY", "SELL"],
            "DailySignal": ["BUY", "SELL"],
            "Close": [100.0, 110.0],
            "Rank": [1.0, 8.0],
            "AlphaScore": [0.25, 0.10],
            "CrossSectionSize": [11, 11],
            "Return126": [0.20, 0.10],
            "Trend200": [0.15, 0.05],
            "SignalReferenceReturn": [0.0, 0.10],
            "HoldingRebalances": [1.0, 2.0],
            "ExitReason": ["", "PROFITABLE_ROTATION"],
            "FinancialStale": [False, False],
            "MomentumFactor": [0.2, 0.1],
            "TrendFactor": [0.3, 0.2],
            "GrowthFactor": [0.1, 0.0],
            "QualityFactor": [0.0, -0.1],
            "RiskControlFactor": [0.4, 0.2],
        }
    )


def test_signal_strengths_center_scores_on_fifty() -> None:
    frame = _signal_frame()
    frame.loc[0, "AlphaScore"] = 0.0
    frame.loc[0, "MomentumFactor"] = -0.5
    result = add_signal_strengths(frame)

    assert result.loc[0, "CompositeStrength"] == pytest.approx(50.0)
    assert result.loc[0, "MomentumFactorStrength"] == pytest.approx(0.0)
    assert result.loc[0, "RankStrength"] == pytest.approx(100.0)
    assert result.loc[1, "RankStrength"] == pytest.approx(30.0)


def test_execution_matching_uses_next_market_open_and_pairs_position() -> None:
    events = add_signal_strengths(_signal_frame())
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-03", "2025-01-06", "2025-01-10", "2025-01-13"]
            ),
            "Ticker": ["A"] * 4,
            "Company": ["Alpha"] * 4,
            "Open": [99.0, 101.0, 109.0, 112.0],
            "Close": [100.0, 103.0, 110.0, 111.0],
        }
    )
    matched, reconciliation = attach_execution_prices(events, prices)
    ledger = build_position_ledger(
        matched,
        prices,
        latest_date=pd.Timestamp("2025-01-13"),
    )

    assert matched["ExecutionDate"].tolist() == [
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-01-13"),
    ]
    assert matched["ExecutionPrice"].tolist() == pytest.approx([101.0, 112.0])
    assert reconciliation["CloseMatches"].all()
    assert len(ledger) == 1
    assert ledger.loc[0, "Status"] == "CLOSED"
    assert ledger.loc[0, "ExecutionPriceReturn"] == pytest.approx(
        112.0 / 101.0 - 1
    )
    assert ledger.loc[0, "PerSharePnL"] == pytest.approx(11.0)

    enriched_events = attach_position_outcomes(matched, ledger)
    sell = enriched_events.loc[
        enriched_events["TradeAction"].eq("SELL")
    ].iloc[0]
    assert sell["PositionId"] == "A-01"
    assert sell["PositionEntryPrice"] == pytest.approx(101.0)
    assert sell["PositionPnLPerShare"] == pytest.approx(11.0)
    assert sell["PositionReturn"] == pytest.approx(112.0 / 101.0 - 1)


def test_open_position_marks_to_latest_close() -> None:
    events = add_signal_strengths(_signal_frame().iloc[[0]].copy())
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-10"]),
            "Ticker": ["A"] * 3,
            "Company": ["Alpha"] * 3,
            "Open": [99.0, 101.0, 109.0],
            "Close": [100.0, 103.0, 110.0],
        }
    )
    matched, _ = attach_execution_prices(events, prices)
    ledger = build_position_ledger(
        matched,
        prices,
        latest_date=pd.Timestamp("2025-01-10"),
    )

    assert ledger.loc[0, "Status"] == "OPEN"
    assert ledger.loc[0, "ExitExecutionPrice"] == pytest.approx(110.0)
    assert ledger.loc[0, "ExecutionPriceReturn"] == pytest.approx(
        110.0 / 101.0 - 1
    )
