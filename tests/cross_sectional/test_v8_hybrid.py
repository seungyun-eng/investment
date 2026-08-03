from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.portfolio import run_portfolio_backtest
from stock_research.cross_sectional.v8_hybrid import (
    V8HybridConfig,
    add_v8_scores,
    generate_v8_targets,
)


def test_v8_maximum_weights_are_fully_invested() -> None:
    config = V8HybridConfig()
    assert config.core_position_weight == pytest.approx(0.10)
    assert (
        config.core_total_weight
        + config.inflection_slots * config.inflection_max_weight
        == pytest.approx(1.0)
    )


def test_add_v8_scores_enforces_large_core_and_inflection_gate() -> None:
    dates = pd.to_datetime(["2025-03-28"] * 3)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["BIG", "MID", "WEAK"],
            "Eligible": [True, True, True],
            "Close": [100.0, 50.0, 20.0],
            "Shares": [2_000.0, 200.0, 500.0],
            "GrowthFactor": [0.4, 0.3, -0.3],
            "QualityFactor": [0.4, 0.2, -0.2],
            "MAFactor": [0.3, 0.3, -0.2],
            "MACDFactor": [0.3, 0.3, -0.2],
            "OBVFactor": [0.3, 0.3, -0.2],
            "MomentumFactor": [0.3, 0.3, -0.2],
            "RiskControlFactor": [0.1, 0.1, -0.2],
            "Trend200": [0.2, 0.2, -0.2],
            "Return126": [0.3, 0.3, -0.3],
            "EpsTtmGrowthYoY": [0.5, 0.4, -0.4],
            "EpsTtmGrowthAcceleration": [0.3, 0.2, -0.2],
            "EbitdaTtmGrowthYoY": [0.5, 0.4, -0.4],
            "EbitdaTtmGrowthAcceleration": [0.3, 0.2, -0.2],
            "DcfPriceGrowthYoY": [0.4, 0.3, -0.3],
        }
    )
    scored = add_v8_scores(frame, V8HybridConfig())
    assert bool(scored.set_index("Ticker").loc["BIG", "CoreQualified"])
    assert bool(
        scored.set_index("Ticker").loc["MID", "InflectionQualified"]
    )
    assert not bool(
        scored.set_index("Ticker").loc["WEAK", "CoreQualified"]
    )


def test_inflection_scales_without_weekly_rotation() -> None:
    config = V8HybridConfig(
        core_slots=1,
        core_total_weight=0.70,
        inflection_slots=1,
        inflection_scout_weight=0.10,
        inflection_confirm_weight=0.20,
        inflection_max_weight=0.30,
        inflection_confirm_weeks=2,
        inflection_max_weeks=3,
    )
    dates = pd.to_datetime(
        ["2025-01-03", "2025-01-10", "2025-01-17", "2025-01-24"]
    )
    rows = []
    for date in dates:
        for ticker, core_rank, inflection_rank in (
            ("CORE", 1.0, 2.0),
            ("FAST", 2.0, 1.0),
        ):
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Company": ticker,
                    "CoreQualified": ticker == "CORE",
                    "CoreRank": core_rank,
                    "CoreScore": 0.4,
                    "InflectionQualified": ticker == "FAST",
                    "InflectionRank": inflection_rank,
                    "InflectionScore": 0.4,
                    "FundamentalAccelerationScore": 0.3,
                    "GrowthFactor": 0.3,
                    "QualityFactor": 0.3,
                    "EpsTtmGrowthYoY": 0.4,
                    "EbitdaTtmGrowthYoY": 0.4,
                    "Trend200": 0.2,
                    "Return126": 0.3,
                }
            )
    targets = generate_v8_targets(pd.DataFrame(rows), config)
    fast = targets.loc[targets["Ticker"].eq("FAST")]
    assert list(fast["TradeAction"]) == ["BUY", "SCALE_UP", "SCALE_UP"]
    assert list(fast["TargetWeight"]) == pytest.approx([0.1, 0.2, 0.3])
    core = targets.loc[targets["Ticker"].eq("CORE")]
    assert not core["TradeAction"].eq("SELL").any()


def test_cash_target_is_not_normalized_by_unrelated_missing_open() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-06",
                ]
            ),
            "Ticker": ["A", "B", "A", "B"],
            "Open": [10.0, 20.0, 10.0, float("nan")],
            "Close": [10.0, 20.0, 10.0, 20.0],
        }
    )
    targets = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-03"]),
            "Ticker": ["A"],
            "TargetWeight": [0.80],
        }
    )
    result = run_portfolio_backtest(
        panel,
        targets,
        start="2025-01-03",
        end="2025-01-06",
        initial_capital=100.0,
        transaction_cost_bps=0.0,
    )
    last = result.daily.iloc[-1]
    assert last["Equity"] == pytest.approx(100.0)
    assert last["Cash"] == pytest.approx(20.0)
    assert last["GrossExposure"] == pytest.approx(0.80)
