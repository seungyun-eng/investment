from __future__ import annotations

import pandas as pd

from stock_research.cross_sectional.config import StrategyParams
from stock_research.cross_sectional.dynamic_top_n import (
    build_sp500_top_n_membership,
)
from stock_research.cross_sectional.signals import (
    generate_rebalance_targets,
    score_panel,
)


def _params() -> StrategyParams:
    return StrategyParams(
        momentum_weight=0.2,
        trend_weight=0.2,
        growth_weight=0.3,
        quality_weight=0.2,
        risk_control_weight=0.1,
        top_k=1,
        exit_rank=2,
        trend_floor=-0.1,
        momentum_floor=-0.1,
        loss_aware_exit_enabled=True,
        replacement_score_advantage=0.05,
    )


def test_build_sp500_top_n_membership_filters_and_normalizes_aliases() -> None:
    direct = pd.DataFrame(
        {
            "AsOfDate": ["2020-01-01"] * 4,
            "Ticker": ["AAPL", "BABA", "BRK.B", "FB"],
            "Company": ["Apple", "Alibaba", "Berkshire", "Facebook"],
            "Rank": [1, 2, 3, 4],
            "MarketCap": [100.0, 90.0, 80.0, 70.0],
        }
    )
    membership = pd.DataFrame(
        {
            "AsOfDate": ["2020-01-01"] * 3,
            "DataSymbol": ["AAPL", "BRK-B", "META"],
            "Selected": [True, True, True],
        }
    )

    result = build_sp500_top_n_membership(
        direct, membership, top_n=3
    )

    assert result["HistoricalTicker"].tolist() == ["AAPL", "BRK.B", "FB"]
    assert result["DataSymbol"].tolist() == ["AAPL", "BRK-B", "META"]
    assert result["PublishedRank"].tolist() == [1, 3, 4]
    assert result["Rank"].tolist() == [1, 2, 3]


def test_force_universe_exit_overrides_loss_aware_retention() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-03", "2020-01-03", "2020-01-10", "2020-01-10"]
            ),
            "Ticker": ["A", "B", "A", "B"],
            "Eligible": [True, True, False, True],
            "UniverseMember": [True, True, False, True],
            "Close": [100.0, 100.0, 110.0, 105.0],
            "Trend200": [0.1, 0.0, 0.1, 0.1],
            "Return126": [0.2, 0.1, 0.2, 0.2],
            "Drawdown126": [0.0, 0.0, 0.0, 0.0],
            "MomentumFactor": [1.0, 0.0, 0.0, 1.0],
            "TrendFactor": [1.0, 0.0, 0.0, 1.0],
            "GrowthFactor": [1.0, 0.0, 0.0, 1.0],
            "QualityFactor": [1.0, 0.0, 0.0, 1.0],
            "RiskControlFactor": [1.0, 0.0, 0.0, 1.0],
        }
    )
    params = _params()
    scored = score_panel(panel, params)

    retained = generate_rebalance_targets(scored, params)
    strict = generate_rebalance_targets(
        scored, params, force_universe_exit=True
    )

    retained_a = retained.loc[
        retained["Date"].eq(pd.Timestamp("2020-01-10"))
        & retained["Ticker"].eq("A")
    ].iloc[0]
    strict_a = strict.loc[
        strict["Date"].eq(pd.Timestamp("2020-01-10"))
        & strict["Ticker"].eq("A")
    ].iloc[0]
    assert retained_a["TradeAction"] == "HOLD"
    assert strict_a["TradeAction"] == "SELL"
    assert strict_a["ExitReason"] == "UNIVERSE_EXIT"
