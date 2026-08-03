from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.config import StrategyParams
from stock_research.cross_sectional.filing_v7_optimization import (
    FilingV7Policy,
    add_filing_v7_scores,
)


def _base_params() -> StrategyParams:
    return StrategyParams(
        momentum_weight=0.2,
        trend_weight=0.2,
        growth_weight=0.2,
        quality_weight=0.2,
        risk_control_weight=0.2,
        top_k=1,
        exit_rank=2,
        trend_floor=-0.2,
        momentum_floor=-0.3,
    )


def _policy(**overrides: object) -> FilingV7Policy:
    values: dict[str, object] = {
        "filing_weight": 1.0,
        "minimum_filing_coverage": 4,
        "filing_quality_floor": None,
        "red_flag_veto": True,
        "top_k": 1,
        "exit_rank_buffer": 1,
        "hard_stop_return": -0.35,
        "minimum_hold_rebalances": 4,
        "replacement_score_advantage": 0.05,
    }
    values.update(overrides)
    return FilingV7Policy(**values)  # type: ignore[arg-type]


def _panel() -> pd.DataFrame:
    rows = []
    for ticker, base, filing, critical in (
        ("BASE", 0.4, -0.4, False),
        ("FILED", -0.4, 0.4, False),
        ("RISK", 0.5, 0.5, True),
    ):
        rows.append(
            {
                "Date": pd.Timestamp("2025-05-02"),
                "Ticker": ticker,
                "Company": ticker,
                "Eligible": True,
                "UniverseMember": True,
                "Close": 100.0,
                "Trend200": 0.1,
                "Return126": 0.2,
                "Drawdown126": -0.05,
                "MomentumFactor": base,
                "TrendFactor": base,
                "GrowthFactor": base,
                "QualityFactor": base,
                "RiskControlFactor": base,
                "FilingFundamentalFactor": filing,
                "FilingCoverageCount": 6,
                "AvailableDate": pd.Timestamp("2025-05-01"),
                "FiledGoingConcernFlag": critical,
                "FiledMaterialWeaknessFlag": False,
                "FiledRestatementFlag": False,
            }
        )
    return pd.DataFrame(rows)


def test_pure_sec_ranking_ignores_base_v7_score() -> None:
    scored = add_filing_v7_scores(_panel(), _base_params(), _policy())
    ranked = scored.loc[scored["Qualified"]].set_index("Ticker")
    assert ranked.loc["FILED", "Rank"] == pytest.approx(1.0)
    assert ranked.loc["BASE", "Rank"] == pytest.approx(2.0)


def test_red_flag_veto_marks_company_ineligible_and_non_member() -> None:
    scored = add_filing_v7_scores(_panel(), _base_params(), _policy())
    risk = scored.set_index("Ticker").loc["RISK"]
    assert not bool(risk["Qualified"])
    assert not bool(risk["UniverseMember"])
    assert bool(risk["FilingCriticalFlag"])


def test_zero_weight_without_veto_reproduces_base_ranking() -> None:
    policy = _policy(
        filing_weight=0.0,
        minimum_filing_coverage=0,
        red_flag_veto=False,
    )
    scored = add_filing_v7_scores(_panel(), _base_params(), policy)
    assert scored.set_index("Ticker").loc["RISK", "Rank"] == pytest.approx(1.0)
