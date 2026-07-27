from __future__ import annotations

import pandas as pd
import pytest

from stock_research.macro_fear_buy_sp500.config import FearBuyParams
from stock_research.macro_fear_buy_sp500.contributions import (
    ContributionDeploymentPolicy,
)
from stock_research.macro_fear_buy_sp500.mass_optimization import (
    candidate_to_params,
    candidate_to_policy,
    sample_candidates,
    select_category_winners,
)


def test_sample_candidates_are_valid_and_reproducible() -> None:
    params = FearBuyParams(core_weight=0.8, tranche_weight=0.1)
    policy = ContributionDeploymentPolicy(
        mild_fraction=0.0,
        fear_fraction=1.0,
        panic_fraction=1.0,
        cooldown_sessions=21,
    )
    first = sample_candidates(
        25,
        seed=7,
        baseline_params=params,
        baseline_policy=policy,
    )
    second = sample_candidates(
        25,
        seed=7,
        baseline_params=params,
        baseline_policy=policy,
    )
    assert first == second
    for candidate in first:
        parsed_params = candidate_to_params(candidate)
        parsed_policy = candidate_to_policy(candidate)
        assert sum(
            (
                parsed_params.vix_weight,
                parsed_params.macro_weight,
                parsed_params.model_risk_weight,
                parsed_params.drawdown_weight,
                parsed_params.downside_momentum_weight,
            )
        ) == pytest.approx(1.0)
        assert (
            parsed_policy.mild_fraction
            <= parsed_policy.fear_fraction
            <= parsed_policy.panic_fraction
        )


def test_select_category_winners_are_distinct() -> None:
    candidates = pd.DataFrame(
        {
            "CandidateId": [1, 2, 3, 4],
            "ReturnScore": [4.0, 3.0, 2.0, 1.0],
            "BalancedScore": [4.0, 5.0, 3.0, 2.0],
            "SafetyScore": [4.0, 3.0, 6.0, 2.0],
            "WorstDevelopmentFoldProfitRatio": [1.0, 1.0, 1.0, 1.0],
        }
    )
    winners = select_category_winners(candidates)
    assert winners["SelectionCategory"].tolist() == [
        "Return",
        "Balanced",
        "Safety",
    ]
    assert winners["CandidateId"].tolist() == [1, 2, 3]
