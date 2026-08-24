from __future__ import annotations

import pandas as pd

from stock_research.tsla_integrated.annual_floor import (
    chained_return_metrics,
    rank_cagr_candidates,
    rank_annual_floor_candidates,
)


def test_hard_floor_candidate_ranks_before_higher_average_failure() -> None:
    candidates = pd.DataFrame(
        {
            "CandidateID": [1, 2],
            "ROI_2019": [45.0, 200.0],
            "ROI_2020": [41.0, 10.0],
            "MDD_2019": [-20.0, -10.0],
            "MDD_2020": [-30.0, -10.0],
        }
    )
    ranked = rank_annual_floor_candidates(candidates, [2019, 2020])
    assert ranked.iloc[0]["CandidateID"] == 1
    assert bool(ranked.iloc[0]["AnnualFloorEligible"])


def test_fallback_maximizes_worst_year_when_no_candidate_hits_floor() -> None:
    candidates = pd.DataFrame(
        {
            "CandidateID": [1, 2],
            "ROI_2019": [100.0, 35.0],
            "ROI_2020": [-20.0, 30.0],
            "MDD_2019": [-20.0, -20.0],
            "MDD_2020": [-20.0, -20.0],
        }
    )
    ranked = rank_annual_floor_candidates(candidates, [2019, 2020])
    assert not ranked["AnnualFloorEligible"].any()
    assert ranked.iloc[0]["CandidateID"] == 2
    assert ranked.iloc[0]["WorstDevelopmentYearROI"] == 30.0


def test_roi_floor_is_primary_and_drawdown_is_reported_separately() -> None:
    candidates = pd.DataFrame(
        {
            "CandidateID": [1, 2],
            "ROI_2019": [50.0, 39.0],
            "MDD_2019": [-45.0, -10.0],
        }
    )
    ranked = rank_annual_floor_candidates(candidates, [2019])
    assert ranked.iloc[0]["CandidateID"] == 1
    assert bool(ranked.iloc[0]["ROIFloorEligible"])
    assert not bool(ranked.iloc[0]["DrawdownEligible"])
    assert not bool(ranked.iloc[0]["AnnualFloorEligible"])


def test_chained_return_uses_compounding_not_arithmetic_average() -> None:
    metrics = chained_return_metrics([100.0, -50.0])
    assert metrics["GrowthMultiple"] == 1.0
    assert metrics["CumulativeROI"] == 0.0
    assert metrics["CAGR"] == 0.0
    assert metrics["ArithmeticMeanROI"] == 25.0


def test_cagr_ranking_rejects_a_ruined_candidate() -> None:
    candidates = pd.DataFrame(
        {
            "CandidateID": [1, 2],
            "ROI_2019": [500.0, 50.0],
            "ROI_2020": [-100.0, 50.0],
        }
    )
    ranked = rank_cagr_candidates(candidates, [2019, 2020])
    assert ranked.iloc[0]["CandidateID"] == 2
    assert ranked.iloc[0]["DevelopmentCAGR"] == 50.0
