from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.cross_sectional.overfitting_diagnostics import (
    calculate_coarse_cscv_pbo,
    classify_dsr,
    deflated_sharpe_ratio,
    estimate_effective_trials,
    expected_maximum_sharpe,
)


def _candidate_frame() -> pd.DataFrame:
    rows = []
    for candidate in range(1, 9):
        rows.append(
            {
                "Candidate": candidate,
                "momentum_weight": 0.1 + candidate / 100,
                "trend_weight": 0.2,
                "growth_weight": 0.3,
                "quality_weight": 0.2,
                "risk_control_weight": 0.2 - candidate / 100,
                "top_k": [3, 5, 8][candidate % 3],
                "exit_rank": [5, 9, 16][candidate % 3],
                "trend_floor": [-0.2, -0.1, 0.0, 0.05][candidate % 4],
                "momentum_floor": [-0.35, -0.15, 0.0, 0.1][candidate % 4],
                "TrainROI": candidate * 10,
                "TrainCAGR": candidate,
                "TrainSharpe": candidate / 10,
                "Fold1ExcessCAGR": candidate,
                "Fold2ExcessCAGR": candidate + 0.1,
                "Fold3ExcessCAGR": candidate + 0.2,
                "SelectionExcessCAGR": candidate + 0.3,
                "SelectionROI": candidate * 11,
                "SelectionCAGR": candidate * 1.1,
                "SelectionSharpe": candidate / 9,
            }
        )
    return pd.DataFrame(rows)


def test_expected_maximum_sharpe_increases_with_trials() -> None:
    small = expected_maximum_sharpe(0.2, 10)
    large = expected_maximum_sharpe(0.2, 2_000)
    assert 0 < small < large


def test_dsr_is_probability_and_declines_with_higher_benchmark() -> None:
    returns = pd.Series(
        np.tile([0.01, -0.004, 0.006, -0.002, 0.008], 50)
    )
    low, *_ = deflated_sharpe_ratio(1.5, 0.3, returns)
    high, *_ = deflated_sharpe_ratio(1.5, 1.0, returns)
    assert 0 <= high < low <= 1
    assert classify_dsr(low) in {"STRONG", "AMBIGUOUS", "WEAK"}


def test_effective_trials_uses_nine_unique_dimensions() -> None:
    candidates = _candidate_frame()
    effective, details = estimate_effective_trials(candidates)
    assert 1 <= effective <= len(candidates)
    assert details.iloc[0]["SearchDimensions"] == 9
    assert details.iloc[0]["UniqueNineDimensionalTuples"] == len(candidates)


def test_coarse_cscv_reports_six_symmetric_splits() -> None:
    summary, splits = calculate_coarse_cscv_pbo(_candidate_frame())
    assert len(splits) == 6
    assert summary.iloc[0]["Resolution"] == 1 / 6
    assert 0 <= summary.iloc[0]["PBO"] <= 1
