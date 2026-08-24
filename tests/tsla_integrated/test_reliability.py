from __future__ import annotations

import numpy as np

from stock_research.tsla_integrated.reliability import (
    binary_signal_metrics,
    block_permutation_auc_pvalue,
    count_positive_runs,
    expected_calibration_error,
    moving_block_bootstrap_auc,
)


def test_positive_runs_count_independent_clusters() -> None:
    assert count_positive_runs(np.array([0, 1, 1, 0, 1, 0, 1, 1])) == 3


def test_calibrated_probabilities_have_low_error() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(target, probability, bins=2) == 0.0
    metrics = binary_signal_metrics(target, probability)
    assert metrics["ROCAUC"] == 1.0
    assert metrics["BrierSkillVsClimatology"] == 1.0


def test_dependence_aware_statistics_are_deterministic() -> None:
    target = np.tile([0, 0, 1, 1], 30)
    probability = target * 0.7 + 0.15
    first = moving_block_bootstrap_auc(
        target, probability, block_length=8, repetitions=20, random_seed=7
    )
    second = moving_block_bootstrap_auc(
        target, probability, block_length=8, repetitions=20, random_seed=7
    )
    np.testing.assert_array_equal(first, second)
    assert first.min() == 1.0
    pvalue = block_permutation_auc_pvalue(
        target, probability, block_length=8, repetitions=20, random_seed=7
    )
    assert 0.0 < pvalue <= 1.0
