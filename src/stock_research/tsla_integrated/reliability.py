from __future__ import annotations

"""Dependence-aware reliability statistics for overlapping market labels."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def count_positive_runs(target: np.ndarray) -> int:
    values = np.asarray(target, dtype=bool)
    if not len(values):
        return 0
    return int(values[0]) + int(np.count_nonzero(values[1:] & ~values[:-1]))


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    y = np.asarray(target, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return float(abs(y.mean() - p.mean()))
    error = 0.0
    for index in range(len(edges) - 1):
        upper_inclusive = index == len(edges) - 2
        selected = (p >= edges[index]) & (
            p <= edges[index + 1] if upper_inclusive else p < edges[index + 1]
        )
        if selected.any():
            error += selected.mean() * abs(y[selected].mean() - p[selected].mean())
    return float(error)


def moving_block_bootstrap_auc(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    random_seed: int,
) -> np.ndarray:
    """Resample paired contiguous blocks to retain local serial dependence."""

    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or len(y) < block_length:
        raise ValueError("target/probability lengths must match and exceed block length.")
    if repetitions < 1 or block_length < 2:
        raise ValueError("repetitions must be positive and block_length >= 2.")
    rng = np.random.default_rng(random_seed)
    starts = np.arange(0, len(y) - block_length + 1)
    values: list[float] = []
    for _ in range(repetitions):
        sampled: list[int] = []
        while len(sampled) < len(y):
            start = int(rng.choice(starts))
            sampled.extend(range(start, start + block_length))
        indices = np.asarray(sampled[: len(y)], dtype=int)
        if np.unique(y[indices]).size == 2:
            values.append(float(roc_auc_score(y[indices], p[indices])))
    return np.asarray(values, dtype=float)


def block_permutation_auc_pvalue(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    random_seed: int,
) -> float:
    """Shuffle target blocks and test whether observed AUC is unusually high."""

    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or len(y) < block_length:
        raise ValueError("target/probability lengths must match and exceed block length.")
    rng = np.random.default_rng(random_seed)
    blocks = [y[start : start + block_length] for start in range(0, len(y), block_length)]
    observed = float(roc_auc_score(y, p))
    exceedances = 0
    valid = 0
    for _ in range(repetitions):
        order = rng.permutation(len(blocks))
        shuffled = np.concatenate([blocks[index] for index in order])[: len(y)]
        if np.unique(shuffled).size < 2:
            continue
        valid += 1
        exceedances += roc_auc_score(shuffled, p) >= observed
    return float((exceedances + 1) / (valid + 1))


def binary_signal_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, p))
    climatology_brier = prevalence * (1.0 - prevalence)
    return {
        "Observations": float(len(y)),
        "PositiveRate": prevalence,
        "PositiveRuns": float(count_positive_runs(y)),
        "ROCAUC": float(roc_auc_score(y, p)),
        "AveragePrecision": float(average_precision_score(y, p)),
        "BrierScore": brier,
        "BrierSkillVsClimatology": 1.0 - brier / climatology_brier,
        "ExpectedCalibrationError": expected_calibration_error(y, p),
    }
