from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .filing_v7_optimization import FilingV7Policy

# Maps a FilingV7Policy field name to its optimization_grid axis key
# (config/cross_sectional/filing_v7_optimization.json:32-39), used by
# parameter_neighbor_stats() to find "one grid-step away on exactly one
# axis" neighbors for a candidate.
_POLICY_FIELD_TO_GRID_AXIS = {
    "filing_weight": "filing_weights",
    "minimum_filing_coverage": "minimum_filing_coverage",
    "filing_quality_floor": "filing_quality_floors",
    "top_k": "top_k",
    "exit_rank_buffer": "exit_rank_buffers",
    "hard_stop_return": "hard_stop_returns",
    "minimum_hold_rebalances": "minimum_hold_rebalances",
    "replacement_score_advantage": "replacement_score_advantages",
}


def policy_from_candidate_row(row: pd.Series) -> FilingV7Policy:
    """Reconstruct a FilingV7Policy from a candidates.csv row (every field
    is already flattened in via asdict(policy) in _candidate_metrics)."""

    return FilingV7Policy(
        filing_weight=float(row["filing_weight"]),
        minimum_filing_coverage=int(row["minimum_filing_coverage"]),
        filing_quality_floor=(
            float(row["filing_quality_floor"])
            if pd.notna(row["filing_quality_floor"])
            else None
        ),
        red_flag_veto=bool(row["red_flag_veto"]),
        top_k=int(row["top_k"]),
        exit_rank_buffer=int(row["exit_rank_buffer"]),
        hard_stop_return=float(row["hard_stop_return"]),
        minimum_hold_rebalances=int(row["minimum_hold_rebalances"]),
        replacement_score_advantage=float(row["replacement_score_advantage"]),
    )


@dataclass(frozen=True)
class EligibilityThresholds:
    """Defaults per the plan's spec; documented (not silently tuned) if
    ever adjusted for a different data length/universe."""

    minimum_fold_win_rate: float = 0.60
    minimum_worst_fold_cagr: float = 0.0
    maximum_mdd_slack_pp: float = 5.0
    minimum_parameter_neighbor_pass_rate: float = 0.60


@dataclass(frozen=True)
class SelectionResult:
    selector: str
    selected_policy: FilingV7Policy | None
    selected_row: dict[str, Any] | None
    total_candidates: int
    eligible_count: int
    fallback_triggered: bool
    fallback_reason: str | None
    eligibility_table: pd.DataFrame


def compute_base_eligibility(
    candidates: pd.DataFrame,
    thresholds: EligibilityThresholds,
    *,
    baseline_mdd: float,
) -> pd.Series:
    """Eligibility per the spec, excluding ParameterNeighborPassRate (that
    requires eligibility to already be computed for every candidate, so
    it's applied as a second pass by parameter_neighbor_stats())."""

    return (
        candidates["FoldWinRate"].ge(thresholds.minimum_fold_win_rate)
        & candidates["WorstFoldCAGR"].ge(thresholds.minimum_worst_fold_cagr)
        & candidates["MedianFoldCAGR"].gt(candidates["SPYMedianFoldCAGR"])
        & candidates["TrainMaxDrawdown"].ge(baseline_mdd - thresholds.maximum_mdd_slack_pp)
        & candidates["Cost25bpCAGR"].gt(candidates["TrainSPYCAGR"])
    )


def parameter_neighbor_stats(
    candidates: pd.DataFrame,
    base_eligible: pd.Series,
    optimization_grid: dict[str, list[Any]],
) -> pd.DataFrame:
    """For each candidate, find candidates one grid-step away on exactly
    one axis (all other axes identical) and summarize their outcomes.

    "filing_weight 0.08 is best but 0.05 and 0.10 are both bad" is exactly
    what NeighborPassRate/NeighborWorstCAGR are meant to catch -- a single
    sharp peak in an otherwise bad neighborhood is the signature of
    overfitting to this window's noise, not a real, reproducible edge.
    """

    grid_values = {
        field: sorted(
            {v for v in optimization_grid[axis] if v is not None},
            key=lambda v: (v is None, v),
        )
        for field, axis in _POLICY_FIELD_TO_GRID_AXIS.items()
        if axis in optimization_grid
    }
    key_columns = list(grid_values)
    lookup = {
        tuple(row[column] for column in key_columns): index
        for index, row in candidates[key_columns].iterrows()
    }

    rows = []
    for index, row in candidates.iterrows():
        key = tuple(row[column] for column in key_columns)
        neighbor_indices: list[int] = []
        for field in key_columns:
            values = grid_values[field]
            current = row[field]
            try:
                position = values.index(current)
            except ValueError:
                continue
            for neighbor_position in (position - 1, position + 1):
                if 0 <= neighbor_position < len(values):
                    neighbor_key = list(key)
                    neighbor_key[key_columns.index(field)] = values[neighbor_position]
                    neighbor_index = lookup.get(tuple(neighbor_key))
                    if neighbor_index is not None and neighbor_index != index:
                        neighbor_indices.append(neighbor_index)
        neighbor_indices = sorted(set(neighbor_indices))
        neighbor_cagrs = candidates.loc[neighbor_indices, "MedianFoldCAGR"] if neighbor_indices else pd.Series(dtype=float)
        neighbor_worst = candidates.loc[neighbor_indices, "WorstFoldCAGR"] if neighbor_indices else pd.Series(dtype=float)
        neighbor_eligible = base_eligible.loc[neighbor_indices] if neighbor_indices else pd.Series(dtype=bool)
        rows.append(
            {
                "Candidate": row["Candidate"],
                "NeighborCount": len(neighbor_indices),
                "NeighborMedianCAGR": (
                    float(neighbor_cagrs.median()) if len(neighbor_cagrs) else np.nan
                ),
                "NeighborWorstCAGR": (
                    float(neighbor_worst.min()) if len(neighbor_worst) else np.nan
                ),
                "ParameterNeighborPassRate": (
                    float(neighbor_eligible.mean()) if len(neighbor_eligible) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def select_legacy(candidates: pd.DataFrame) -> SelectionResult:
    """Selector A: BEST_TRAIN_ONLY_NO_STRICT_PASS, unchanged, kept only as
    the Phase 4 diagnostic comparison point -- not the production default."""

    ranked = candidates.sort_values(
        ["PassStrict", "ConstraintPenalty", "Objective"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    winner = ranked.iloc[0]
    return SelectionResult(
        selector="A_LEGACY_BEST_TRAIN_ONLY",
        selected_policy=policy_from_candidate_row(winner),
        selected_row=winner.to_dict(),
        total_candidates=len(candidates),
        eligible_count=int(candidates["PassStrict"].sum()),
        fallback_triggered=not bool(winner["PassStrict"]),
        fallback_reason=(
            None if bool(winner["PassStrict"])
            else "No candidate passed the strict train-only constraints; "
            "picked the closest-to-passing by ConstraintPenalty."
        ),
        eligibility_table=candidates[["Candidate", "PassStrict", "ConstraintPenalty", "Objective"]],
    )


def select_stable_region(
    candidates: pd.DataFrame,
    optimization_grid: dict[str, list[Any]],
    *,
    baseline_policy: FilingV7Policy,
    baseline_mdd: float,
    thresholds: EligibilityThresholds = EligibilityThresholds(),
) -> SelectionResult:
    """Selector B: strict eligibility filter, then prefer candidates whose
    immediate parameter-grid neighbors also pass -- a single sharp peak
    surrounded by failing neighbors is treated as noise, not signal.
    """

    base_eligible = compute_base_eligibility(
        candidates, thresholds, baseline_mdd=baseline_mdd
    )
    neighbor_stats = parameter_neighbor_stats(
        candidates, base_eligible, optimization_grid
    )
    merged = candidates.merge(neighbor_stats, on="Candidate", how="left")
    merged["BaseEligible"] = base_eligible.to_numpy()
    merged["Eligible"] = merged["BaseEligible"] & merged[
        "ParameterNeighborPassRate"
    ].ge(thresholds.minimum_parameter_neighbor_pass_rate)

    eligible = merged.loc[merged["Eligible"]]
    if eligible.empty:
        # Per the spec's default: fall back to the frozen Phase 0 baseline
        # policy, not the closest-to-passing candidate and not a silent
        # cash/no-candidate state.
        return SelectionResult(
            selector="B_STRICT_STABLE_REGION",
            selected_policy=baseline_policy,
            selected_row=None,
            total_candidates=len(candidates),
            eligible_count=0,
            fallback_triggered=True,
            fallback_reason=(
                "No candidate passed the strict eligibility filter "
                "(FoldWinRate/WorstFoldCAGR/MedianFoldCAGR-vs-SPY/MDD-slack/"
                "Cost25bp/ParameterNeighborPassRate); fell back to the "
                "frozen Phase 0 baseline policy."
            ),
            eligibility_table=merged,
        )
    winner = eligible.sort_values(
        ["ParameterNeighborPassRate", "MedianFoldCAGR"], ascending=[False, False]
    ).iloc[0]
    return SelectionResult(
        selector="B_STRICT_STABLE_REGION",
        selected_policy=policy_from_candidate_row(winner),
        selected_row=winner.to_dict(),
        total_candidates=len(candidates),
        eligible_count=int(merged["Eligible"].sum()),
        fallback_triggered=False,
        fallback_reason=None,
        eligibility_table=merged,
    )


def select_top_n_consensus(
    candidates: pd.DataFrame,
    optimization_grid: dict[str, list[Any]],
    *,
    baseline_policy: FilingV7Policy,
    baseline_mdd: float,
    top_n: int = 20,
    thresholds: EligibilityThresholds = EligibilityThresholds(),
) -> SelectionResult:
    """Selector C: median (numeric) / mode (categorical-like) of the
    parameters across the top-N eligible candidates by MedianFoldCAGR,
    forming ONE consensus policy -- mirrors
    tsla_integrated/optimization.py::consensus_execution_params's pattern
    from earlier this session, ported to FilingV7Policy's fields.
    """

    base_eligible = compute_base_eligibility(
        candidates, thresholds, baseline_mdd=baseline_mdd
    )
    eligible = candidates.loc[base_eligible].sort_values(
        "MedianFoldCAGR", ascending=False
    )
    if eligible.empty:
        return SelectionResult(
            selector="C_TOP_N_PARAMETER_CONSENSUS",
            selected_policy=baseline_policy,
            selected_row=None,
            total_candidates=len(candidates),
            eligible_count=0,
            fallback_triggered=True,
            fallback_reason="No candidate passed base eligibility; fell back to baseline.",
            eligibility_table=candidates.assign(Eligible=base_eligible),
        )
    pool = eligible.head(top_n)
    consensus_policy = FilingV7Policy(
        filing_weight=float(pool["filing_weight"].median()),
        minimum_filing_coverage=int(round(pool["minimum_filing_coverage"].median())),
        filing_quality_floor=(
            float(pool["filing_quality_floor"].median())
            if pool["filing_quality_floor"].notna().any()
            else None
        ),
        red_flag_veto=bool(pool["red_flag_veto"].mode().iloc[0]),
        top_k=int(round(pool["top_k"].median())),
        exit_rank_buffer=int(round(pool["exit_rank_buffer"].median())),
        hard_stop_return=float(pool["hard_stop_return"].median()),
        minimum_hold_rebalances=int(round(pool["minimum_hold_rebalances"].median())),
        replacement_score_advantage=float(
            pool["replacement_score_advantage"].median()
        ),
    )
    return SelectionResult(
        selector="C_TOP_N_PARAMETER_CONSENSUS",
        selected_policy=consensus_policy,
        selected_row={"PoolSize": len(pool), **consensus_policy.__dict__},
        total_candidates=len(candidates),
        eligible_count=len(eligible),
        fallback_triggered=False,
        fallback_reason=None,
        eligibility_table=candidates.assign(Eligible=base_eligible),
    )


def select_top_n_rank_ensemble(
    candidates: pd.DataFrame,
    scored_panels: dict[int, pd.DataFrame],
    *,
    top_k: int,
    baseline_policy: FilingV7Policy,
    baseline_mdd: float,
    top_n: int = 20,
    thresholds: EligibilityThresholds = EligibilityThresholds(),
) -> tuple[SelectionResult, pd.DataFrame | None]:
    """Selector D: average each eligible candidate's per-ticker daily Rank
    (not held/not-held union), take top_k from the averaged ranking.

    `scored_panels` maps Candidate -> the frame produced by
    add_filing_v7_scores(...) for that candidate (has Date/Ticker/Rank).
    top_k stays fixed at the base_v7_params value -- the ensemble changes
    which names are considered, never how many are held.
    """

    base_eligible = compute_base_eligibility(
        candidates, thresholds, baseline_mdd=baseline_mdd
    )
    eligible = candidates.loc[base_eligible].sort_values(
        "MedianFoldCAGR", ascending=False
    )
    if eligible.empty:
        return (
            SelectionResult(
                selector="D_TOP_N_RANK_ENSEMBLE",
                selected_policy=baseline_policy,
                selected_row=None,
                total_candidates=len(candidates),
                eligible_count=0,
                fallback_triggered=True,
                fallback_reason="No candidate passed base eligibility; fell back to baseline.",
                eligibility_table=candidates.assign(Eligible=base_eligible),
            ),
            None,
        )
    pool_ids = eligible.head(top_n)["Candidate"].tolist()
    frames = [scored_panels[cid][["Date", "Ticker", "Rank"]] for cid in pool_ids if cid in scored_panels]
    if not frames:
        raise ValueError("No scored panels available for the eligible pool")
    stacked = pd.concat(frames, ignore_index=True)
    averaged = (
        stacked.groupby(["Date", "Ticker"])["Rank"]
        .mean()
        .reset_index()
        .rename(columns={"Rank": "AverageRank"})
    )
    averaged["EnsembleRank"] = averaged.groupby("Date")["AverageRank"].rank(
        method="first"
    )
    ensemble_targets = averaged.loc[averaged["EnsembleRank"].le(top_k)].copy()
    ensemble_targets["TargetWeight"] = 1.0 / top_k
    return (
        SelectionResult(
            selector="D_TOP_N_RANK_ENSEMBLE",
            selected_policy=None,  # ensemble has no single policy; see ensemble_targets
            selected_row={"PoolSize": len(pool_ids), "PoolCandidateIds": pool_ids},
            total_candidates=len(candidates),
            eligible_count=len(eligible),
            fallback_triggered=False,
            fallback_reason=None,
            eligibility_table=candidates.assign(Eligible=base_eligible),
        ),
        ensemble_targets[["Date", "Ticker", "TargetWeight"]],
    )
