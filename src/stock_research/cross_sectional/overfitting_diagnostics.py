from __future__ import annotations

import itertools
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, rankdata, skew, spearmanr

from stock_research.io_utils import atomic_to_csv

EULER_MASCHERONI = 0.5772156649015329
ANNUALIZATION = 252

PARAMETER_COLUMNS = [
    "momentum_weight",
    "trend_weight",
    "growth_weight",
    "quality_weight",
    "risk_control_weight",
    "top_k",
    "exit_rank",
    "trend_floor",
    "momentum_floor",
]
TRAINING_BLOCK_COLUMNS = [
    "Fold1ExcessCAGR",
    "Fold2ExcessCAGR",
    "Fold3ExcessCAGR",
]
CSCV_BLOCK_COLUMNS = [
    *TRAINING_BLOCK_COLUMNS,
    "SelectionExcessCAGR",
]
BLOCK_LABELS = [
    "TRAIN_2020_2021",
    "TRAIN_2022",
    "TRAIN_2023_2024",
    "SELECTION_2025_CONTAMINATED",
]
BLOCK_YEARS = np.array([2.0, 1.0, 2.0, 1.0])
CANDIDATE_REQUIRED_COLUMNS = {
    "Candidate",
    "TrainROI",
    "TrainCAGR",
    "TrainSharpe",
    "SelectionROI",
    "SelectionCAGR",
    "SelectionSharpe",
    *PARAMETER_COLUMNS,
    *CSCV_BLOCK_COLUMNS,
}


@dataclass(frozen=True)
class OverfittingDiagnostic:
    schema: pd.DataFrame
    dsr: pd.DataFrame
    effective_trials: pd.DataFrame
    pbo_summary: pd.DataFrame
    cscv_splits: pd.DataFrame
    rank_correlation: pd.DataFrame
    candidate_ranks: pd.DataFrame


def input_schema() -> pd.DataFrame:
    rows = [
        ("Candidate", "integer", "candidate id"),
        *[
            (column, "float/integer", "one of nine sampled dimensions")
            for column in PARAMETER_COLUMNS
        ],
        ("TrainROI", "float percent", "2020-2024 net ROI"),
        ("TrainCAGR", "float percent", "2020-2024 CAGR"),
        ("TrainSharpe", "annualized float", "2020-2024 Sharpe"),
        (
            "Fold1ExcessCAGR",
            "float percent",
            "2020-2021 excess CAGR block",
        ),
        ("Fold2ExcessCAGR", "float percent", "2022 excess CAGR block"),
        (
            "Fold3ExcessCAGR",
            "float percent",
            "2023-2024 excess CAGR block",
        ),
        (
            "SelectionExcessCAGR",
            "float percent",
            "2025 contaminated selection block",
        ),
        ("SelectionROI", "float percent", "2025 contaminated net ROI"),
        ("SelectionCAGR", "float percent", "2025 contaminated CAGR"),
        (
            "SelectionSharpe",
            "annualized float",
            "2025 contaminated selection Sharpe",
        ),
        (
            "scenario_equity: Date",
            "ISO date",
            "selected strategy training equity date",
        ),
        (
            "scenario_equity: Equity",
            "float",
            "selected strategy end-of-day equity",
        ),
    ]
    return pd.DataFrame(rows, columns=["Column", "Type", "Use"])


def validate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(CANDIDATE_REQUIRED_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate data is missing columns: {missing}")
    result = candidates.copy()
    numeric = sorted(CANDIDATE_REQUIRED_COLUMNS - {"Candidate"})
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["Candidate"] = pd.to_numeric(
        result["Candidate"],
        errors="raise",
        downcast="integer",
    )
    if result["Candidate"].duplicated().any():
        raise ValueError("Candidate ids must be unique")
    if result[numeric].isna().any().any():
        raise ValueError("Required candidate metrics contain missing values")
    return result.reset_index(drop=True)


def selected_training_returns(
    equity: pd.DataFrame,
    *,
    scenario: str,
    period: str,
) -> pd.Series:
    required = {"Scenario", "Period", "Date", "Equity"}
    missing = sorted(required - set(equity.columns))
    if missing:
        raise ValueError(f"Equity data is missing columns: {missing}")
    selected = equity.loc[
        equity["Scenario"].eq(scenario) & equity["Period"].eq(period)
    ].copy()
    if len(selected) < 3:
        raise ValueError(
            f"Insufficient equity rows for {scenario}/{period}: "
            f"{len(selected)}"
        )
    selected["Date"] = pd.to_datetime(selected["Date"], errors="raise")
    selected["Equity"] = pd.to_numeric(
        selected["Equity"],
        errors="raise",
    )
    selected = selected.sort_values("Date")
    returns = selected["Equity"].pct_change(fill_method=None).dropna()
    if len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError("Selected training returns are not usable")
    return returns


def expected_maximum_sharpe(
    sharpe_standard_deviation: float,
    number_of_trials: float,
) -> float:
    if sharpe_standard_deviation < 0:
        raise ValueError("Sharpe standard deviation cannot be negative")
    if number_of_trials <= 1:
        return 0.0
    max_z = (
        (1 - EULER_MASCHERONI)
        * norm.ppf(1 - 1 / number_of_trials)
        + EULER_MASCHERONI
        * norm.ppf(1 - 1 / (number_of_trials * math.e))
    )
    return float(sharpe_standard_deviation * max_z)


def deflated_sharpe_ratio(
    observed_annualized_sharpe: float,
    expected_max_annualized_sharpe: float,
    returns: pd.Series,
    *,
    annualization: int = ANNUALIZATION,
) -> tuple[float, float, float, float]:
    values = pd.Series(returns, dtype=float).dropna()
    if len(values) < 3:
        raise ValueError("DSR requires at least three return observations")
    observed = observed_annualized_sharpe / math.sqrt(annualization)
    benchmark = expected_max_annualized_sharpe / math.sqrt(annualization)
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    denominator_squared = (
        1
        - sample_skew * observed
        + (sample_kurtosis - 1) * observed**2 / 4
    )
    if denominator_squared <= 0:
        raise ValueError("Non-normal Sharpe variance term is not positive")
    z_score = (
        (observed - benchmark)
        * math.sqrt(len(values) - 1)
        / math.sqrt(denominator_squared)
    )
    return (
        float(norm.cdf(z_score)),
        float(z_score),
        sample_skew,
        sample_kurtosis,
    )


def estimate_effective_trials(
    candidates: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    frame = validate_candidates(candidates)
    block_values = frame[TRAINING_BLOCK_COLUMNS].to_numpy(dtype=float)
    correlations = np.corrcoef(block_values)
    upper = correlations[np.triu_indices(len(correlations), k=1)]
    average_correlation = float(np.nanmean(upper))
    bounded_correlation = float(np.clip(average_correlation, 0.0, 1.0))
    nominal = float(len(frame))
    effective = bounded_correlation + (1 - bounded_correlation) * nominal
    diagnostics = pd.DataFrame(
        [
            {
                "Method": "BAILEY_AVERAGE_CORRELATION_THREE_TRAIN_FOLD_PROXY",
                "NominalTrials": nominal,
                "SearchDimensions": len(PARAMETER_COLUMNS),
                "UniqueNineDimensionalTuples": int(
                    len(frame[PARAMETER_COLUMNS].drop_duplicates())
                ),
                "PerformanceBlocks": len(TRAINING_BLOCK_COLUMNS),
                "AveragePairwiseCorrelation": average_correlation,
                "BoundedAverageCorrelation": bounded_correlation,
                "EffectiveTrials": effective,
                "Limitation": (
                    "Three aggregate training folds are used because "
                    "candidate-by-date returns were not saved."
                ),
            }
        ]
    )
    return effective, diagnostics


def calculate_dsr(
    candidates: pd.DataFrame,
    selected_returns: pd.Series,
    *,
    selected_candidate: int,
    nominal_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = validate_candidates(candidates)
    selected = frame.loc[frame["Candidate"].eq(selected_candidate)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected candidate {selected_candidate} exactly once"
        )
    observed = float(selected["TrainSharpe"].iloc[0])
    trial_variance = float(frame["TrainSharpe"].var(ddof=1))
    trial_standard_deviation = math.sqrt(trial_variance)
    effective, effective_details = estimate_effective_trials(frame)
    proxy_sharpe = float(
        selected_returns.mean()
        / selected_returns.std(ddof=1)
        * math.sqrt(ANNUALIZATION)
    )
    rows: list[dict[str, object]] = []
    for label, trial_count in (
        ("NOMINAL_2000", float(nominal_trials)),
        ("EFFECTIVE_9D_THREE_TRAIN_FOLD_PROXY", effective),
    ):
        expected = expected_maximum_sharpe(
            trial_standard_deviation,
            trial_count,
        )
        dsr, z_score, sample_skew, sample_kurtosis = (
            deflated_sharpe_ratio(
                observed,
                expected,
                selected_returns,
            )
        )
        rows.append(
            {
                "Scenario": label,
                "NumberOfTrials": trial_count,
                "SearchDimensions": len(PARAMETER_COLUMNS),
                "ObservedTrainSharpe": observed,
                "ReturnProxySharpe": proxy_sharpe,
                "ProxySharpeDifference": proxy_sharpe - observed,
                "AcrossTrialSharpeVariance": trial_variance,
                "ExpectedMaximumSharpeUnderZeroNull": expected,
                "ReturnObservations": len(selected_returns),
                "ReturnSkewness": sample_skew,
                "ReturnPearsonKurtosis": sample_kurtosis,
                "DSRZScore": z_score,
                "DSRProbability": dsr,
                "DSRClassification": classify_dsr(dsr),
            }
        )
    return pd.DataFrame(rows), effective_details


def calculate_coarse_cscv_pbo(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = validate_candidates(candidates)
    blocks = frame[CSCV_BLOCK_COLUMNS].to_numpy(dtype=float).T
    rows: list[dict[str, object]] = []
    block_indices = range(len(CSCV_BLOCK_COLUMNS))
    for split_number, in_sample in enumerate(
        itertools.combinations(block_indices, len(BLOCK_LABELS) // 2),
        start=1,
    ):
        out_of_sample = tuple(
            index for index in block_indices if index not in in_sample
        )
        in_metric = np.average(
            blocks[list(in_sample)],
            axis=0,
            weights=BLOCK_YEARS[list(in_sample)],
        )
        out_metric = np.average(
            blocks[list(out_of_sample)],
            axis=0,
            weights=BLOCK_YEARS[list(out_of_sample)],
        )
        winner_index = int(np.nanargmax(in_metric))
        ranks = rankdata(out_metric, method="average")
        percentile = float(
            (ranks[winner_index] - 1) / (len(frame) - 1)
        )
        clipped = float(np.clip(percentile, 1e-12, 1 - 1e-12))
        logit = float(math.log(clipped / (1 - clipped)))
        rows.append(
            {
                "Split": split_number,
                "ISBlocks": "|".join(BLOCK_LABELS[index] for index in in_sample),
                "OOSBlocks": "|".join(
                    BLOCK_LABELS[index] for index in out_of_sample
                ),
                "ISWinnerCandidate": int(
                    frame.iloc[winner_index]["Candidate"]
                ),
                "ISMetric": float(in_metric[winner_index]),
                "OOSMetric": float(out_metric[winner_index]),
                "OOSPercentile": percentile,
                "OOSLogit": logit,
                "BelowOOSMedian": logit <= 0,
            }
        )
    splits = pd.DataFrame(rows)
    pbo = float(splits["BelowOOSMedian"].mean())
    summary = pd.DataFrame(
        [
            {
                "Method": "COARSE_CSCV_S4_AGGREGATE_BLOCKS",
                "PBO": pbo,
                "Classification": classify_pbo(pbo),
                "Splits": len(splits),
                "Resolution": 1 / len(splits),
                "ExactCandidateByDateMatrixAvailable": False,
                "ValidationIsFresh": False,
                "Limitation": (
                    "Uses three stored training folds plus contaminated 2025 "
                    "selection metrics; it is not full daily-return CSCV."
                ),
            }
        ]
    )
    return summary, splits


def calculate_rank_diagnostics(
    candidates: pd.DataFrame,
    *,
    selected_candidate: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = validate_candidates(candidates)
    correlation = spearmanr(
        frame["TrainSharpe"],
        frame["SelectionSharpe"],
        nan_policy="raise",
    )
    train_winner_index = int(frame["TrainSharpe"].to_numpy().argmax())
    selected_index = int(
        frame.index[frame["Candidate"].eq(selected_candidate)][0]
    )
    summary = pd.DataFrame(
        [
            {
                "ISMetric": "TrainSharpe",
                "OOSLabel": "SelectionSharpe_2025_CONTAMINATED",
                "SpearmanRho": float(correlation.statistic),
                "PValue": float(correlation.pvalue),
                "Candidates": len(frame),
                "Interpretation": classify_rank_correlation(
                    float(correlation.statistic)
                ),
            }
        ]
    )
    rows = []
    for label, index in (
        ("MAX_TRAIN_SHARPE", train_winner_index),
        ("SELECTED_CANDIDATE", selected_index),
    ):
        rows.append(
            {
                "Role": label,
                "Candidate": int(frame.iloc[index]["Candidate"]),
                "TrainSharpe": float(frame.iloc[index]["TrainSharpe"]),
                "TrainSharpePercentile": _percentile(
                    frame["TrainSharpe"],
                    index,
                ),
                "SelectionSharpe": float(
                    frame.iloc[index]["SelectionSharpe"]
                ),
                "SelectionSharpePercentile": _percentile(
                    frame["SelectionSharpe"],
                    index,
                ),
                "SelectionExcessCAGR": float(
                    frame.iloc[index]["SelectionExcessCAGR"]
                ),
                "SelectionExcessCAGRPercentile": _percentile(
                    frame["SelectionExcessCAGR"],
                    index,
                ),
            }
        )
    return summary, pd.DataFrame(rows)


def create_is_oos_scatter_svg(
    candidates: pd.DataFrame,
    path: str | Path,
    *,
    selected_candidate: int,
) -> Path:
    frame = validate_candidates(candidates)
    width, height = 960, 620
    left, right, top, bottom = 80, 35, 35, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    x = frame["TrainSharpe"].to_numpy(dtype=float)
    y = frame["SelectionSharpe"].to_numpy(dtype=float)
    x_min, x_max = _padded_range(x)
    y_min, y_max = _padded_range(y)

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    selected_mask = frame["Candidate"].eq(selected_candidate)
    train_winner = int(frame["TrainSharpe"].idxmax())
    circles = []
    for row in frame.itertuples(index=True):
        if row.Candidate == selected_candidate or row.Index == train_winner:
            continue
        circles.append(
            (
                f'<circle cx="{x_coord(row.TrainSharpe):.2f}" '
                f'cy="{y_coord(row.SelectionSharpe):.2f}" r="2.1" '
                'class="candidate"/>'
            )
        )
    selected = frame.loc[selected_mask].iloc[0]
    winner = frame.loc[train_winner]
    x_ticks = np.linspace(x_min, x_max, 6)
    y_ticks = np.linspace(y_min, y_max, 6)
    x_grid = "\n".join(
        (
            f'<line x1="{x_coord(value):.2f}" y1="{top}" '
            f'x2="{x_coord(value):.2f}" y2="{height-bottom}" '
            'class="grid"/>'
            f'<text x="{x_coord(value):.2f}" y="{height-bottom+28}" '
            f'class="tick" text-anchor="middle">{value:.2f}</text>'
        )
        for value in x_ticks
    )
    y_grid = "\n".join(
        (
            f'<line x1="{left}" y1="{y_coord(value):.2f}" '
            f'x2="{width-right}" y2="{y_coord(value):.2f}" '
            'class="grid"/>'
            f'<text x="{left-12}" y="{y_coord(value)+4:.2f}" '
            f'class="tick" text-anchor="end">{value:.2f}</text>'
        )
        for value in y_ticks
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}"
height="{height}" viewBox="0 0 {width} {height}"
role="img" aria-labelledby="title description">
<title id="title">Train Sharpe versus contaminated 2025 selection Sharpe</title>
<desc id="description">Scatter of 2000 V6-B candidates. Candidate 1931 and the
maximum train-Sharpe candidate are highlighted.</desc>
<style>
.background {{ fill: #111827; }}
.axis {{ stroke: #9ca3af; stroke-width: 1.2; }}
.grid {{ stroke: #374151; stroke-width: 1; }}
.tick, .label, .annotation {{ fill: #e5e7eb; font-family: sans-serif; }}
.tick {{ font-size: 12px; }}
.label {{ font-size: 15px; }}
.annotation {{ font-size: 13px; }}
.candidate {{ fill: #60a5fa; fill-opacity: .25; }}
.selected {{ fill: #f59e0b; stroke: #fff; stroke-width: 1.5; }}
.winner {{ fill: #34d399; stroke: #fff; stroke-width: 1.5; }}
</style>
<rect width="{width}" height="{height}" class="background"/>
{x_grid}
{y_grid}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}"
y2="{height-bottom}" class="axis"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"
class="axis"/>
{"".join(circles)}
<circle cx="{x_coord(float(selected.TrainSharpe)):.2f}"
cy="{y_coord(float(selected.SelectionSharpe)):.2f}" r="7"
class="selected"/>
<circle cx="{x_coord(float(winner.TrainSharpe)):.2f}"
cy="{y_coord(float(winner.SelectionSharpe)):.2f}" r="7"
class="winner"/>
<text x="{x_coord(float(selected.TrainSharpe))+10:.2f}"
y="{y_coord(float(selected.SelectionSharpe))-10:.2f}"
class="annotation">#1931 selected</text>
<text x="{x_coord(float(winner.TrainSharpe))-10:.2f}"
y="{y_coord(float(winner.SelectionSharpe))+24:.2f}"
class="annotation" text-anchor="end">#{int(winner.Candidate)} max train SR</text>
<text x="{left+plot_width/2:.2f}" y="{height-24}" class="label"
text-anchor="middle">Training Sharpe (2020–2024)</text>
<text x="22" y="{top+plot_height/2:.2f}" class="label"
text-anchor="middle"
transform="rotate(-90 22 {top+plot_height/2:.2f})">2025 selection Sharpe
(not fresh OOS)</text>
</svg>
"""
    return _atomic_text(svg, Path(path))


def build_overfitting_diagnostic(
    candidates: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    selected_candidate: int,
    nominal_trials: int,
    scenario: str,
    period: str,
) -> OverfittingDiagnostic:
    frame = validate_candidates(candidates)
    returns = selected_training_returns(
        equity,
        scenario=scenario,
        period=period,
    )
    dsr, effective = calculate_dsr(
        frame,
        returns,
        selected_candidate=selected_candidate,
        nominal_trials=nominal_trials,
    )
    pbo_summary, splits = calculate_coarse_cscv_pbo(frame)
    rank_summary, candidate_ranks = calculate_rank_diagnostics(
        frame,
        selected_candidate=selected_candidate,
    )
    return OverfittingDiagnostic(
        schema=input_schema(),
        dsr=dsr,
        effective_trials=effective,
        pbo_summary=pbo_summary,
        cscv_splits=splits,
        rank_correlation=rank_summary,
        candidate_ranks=candidate_ranks,
    )


def write_overfitting_outputs(
    diagnostic: OverfittingDiagnostic,
    candidates: pd.DataFrame,
    output_dir: str | Path,
    *,
    selected_candidate: int,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "schema": atomic_to_csv(
            diagnostic.schema,
            destination / "input_schema.csv",
            index=False,
        ),
        "dsr": atomic_to_csv(
            diagnostic.dsr,
            destination / "dsr_results.csv",
            index=False,
        ),
        "effective_trials": atomic_to_csv(
            diagnostic.effective_trials,
            destination / "effective_trials.csv",
            index=False,
        ),
        "pbo": atomic_to_csv(
            diagnostic.pbo_summary,
            destination / "pbo_summary.csv",
            index=False,
        ),
        "splits": atomic_to_csv(
            diagnostic.cscv_splits,
            destination / "cscv_splits.csv",
            index=False,
        ),
        "rank_correlation": atomic_to_csv(
            diagnostic.rank_correlation,
            destination / "rank_correlation.csv",
            index=False,
        ),
        "candidate_ranks": atomic_to_csv(
            diagnostic.candidate_ranks,
            destination / "candidate_ranks.csv",
            index=False,
        ),
    }
    outputs["scatter"] = create_is_oos_scatter_svg(
        candidates,
        destination / "is_vs_2025_sharpe_scatter.svg",
        selected_candidate=selected_candidate,
    )
    return outputs


def classify_dsr(value: float) -> str:
    if value >= 0.95:
        return "STRONG"
    if value >= 0.5:
        return "AMBIGUOUS"
    return "WEAK"


def classify_pbo(value: float) -> str:
    if value <= 0.25:
        return "LOW"
    if value <= 0.5:
        return "CAUTION"
    return "OVERFITTING_MORE_LIKELY"


def classify_rank_correlation(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.2:
        return "VERY_WEAK"
    if absolute < 0.4:
        return "WEAK"
    if absolute < 0.6:
        return "MODERATE"
    return "STRONG"


def _percentile(values: pd.Series, index: int) -> float:
    ranks = rankdata(values.to_numpy(dtype=float), method="average")
    return float((ranks[index] - 1) / (len(values) - 1))


def _padded_range(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    padding = max((maximum - minimum) * 0.05, 0.01)
    return minimum - padding, maximum + padding


def _atomic_text(content: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
