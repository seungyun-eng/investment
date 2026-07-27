from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_fear_buy_sp500.config import FearBuyParams
from stock_research.macro_fear_buy_sp500.contributions import (
    ContributionConfig,
    ContributionDeploymentPolicy,
    run_contribution_backtest,
)
from stock_research.macro_fear_buy_sp500.features import build_fear_features
from stock_research.macro_fear_buy_sp500.mass_optimization import (
    add_selection_scores,
    candidate_to_params,
    candidate_to_policy,
    constant_signals,
    evaluate_candidate_batch,
    evaluate_frozen_candidate,
    initialize_worker,
    sample_candidates,
    select_category_winners,
)
from stock_research.macro_fear_buy_sp500.strategy import (
    generate_fear_buy_signals,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate at least 100,000 fear-buy parameter candidates on "
            "development data, then freeze three category winners."
        )
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--selected-params", type=Path)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--shortlist-size", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--initial", type=float, default=40_000.0)
    parser.add_argument("--monthly", type=float, default=4_000.0)
    parser.add_argument("--development-end", default="2016-12-30")
    parser.add_argument("--holdout-start", default="2017-01-03")
    return parser.parse_args()


def _latest(folder: Path, pattern: str) -> Path:
    hits = list(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return max(hits, key=lambda path: path.stat().st_mtime)


def _atomic_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".json.tmp",
        prefix=path.stem + "_",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _batch(
    candidates: list[dict[str, float | int]],
    size: int,
) -> list[list[tuple[int, dict[str, float | int]]]]:
    indexed = list(enumerate(candidates, start=1))
    return [
        indexed[offset : offset + size]
        for offset in range(0, len(indexed), size)
    ]


def _shortlist_ids(
    screening: pd.DataFrame,
    benchmark_profit: float,
    *,
    size: int,
) -> list[int]:
    scored = screening.copy()
    scored["DevelopmentProfitRatio"] = (
        scored["DevelopmentNetProfit"] / benchmark_profit
    )
    positive_drawdown = -scored["DevelopmentMDD(%)"].clip(upper=-1e-6)
    scored["PreliminaryCalmar"] = (
        scored["DevelopmentXIRR(%)"] / positive_drawdown
    )
    scored["PreliminaryBalanced"] = (
        0.55 * scored["DevelopmentProfitRatio"]
        + 0.25 * scored["DevelopmentSharpe"]
        + 0.20 * scored["PreliminaryCalmar"]
    )
    selected: set[int] = set()
    selected.update(
        scored.nlargest(size, "DevelopmentProfitRatio")[
            "CandidateId"
        ].astype(int)
    )
    selected.update(
        scored.nlargest(size, "PreliminaryBalanced")[
            "CandidateId"
        ].astype(int)
    )
    safety_pool = scored[scored["DevelopmentProfitRatio"] >= 0.85]
    selected.update(
        safety_pool.nlargest(size, "DevelopmentMDD(%)")[
            "CandidateId"
        ].astype(int)
    )
    selected.add(1)
    return sorted(selected)


def _benchmark_metrics(
    features: pd.DataFrame,
    baseline_params: FearBuyParams,
    config: ContributionConfig,
    *,
    start: str | None,
    end: str | None,
) -> tuple[dict[str, float], object]:
    period = features.copy()
    if start:
        period = period[period["Date"] >= pd.Timestamp(start)]
    if end:
        period = period[period["Date"] <= pd.Timestamp(end)]
    signals = generate_fear_buy_signals(
        period.reset_index(drop=True),
        baseline_params,
    )
    result = run_contribution_backtest(
        constant_signals(signals),
        baseline_params,
        config,
        name="MonthlyBuyHold",
        core_weight_override=1.0,
        invest_contributions_without_signal=True,
    )
    summary = result.summary
    return {
        "FinalValue": summary.final_value,
        "NetProfit": summary.net_profit,
        "ROI(%)": summary.roi_percent,
        "XIRR(%)": summary.money_weighted_return_percent,
        "MDD(%)": summary.max_drawdown_percent,
        "Sharpe": summary.sharpe_ratio,
    }, result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.count < 100_000:
        print(
            "Warning: requested candidate count is below the 100,000-run "
            "research target.",
            file=sys.stderr,
        )
    paths = load_paths(args.stock_root)
    research_folder = paths.results / "SP500" / "macro_fear_buy_sp500"
    output_folder = (
        research_folder / "monthly_contributions" / "massive_optimization"
    )
    checkpoint_path = output_folder / (
        f"screening_checkpoint_v2_seed{args.seed}_count{args.count}.csv"
    )
    prediction_path = (
        args.predictions
        or _latest(
            paths.results / "SP500" / "macro_momentum_sp500",
            "oos_predictions_*.csv",
        )
    ).resolve()
    selected_path = (
        args.selected_params
        or _latest(research_folder, "selected_params_*.json")
    ).resolve()
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if bool(selected_payload.get("quick_diagnostic")):
        raise ValueError("Refusing parameters from a quick diagnostic run.")
    baseline_params = FearBuyParams(**selected_payload["strategy"])
    baseline_policy = ContributionDeploymentPolicy(
        mild_fraction=0.0,
        fear_fraction=1.0,
        panic_fraction=1.0,
        cooldown_sessions=21,
    )
    config = ContributionConfig(
        initial_lump_sum=args.initial,
        monthly_contribution=args.monthly,
        transaction_cost_bps=float(
            selected_payload["research"]["transaction_cost_bps"]
        ),
        slippage_bps=float(selected_payload["research"]["slippage_bps"]),
    )
    predictions = pd.read_csv(prediction_path, parse_dates=["Date"])
    baseline_features = build_fear_features(predictions, baseline_params)
    benchmark_periods = {
        "Development": (None, args.development_end),
        "DevelopmentCrisis": (None, "2011-12-30"),
        "DevelopmentBull": ("2012-01-03", args.development_end),
        "Holdout": (args.holdout_start, None),
        "Full": (None, None),
    }
    benchmark_metrics: dict[str, dict[str, float]] = {}
    benchmark_results: dict[str, object] = {}
    for period, (start, end) in benchmark_periods.items():
        metrics, result = _benchmark_metrics(
            baseline_features,
            baseline_params,
            config,
            start=start,
            end=end,
        )
        benchmark_metrics[period] = metrics
        benchmark_results[period] = result

    candidates = sample_candidates(
        args.count,
        seed=args.seed,
        baseline_params=baseline_params,
        baseline_policy=baseline_policy,
    )
    screening_rows: list[dict[str, float | int]] = []
    if checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        screening_rows = checkpoint.to_dict(orient="records")
    completed_ids = {
        int(row["CandidateId"])
        for row in screening_rows
    }
    remaining_indexed = [
        (candidate_id, candidate)
        for candidate_id, candidate in enumerate(candidates, start=1)
        if candidate_id not in completed_ids
    ]
    screening_batches = [
        remaining_indexed[offset : offset + args.batch_size]
        for offset in range(0, len(remaining_indexed), args.batch_size)
    ]
    worker_config = {
        "initial_lump_sum": config.initial_lump_sum,
        "monthly_contribution": config.monthly_contribution,
        "transaction_cost_bps": config.transaction_cost_bps,
        "slippage_bps": config.slippage_bps,
    }
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(
            str(prediction_path),
            asdict(baseline_params),
            worker_config,
            args.development_end,
            False,
        ),
    ) as executor:
        for batch_number, batch_rows in enumerate(
            executor.map(evaluate_candidate_batch, screening_batches),
            start=1,
        ):
            screening_rows.extend(batch_rows)
            if (
                args.checkpoint_every > 0
                and len(screening_rows) % args.checkpoint_every
                < args.batch_size
            ):
                atomic_to_csv(
                    pd.DataFrame(screening_rows),
                    checkpoint_path,
                    index=False,
                )
            if batch_number % max(1, 1_000 // args.batch_size) == 0:
                print(
                    f"Screened={len(screening_rows)}/{args.count}",
                    flush=True,
                )

    screening_frame = pd.DataFrame(screening_rows)
    atomic_to_csv(screening_frame, checkpoint_path, index=False)
    shortlist_ids = _shortlist_ids(
        screening_frame,
        benchmark_metrics["Development"]["NetProfit"],
        size=args.shortlist_size,
    )
    indexed_shortlist = [
        (candidate_id, candidates[candidate_id - 1])
        for candidate_id in shortlist_ids
    ]
    refinement_batches = [
        indexed_shortlist[offset : offset + args.batch_size]
        for offset in range(0, len(indexed_shortlist), args.batch_size)
    ]
    rows: list[dict[str, float | int]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(
            str(prediction_path),
            asdict(baseline_params),
            worker_config,
            args.development_end,
            True,
        ),
    ) as executor:
        for batch_number, batch_rows in enumerate(
            executor.map(evaluate_candidate_batch, refinement_batches),
            start=1,
        ):
            rows.extend(batch_rows)
            if batch_number % max(1, 250 // args.batch_size) == 0:
                print(
                    f"Refined={len(rows)}/{len(indexed_shortlist)}",
                    flush=True,
                )

    candidate_frame = add_selection_scores(
        pd.DataFrame(rows),
        benchmark_metrics,
    )
    winners = select_category_winners(candidate_frame)
    evaluation_rows: list[dict[str, object]] = []
    frozen_payloads: list[dict[str, object]] = []
    evaluation_periods = {
        "Development": (None, args.development_end),
        "Holdout": (args.holdout_start, None),
        "Full": (None, None),
    }
    for winner in winners.to_dict(orient="records"):
        category = str(winner["SelectionCategory"])
        candidate = {
            name: winner[name]
            for name in candidates[0]
        }
        params = candidate_to_params(candidate)
        policy = candidate_to_policy(candidate)
        frozen_payloads.append(
            {
                "selection_category": category,
                "candidate_id": int(winner["CandidateId"]),
                "strategy": asdict(params),
                "deployment_policy": asdict(policy),
            }
        )
        for period, (start, end) in evaluation_periods.items():
            period_features = baseline_features.copy()
            if start:
                period_features = period_features[
                    period_features["Date"] >= pd.Timestamp(start)
                ]
            if end:
                period_features = period_features[
                    period_features["Date"] <= pd.Timestamp(end)
                ]
            _, _, result = evaluate_frozen_candidate(
                period_features.reset_index(drop=True),
                candidate,
                config,
                name=f"{category}Frozen",
            )
            benchmark = benchmark_results[period]
            summary = result.summary
            benchmark_summary = benchmark.summary
            evaluation_rows.append(
                {
                    "SelectionCategory": category,
                    "CandidateId": int(winner["CandidateId"]),
                    "Period": period,
                    "Portfolio": "Strategy",
                    "FinalValue": summary.final_value,
                    "NetProfit": summary.net_profit,
                    "ProfitRatioVsBuyHold": (
                        summary.net_profit / benchmark_summary.net_profit
                    ),
                    "ROI(%)": summary.roi_percent,
                    "XIRR(%)": summary.money_weighted_return_percent,
                    "MDD(%)": summary.max_drawdown_percent,
                    "Sharpe": summary.sharpe_ratio,
                    "AverageExposure(%)": (
                        summary.average_exposure_percent
                    ),
                    "Trades": summary.trade_count,
                }
            )
            evaluation_rows.append(
                {
                    "SelectionCategory": category,
                    "CandidateId": int(winner["CandidateId"]),
                    "Period": period,
                    "Portfolio": "MonthlyBuyHold",
                    "FinalValue": benchmark_summary.final_value,
                    "NetProfit": benchmark_summary.net_profit,
                    "ProfitRatioVsBuyHold": 1.0,
                    "ROI(%)": benchmark_summary.roi_percent,
                    "XIRR(%)": (
                        benchmark_summary.money_weighted_return_percent
                    ),
                    "MDD(%)": benchmark_summary.max_drawdown_percent,
                    "Sharpe": benchmark_summary.sharpe_ratio,
                    "AverageExposure(%)": (
                        benchmark_summary.average_exposure_percent
                    ),
                    "Trades": benchmark_summary.trade_count,
                }
            )

    evaluation = pd.DataFrame(evaluation_rows)
    timestamp = datetime.now(UTC).astimezone().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    screening_path = atomic_to_csv(
        screening_frame,
        output_folder / f"development_screening_{timestamp}.csv",
        index=False,
    )
    candidate_path = atomic_to_csv(
        candidate_frame.sort_values("ReturnScore", ascending=False),
        output_folder / f"development_candidates_{timestamp}.csv",
        index=False,
    )
    winner_path = atomic_to_csv(
        winners,
        output_folder / f"development_winners_{timestamp}.csv",
        index=False,
    )
    evaluation_path = atomic_to_csv(
        evaluation,
        output_folder / f"frozen_evaluation_{timestamp}.csv",
        index=False,
    )
    manifest_path = _atomic_json(
        {
            "candidate_count": args.count,
            "seed": args.seed,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "shortlist_size_per_objective": args.shortlist_size,
            "refined_candidate_count": len(indexed_shortlist),
            "screening_checkpoint": str(checkpoint_path),
            "selected_on": f"Date <= {args.development_end}",
            "untouched_holdout_start": args.holdout_start,
            "prediction_source": str(prediction_path),
            "baseline_parameter_source": str(selected_path),
            "selection_categories": frozen_payloads,
            "selection_scores": {
                "ReturnScore": (
                    "0.75*DevelopmentProfitRatio + "
                    "0.25*WorstDevelopmentFoldProfitRatio"
                ),
                "BalancedScore": (
                    "0.40*DevelopmentProfitRatio + "
                    "0.25*WorstDevelopmentFoldProfitRatio + "
                    "0.20*DevelopmentSharpe + "
                    "0.15*DevelopmentCalmar"
                ),
                "SafetyScore": (
                    "eligible if development profit ratio >=0.85 and "
                    "worst fold >=0.60; then 0.60*MDD improvement + "
                    "0.25*Sharpe + 0.15*profit ratio"
                ),
            },
            "files": {
                "screening": str(screening_path),
                "candidates": str(candidate_path),
                "winners": str(winner_path),
                "evaluation": str(evaluation_path),
            },
        },
        output_folder / f"manifest_{timestamp}.json",
    )
    print(winners.to_string(index=False))
    print(evaluation.to_string(index=False))
    print(f"screening={screening_path}")
    print(f"candidates={candidate_path}")
    print(f"winners={winner_path}")
    print(f"evaluation={evaluation_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
