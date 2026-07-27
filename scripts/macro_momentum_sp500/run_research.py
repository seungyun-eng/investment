from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.evaluation import nested_walk_forward
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.macro_momentum_sp500.portfolio import (
    allocation_sensitivity,
    performance_table,
    run_portfolio_comparison,
    stateful_allocation_sensitivity,
    trade_cycle_diagnostics,
)
from stock_research.macro_momentum_sp500.reporting import generate_research_report
from stock_research.macro_momentum_sp500.robustness import (
    block_bootstrap_excess_return,
    yearly_portfolio_metrics,
)
from stock_research.macro_momentum_sp500.targets import build_targets
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run point-in-time macro + momentum SPY walk-forward research."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/macro_momentum_sp500/research.json"),
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--first-test-year", type=int, default=None)
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short diagnostic run; not a final research result.",
    )
    return parser.parse_args()


def _save_outputs(
    output_folder: Path,
    timestamp: str,
    *,
    walk_forward,
    portfolios,
    sensitivity: pd.DataFrame,
    yearly: pd.DataFrame,
    bootstrap: pd.DataFrame,
    baseline_bootstrap: pd.DataFrame,
    config,
) -> dict[str, Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions": atomic_to_csv(
            walk_forward.predictions,
            output_folder / f"oos_predictions_{timestamp}.csv",
            index=False,
        ),
        "selections": atomic_to_csv(
            walk_forward.selections,
            output_folder / f"model_selections_{timestamp}.csv",
            index=False,
        ),
        "candidate_scores": atomic_to_csv(
            walk_forward.candidate_scores,
            output_folder / f"candidate_scores_{timestamp}.csv",
            index=False,
        ),
        "metrics": atomic_to_csv(
            walk_forward.metrics,
            output_folder / f"predictive_metrics_{timestamp}.csv",
            index=False,
        ),
        "calibration": atomic_to_csv(
            walk_forward.calibration,
            output_folder / f"risk_calibration_{timestamp}.csv",
            index=False,
        ),
        "feature_importance": atomic_to_csv(
            walk_forward.feature_importance,
            output_folder / f"oos_feature_importance_{timestamp}.csv",
            index=False,
        ),
        "portfolio_summary": atomic_to_csv(
            performance_table(portfolios),
            output_folder / f"portfolio_summary_{timestamp}.csv",
            index=False,
        ),
        "portfolio_daily": atomic_to_csv(
            portfolios["MacroMomentum"].daily,
            output_folder / f"portfolio_daily_{timestamp}.csv",
            index=False,
        ),
        "portfolio_trades": atomic_to_csv(
            portfolios["MacroMomentum"].trades,
            output_folder / f"portfolio_trades_{timestamp}.csv",
            index=False,
        ),
        "stateful_portfolio_daily": atomic_to_csv(
            portfolios["StatefulMacro"].daily,
            output_folder / f"stateful_portfolio_daily_{timestamp}.csv",
            index=False,
        ),
        "stateful_portfolio_trades": atomic_to_csv(
            portfolios["StatefulMacro"].trades,
            output_folder / f"stateful_portfolio_trades_{timestamp}.csv",
            index=False,
        ),
        "stateful_trade_cycles": atomic_to_csv(
            trade_cycle_diagnostics(portfolios["StatefulMacro"]),
            output_folder / f"stateful_trade_cycles_{timestamp}.csv",
            index=False,
        ),
        "baseline_trade_cycles": atomic_to_csv(
            trade_cycle_diagnostics(portfolios["MacroMomentum"]),
            output_folder / f"baseline_trade_cycles_{timestamp}.csv",
            index=False,
        ),
        "sensitivity": atomic_to_csv(
            sensitivity,
            output_folder / f"allocation_sensitivity_{timestamp}.csv",
            index=False,
        ),
        "yearly": atomic_to_csv(
            yearly,
            output_folder / f"yearly_robustness_{timestamp}.csv",
            index=False,
        ),
        "bootstrap": atomic_to_csv(
            bootstrap,
            output_folder / f"block_bootstrap_{timestamp}.csv",
            index=False,
        ),
        "baseline_bootstrap": atomic_to_csv(
            baseline_bootstrap,
            output_folder / f"baseline_block_bootstrap_{timestamp}.csv",
            index=False,
        ),
    }
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": config.as_dict(),
        "files": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = output_folder / f"manifest_{timestamp}.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    temporary.replace(manifest_path)
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    config = load_research_config(args.config)
    overrides: dict[str, object] = {}
    if args.first_test_year is not None:
        overrides["first_test_year"] = args.first_test_year
    if args.search_budget is not None:
        overrides["search_budget_per_task"] = args.search_budget
    if overrides:
        config = replace(config, **overrides)
    if args.quick:
        config = replace(
            config,
            first_test_year=max(config.first_test_year, 2018),
            search_budget_per_task=min(6, config.search_budget_per_task),
            inner_validation_years=2,
        )

    print("Stage=load point-in-time data")
    data = load_research_data(paths.macro, config)
    if args.start:
        data = data[data["Date"] >= pd.Timestamp(args.start)]
    if args.end:
        data = data[data["Date"] <= pd.Timestamp(args.end)]
    data = data.reset_index(drop=True)
    print(f"Rows={len(data)} Range={data['Date'].min():%Y-%m-%d}..{data['Date'].max():%Y-%m-%d}")

    print("Stage=build features and multi-horizon targets")
    features = build_features(data, config)
    targets = build_targets(data, config)
    print(f"FeatureColumns={len(features.columns)} TargetColumns={len(targets.columns)}")

    print("Stage=nested walk-forward model selection and OOS prediction")
    walk_forward = nested_walk_forward(features, targets, config)
    print(
        f"OOSRows={len(walk_forward.predictions)} "
        f"CandidateInnerFolds={walk_forward.candidate_scores['InnerFoldCount'].sum()}"
    )

    print("Stage=portfolio comparison and robustness")
    portfolios = run_portfolio_comparison(
        walk_forward.predictions,
        config,
        v2_folder=paths.results / "SP500" / "macro_sp500_v2",
    )
    sensitivity = pd.concat(
        [
            allocation_sensitivity(walk_forward.predictions, config),
            stateful_allocation_sensitivity(walk_forward.predictions, config),
        ],
        ignore_index=True,
        sort=False,
    )
    yearly = yearly_portfolio_metrics(
        {name: result.daily for name, result in portfolios.items()}
    )
    bootstrap = block_bootstrap_excess_return(
        portfolios["StatefulMacro"].daily,
        portfolios["BuyHold"].daily,
        seed=config.random_seed,
    )
    baseline_bootstrap = block_bootstrap_excess_return(
        portfolios["MacroMomentum"].daily,
        portfolios["BuyHold"].daily,
        seed=config.random_seed,
    )

    output_folder = paths.results / "SP500" / "macro_momentum_sp500"
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    outputs = _save_outputs(
        output_folder,
        timestamp,
        walk_forward=walk_forward,
        portfolios=portfolios,
        sensitivity=sensitivity,
        yearly=yearly,
        bootstrap=bootstrap,
        baseline_bootstrap=baseline_bootstrap,
        config=config,
    )
    report_sources = dict(features.attrs.get("sources", {}))
    prior_v2 = portfolios.get("PriorV2")
    if prior_v2 is not None:
        report_sources["PriorV2Benchmark"] = prior_v2.source
    report = generate_research_report(
        walk_forward,
        portfolios,
        config,
        output_folder,
        sensitivity=sensitivity,
        yearly_metrics=yearly,
        bootstrap=bootstrap,
        sources=report_sources,
    )
    outputs["report"] = report
    print(performance_table(portfolios).to_string(index=False))
    print(walk_forward.metrics[walk_forward.metrics["Period"] == "All"].to_string(index=False))
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
