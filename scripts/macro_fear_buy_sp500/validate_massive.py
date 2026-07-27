from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_fear_buy_sp500.config import FearBuyParams
from stock_research.macro_fear_buy_sp500.contributions import (
    ContributionConfig,
    run_contribution_backtest,
)
from stock_research.macro_fear_buy_sp500.features import build_fear_features
from stock_research.macro_fear_buy_sp500.mass_optimization import (
    candidate_features,
    constant_signals,
    evaluate_frozen_candidate,
)
from stock_research.macro_fear_buy_sp500.strategy import (
    generate_fear_buy_signals,
)
from stock_research.macro_fear_buy_sp500.validation import (
    flow_adjusted_block_bootstrap,
    yearly_flow_adjusted_comparison,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen 100k-search winners with flow-adjusted annual "
            "comparisons and block-bootstrap uncertainty."
        )
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--mass-manifest", type=Path)
    parser.add_argument("--initial", type=float, default=40_000.0)
    parser.add_argument("--monthly", type=float, default=4_000.0)
    parser.add_argument("--development-end", default="2016-12-30")
    parser.add_argument("--holdout-start", default="2017-01-03")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
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


def _candidate_payload(
    frozen: dict[str, object],
) -> dict[str, float | int]:
    payload = dict(frozen["strategy"])
    payload.update(dict(frozen["deployment_policy"]))
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    output_root = (
        paths.results
        / "SP500"
        / "macro_fear_buy_sp500"
        / "monthly_contributions"
        / "massive_optimization"
    )
    manifest_path = (
        args.mass_manifest
        or _latest(output_root, "manifest_*.json")
    ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["candidate_count"]) < 100_000:
        raise ValueError("Refusing a run with fewer than 100,000 candidates.")
    predictions = pd.read_csv(
        Path(manifest["prediction_source"]),
        parse_dates=["Date"],
    )
    baseline_params = FearBuyParams(
        **manifest["selection_categories"][0]["strategy"]
    )
    features = build_fear_features(predictions, baseline_params)
    config = ContributionConfig(
        initial_lump_sum=args.initial,
        monthly_contribution=args.monthly,
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
    )
    periods = {
        "Development": (None, args.development_end),
        "Holdout": (args.holdout_start, None),
        "Full": (None, None),
    }
    bootstrap_rows: list[pd.DataFrame] = []
    yearly_rows: list[pd.DataFrame] = []
    for frozen in manifest["selection_categories"]:
        category = str(frozen["selection_category"])
        candidate = _candidate_payload(frozen)
        params = FearBuyParams(**frozen["strategy"])
        for period, (start, end) in periods.items():
            period_features = features.copy()
            if start:
                period_features = period_features[
                    period_features["Date"] >= pd.Timestamp(start)
                ]
            if end:
                period_features = period_features[
                    period_features["Date"] <= pd.Timestamp(end)
                ]
            period_features = period_features.reset_index(drop=True)
            _, _, strategy = evaluate_frozen_candidate(
                period_features,
                candidate,
                config,
                name=f"{category}{period}",
            )
            signals = generate_fear_buy_signals(
                candidate_features(period_features, params),
                params,
            )
            benchmark = run_contribution_backtest(
                constant_signals(signals),
                params,
                config,
                name="MonthlyBuyHold",
                core_weight_override=1.0,
                invest_contributions_without_signal=True,
            )
            bootstrap = flow_adjusted_block_bootstrap(
                strategy.daily,
                benchmark.daily,
                samples=args.bootstrap_samples,
            )
            bootstrap.insert(0, "Period", period)
            bootstrap.insert(0, "SelectionCategory", category)
            bootstrap_rows.append(bootstrap)
            yearly = yearly_flow_adjusted_comparison(
                strategy.daily,
                benchmark.daily,
            )
            yearly.insert(0, "Period", period)
            yearly.insert(0, "SelectionCategory", category)
            yearly_rows.append(yearly)

    bootstrap_frame = pd.concat(bootstrap_rows, ignore_index=True)
    yearly_frame = pd.concat(yearly_rows, ignore_index=True)
    timestamp = datetime.now(UTC).astimezone().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    bootstrap_path = atomic_to_csv(
        bootstrap_frame,
        output_root / f"frozen_bootstrap_{timestamp}.csv",
        index=False,
    )
    yearly_path = atomic_to_csv(
        yearly_frame,
        output_root / f"frozen_yearly_{timestamp}.csv",
        index=False,
    )
    manifest_output = _atomic_json(
        {
            "source_mass_manifest": str(manifest_path),
            "bootstrap_block_days": 21,
            "bootstrap_samples": args.bootstrap_samples,
            "flow_adjusted": True,
            "selected_on": f"Date <= {args.development_end}",
            "untouched_holdout_start": args.holdout_start,
            "files": {
                "bootstrap": str(bootstrap_path),
                "yearly": str(yearly_path),
            },
        },
        output_root / f"validation_manifest_{timestamp}.json",
    )
    print(bootstrap_frame.to_string(index=False))
    print(f"bootstrap={bootstrap_path}")
    print(f"yearly={yearly_path}")
    print(f"manifest={manifest_output}")


if __name__ == "__main__":
    main()
