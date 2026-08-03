from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import settings_from_dict
from stock_research.cross_sectional.filing_v7_optimization import (
    run_filing_v7_optimization,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize V7-3 with PIT SEC scores and red-flag vetoes."
    )
    parser.add_argument(
        "--universe-config",
        default="config/cross_sectional/known_2020_growth_16_v7.json",
    )
    parser.add_argument(
        "--optimization-config",
        default="config/cross_sectional/filing_v7_optimization.json",
    )
    parser.add_argument("--filing-features")
    parser.add_argument("--spy")
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    universe = json.loads(
        _resolve(paths.repo_root, args.universe_config).read_text(encoding="utf-8")
    )
    optimization = json.loads(
        _resolve(paths.repo_root, args.optimization_config).read_text(
            encoding="utf-8"
        )
    )
    filing_features = (
        Path(args.filing_features).expanduser().resolve()
        if args.filing_features
        else paths.processed
        / "SEC Filings"
        / "known_2020_growth_16"
        / "point_in_time_features.csv"
    )
    spy = (
        Path(args.spy).expanduser().resolve()
        if args.spy
        else paths.macro / "SPY Adjusted Historical Data.csv"
    )
    artifacts = run_filing_v7_optimization(
        paths,
        settings_from_dict(universe["research"]),
        universe_config=universe,
        optimization_config=optimization,
        filing_features_path=filing_features,
        spy_path=spy,
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Selection: {artifacts.selection_json}")
    print(f"Period summary: {artifacts.period_summary_csv}")
    print(f"Calendar returns: {artifacts.calendar_returns_csv}")


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


if __name__ == "__main__":
    main()
