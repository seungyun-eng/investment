from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import settings_from_dict
from stock_research.cross_sectional.research import load_selected_strategy
from stock_research.cross_sectional.v7_technical import (
    run_v7_technical_variants,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-weight V7-2/3/4 technical variants with five "
            "equal-weight positions on the 575-name PIT proxy."
        )
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--ticker-config", required=True)
    parser.add_argument("--backfill-status", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-ready-count", type=int, default=575)
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    strategy_path = _resolve(paths.repo_root, args.strategy)
    payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    settings = settings_from_dict(payload["settings"])
    artifacts = run_v7_technical_variants(
        paths,
        settings,
        load_selected_strategy(strategy_path),
        ticker_config_path=_resolve(paths.repo_root, args.ticker_config),
        backfill_status_path=_resolve(
            paths.repo_root, args.backfill_status
        ),
        membership_path=_resolve(paths.repo_root, args.membership),
        frozen_strategy_path=strategy_path,
        expected_ready_count=args.expected_ready_count,
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Variant summary: {artifacts.variant_summary_csv}")
    print(
        "Concentration summary: "
        f"{artifacts.concentration_summary_csv}"
    )
    print(f"Ticker contributions: {artifacts.ticker_contributions_csv}")
    print(f"Feature coverage: {artifacts.feature_coverage_csv}")
    print(f"Manifest: {artifacts.manifest_json}")


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


if __name__ == "__main__":
    main()
