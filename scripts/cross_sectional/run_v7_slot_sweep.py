from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import settings_from_dict
from stock_research.cross_sectional.research import load_selected_strategy
from stock_research.cross_sectional.v7_slot_sweep import run_v7_slot_sweep
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep V7-3 from one to ten positions and compare with "
            "dividend-adjusted SPY buy-and-hold."
        )
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--ticker-config", required=True)
    parser.add_argument("--backfill-status", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--spy", required=True)
    parser.add_argument("--minimum-top-k", type=int, default=1)
    parser.add_argument("--maximum-top-k", type=int, default=10)
    parser.add_argument("--exit-buffer", type=int, default=4)
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
    artifacts = run_v7_slot_sweep(
        paths,
        settings_from_dict(payload["settings"]),
        load_selected_strategy(strategy_path),
        ticker_config_path=_resolve(paths.repo_root, args.ticker_config),
        backfill_status_path=_resolve(
            paths.repo_root, args.backfill_status
        ),
        membership_path=_resolve(paths.repo_root, args.membership),
        frozen_strategy_path=strategy_path,
        spy_path=_resolve(paths.repo_root, args.spy),
        minimum_top_k=args.minimum_top_k,
        maximum_top_k=args.maximum_top_k,
        exit_buffer=args.exit_buffer,
        expected_ready_count=args.expected_ready_count,
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Slot summary: {artifacts.slot_summary_csv}")
    print(f"Balanced ranking: {artifacts.balanced_ranking_csv}")
    print(f"Concentration: {artifacts.concentration_csv}")
    print(f"Equity: {artifacts.equity_csv}")
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
