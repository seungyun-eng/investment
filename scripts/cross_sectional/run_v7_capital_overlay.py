from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import settings_from_dict
from stock_research.cross_sectional.research import load_selected_strategy
from stock_research.cross_sectional.v7_capital_overlay import (
    run_v7_capital_overlay,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a cash/leverage/strict-bottom-five-short overlay while "
            "leaving V7-3 stock selection unchanged."
        )
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--ticker-config", required=True)
    parser.add_argument("--backfill-status", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--spy", required=True)
    parser.add_argument("--overlay-config", required=True)
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
    strategy_payload = json.loads(
        strategy_path.read_text(encoding="utf-8")
    )
    overlay_config_path = _resolve(paths.repo_root, args.overlay_config)
    overlay_config = json.loads(
        overlay_config_path.read_text(encoding="utf-8")
    )
    artifacts = run_v7_capital_overlay(
        paths,
        settings_from_dict(strategy_payload["settings"]),
        load_selected_strategy(strategy_path),
        ticker_config_path=_resolve(paths.repo_root, args.ticker_config),
        backfill_status_path=_resolve(
            paths.repo_root,
            args.backfill_status,
        ),
        membership_path=_resolve(paths.repo_root, args.membership),
        frozen_strategy_path=strategy_path,
        spy_path=_resolve(paths.repo_root, args.spy),
        overlay_config=overlay_config,
        expected_ready_count=args.expected_ready_count,
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Candidate summary: {artifacts.candidate_summary_csv}")
    print(f"Period summary: {artifacts.period_summary_csv}")
    print(f"Equity: {artifacts.equity_csv}")
    print(f"Signal exposure: {artifacts.signal_exposure_csv}")
    print(f"Short selections: {artifacts.short_selections_csv}")
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
