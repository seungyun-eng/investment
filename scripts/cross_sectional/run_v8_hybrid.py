from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import settings_from_dict
from stock_research.cross_sectional.v8_hybrid import run_v8_hybrid
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed V8 hybrid long-term core plus staged "
            "fundamental-inflection prototype."
        )
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--ticker-config", required=True)
    parser.add_argument("--backfill-status", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--spy", required=True)
    parser.add_argument("--v7-slot-summary")
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
    artifacts = run_v8_hybrid(
        paths,
        settings_from_dict(payload["settings"]),
        ticker_config_path=_resolve(paths.repo_root, args.ticker_config),
        backfill_status_path=_resolve(
            paths.repo_root,
            args.backfill_status,
        ),
        membership_path=_resolve(paths.repo_root, args.membership),
        spy_path=_resolve(paths.repo_root, args.spy),
        frozen_strategy_path=strategy_path,
        expected_ready_count=args.expected_ready_count,
        v7_slot_summary_path=(
            _resolve(paths.repo_root, args.v7_slot_summary)
            if args.v7_slot_summary
            else None
        ),
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Report: {artifacts.report_html}")
    print(f"Summary: {artifacts.summary_csv}")
    print(f"Period summary: {artifacts.period_summary_csv}")
    print(f"Trades: {artifacts.trade_events_csv}")
    print(f"Positions: {artifacts.position_ledger_csv}")


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


if __name__ == "__main__":
    main()
