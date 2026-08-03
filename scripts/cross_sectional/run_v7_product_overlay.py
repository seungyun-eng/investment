from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.v7_product_overlay import (
    run_v7_product_overlay_optimization,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a cash-funded mix of V7-3 and daily-reset 2X product "
            "proxies without using negative cash or a margin loan."
        )
    )
    parser.add_argument("--base-equity", required=True)
    parser.add_argument("--base-series", default="V7_3_RISK_SELECTED")
    parser.add_argument("--base-period", default="FULL_2020_2026")
    parser.add_argument("--spy", required=True)
    parser.add_argument("--vix", required=True)
    parser.add_argument("--fed-funds", required=True)
    parser.add_argument("--optimization-config", required=True)
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args()
    paths = load_paths(args.stock_root)
    config_path = _resolve(paths.repo_root, args.optimization_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts = run_v7_product_overlay_optimization(
        paths,
        base_equity_path=_resolve(paths.repo_root, args.base_equity),
        base_series=args.base_series,
        base_period=args.base_period,
        spy_path=_resolve(paths.repo_root, args.spy),
        vix_path=_resolve(paths.repo_root, args.vix),
        fed_funds_path=_resolve(paths.repo_root, args.fed_funds),
        optimization_config=config,
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Candidates: {artifacts.candidate_summary_csv}")
    print(f"Periods: {artifacts.period_summary_csv}")
    print(f"Stress tests: {artifacts.stress_summary_csv}")
    print(f"Calendar returns: {artifacts.calendar_returns_csv}")
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
