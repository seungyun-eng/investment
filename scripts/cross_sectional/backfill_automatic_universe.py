from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.automatic_backfill import (
    find_latest_automatic_universe,
    load_automatic_backfill_settings,
    run_automatic_backfill,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill price history and quarterly financial statements for "
            "a rule-based automatic universe. Existing valid files are skipped."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/automatic_backfill.json",
    )
    parser.add_argument("--universe")
    parser.add_argument(
        "--base-ticker-config",
        default="config/tickers.json",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--stock-root")
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--stage",
        choices=["all", "price", "financial"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    settings = load_automatic_backfill_settings(
        paths.repo_root / args.config
    )
    universe_path = (
        Path(args.universe).expanduser().resolve()
        if args.universe
        else find_latest_automatic_universe(paths)
    )
    artifacts = run_automatic_backfill(
        paths,
        settings,
        universe_path=universe_path,
        base_ticker_config_path=(
            paths.repo_root / args.base_ticker_config
        ),
        output_dir=args.output_dir,
        selected_tickers=args.ticker,
        limit=args.limit,
        stage=args.stage,
        force=args.force,
    )
    status = pd.read_csv(artifacts.status)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    counts = manifest["counts"]
    print(f"Output: {artifacts.output_dir}")
    print(f"Batch: {counts['batch']}")
    print(f"V6-ready universe: {counts['v6_ready']}/{counts['universe']}")
    print(f"Batch failures: {counts['batch_failures']}")
    print(
        status.loc[
            status["RunSelected"],
            [
                "Ticker",
                "PriceAction",
                "PriceValid",
                "FinancialAction",
                "FinancialValid",
                "V6Ready",
                "Errors",
            ],
        ].to_string(index=False)
    )
    print(
        "Use this ticker config for V6 after readiness is sufficient: "
        f"{artifacts.ticker_config}"
    )


if __name__ == "__main__":
    main()
