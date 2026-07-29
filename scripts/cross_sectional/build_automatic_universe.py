from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from stock_research.cross_sectional.automatic_universe import (
    generate_automatic_universe,
    load_automatic_universe_settings,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rule-based current US-listed equity universe from "
            "Nasdaq directory and screener snapshots."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/automatic_universe.json",
    )
    parser.add_argument("--ticker-config", default="config/tickers.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    settings = load_automatic_universe_settings(
        paths.repo_root / args.config
    )
    artifacts = generate_automatic_universe(
        paths,
        settings,
        ticker_config_path=paths.repo_root / args.ticker_config,
        output_dir=args.output_dir,
    )
    selected = pd.read_csv(artifacts.selected_universe)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    print(f"Output: {artifacts.output_dir}")
    print(f"Selected: {len(selected)}")
    print(f"V6-ready with current local data: {int(selected['V6Ready'].sum())}")
    print(f"Backfill required: {int((~selected['V6Ready']).sum())}")
    print(
        selected[
            [
                "DataSymbol",
                "SecurityName",
                "Exchange",
                "LastSale",
                "MarketCap",
                "DollarVolume",
                "V6Ready",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    print(
        "Caveat: "
        + manifest["methodology"]["survivorship_caveat"]
    )


if __name__ == "__main__":
    main()
