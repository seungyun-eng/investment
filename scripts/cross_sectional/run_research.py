from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from stock_research.cross_sectional.config import load_settings
from stock_research.cross_sectional.research import run_research
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a cross-sectional rank strategy on 2020-2024, then "
            "validate frozen parameters on 2025 and 2026."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/research.json",
    )
    parser.add_argument("--ticker-config", default="config/tickers.json")
    parser.add_argument("--stock-root")
    parser.add_argument("--candidates", type=int)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    settings = load_settings(paths.repo_root / args.config)
    if args.candidates is not None:
        settings = replace(settings, candidate_count=args.candidates)
    result = run_research(
        paths,
        settings,
        ticker_config_path=paths.repo_root / args.ticker_config,
    )
    universe_size = int(
        result["data_audit"]["Status"].eq("INCLUDED").sum()
    )
    print(f"Output: {result['output_dir']}")
    print(f"Universe: {universe_size}")
    print(f"Parameters: {result['params'].as_dict()}")
    print(result["validation_summary"].to_string(index=False))
    print("\nLatest selected names:")
    selected = result["latest_signals"].loc[
        result["latest_signals"]["ModelSelected"]
    ]
    print(
        selected[
            [
                "Ticker",
                "DailySignal",
                "TradeAction",
                "TargetWeight",
                "Rank",
                "AlphaScore",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
