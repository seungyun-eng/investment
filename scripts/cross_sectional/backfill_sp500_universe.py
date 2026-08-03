from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.automatic_backfill import (
    load_automatic_backfill_settings,
    run_automatic_backfill,
)
from stock_research.cross_sectional.sp500_universe import (
    find_latest_sp500_union,
)
from stock_research.paths import ProjectPaths, load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill prices and quarterly financials for the union of annual "
            "and intra-year S&P 500 membership snapshots. Existing valid "
            "files are skipped."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/sp500_backfill.json",
    )
    parser.add_argument("--universe")
    parser.add_argument(
        "--base-ticker-config",
        help=(
            "Optional ticker config. By default, the latest automatic "
            "200-stock config is used, then config/tickers.json."
        ),
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
        _resolve(paths.repo_root, args.config)
    )
    universe_path = (
        Path(args.universe).expanduser().resolve()
        if args.universe
        else find_latest_sp500_union(paths)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            paths.results
            / "Cross_Sectional"
            / "sp500_backfill_runs"
            / (
                datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
                + "_sp500_membership_union_backfill"
            )
        )
    )
    base_ticker_config = (
        _resolve(paths.repo_root, args.base_ticker_config)
        if args.base_ticker_config
        else _latest_base_ticker_config(paths)
    )
    artifacts = run_automatic_backfill(
        paths,
        settings,
        universe_path=universe_path,
        base_ticker_config_path=base_ticker_config,
        output_dir=output_dir,
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
    print(
        "Ready price/financial pairs: "
        f"{counts['v6_ready']}/{counts['universe']}"
    )
    print(f"Batch failures: {counts['batch_failures']}")
    if status["RunSelected"].any():
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
    print(f"Ticker config: {artifacts.ticker_config}")


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else repo_root / candidate
    )


def _latest_base_ticker_config(paths: ProjectPaths) -> Path:
    root = paths.results / "Cross_Sectional" / "backfill_runs"
    candidates = list(root.glob("*/automatic_tickers.json"))
    if candidates:
        return max(
            candidates,
            key=lambda path: (path.stat().st_mtime, path.as_posix()),
        )
    return paths.repo_root / "config" / "tickers.json"


if __name__ == "__main__":
    main()
