from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_research.cross_sectional.pit_universe_builder import (
    generate_pit_universe_source_sample,
    load_pit_universe_settings,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build source-separated V7 PIT market-cap universe samples. "
            "This stage does not run a backtest or download missing "
            "full price histories."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/v7_pit_universe.json",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Optional subset such as --years 2019 2020.",
    )
    parser.add_argument(
        "--ticker-config",
        action="append",
        default=[],
        help="Ticker config used to match the local Back Test folders.",
    )
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    config_path = _resolve(paths.repo_root, args.config)
    settings = load_pit_universe_settings(
        config_path,
        years=args.years,
    )
    ticker_configs = [
        _resolve(paths.repo_root, value)
        for value in args.ticker_config
    ]
    if not ticker_configs:
        ticker_configs.append(paths.repo_root / "config" / "tickers.json")
        latest = _latest_automatic_ticker_config(paths.results)
        if latest is not None:
            ticker_configs.append(latest)
    artifacts = generate_pit_universe_source_sample(
        paths,
        settings,
        ticker_config_paths=ticker_configs,
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Direct published ranking: {artifacts.direct_rankings_csv}")
    print(f"S&P proxy candidates: {artifacts.proxy_candidates_csv}")
    print(f"S&P proxy top 100: {artifacts.proxy_snapshots_csv}")
    print(f"Hybrid sample: {artifacts.hybrid_snapshots_csv}")
    print(f"Source comparison: {artifacts.source_comparison_csv}")
    print(f"Coverage: {artifacts.coverage_csv}")
    print(f"Missing local sample: {artifacts.missing_local_csv}")
    print(f"Fetch log: {artifacts.fetch_log_csv}")
    print(f"Manifest: {artifacts.manifest_json}")


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else repo_root / candidate


def _latest_automatic_ticker_config(results: Path) -> Path | None:
    root = results / "Cross_Sectional" / "backfill_runs"
    candidates = list(root.glob("*/automatic_tickers.json"))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
    )


if __name__ == "__main__":
    main()
