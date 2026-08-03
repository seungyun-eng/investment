from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.sp500_universe import (
    generate_sp500_universe,
    load_sp500_universe_settings,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build annual and change-date S&P 500 membership snapshots plus "
            "the full-period union crawl queue. This command does not "
            "download prices or financials."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/sp500_universe.json",
    )
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    config_path = _resolve(paths.repo_root, args.config)
    settings = load_sp500_universe_settings(
        config_path,
        years=args.years,
    )
    artifacts = generate_sp500_universe(
        paths,
        settings,
        output_dir=args.output_dir,
    )
    membership = pd.read_csv(artifacts.membership_csv)
    manifest = json.loads(
        artifacts.manifest_json.read_text(encoding="utf-8")
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Union crawl queue: {len(pd.read_csv(artifacts.union_csv))}")
    print("Annual membership counts:")
    print(membership.groupby("AsOfDate").size().to_string())
    print(f"Membership: {artifacts.membership_csv}")
    print(f"Membership changes: {artifacts.change_membership_csv}")
    print(f"Union: {artifacts.union_csv}")
    print(
        "Historical source coverage: "
        f"{manifest['source_coverage']['historical_first_date']} to "
        f"{manifest['source_coverage']['historical_last_date']}"
    )


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else repo_root / candidate
    )


if __name__ == "__main__":
    main()
