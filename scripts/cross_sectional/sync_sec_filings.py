from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from stock_research.cross_sectional.sec_filings import (
    settings_from_dict,
    sync_sec_filings,
    universe_tickers,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache SEC submissions, Company Facts, and periodic filing documents "
            "then create point-in-time filing features."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--universe-config", required=True)
    parser.add_argument("--user-agent")
    parser.add_argument("--stock-root")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--refresh-documents", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    config_path = _resolve(paths.repo_root, args.config)
    universe_path = _resolve(paths.repo_root, args.universe_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    artifacts = sync_sec_filings(
        paths,
        tickers=universe_tickers(universe),
        settings=settings_from_dict(config),
        user_agent=args.user_agent or os.getenv("SEC_USER_AGENT", ""),
        output_label=str(config.get("output_label", "sec_filings")),
        refresh_metadata=args.refresh_metadata,
        refresh_documents=args.refresh_documents,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Filing index: {artifacts.filing_index_csv}")
    print(f"PIT features: {artifacts.point_in_time_features_csv}")
    print(f"Audit: {artifacts.download_audit_csv}")


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


if __name__ == "__main__":
    main()
