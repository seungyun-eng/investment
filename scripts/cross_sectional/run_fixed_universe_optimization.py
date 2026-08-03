from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import settings_from_dict
from stock_research.cross_sectional.fixed_universe import (
    run_fixed_universe_optimization,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize V7 inside a fixed user-specified stock universe."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--spy", required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args()
    paths = load_paths(args.stock_root)
    config_path = _resolve(paths.repo_root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts = run_fixed_universe_optimization(
        paths,
        settings_from_dict(config["research"]),
        config=config,
        spy_path=_resolve(paths.repo_root, args.spy),
        reference_manifest_path=(
            _resolve(paths.repo_root, args.reference_manifest)
            if args.reference_manifest
            else None
        ),
        output_dir=args.output_dir,
    )
    print(f"Output: {artifacts.output_dir}")
    print(f"Candidates: {artifacts.candidate_summary_csv}")
    print(f"Periods: {artifacts.period_summary_csv}")
    print(f"Calendar returns: {artifacts.calendar_returns_csv}")
    print(f"Purchase audit: {artifacts.purchase_audit_csv}")
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
