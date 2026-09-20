from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_research.dashboard.forward_shadow import run_forward_shadow
from stock_research.paths import load_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Alpha Desk frozen-model shadow ledgers.")
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--public-json", type=Path)
    parser.add_argument("--cutoff", type=pd.Timestamp)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-refresh-market", action="store_true")
    args = parser.parse_args()
    paths = load_paths(args.stock_root)
    result = run_forward_shadow(
        paths,
        output=args.output,
        public_json=args.public_json,
        refresh_market=not args.no_refresh_market,
        cutoff=args.cutoff,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
