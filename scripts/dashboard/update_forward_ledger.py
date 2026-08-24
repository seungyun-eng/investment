from __future__ import annotations

"""Refresh the point-in-time forward ledger (see
stock_research.dashboard.forward_ledger for the actual logic). Safe to
re-run any time: only the currently-open segment's end date moves forward,
past segments are never recomputed."""

import argparse
import json
from datetime import date

from stock_research.dashboard.forward_ledger import LEDGER_FILENAME, LEDGER_RESULTS_FOLDER, build_forward_ledger
from stock_research.dashboard.today import _atomic_json
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the point-in-time forward ledger.")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.stock_root)
    payload = build_forward_ledger(paths, end=args.end)

    output_path = paths.results / LEDGER_RESULTS_FOLDER / LEDGER_FILENAME
    _atomic_json(output_path, payload)
    series = payload["series"]
    print(json.dumps({
        "output": str(output_path),
        "rows": len(series),
        "latest_nav": series[-1]["nav"] if series else None,
    }, indent=2))


if __name__ == "__main__":
    main()
