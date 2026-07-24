from __future__ import annotations

import argparse

from stock_research.data_loading import load_processed
from stock_research.legacy_grid_search import run_legacy_grid_search
from stock_research.paths import load_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the older tranche/grid-search experiment."
    )
    parser.add_argument("company")
    parser.add_argument("--output", default="legacy_grid_search_results.csv")
    args = parser.parse_args()

    paths = load_paths()
    frame = load_processed(paths.processed, args.company)
    output = paths.results / args.company / args.output
    result = run_legacy_grid_search(frame, output)
    print(result)
    print(output)


if __name__ == "__main__":
    main()
