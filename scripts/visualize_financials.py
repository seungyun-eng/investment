from __future__ import annotations

import argparse

from stock_research.paths import load_paths
from stock_research.visualization import create_financial_charts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()

    paths = load_paths()
    workbook = paths.financial_raw / "Summary" / f"{args.ticker.upper()}_analysis_Q.xlsx"
    output = paths.financial_raw / "Summary" / "Visual" / args.ticker.upper()
    for path in create_financial_charts(workbook, output):
        print(path)


if __name__ == "__main__":
    main()
