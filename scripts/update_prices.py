from __future__ import annotations

import argparse

from stock_research.market_data import update_all_prices
from stock_research.paths import load_paths
from stock_research.tickers import load_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", help="Repeat to select tickers.")
    parser.add_argument("--config", default="config/tickers.json")
    args = parser.parse_args()

    paths = load_paths()
    tickers = load_tickers(paths.repo_root / args.config)
    outputs = update_all_prices(tickers, paths.raw_prices, args.ticker)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
