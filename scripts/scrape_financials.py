from __future__ import annotations

import argparse

from stock_research.financial_crawler import scrape_financials
from stock_research.paths import load_paths
from stock_research.tickers import load_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--config", default="config/tickers.json")
    parser.add_argument("--frequency", default="Q", choices=["Q", "A"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wait", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--page-load-timeout", type=float, default=30.0)
    parser.add_argument("--transport", default="http", choices=["http", "selenium"])
    parser.add_argument("--request-pause", type=float, default=0.25)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    args = parser.parse_args()

    paths = load_paths()
    tickers = load_tickers(paths.repo_root / args.config)
    outputs = scrape_financials(
        tickers,
        paths.financial_raw,
        selected=args.ticker,
        frequency=args.frequency,
        headless=args.headless,
        wait_seconds=args.wait,
        retries=args.retries,
        page_load_timeout_seconds=args.page_load_timeout,
        transport=args.transport,
        request_pause_seconds=args.request_pause,
        retry_backoff_seconds=args.retry_backoff,
    )
    for output in outputs:
        print(output.name)


if __name__ == "__main__":
    main()
