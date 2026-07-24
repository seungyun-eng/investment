from __future__ import annotations

import argparse

from stock_research.financial_analysis import analyze_company
from stock_research.paths import load_paths
from stock_research.tickers import load_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--config", default="config/tickers.json")
    args = parser.parse_args()

    paths = load_paths()
    configs = load_tickers(paths.repo_root / args.config)
    selected = {ticker.upper() for ticker in args.ticker} if args.ticker else None
    summary = paths.financial_raw / "Summary"
    for ticker, config in configs.items():
        if selected and ticker not in selected:
            continue
        try:
            output = analyze_company(
                config,
                raw_price_root=paths.raw_prices,
                processed_root=paths.processed,
                financial_root=paths.financial_raw,
                output_root=summary,
            )
            print(output)
        except Exception as exc:
            print(f"WARN {ticker}: {exc}")


if __name__ == "__main__":
    main()
