from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_research.cross_sectional.config import load_settings
from stock_research.cross_sectional.research import load_selected_strategy
from stock_research.cross_sectional.trade_reporting import (
    latest_loss_protected_strategy,
)
from stock_research.cross_sectional.v6_reporting import (
    generate_v6_comparison,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare V6 rule ablations with V5 factor weights and entry "
            "filters frozen."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/research_loss_protected_v5.json",
    )
    parser.add_argument("--strategy")
    parser.add_argument("--ticker-config", default="config/tickers.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    strategy_path = (
        Path(args.strategy).expanduser().resolve()
        if args.strategy
        else latest_loss_protected_strategy(paths.results)
    )
    settings = load_settings(paths.repo_root / args.config)
    params = load_selected_strategy(strategy_path)
    artifacts = generate_v6_comparison(
        paths,
        settings,
        params,
        ticker_config_path=paths.repo_root / args.ticker_config,
        output_dir=args.output_dir,
    )
    print(f"Base strategy: {strategy_path}")
    print(f"Output: {artifacts.output_dir}")
    print(f"Summary: {artifacts.summary_csv}")
    print(f"Sensitivity: {artifacts.sensitivity_csv}")
    print(f"Equity: {artifacts.equity_csv}")
    print(f"Events: {artifacts.events_csv}")
    print(f"Ledger: {artifacts.ledger_csv}")
    print(f"HTML: {artifacts.html_report}")


if __name__ == "__main__":
    main()
