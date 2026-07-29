from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_research.cross_sectional.trade_reporting import (
    generate_v5_trade_report,
    latest_loss_protected_strategy,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a V5 price, full-entry/full-exit, and signal-strength "
            "report from a frozen research run."
        )
    )
    parser.add_argument(
        "--strategy",
        help=(
            "Path to selected_strategy.json. Defaults to the latest "
            "loss_protected_v5 research run."
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    strategy = (
        Path(args.strategy).expanduser().resolve()
        if args.strategy
        else latest_loss_protected_strategy(paths.results)
    )
    artifacts = generate_v5_trade_report(
        paths,
        strategy,
        output_dir=args.output_dir,
    )
    print(f"Strategy: {strategy}")
    print(f"Output: {artifacts.output_dir}")
    print(f"HTML: {artifacts.html_report}")
    print(f"PDF: {artifacts.pdf_report}")
    print(f"Positions: {artifacts.position_ledger}")
    print(f"Trade events: {artifacts.trade_events}")
    print(f"Weekly signals: {artifacts.weekly_signals}")
    print(f"Price reconciliation: {artifacts.reconciliation}")


if __name__ == "__main__":
    main()
