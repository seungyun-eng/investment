from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.live_dashboard import (
    build_dashboard_payload,
    render_dashboard_html,
    write_dashboard,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained interactive HTML dashboard for the frozen "
            "Top 10 plus watchlist model."
        )
    )
    parser.add_argument("--signal-run")
    parser.add_argument("--optimization-run")
    parser.add_argument("--output")
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args()
    paths = load_paths(args.stock_root)
    cross_sectional = paths.results / "Cross_Sectional"
    signal_run = (
        Path(args.signal_run).expanduser().resolve()
        if args.signal_run
        else _latest_directory(
            cross_sectional / "live_top10_watchlist_signals", "*"
        )
    )
    optimization_run = (
        Path(args.optimization_run).expanduser().resolve()
        if args.optimization_run
        else _latest_directory(
            cross_sectional / "filing_v7_optimization",
            "*_live_top10_plus_watchlist_cash",
        )
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else signal_run / "live_top10_watchlist_dashboard.html"
    )
    payload = build_dashboard_payload(
        pd.read_csv(signal_run / "signal_history.csv"),
        pd.read_csv(optimization_run / "equity.csv"),
        pd.read_csv(signal_run / "top10_plus_watchlist_membership.csv"),
        pd.read_csv(signal_run / "universe_readiness.csv"),
        json.loads((signal_run / "manifest.json").read_text(encoding="utf-8")),
    )
    write_dashboard(output, render_dashboard_html(payload))
    print(f"Dashboard: {output}")
    print(f"Signal dates: {len(payload['scoreDates'])}")
    print(f"Score universe: {len(payload['scoreDates'][-1]['rows'])}")


def _latest_directory(root: Path, pattern: str) -> Path:
    candidates = sorted(
        (path for path in root.glob(pattern) if path.is_dir()),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directory matched {root / pattern}")
    return candidates[-1]


if __name__ == "__main__":
    main()
