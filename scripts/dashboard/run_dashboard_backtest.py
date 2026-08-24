from __future__ import annotations

"""Run the dashboard's Filing V7 (top_k=5 by default) engine across several
date ranges from the command line, without opening the browser dashboard.

Uses the exact same universe (config/cross_sectional/dashboard_universe.json,
editable from the dashboard UI or by hand) and the exact same backtest engine
(stock_research.dashboard.engine.run_backtest) as the dashboard itself, so
results here always match what the dashboard would show for the same
start/end/top_k.  Signals and exits remain weekly; unchanged positions reset
to equal weight on the first weekly signal of each calendar month.

Examples
--------
Sweep a few fixed periods at top_k=5 (the dashboard default):

    python scripts/dashboard/run_dashboard_backtest.py

Explicit periods and a different top_k:

    python scripts/dashboard/run_dashboard_backtest.py --top-k 7 \\
        --periods 2021-01-04:2022-12-30 2023-01-02:2024-12-31 2025-01-02:2026-07-29

Output: one row per period in artifacts/dashboard_backtests/<timestamp>/period_summary.csv,
plus each period's full trade log alongside it.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stock_research.dashboard import engine  # noqa: E402
from stock_research.dashboard.state import load_state  # noqa: E402
from stock_research.io_utils import atomic_to_csv  # noqa: E402
from stock_research.paths import load_paths  # noqa: E402

DEFAULT_PERIODS = [
    ("2021-01-04", "2022-12-30"),
    ("2023-01-02", "2024-12-31"),
    ("2025-01-02", "2026-07-29"),
    ("2021-01-04", "2026-07-29"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--periods",
        nargs="*",
        metavar="START:END",
        help="One or more START:END (YYYY-MM-DD) periods. Defaults to a fixed 4-period sweep.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Overrides the dashboard's configured top_k for every period.")
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _parse_periods(raw: list[str] | None) -> list[tuple[str, str]]:
    if not raw:
        return DEFAULT_PERIODS
    periods = []
    for item in raw:
        start, end = item.split(":")
        periods.append((start.strip(), end.strip()))
    return periods


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    periods = _parse_periods(args.periods)

    paths = load_paths(args.stock_root)
    state = load_state(paths)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "artifacts" / "dashboard_backtests" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for start, end in periods:
        label = f"{start}_{end}"
        print(f"Running {label} (top_k={args.top_k or state.top_k})...", flush=True)
        try:
            result = engine.run_backtest(paths, state, start=start, end=end, top_k=args.top_k)
        except ValueError as exc:
            print(f"  skipped: {exc}")
            summary_rows.append({"Start": start, "End": end, "Error": str(exc)})
            continue
        s = result["summary"]
        summary_rows.append(
            {
                "Start": start,
                "End": end,
                "TopK": result["meta"]["top_k"],
                "CAGR": s["CAGR"],
                "ROI": s["ROI"],
                "Sharpe": s["Sharpe"],
                "MaxDrawdown": s["MaxDrawdown"],
                "AnnualizedTurnover": s["AnnualizedTurnover"],
                "Rebalances": s["Rebalances"],
                "TickerTrades": s["TickerTrades"],
                "TradeCount": len(result["trades"]),
            }
        )
        period_dir = output_dir / label
        period_dir.mkdir(parents=True, exist_ok=True)
        atomic_to_csv(pd.DataFrame(result["trades"]), period_dir / "trades.csv", index=False)
        atomic_to_csv(pd.DataFrame(result["ticker_summary"]), period_dir / "ticker_summary.csv", index=False)
        (period_dir / "backtest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    summary = pd.DataFrame(summary_rows)
    atomic_to_csv(summary, output_dir / "period_summary.csv", index=False)
    print(f"\nDone. Output in: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
