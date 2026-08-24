from __future__ import annotations

"""Standing procedure for changing the live dashboard universe from now on.

Unlike editing config/cross_sectional/dashboard_universe.json directly (what
this session did before building the model registry), this script keeps the
point-in-time forward ledger correct: it collects real data for any added
ticker (unchanged -- data_collection.collect_ticker), removes any dropped
ticker, then records the change as a new dated universe snapshot + registry
entry, and refreshes the forward ledger so the change only affects data from
today forward. It never touches the frozen historical backtest.

Usage:
    python scripts/dashboard/change_universe.py --add UBER=Uber --add INTC=Intel --remove JNJ
"""

import argparse
import json
from datetime import date

from stock_research.dashboard import data_collection
from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard.forward_ledger import build_forward_ledger, registry_dir, LEDGER_FILENAME, LEDGER_RESULTS_FOLDER
from stock_research.dashboard.today import _atomic_json
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add/remove tickers and record the change in the model registry.")
    parser.add_argument("--add", action="append", default=[], metavar="TICKER[=Company Name]")
    parser.add_argument("--remove", action="append", default=[], metavar="TICKER")
    parser.add_argument("--note", default="")
    parser.add_argument("--stock-root")
    return parser.parse_args()


def _split_add(spec: str) -> tuple[str, str]:
    ticker, _, company = spec.partition("=")
    ticker = ticker.upper().strip()
    return ticker, company.strip() or ticker


def main() -> None:
    args = parse_args()
    if not args.add and not args.remove:
        raise SystemExit("Nothing to do: pass at least one --add or --remove.")
    paths = load_paths(args.stock_root)

    added: list[str] = []
    for spec in args.add:
        ticker, company = _split_add(spec)
        result = data_collection.collect_ticker(paths, ticker, company)
        added.append(ticker)
        print(json.dumps({"added": ticker, "price": result["price"], "filings": result["filings"]}, indent=2))

    removed: list[str] = []
    for ticker in args.remove:
        dashboard_state.remove_ticker(paths, ticker.upper().strip())
        removed.append(ticker.upper().strip())

    state = dashboard_state.load_state(paths)
    today = date.today().isoformat()

    snapshots_dir = registry_dir(paths) / "universe_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"{today}.json"
    snapshot_path.write_text(json.dumps(state.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    registry_path = registry_dir(paths) / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    changes = registry["universe_changes"]
    same_day_correction = bool(changes) and changes[-1]["date"] == today
    universe_version = (
        changes[-1].get("universe_version", f"U{len(changes):03d}")
        if same_day_correction
        else f"U{len(changes) + 1:03d}"
    )
    entry = {
        "date": today,
        "snapshot": snapshot_path.name,
        "universe_version": universe_version,
        "added": added,
        "removed": removed,
        "note": args.note,
    }
    if same_day_correction:
        # Re-running the same day (e.g. a correction) replaces today's entry
        # rather than creating a second segment for the same date.
        changes[-1] = entry
    else:
        changes.append(entry)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = build_forward_ledger(paths, end=today)
    output_path = paths.results / LEDGER_RESULTS_FOLDER / LEDGER_FILENAME
    _atomic_json(output_path, payload)

    print(json.dumps({
        "universe_snapshot": str(snapshot_path),
        "registry_entry": entry,
        "forward_ledger": str(output_path),
        "forward_ledger_rows": len(payload["series"]),
    }, indent=2))


if __name__ == "__main__":
    main()
