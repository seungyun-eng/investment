from __future__ import annotations

"""One-time (per model version) freeze: snapshot the currently-live
dashboard backtest into config/dashboard_model_registry/ so future universe
or model edits can never silently rewrite it.

Reads whatever is CURRENTLY on disk (latest_today.json's backtest_summary/
ticker_summary, dashboard_universe.json) rather than recomputing anything,
so the freeze is guaranteed to match what was actually live at freeze time.
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard.today import latest_path
from stock_research.paths import load_paths

REGISTRY_DIR = "config/dashboard_model_registry"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the currently-live dashboard backtest.")
    parser.add_argument("--model-version", required=True, help='e.g. "V7.3"')
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.stock_root)
    registry_dir = paths.repo_root / REGISTRY_DIR

    latest = json.loads(latest_path(paths).read_text(encoding="utf-8-sig"))
    summary = latest["backtest_summary"]
    ticker_summary = latest["ticker_summary"]

    state = dashboard_state.load_state(paths)
    universe = sorted(t.ticker for t in state.tickers)

    frozen = {
        "model_version": args.model_version,
        "universe_version": "U001",
        "status": "FROZEN",
        "frozen_at": date.today().isoformat(),
        "period": {"start": summary["StartDate"], "end": summary["EndDate"]},
        "universe": universe,
        "summary": summary,
        "ticker_summary": ticker_summary,
        "git_commit": _git_commit(paths.repo_root),
    }

    frozen_dir = registry_dir / "frozen_backtests"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = frozen_dir / f"{args.model_version.lower().replace('.', '_')}.json"
    if frozen_path.exists():
        raise SystemExit(
            f"{frozen_path} already exists and is FROZEN -- refusing to overwrite. "
            "Freeze a new model_version instead of re-freezing an existing one."
        )
    frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshots_dir = registry_dir / "universe_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    baseline_date = date.today().isoformat()
    snapshot_path = snapshots_dir / f"{baseline_date}.json"
    snapshot_path.write_text(
        json.dumps(state.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    registry_path = registry_dir / "registry.json"
    registry = {
        "active_model_version": args.model_version,
        "forward_ledger_start": baseline_date,
        "universe_changes": [
            {
                "date": baseline_date,
                "snapshot": snapshot_path.name,
                "universe_version": "U001",
                "added": [],
                "removed": [],
                "note": "forward ledger baseline",
            }
        ],
    }
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "frozen_backtest": str(frozen_path),
                "universe_snapshot": str(snapshot_path),
                "registry": str(registry_path),
                "roi": summary["ROI"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
