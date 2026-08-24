from __future__ import annotations

"""Signal-health diagnostics for an already-frozen backtest, written to a
SEPARATE sidecar file -- frozen_backtests/<model>.json is never opened for
writing here. A frozen artifact's meaning is "the validation result exactly
as it stood the day it was frozen"; even a byte-identical rewrite of that
file would blur that guarantee. Diagnostics computed *about* the frozen
period afterward belong in their own file, stamped with a hash of the
frozen artifact they describe so it's always traceable which frozen result
they're diagnosing.

Usage:
    python scripts/dashboard/build_frozen_validation_diagnostics.py --model-version V7.3
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from stock_research.dashboard import engine
from stock_research.dashboard.signal_health import alpha_vs_universe_ew_series, build_signal_health
from stock_research.dashboard.state import state_from_snapshot
from stock_research.paths import load_paths

REGISTRY_DIR = "config/dashboard_model_registry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build (or refresh) the frozen backtest's signal-health sidecar.")
    parser.add_argument("--model-version", required=True, help='e.g. "V7.3"')
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.stock_root)
    registry_dir = paths.repo_root / REGISTRY_DIR

    frozen_path = registry_dir / "frozen_backtests" / f"{args.model_version.lower().replace('.', '_')}.json"
    if not frozen_path.exists():
        raise SystemExit(f"{frozen_path} does not exist -- freeze the backtest first (freeze_backtest.py).")
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)

    snapshot_path = registry_dir / "universe_snapshots" / f"{frozen['frozen_at']}.json"
    state = state_from_snapshot(snapshot_path)

    panel = engine.build_scored_panel(
        paths, state, start=frozen["period"]["start"], end=frozen["period"]["end"], top_k=5
    )
    health = build_signal_health(panel.scored, panel.factored, k=panel.top_k)

    # Same weekly signal dates as the EW curve, sampled from the model's own
    # (daily) equity curve, so "when was the alpha actually earned" compares
    # two curves on the same date grid instead of just one endpoint total.
    portfolio = engine.simulate_portfolio(panel, start=frozen["period"]["start"], end=frozen["period"]["end"])
    equity_by_date = {row["date"]: row["equity"] for row in portfolio["equity_curve"]}
    ew_curve = health["universe_equal_weight_curve_gross"]
    model_curve = [{"date": row["date"], "nav": equity_by_date[row["date"]]} for row in ew_curve if row["date"] in equity_by_date]
    alpha_series = alpha_vs_universe_ew_series(model_curve, ew_curve)

    diagnostics = {
        "model_version": frozen["model_version"],
        "universe_version": frozen.get("universe_version"),
        "period": frozen["period"],
        "source_artifact": frozen_path.name,
        "source_artifact_hash": hashlib.sha256(frozen_bytes).hexdigest(),
        "generated_at": date.today().isoformat(),
        "signal_health": health,
        "alpha_vs_universe_ew": alpha_series,
    }

    output_path = registry_dir / "frozen_backtests" / f"{args.model_version.lower().replace('.', '_')}_validation_diagnostics.json"
    output_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "source_artifact_hash": diagnostics["source_artifact_hash"],
        "completed_weeks": health["completed_weeks"],
        "portfolio_ic_rolling": health["portfolio_ic_rolling"],
    }, indent=2))


if __name__ == "__main__":
    main()
