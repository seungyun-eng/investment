from __future__ import annotations

"""Research-only factor variants against the frozen V7.3 universe.

This never writes the frozen artifact or changes the live model.  The four
variants are predeclared hypotheses from the frozen signal-health diagnostic:
Quality has the strongest recent IC while Trend weakened.  Candidate selection
uses the pre-2026 window only; 2026 is kept as an out-of-sample report.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import (
    generate_filing_v7_targets,
)
from stock_research.cross_sectional.portfolio import (
    prepare_market,
    run_portfolio_backtest,
)
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine
from stock_research.dashboard.signal_health import build_signal_health
from stock_research.dashboard.state import state_from_snapshot
from stock_research.paths import load_paths


REGISTRY_DIR = Path("config/dashboard_model_registry")
FROZEN_PATH = REGISTRY_DIR / "frozen_backtests/v7_3.json"
SNAPSHOT_PATH = REGISTRY_DIR / "universe_snapshots/2026-08-12.json"
OUTPUT_PATH = Path("artifacts/frozen_factor_candidate_study/results.json")
SELECTION_END = "2025-12-31"
OOS_START = "2026-01-02"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary(result: object) -> dict[str, float | int]:
    summary = result.summary
    return {
        "ROI": summary.roi_percent,
        "CAGR": summary.cagr_percent,
        "MaxDrawdown": summary.max_drawdown_percent,
        "Sharpe": summary.sharpe_ratio,
        "AnnualizedTurnover": summary.annualized_turnover,
        "TickerTrades": summary.ticker_trades,
        "Rebalances": summary.rebalance_count,
    }


def _health_summary(scored: pd.DataFrame, factored: pd.DataFrame) -> dict[str, float | None]:
    health = build_signal_health(scored, factored, k=5)
    return {
        "IC13W": health["portfolio_ic_rolling"]["13W"],
        "IC26W": health["portfolio_ic_rolling"]["26W"],
        "Spread13W": health["top_k_spread_rolling"]["13W"],
        "Spread26W": health["top_k_spread_rolling"]["26W"],
    }


def main() -> None:
    paths = load_paths()
    frozen_bytes = FROZEN_PATH.read_bytes()
    frozen = json.loads(frozen_bytes)
    state = state_from_snapshot(SNAPSHOT_PATH)
    start, end = frozen["period"]["start"], frozen["period"]["end"]
    panel = engine.build_scored_panel(paths, state, start=start, end=end, top_k=5)
    market = prepare_market(panel.factored, start=start, end=end)
    base = engine._base_params()
    policy = engine._policy(5)

    # Predeclared variants: keep every exit, filing, timing, universe and cost
    # rule identical. Only the five technical factor weights differ.
    candidates = {
        "baseline": base,
        "quality_plus_8pt_trend_minus_8pt": replace(
            base, quality_weight=base.quality_weight + 0.08, trend_weight=base.trend_weight - 0.08
        ),
        "quality_plus_12pt_momentum_minus_6pt_trend_minus_6pt": replace(
            base,
            quality_weight=base.quality_weight + 0.12,
            momentum_weight=base.momentum_weight - 0.06,
            trend_weight=base.trend_weight - 0.06,
        ),
        "quality_plus_8pt_growth_plus_4pt_trend_minus_12pt": replace(
            base,
            quality_weight=base.quality_weight + 0.08,
            growth_weight=base.growth_weight + 0.04,
            trend_weight=base.trend_weight - 0.12,
        ),
    }
    rows: list[dict[str, object]] = []
    signal_dates = pd.Index(panel.scored["Date"].unique())
    weekly_factored = panel.factored.loc[panel.factored["Date"].isin(signal_dates)].copy()
    for name, params in candidates.items():
        scored, targets = generate_filing_v7_targets(weekly_factored, params, policy)
        execution = monthly_weight_reset_targets(targets)
        full = run_portfolio_backtest({}, execution, start=start, end=end, initial_capital=panel.settings.initial_capital, transaction_cost_bps=panel.settings.transaction_cost_bps, prepared_market=market)
        oos = run_portfolio_backtest({}, execution, start=OOS_START, end=end, initial_capital=panel.settings.initial_capital, transaction_cost_bps=panel.settings.transaction_cost_bps, prepared_market=prepare_market(panel.factored, start=OOS_START, end=end))
        selection = scored.loc[scored["Date"].between(start, SELECTION_END)].copy()
        rows.append({
            "candidate": name,
            "weights": params.factor_weights,
            "selection_health": _health_summary(selection, panel.factored),
            "full_period_report_only": _summary(full),
            "oos_2026": _summary(oos),
        })
    payload = {
        "research_only": True,
        "generated_at": date.today().isoformat(),
        "frozen_artifact": str(FROZEN_PATH),
        "frozen_artifact_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "universe_snapshot": str(SNAPSHOT_PATH),
        "period": frozen["period"],
        "selection_period": {"start": start, "end": SELECTION_END},
        "oos_period": {"start": OOS_START, "end": end},
        "constant_conditions": "U001 snapshot; Friday weekly signals; monthly first-signal weight reset; next-session-open execution; 10 bps transaction cost; Top 5; all exit and filing rules unchanged.",
        "selection_rule": "This report does not promote a candidate. A candidate needs stronger pre-2026 IC and spread plus an improved 2026 OOS CAGR/Sharpe without materially worse MDD or turnover.",
        "candidates": rows,
    }
    _atomic_json(OUTPUT_PATH, payload)
    print(json.dumps({"output": str(OUTPUT_PATH), "candidates": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
