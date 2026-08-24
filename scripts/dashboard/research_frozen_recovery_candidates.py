from __future__ import annotations

"""Research-only, fundamentally-confirmed low-momentum candidates.

This is deliberately separate from the frozen model and production dashboard.
It reuses the production filing V7 scoring, target-generation and execution
functions.  The only experimental inputs are factor weights, the 126-session
entry floor, and an optional *recovery overlay*: a modest score bonus only for
stocks with above-median Quality and Growth that are still in an intact trend
but have below-median six-month momentum.  It is not a recommendation engine.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
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
OUTPUT_PATH = Path("artifacts/frozen_recovery_candidate_study/results.json")
PERIODS = {
    "selection_2024": ("2024-01-02", "2024-12-31"),
    "selection_2025": ("2025-01-02", "2025-12-31"),
    "oos_2026": ("2026-01-02", "2026-08-12"),
}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem, suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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


def _health(scored: pd.DataFrame, factored: pd.DataFrame) -> dict[str, float | None]:
    report = build_signal_health(scored, factored, k=5)
    return {
        "IC13W": report["portfolio_ic_rolling"]["13W"],
        "IC26W": report["portfolio_ic_rolling"]["26W"],
        "Spread13W": report["top_k_spread_rolling"]["13W"],
        "Spread26W": report["top_k_spread_rolling"]["26W"],
    }


def _recovery_overlay(panel: pd.DataFrame, strength: float) -> pd.DataFrame:
    """Apply a bounded bonus only to financially supported, modest momentum.

    The overlay cannot rescue a broken long-term trend: the normal Trend200
    and Return126 floors are still enforced later by generate_filing_v7_targets.
    """

    frame = panel.copy()
    eligible = frame["Eligible"].fillna(False)
    columns = ["QualityFactor", "GrowthFactor", "Return126", "Trend200"]
    ranks = frame.loc[:, columns].groupby(frame["Date"], sort=False).rank(
        pct=True, method="average"
    )
    support = ranks["QualityFactor"].ge(0.60) & ranks["GrowthFactor"].ge(0.60)
    moderate_momentum = ranks["Return126"].between(0.20, 0.60)
    intact_trend = pd.to_numeric(frame["Trend200"], errors="coerce").ge(-0.10)
    # Within the allowed band, lower momentum receives a larger but bounded
    # (+0.0 to +0.4 before strength) offset.  It is measured in the same
    # centred-rank scale as the underlying technical factors.
    pullback = ((0.60 - ranks["Return126"]) / 0.40).clip(0.0, 1.0) * 0.40
    bonus = (pullback * strength).where(
        eligible & support & moderate_momentum & intact_trend, 0.0
    )
    frame["RecoveryOverlay"] = bonus
    frame["MomentumFactor"] = pd.to_numeric(
        frame["MomentumFactor"], errors="coerce"
    ).add(bonus, fill_value=0.0)
    return frame


def _candidate_grid(base: object) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = [
        {"name": "baseline", "params": base, "overlay_strength": 0.0}
    ]
    # Predeclared 18-cell weight/floor grid.  Lower momentum weight is moved
    # only to Quality/Growth; Trend, risk control, exits, filing rules and
    # execution remain unchanged.
    for momentum in (0.15, 0.10, 0.05):
        released = base.momentum_weight - momentum
        for quality_share in (0.50, 0.75):
            params = replace(
                base,
                momentum_weight=momentum,
                quality_weight=base.quality_weight + released * quality_share,
                growth_weight=base.growth_weight + released * (1 - quality_share),
            )
            for floor in (-0.35, -0.45, -0.55):
                candidates.append(
                    {
                        "name": (
                            f"qg_weight_m{momentum:.2f}_qshare{quality_share:.2f}"
                            f"_floor{floor:.2f}"
                        ),
                        "params": replace(params, momentum_floor=floor),
                        "overlay_strength": 0.0,
                    }
                )
    # Six direct tests of the user's recovery hypothesis.  The weights above
    # reduce momentum from 20% to 10%; the overlay is a limited confirmation,
    # not a reversal of all price information.
    recovery_released = base.momentum_weight - 0.10
    recovery_base = replace(
        base,
        momentum_weight=0.10,
        quality_weight=base.quality_weight + recovery_released * 0.60,
        growth_weight=base.growth_weight + recovery_released * 0.40,
    )
    for strength in (0.25, 0.50, 0.75):
        for floor in (-0.45, -0.55):
            candidates.append(
                {
                    "name": f"confirmed_recovery_s{strength:.2f}_floor{floor:.2f}",
                    "params": replace(recovery_base, momentum_floor=floor),
                    "overlay_strength": strength,
                }
            )
    return candidates


def _selection_characteristics(scored: pd.DataFrame) -> dict[str, float | int]:
    top = scored.loc[scored["Rank"].le(5) & scored["Qualified"]].copy()
    return {
        "MeanReturn126": float(pd.to_numeric(top["Return126"], errors="coerce").mean()),
        "MeanQualityFactor": float(pd.to_numeric(top["QualityFactor"], errors="coerce").mean()),
        "MeanGrowthFactor": float(pd.to_numeric(top["GrowthFactor"], errors="coerce").mean()),
        "WeeklyTop5Observations": int(len(top)),
    }


def main() -> None:
    paths = load_paths()
    frozen_bytes = FROZEN_PATH.read_bytes()
    frozen = json.loads(frozen_bytes)
    start, end = frozen["period"]["start"], frozen["period"]["end"]
    state = state_from_snapshot(SNAPSHOT_PATH)
    panel = engine.build_scored_panel(paths, state, start=start, end=end, top_k=5)
    market_by_period = {
        name: prepare_market(panel.factored, start=period_start, end=period_end)
        for name, (period_start, period_end) in PERIODS.items()
    }
    signal_dates = pd.Index(panel.scored["Date"].unique())
    weekly_factored = panel.factored.loc[panel.factored["Date"].isin(signal_dates)].copy()
    base = engine._base_params()
    policy = engine._policy(5)
    rows: list[dict[str, object]] = []
    for candidate in _candidate_grid(base):
        candidate_panel = weekly_factored
        overlay_strength = float(candidate["overlay_strength"])
        if overlay_strength:
            candidate_panel = _recovery_overlay(weekly_factored, overlay_strength)
        scored, targets = generate_filing_v7_targets(
            candidate_panel, candidate["params"], policy
        )
        execution = monthly_weight_reset_targets(targets)
        period_results = {}
        for label, (period_start, period_end) in PERIODS.items():
            result = run_portfolio_backtest(
                {}, execution,
                start=period_start, end=period_end,
                initial_capital=panel.settings.initial_capital,
                transaction_cost_bps=panel.settings.transaction_cost_bps,
                prepared_market=market_by_period[label],
            )
            period_results[label] = _summary(result)
        selection = scored.loc[scored["Date"].between("2024-01-02", "2025-12-31")]
        rows.append(
            {
                "candidate": candidate["name"],
                "weights": candidate["params"].factor_weights,
                "momentum_floor": candidate["params"].momentum_floor,
                "recovery_overlay_strength": overlay_strength,
                "selection_health": _health(selection, panel.factored),
                "selection_characteristics": _selection_characteristics(selection),
                "period_results": period_results,
            }
        )
    payload = {
        "research_only": True,
        "generated_at": date.today().isoformat(),
        "frozen_artifact": str(FROZEN_PATH),
        "frozen_artifact_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "universe_snapshot": str(SNAPSHOT_PATH),
        "period": frozen["period"],
        "constant_conditions": (
            "U001 snapshot; Friday weekly signal; Top 5; monthly first-signal "
            "weight reset; next-session-open execution; 10 bps transaction cost; "
            "same filing gates and exit rules."
        ),
        "selection_protocol": (
            "2024 and 2025 are selection diagnostics.  2026 is held out as a "
            "report-only OOS period; no candidate is promoted by this report."
        ),
        "candidate_count": len(rows),
        "candidates": rows,
    }
    _atomic_json(OUTPUT_PATH, payload)
    print(json.dumps({"output": str(OUTPUT_PATH), "candidates": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
