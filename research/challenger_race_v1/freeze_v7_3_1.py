"""Freeze V7.3.1 -- V7.3 recomputed on the repaired filing code.

V7.3 was frozen 2026-08-13 against a filing pipeline that (a) skipped every Q4
in trailing-twelve-month sums, because most US filers only report Q4 inside the
annual 10-K, and (b) read a 10-for-1 stock split as ~1000% share dilution. Both
are now fixed, plus two KeyError crash paths in the display-only valuation block.
The ranking those fundamentals feed therefore changed, and V7.3's headline
213.72% no longer describes what the code produces.

`scripts/dashboard/freeze_backtest.py` deliberately refuses to overwrite an
existing frozen artifact, so this writes a NEW version and leaves v7_3.json as
the historical record. It also does not go through that script, because that
script reads the live `latest_today.json` -- which would require re-running the
whole dashboard publish. This recomputes the same backtest directly instead, and
records that difference in `provenance`.

Nothing is committed and nothing is deployed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.portfolio import prepare_market, run_portfolio_backtest
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import FROZEN_PATH, REGISTRY_DIR, ROOT, atomic_json, frozen_signal_days, load_panel

NEW_VERSION = "V7.3.1"
NEW_PATH = REGISTRY_DIR / "frozen_backtests" / "v7_3_1.json"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    if NEW_PATH.exists():
        raise SystemExit("%s already exists and is FROZEN -- refusing to overwrite." % NEW_PATH)

    bundle = load_panel()
    panel, start, end = bundle["panel"], bundle["start"], bundle["end"]
    signal_days = frozen_signal_days(panel.factored, start, end, panel.settings.rebalance_weekday)
    scored, targets = generate_filing_v7_targets(signal_days, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])
    execution = monthly_weight_reset_targets(targets)
    result = run_portfolio_backtest(
        {}, execution, start=start, end=end,
        initial_capital=panel.settings.initial_capital,
        transaction_cost_bps=panel.settings.transaction_cost_bps,
        prepared_market=prepare_market(panel.factored, start=start, end=end),
        record_attribution=True,
    )
    summary = result.summary
    previous = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    payload = {
        "model_version": NEW_VERSION,
        "universe_version": previous["universe_version"],
        "status": "FROZEN",
        "frozen_at": date.today().isoformat(),
        "supersedes": {
            "model_version": previous["model_version"],
            "frozen_at": previous["frozen_at"],
            "ROI": previous["summary"]["ROI"],
            "reason": "filing pipeline repaired: Q4 derivation in TTM sums, "
                      "split-adjusted share counts, and two KeyError crash paths "
                      "in the display-only valuation block",
        },
        "period": previous["period"],
        "universe": previous["universe"],
        "summary": {
            "StartDate": summary.start_date,
            "EndDate": summary.end_date,
            "InitialCapital": summary.initial_capital,
            "TotalInjected": summary.total_injected,
            "FinalValue": summary.final_value,
            "ROI": summary.roi_percent,
            "CAGR": summary.cagr_percent,
            "MaxDrawdown": summary.max_drawdown_percent,
            "Sharpe": summary.sharpe_ratio,
            "TurnoverMultiple": summary.turnover_multiple,
            "AnnualizedTurnover": summary.annualized_turnover,
            "TickerTrades": summary.ticker_trades,
            "Rebalances": summary.rebalance_count,
        },
        "ticker_summary": engine._ticker_summary_payload(targets, result),
        "git_commit": git_commit(),
        "provenance": {
            "recomputed_by": "research/challenger_race_v1/freeze_v7_3_1.py",
            "not_via": "scripts/dashboard/freeze_backtest.py (that script reads the "
                       "live latest_today.json; this recomputes the backtest directly)",
            "signal_dates": "frozen_signal_days shim -- the post-freeze production "
                            "helper drops an incomplete current week and lands on a "
                            "different weekly grid",
            "working_tree_uncommitted": True,
            "tests": "pytest tests/ (excluding untracked, pre-broken tests/tesla_v3) "
                     "= 513 passed at freeze time",
        },
    }
    atomic_json(NEW_PATH, payload)
    print("wrote %s" % NEW_PATH)
    print("  ROI  %.2f%%  (was %.2f%%)" % (summary.roi_percent, previous["summary"]["ROI"]))
    print("  CAGR %.2f%%  (was %.2f%%)" % (summary.cagr_percent, previous["summary"]["CAGR"]))
    print("  MDD  %.2f%%  (was %.2f%%)" % (summary.max_drawdown_percent, previous["summary"]["MaxDrawdown"]))
    print("  Sharpe %.3f (was %.3f)" % (summary.sharpe_ratio, previous["summary"]["Sharpe"]))


if __name__ == "__main__":
    main()
