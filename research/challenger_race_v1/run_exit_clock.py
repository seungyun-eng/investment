"""If 2026 was a timing failure, which timing rule caused it?

The decomposition says the 2026 picks were fine held passively (+11.1%) and the
model's own entries and exits turned that into -3.0%.  V7.3 has four rules that
can delay a rotation, all of them added to *protect* losers:

  minimum_hold_rebalances = 4   a new holding cannot be replaced for 4 weeks
  minimum_exit_gain       = 1%  a rank-based rotation only fires at a profit
  replacement_score_advantage   a challenger must beat the holding by 0.05
  exit_rank buffer              a holding survives until it falls past rank 9

Each is switched off one at a time, selection untouched, and scored on 2024,
2025 and 2026 separately.  A rule that helps in 2024/2025 and hurts in 2026 is
the regime-dependent one; a rule that hurts everywhere is just wrong.

Research only.  Nothing committed, nothing deployed.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.portfolio import prepare_market
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import OUT, atomic_csv, frozen_signal_days, load_panel
from engine_bits import backtest_nav, stats

YEARS = {"2024": ("2024-01-02", "2024-12-31"),
         "2025": ("2025-01-02", "2025-12-31"),
         "2026_OOS": ("2026-01-02", None)}


def main():
    bundle = load_panel()
    panel, start, end = bundle["panel"], bundle["start"], bundle["end"]
    capital = panel.settings.initial_capital
    market = prepare_market(panel.factored, start=start, end=end)
    signal_days = frozen_signal_days(panel.factored, start, end, panel.settings.rebalance_weekday)
    base = engine._base_params()
    policy = engine._policy(5)

    # generate_filing_v7_targets calls policy.strategy_params(base_params), which
    # OVERRIDES top_k / exit_rank / hard_stop / minimum_hold_rebalances /
    # replacement_score_advantage from the policy.  Those four must be varied on
    # the policy; minimum_exit_gain and loss_aware_exit_enabled live on the params.
    variants = {
        "baseline (V7.3 as shipped)": (base, policy),
        "no minimum hold (4w -> 1w)": (base, replace(policy, minimum_hold_rebalances=1)),
        "no minimum exit gain (1% -> 0%)": (replace(base, minimum_exit_gain=0.0), policy),
        "no replacement hurdle (0.05 -> 0)": (base, replace(policy, replacement_score_advantage=0.0)),
        "tighter exit rank (buffer 4 -> 1)": (base, replace(policy, exit_rank_buffer=1)),
        "wider exit rank (buffer 4 -> 8)": (base, replace(policy, exit_rank_buffer=8)),
        "loss-aware exits off entirely": (replace(base, loss_aware_exit_enabled=False), policy),
        "all delays off": (replace(base, minimum_exit_gain=0.0),
                           replace(policy, minimum_hold_rebalances=1, replacement_score_advantage=0.0)),
        "all delays off + tight exit rank": (replace(base, minimum_exit_gain=0.0),
                                             replace(policy, minimum_hold_rebalances=1,
                                                     replacement_score_advantage=0.0, exit_rank_buffer=1)),
    }

    rows = []
    for name, (params, variant_policy) in variants.items():
        scored, targets = generate_filing_v7_targets(signal_days, params, variant_policy)
        targets["Date"] = pd.to_datetime(targets["Date"])
        execution = monthly_weight_reset_targets(targets)
        row = {"variant": name}
        full, extra = backtest_nav(execution, panel.factored, start, end, capital, market)
        row["full_CAGR"] = round(stats(full)["CAGR"], 1)
        row["full_Sharpe"] = round(stats(full)["Sharpe"], 2)
        row["full_MDD"] = round(stats(full)["MDD"], 1)
        row["turnover"] = round(extra["AnnualizedTurnover"], 1)
        for label, (a, b) in YEARS.items():
            b = b or end
            nav, _ = backtest_nav(execution, panel.factored, a, b, capital,
                                  prepare_market(panel.factored, start=a, end=b))
            row[label] = round(stats(nav)["ROI"], 1)
        rows.append(row)

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "diag_exit_clock.csv", table)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
