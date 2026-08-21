"""Both open questions, on the longest panels each can actually use.

Q1 (rotation speed): only 2.6 years of evidence so far, and effectively one
    regime flip.  The price-only engine runs back to 2008, so the exit-rank
    buffer is swept there instead -- 19 calendar years, four bear markets.

Q2 (growth vs quality): needs the fundamental factors, which need the SEC
    filing features.  Those start 2019-01-09, so the panel is rebuilt for
    2020-01-02..2026-08-12 -- 6.6 years covering COVID, 2021, the 2022 bear,
    2023-24, 2025 and 2026, instead of the 2.6 years used in Part 3.

Same universe snapshot, 10 bps, next-open execution, monthly weight reset.
Research only.  Nothing committed, nothing deployed.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.portfolio import prepare_market
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine
from stock_research.dashboard.state import state_from_snapshot
from stock_research.paths import load_paths

from common import OUT, SNAPSHOT_PATH, atomic_csv, frozen_signal_days
from engine_bits import backtest_nav, stats

LONG_START = "2020-01-02"
LONG_END = "2026-08-12"
LONG_CACHE = OUT / "panel_2020_2026.pkl"
YEARS = [str(y) for y in range(2020, 2027)]


def load_long_panel(force: bool = False):
    if LONG_CACHE.exists() and not force:
        with LONG_CACHE.open("rb") as fh:
            return pickle.load(fh)
    state = state_from_snapshot(SNAPSHOT_PATH)
    panel = engine.build_scored_panel(load_paths(), state, start=LONG_START, end=LONG_END, top_k=5)
    with LONG_CACHE.open("wb") as fh:
        pickle.dump(panel, fh)
    return panel


def year_bounds(year: str, end: str):
    return ("%s-01-01" % year, min("%s-12-31" % year, end))


def score_by_year(execution, factored, capital, markets, start, end, market):
    nav, extra = backtest_nav(execution, factored, start, end, capital, market)
    row = {**{k: round(v, 2) for k, v in stats(nav).items()},
           "turnover": round(extra["AnnualizedTurnover"], 1)}
    for year, (a, b) in markets.items():
        piece, _ = backtest_nav(execution, factored, a, b, capital,
                                prepare_market(factored, start=a, end=b))
        row[year] = round(stats(piece)["ROI"], 1)
    return row


# --------------------------------------------------------------------------- #
# Q1: rotation speed, 2008-2026, price-only engine
# --------------------------------------------------------------------------- #
def rotation_speed_long():
    from stock_research.cross_sectional.signals import generate_rebalance_targets, score_panel
    from run_long import END, START, build_long_panel

    settings = engine._settings(START, "2019-12-31", "2020-01-02", END, 8)
    panel = build_long_panel()
    market = prepare_market(panel, start=START, end=END)
    capital = engine.INITIAL_CAPITAL
    signal_days = frozen_signal_days(panel, START, END, settings.rebalance_weekday)
    base = engine._base_params()

    years = {str(y): (("%d-01-01" % y), min("%d-12-31" % y, END)) for y in range(2008, 2027)}
    rows = []
    for buffer in (1, 2, 4, 6, 8, 12):
        params = replace(base, top_k=5, exit_rank=5 + buffer,
                         profit_rotation_exit_rank=5 + buffer,
                         conviction_exit_rank=5 + buffer + 3)
        scored = score_panel(signal_days, params)
        targets = generate_rebalance_targets(scored, params)
        targets["Date"] = pd.to_datetime(targets["Date"])
        execution = monthly_weight_reset_targets(targets)
        row = {"exit_buffer": buffer}
        row.update(score_by_year(execution, panel, capital, years, START, END, market))
        rows.append(row)
    table = pd.DataFrame(rows)
    atomic_csv(OUT / "q1_rotation_speed_2008_2026.csv", table)
    return table


# --------------------------------------------------------------------------- #
# Q2: growth vs quality weight, 2020-2026, full engine
# --------------------------------------------------------------------------- #
def growth_quality_sweep(panel):
    market = prepare_market(panel.factored, start=LONG_START, end=LONG_END)
    capital = panel.settings.initial_capital
    signal_days = frozen_signal_days(panel.factored, LONG_START, LONG_END,
                                     panel.settings.rebalance_weekday)
    base = engine._base_params()
    policy = engine._policy(5)
    pool = base.growth_weight + base.quality_weight
    years = {y: year_bounds(y, LONG_END) for y in YEARS}

    rows = []
    for growth_share in (1.0, 0.75, 0.5, 0.387, 0.25, 0.0):
        growth = pool * growth_share
        quality = pool - growth
        params = replace(base, growth_weight=growth, quality_weight=quality)
        scored, targets = generate_filing_v7_targets(signal_days, params, policy)
        targets["Date"] = pd.to_datetime(targets["Date"])
        execution = monthly_weight_reset_targets(targets)
        row = {"growth_w": round(growth, 3), "quality_w": round(quality, 3),
               "growth_share_of_pool": round(growth_share, 3)}
        row.update(score_by_year(execution, panel.factored, capital, years,
                                 LONG_START, LONG_END, market))
        rows.append(row)
    table = pd.DataFrame(rows)
    atomic_csv(OUT / "q2_growth_quality_2020_2026.csv", table)
    return table


def rotation_speed_fundamental(panel):
    """Q1 cross-check on the real engine over 6.6 years instead of 2.6."""
    market = prepare_market(panel.factored, start=LONG_START, end=LONG_END)
    capital = panel.settings.initial_capital
    signal_days = frozen_signal_days(panel.factored, LONG_START, LONG_END,
                                     panel.settings.rebalance_weekday)
    base = engine._base_params()
    years = {y: year_bounds(y, LONG_END) for y in YEARS}
    rows = []
    for buffer in (1, 2, 4, 6, 8, 12):
        policy = replace(engine._policy(5), exit_rank_buffer=buffer)
        scored, targets = generate_filing_v7_targets(signal_days, base, policy)
        targets["Date"] = pd.to_datetime(targets["Date"])
        execution = monthly_weight_reset_targets(targets)
        row = {"exit_buffer": buffer}
        row.update(score_by_year(execution, panel.factored, capital, years,
                                 LONG_START, LONG_END, market))
        rows.append(row)
    table = pd.DataFrame(rows)
    atomic_csv(OUT / "q1_rotation_speed_2020_2026.csv", table)
    return table


def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)

    print("=== Q1a: rotation speed, price-only engine, 2008-2026 (19 years) ===")
    print(rotation_speed_long().to_string(index=False))

    panel = load_long_panel()
    print()
    print("panel 2020-2026 built:", panel.factored["Date"].min().date(), "->",
          panel.factored["Date"].max().date(), panel.factored["Ticker"].nunique(), "tickers")

    print()
    print("=== Q1b: rotation speed, full engine, 2020-2026 (6.6 years) ===")
    print(rotation_speed_fundamental(panel).to_string(index=False))

    print()
    print("=== Q2: growth vs quality weight, full engine, 2020-2026 ===")
    print("   (V7.3 ships growth 0.196 / quality 0.309 -- growth_share_of_pool 0.387)")
    print(growth_quality_sweep(panel).to_string(index=False))


if __name__ == "__main__":
    main()
