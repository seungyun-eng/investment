"""Validate the one survivor: min(V2, SPY-vol-driven vol target).

It was picked after looking at the full 2008-2026 result and its target vol
(15%) was chosen the same way, so the headline number is in-sample. Three
things have to hold before it is worth shadowing:

  1. SENSITIVITY  -- it must work across a range of target vols, not at one.
  2. SPLIT SAMPLE -- it must beat its control in BOTH halves of the period,
                     not just in the half that happens to contain 2022.
  3. CONTROL      -- in every cell, versus simply holding that rule's own mean
                     exposure flat. Cutting drawdown by holding less is not an
                     edge; only the timing is.

A rule that only clears the bar in one half, or at one parameter value, is
curve fit and gets rejected here.

Research only. Nothing committed, nothing deployed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.portfolio import prepare_market
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import OUT, atomic_csv, frozen_signal_days, load_panel
from engine_bits import V2_BASELINE, backtest_nav, spy_close, stats, v2_weights
from run_long import END, START, build_long_panel, v7_price_targets
from run_long_addendum import spy_driven_exposure
from run_long_defense import apply_exposure

HALVES = {"first half 2008-2017H1": (START, "2017-06-30"),
          "second half 2017H2-2026": ("2017-07-01", END)}


def evaluate(nav: pd.Series, weights: pd.Series, label: str, rule: str, target) -> dict:
    defended = apply_exposure(nav, weights)
    flat = apply_exposure(nav, pd.Series(float(weights.mean()), index=nav.index))
    s, f = stats(defended), stats(flat)
    return {"window": label, "rule": rule, "target_vol": target,
            "mean_exposure": round(float(weights.mean()), 3),
            "CAGR": round(s["CAGR"], 2), "MDD": round(s["MDD"], 2),
            "Sharpe": round(s["Sharpe"], 3), "Calmar": round(s["Calmar"], 3),
            "Sharpe_gain": round(s["Sharpe"] - f["Sharpe"], 3),
            "Calmar_gain": round(s["Calmar"] - f["Calmar"], 3)}


def sweep(nav: pd.Series, label: str) -> list[dict]:
    close = spy_close()
    v2 = pd.Series(v2_weights(close.to_numpy(), V2_BASELINE), index=close.index)
    v2 = v2.reindex(nav.index).ffill().fillna(1.0)
    rows = [evaluate(nav, v2, label, "V2 alone", None)]
    for target in (0.10, 0.12, 0.15, 0.18, 0.22):
        vt = spy_driven_exposure(nav.index, target)
        rows.append(evaluate(nav, vt, label, "SPY-vol target alone", target))
        combined = pd.concat([v2, vt], axis=1).min(axis=1)
        rows.append(evaluate(nav, combined, label, "min(V2, SPY-vol target)", target))
    return rows


def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)

    settings = engine._settings(START, "2019-12-31", "2020-01-02", END, 8)
    panel = build_long_panel()
    _, targets = v7_price_targets(panel, settings)
    market = prepare_market(panel, start=START, end=END)
    nav, _ = backtest_nav(monthly_weight_reset_targets(targets), panel, START, END,
                          engine.INITIAL_CAPITAL, market)

    rows = sweep(nav, "full 2008-2026")
    for label, (a, b) in HALVES.items():
        piece = nav.loc[a:b]
        rows += sweep(piece, label)

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "survivor_validation.csv", table)

    for window in ["full 2008-2026", *HALVES]:
        print("=== %s ===" % window)
        view = table.loc[table["window"].eq(window)].drop(columns=["window"])
        print(view.to_string(index=False))
        print()

    combined = table.loc[table["rule"].eq("min(V2, SPY-vol target)")]
    print("=== verdict: does min(V2, SPY-vol target) beat its own flat control? ===")
    for target in sorted(combined["target_vol"].unique()):
        cells = combined.loc[combined["target_vol"].eq(target)]
        halves = cells.loc[cells["window"].isin(HALVES)]
        ok_sharpe = bool((halves["Sharpe_gain"] > 0).all())
        ok_calmar = bool((halves["Calmar_gain"] > 0).all())
        print("  target %.0f%%: both halves positive -- Sharpe %s, Calmar %s  "
              "(gains %s / %s)"
              % (target * 100, "YES" if ok_sharpe else "NO", "YES" if ok_calmar else "NO",
                 list(halves["Sharpe_gain"]), list(halves["Calmar_gain"])))

    # secondary: the live fundamental book, 2024-2026 V7.3.1
    print()
    print("=== secondary check on the live V7.3.1 book (2024-2026) ===")
    from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
    bundle = load_panel()
    fp, fstart, fend = bundle["panel"], bundle["start"], bundle["end"]
    sd = frozen_signal_days(fp.factored, fstart, fend, fp.settings.rebalance_weekday)
    _, ft = generate_filing_v7_targets(sd, engine._base_params(), engine._policy(5))
    ft["Date"] = pd.to_datetime(ft["Date"])
    fnav, _ = backtest_nav(monthly_weight_reset_targets(ft), fp.factored, fstart, fend,
                           fp.settings.initial_capital,
                           prepare_market(fp.factored, start=fstart, end=fend))
    live = pd.DataFrame(sweep(fnav, "V7.3.1 book 2024-2026")).drop(columns=["window"])
    atomic_csv(OUT / "survivor_validation_live_book.csv", live)
    print(live.to_string(index=False))


if __name__ == "__main__":
    main()
