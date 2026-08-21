"""V7.3 + V2 over 2020-2026 -- the window that actually contains 2022.

The frozen backtest starts 2024-01-02, and V2 fired exactly twice in it. 2022,
where V2 fired 12 times and spent 182 days defensive, sits outside it. So the
frozen window cannot say whether the overlay is worth anything.

SEC filing features start 2019-01-09, so 2020-01-02 is the earliest the real
V7.3 engine can run. That covers COVID, the 2021 melt-up, the 2022 bear, 2023-24,
the 2025 tariff drawdown and 2026.

Read as research, not as a new baseline:
  - the 31-name universe is the 2026-08-12 snapshot run backwards, so 2020-2021
    carry survivorship bias and a thinner effective cross-section (PLTR, SNOW, U
    and ARRY only listed in late 2020 and need 200 sessions before they qualify).
  - the frozen artifact and its period are untouched.

Research only. Nothing committed, nothing deployed.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.portfolio import prepare_market
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import OUT, atomic_csv, atomic_json, frozen_signal_days
from engine_bits import (V2_BASELINE, V2_TUNED, backtest_nav, buy_and_hold, qqq_close,
                         spy_close, stats, v2_overlay_nav, v2_weights)
from run_a import equal_weight_targets
from run_long_addendum import spy_driven_exposure
from run_long_defense import apply_exposure

START = "2020-01-02"
END = "2026-08-12"
YEARS = [str(y) for y in range(2020, 2027)]
STRESS = {"covid_2020Q1": ("2020-02-14", "2020-04-30"),
          "bear_2022": ("2022-01-03", "2022-12-30"),
          "tariff_2025Q2": ("2025-02-14", "2025-05-14")}


def year_slices(nav: pd.Series) -> dict:
    out = {}
    for year in YEARS:
        piece = nav.loc["%s-01-01" % year:"%s-12-31" % year]
        if len(piece) > 30:
            out[year] = round(stats(piece)["ROI"], 1)
    for label, (a, b) in STRESS.items():
        piece = nav.loc[a:b]
        if len(piece) > 20:
            out[label] = round(stats(piece)["ROI"], 1)
    return out


def main():
    panel = pickle.load(open(OUT / "panel_2020_2026.pkl", "rb"))
    capital = panel.settings.initial_capital
    market = prepare_market(panel.factored, start=START, end=END)
    signal_days = frozen_signal_days(panel.factored, START, END, panel.settings.rebalance_weekday)
    scored, targets = generate_filing_v7_targets(signal_days, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])

    base, base_extra = backtest_nav(monthly_weight_reset_targets(targets), panel.factored,
                                    START, END, capital, market)
    curves = {"V7.3 alone (no overlay)": base}
    info = {}

    for label, params in (("V7.3 + V2 (original params)", V2_BASELINE),
                          ("V7.3 + V2 (tuned params)", V2_TUNED)):
        nav, meta = v2_overlay_nav(base, params, charge_cost=True)
        curves[label] = nav
        info[label] = meta

    close = spy_close()
    v2 = pd.Series(v2_weights(close.to_numpy(), V2_BASELINE), index=close.index)
    v2 = v2.reindex(base.index).ffill().fillna(1.0)
    for target in (0.12, 0.15):
        combined = pd.concat([v2, spy_driven_exposure(base.index, target)], axis=1).min(axis=1)
        curves["V7.3 + min(V2, SPY-vol %d%%)" % int(target * 100)] = apply_exposure(base, combined)

    ew, _ = backtest_nav(equal_weight_targets(scored), panel.factored, START, END, capital, market)
    curves["CONTROL universe equal weight"] = ew
    for label, series in (("CONTROL SPY buy&hold", spy_close()), ("CONTROL QQQ buy&hold", qqq_close())):
        curves[label] = buy_and_hold(series, base.index, capital)

    rows = []
    for name, nav in curves.items():
        row = {"strategy": name, **{k: round(v, 2) for k, v in stats(nav).items()}}
        row.update(year_slices(nav))
        if name in info:
            row["V2_episodes"] = info[name]["DefensiveEpisodes"]
            row["V2_defensive_days"] = info[name]["DefensiveDays"]
        rows.append(row)
    table = pd.DataFrame(rows)
    atomic_csv(OUT / "v2_with_2022_results.csv", table)
    frame = pd.DataFrame(curves)
    frame.index.name = "Date"
    atomic_csv(OUT / "v2_with_2022_curves.csv", frame.reset_index())
    atomic_json(OUT / "v2_with_2022_windows.json", {n: year_slices(c) for n, c in curves.items()})

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print("=== V7.3 + V2, 2020-01-02 .. 2026-08-12 (10 bps, cost charged on every V2 weight change) ===")
    print(table[["strategy", "ROI", "CAGR", "MDD", "Sharpe", "Calmar"]].to_string(index=False))
    print()
    print("=== by year, and the three stress windows ===")
    cols = ["strategy"] + [c for c in table.columns if c in YEARS or c in STRESS]
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
