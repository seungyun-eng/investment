"""2026: wrong stocks, or right stocks at the wrong time?

The ablation showed the fundamental/filing side produced all of 2025 and then
went negative in 2026.  That does not say WHICH kind of failure it is, and the
two have opposite fixes:

  selection failure -> the names it picked simply did not go up.  Fix the score.
  timing failure    -> the names went up, but not while we held them.  Fix the
                       entry/exit clock, not the score.

Three measurements separate them:

  1. Factor rank IC by horizon (1w/1m/3m/6m) and by year.  A score that is
     negative at short horizon and positive at long horizon is early, not wrong.
  2. Selection-vs-timing decomposition.  Compare what the model actually earned
     against holding the exact same names, equal weight, from first selection to
     the end of the window.  The gap is what the entry/exit clock cost.
  3. Per-entry timing: for every 2026 entry, the run-up before it bought, the
     return it captured, and the return it left on the table after selling.

Research only.  Nothing is committed or deployed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.portfolio import prepare_market
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import OUT, atomic_csv, atomic_json, frozen_signal_days, load_panel
from engine_bits import backtest_nav, stats

HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}
FACTORS = ["MomentumFactor", "TrendFactor", "RiskControlFactor", "GrowthFactor",
           "QualityFactor", "FilingFundamentalFactor", "FilingDurabilityFactor",
           "BaseV7Score", "AlphaScore"]


def forward_returns(factored: pd.DataFrame) -> pd.DataFrame:
    """Forward return at each horizon for every ticker/date, from daily closes."""
    wide = factored.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    frames = []
    for label, horizon in HORIZONS.items():
        fwd = (wide.shift(-horizon) / wide - 1.0).stack().rename("Fwd" + label).reset_index()
        fwd.columns = ["Date", "Ticker", "Fwd" + label]
        frames.append(fwd.set_index(["Date", "Ticker"]))
    return pd.concat(frames, axis=1).reset_index()


def factor_ic(scored: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    fwd["Date"] = pd.to_datetime(fwd["Date"])
    merged = frame.merge(fwd, on=["Date", "Ticker"], how="left")
    merged["Year"] = merged["Date"].dt.year
    rows = []
    for (year, factor), _ in [((y, f), None) for y in sorted(merged["Year"].unique()) for f in FACTORS]:
        if factor not in merged.columns:
            continue
        row = {"Year": year, "Factor": factor}
        for label in HORIZONS:
            column = "Fwd" + label
            values = []
            for _, group in merged.loc[merged["Year"].eq(year)].groupby("Date"):
                pair = group[[factor, column]].dropna()
                if len(pair) >= 8 and pair[factor].nunique() > 1:
                    values.append(pair[factor].corr(pair[column], method="spearman"))
            row[label] = float(np.mean(values)) if values else np.nan
            row[label + "_n"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def selection_vs_timing(scored: pd.DataFrame, targets: pd.DataFrame, factored: pd.DataFrame,
                        start: str, end: str, capital: float) -> pd.DataFrame:
    """What the clock cost. `actual` is the traded result. `hold_from_first`
    buys each selected name at its first selection and never sells. `hold_all`
    buys every name ever selected, at the window start."""
    wide = factored.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().loc[start:end]

    window = targets.loc[targets["Date"].between(start, end)]
    picked = window.loc[window["TargetWeight"] > 0]
    names = sorted(picked["Ticker"].astype(str).unique())
    first_pick = picked.groupby("Ticker")["Date"].min()

    rows = []
    market = prepare_market(factored, start=start, end=end)
    actual, _ = backtest_nav(monthly_weight_reset_targets(targets), factored, start, end, capital, market)
    rows.append({"variant": "actual (model entries and exits)", "names": len(names),
                 **{k: round(v, 2) for k, v in stats(actual).items()}})

    # hold every selected name from the window start, equal weight
    available = [t for t in names if t in wide.columns and wide[t].notna().any()]
    curve = wide[available].ffill()
    normed = curve / curve.iloc[0]
    rows.append({"variant": "hold_all (same names, bought day 1)", "names": len(available),
                 **{k: round(v, 2) for k, v in stats(capital * normed.mean(axis=1)).items()}})

    # buy each name at its first selection, hold to the end, equal weight
    legs = []
    for ticker in available:
        entry = pd.Timestamp(first_pick[ticker])
        series = wide[ticker].ffill().loc[entry:]
        if series.dropna().empty:
            continue
        leg = (series / series.iloc[0]).reindex(wide.index).fillna(1.0)
        legs.append(leg)
    if legs:
        rows.append({"variant": "hold_from_first (bought when first picked, never sold)",
                     "names": len(legs),
                     **{k: round(v, 2) for k, v in stats(capital * pd.concat(legs, axis=1).mean(axis=1)).items()}})
    return pd.DataFrame(rows)


def entry_timing(targets: pd.DataFrame, factored: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Per-holding: run-up bought into, return captured, return given up after."""
    wide = factored.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    window = targets.loc[targets["Date"].between(start, end)].copy()
    window["Date"] = pd.to_datetime(window["Date"])
    held = {}
    for date, group in window.groupby("Date", sort=True):
        names = set(group.loc[group["TargetWeight"] > 0, "Ticker"].astype(str))
        for ticker in names:
            held.setdefault(ticker, []).append(date)

    last = wide.index[wide.index <= pd.Timestamp(end)][-1]
    rows = []
    for ticker, dates in held.items():
        if ticker not in wide.columns:
            continue
        series = wide[ticker].ffill()
        entry, exit_ = dates[0], dates[-1]
        def px(d):
            sub = series.loc[:d].dropna()
            return float(sub.iloc[-1]) if len(sub) else np.nan
        before = series.loc[:entry].dropna()
        runup = (px(entry) / float(before.iloc[-64]) - 1.0) if len(before) > 64 else np.nan
        captured = px(exit_) / px(entry) - 1.0
        after = px(last) / px(exit_) - 1.0
        full = px(last) / px(entry) - 1.0
        rows.append({"Ticker": ticker, "FirstHeld": entry.date(), "LastHeld": exit_.date(),
                     "WeeksHeld": len(dates), "RunUp63dBeforeEntry": runup,
                     "ReturnWhileHeld": captured, "ReturnAfterLastHeld": after,
                     "ReturnIfNeverSold": full})
    return pd.DataFrame(rows).sort_values("ReturnWhileHeld")


def main():
    bundle = load_panel()
    panel, start, end = bundle["panel"], bundle["start"], bundle["end"]
    capital = panel.settings.initial_capital
    signal_days = frozen_signal_days(panel.factored, start, end, panel.settings.rebalance_weekday)
    scored, targets = generate_filing_v7_targets(signal_days, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])

    fwd = forward_returns(panel.factored)
    ic = factor_ic(scored, fwd)
    atomic_csv(OUT / "diag_factor_ic_by_year.csv", ic)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("=== 1. factor rank IC by horizon and year (mean of weekly cross-sectional Spearman) ===")
    pivot = ic.pivot(index="Factor", columns="Year", values=["1w", "1m", "3m", "6m"])
    print(pivot.round(3).to_string())

    print()
    print("=== 2. selection vs timing, 2026 (2026-01-02 .. %s) ===" % end)
    decomposition = selection_vs_timing(scored, targets, panel.factored, "2026-01-02", end, capital)
    atomic_csv(OUT / "diag_selection_vs_timing_2026.csv", decomposition)
    print(decomposition.to_string(index=False))

    print()
    print("=== same decomposition for 2025, as the control ===")
    d2025 = selection_vs_timing(scored, targets, panel.factored, "2025-01-02", "2025-12-31", capital)
    atomic_csv(OUT / "diag_selection_vs_timing_2025.csv", d2025)
    print(d2025.to_string(index=False))

    print()
    print("=== 3. per-holding entry timing, 2026 ===")
    timing = entry_timing(targets, panel.factored, "2026-01-02", end)
    atomic_csv(OUT / "diag_entry_timing_2026.csv", timing)
    print(timing.round(3).to_string(index=False))

    summary = {
        "median_runup_before_entry_2026": float(timing["RunUp63dBeforeEntry"].median()),
        "median_return_while_held_2026": float(timing["ReturnWhileHeld"].median()),
        "median_return_after_sold_2026": float(timing["ReturnAfterLastHeld"].median()),
        "median_return_if_never_sold_2026": float(timing["ReturnIfNeverSold"].median()),
    }
    atomic_json(OUT / "diag_2026_summary.json", summary)
    print()
    print({k: round(v, 4) for k, v in summary.items()})


if __name__ == "__main__":
    main()
