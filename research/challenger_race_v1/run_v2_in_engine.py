"""V7.3 + V2 with the overlay INSIDE the execution engine.

Every earlier run in this study applied V2 to a finished NAV curve, which forced
a choice between two close-to-close conventions that differ by more than the
effect being measured (see the ERRATUM). This removes the choice: V2's exposure
becomes part of the target weights, so `run_portfolio_backtest` executes it at
the next session's OPEN and charges the same 10 bps it charges every other trade.

Timing, stated explicitly:
  `v2_weights` sets weight[i] from day i-1 data, so the decision is known at
  close[i-1]. A target row is therefore emitted on date i-1 carrying weight[i],
  and the engine fills it at open[i] -- identical to how a Friday-close V7.3
  signal fills at Monday's open.

A target row is emitted whenever the V7.3 selection executes OR V2's exposure
changes, so both kinds of trade are costed.

Research only.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.portfolio import prepare_market, run_portfolio_backtest
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import COST_BPS, OUT, atomic_csv, frozen_signal_days, load_panel
from engine_bits import V2_BASELINE, V2_TUNED, spy_close, stats, v2_weights

WINDOWS = {"2024-2026 (frozen window)": ("2024-01-02", "2026-08-12"),
           "2020-2026": ("2020-01-02", "2026-08-12")}
YEARS = [str(y) for y in range(2020, 2027)]


def v2_signal_weights(calendar: pd.DatetimeIndex, params: dict) -> pd.Series:
    """Exposure to be in force from the NEXT open, indexed by signal date."""
    close = spy_close()
    raw = pd.Series(v2_weights(close.to_numpy(), params), index=close.index)
    aligned = raw.reindex(calendar).ffill().fillna(1.0)
    # weight[i] is decided at close[i-1]; as a signal-date series that is w shifted back one.
    return aligned.shift(-1).ffill().fillna(1.0)


def overlay_targets(execution: pd.DataFrame, calendar: pd.DatetimeIndex,
                    params: dict | None) -> pd.DataFrame:
    """Scale the executed V7.3 target set by V2's exposure, and re-emit whenever
    either the selection or the exposure changes."""
    execution = execution.copy()
    execution["Date"] = pd.to_datetime(execution["Date"])
    by_date = {d: g for d, g in execution.groupby("Date", sort=True)}
    weights = (v2_signal_weights(calendar, params) if params
               else pd.Series(1.0, index=calendar))

    rows = []
    current: pd.DataFrame | None = None
    previous_exposure: float | None = None
    for date in calendar:
        selection_changed = date in by_date
        if selection_changed:
            current = by_date[date]
        if current is None:
            continue
        exposure = float(weights.get(date, 1.0))
        exposure_changed = previous_exposure is None or abs(exposure - previous_exposure) > 1e-9
        if not (selection_changed or exposure_changed):
            continue
        block = current.copy()
        block["Date"] = date
        block["TargetWeight"] = pd.to_numeric(block["TargetWeight"], errors="coerce").fillna(0.0) * exposure
        rows.append(block)
        previous_exposure = exposure
    return pd.concat(rows, ignore_index=True) if rows else execution.iloc[0:0]


def run(panel, start, end):
    capital = panel.settings.initial_capital
    market = prepare_market(panel.factored, start=start, end=end)
    calendar = market.dates
    signal_days = frozen_signal_days(panel.factored, start, end, panel.settings.rebalance_weekday)
    _, targets = generate_filing_v7_targets(signal_days, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])
    execution = monthly_weight_reset_targets(targets)

    out = []
    for label, params in (("V7.3 alone", None),
                          ("V7.3 + V2 (original params)", V2_BASELINE),
                          ("V7.3 + V2 (tuned params)", V2_TUNED)):
        combined = overlay_targets(execution, calendar, params)
        result = run_portfolio_backtest({}, combined, start=start, end=end,
                                        initial_capital=capital,
                                        transaction_cost_bps=COST_BPS,
                                        prepared_market=market)
        daily = result.daily.copy()
        daily["Date"] = pd.to_datetime(daily["Date"])
        nav = daily.set_index("Date")["Equity"]
        row = {"strategy": label, **{k: round(v, 2) for k, v in stats(nav).items()},
               "turnover": round(result.summary.annualized_turnover, 1),
               "rebalances": result.summary.rebalance_count}
        for year in YEARS:
            piece = nav.loc["%s-01-01" % year:"%s-12-31" % year]
            if len(piece) > 30:
                row[year] = round(stats(piece)["ROI"], 1)
        out.append(row)
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    panels = {"2024-2026 (frozen window)": load_panel()["panel"],
              "2020-2026": pickle.load(open(OUT / "panel_2020_2026.pkl", "rb"))}
    frames = []
    for label, (start, end) in WINDOWS.items():
        table = run(panels[label], start, end)
        table.insert(0, "window", label)
        frames.append(table)
        print("=== %s — V2 executed at next open, 10 bps on every trade ===" % label)
        print(table.drop(columns=["window"]).to_string(index=False))
        print()
    atomic_csv(OUT / "v2_in_engine_results.csv", pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    main()
