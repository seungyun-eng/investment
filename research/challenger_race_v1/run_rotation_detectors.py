"""Four detectors for "should I slow the rotation down this week?"

The trailing-IC detector already failed its shuffle control (1.8th percentile --
worse than freezing at random).  Three more candidates are tested the same way,
because the question the rotation-speed result raises is worth more than one
attempt:

  ic_negative     trailing 13w realised rank IC < 0            (already failed)
  spy_bear        SPY below its 200-day moving average
  spy_highvol     SPY 60d realised vol above its own trailing 2y 75th pct
  low_dispersion  cross-sectional std of 21d returns below its trailing median
                  -- "there is nothing to choose between these names this week"

Every flag reads only data available at the signal date.  Each is scored
against 500 shuffles that freeze the SAME number of weeks at random dates: a
detector only counts if it lands in the top tail of its own shuffle.

Research only.  Nothing committed, nothing deployed.
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
from engine_bits import backtest_nav, spy_close, stats
from run_rotation_regime import freeze_flags, run_with_frozen, weekly_ic

SEED = 20260820
DRAWS = 500


def spy_flags(dates) -> tuple[dict, dict]:
    close = spy_close()
    ma200 = close.rolling(200).mean()
    bear = (close < ma200)
    vol = close.pct_change().rolling(60).std(ddof=1) * np.sqrt(252)
    threshold = vol.rolling(504).quantile(0.75)
    highvol = (vol > threshold)
    bear_map, vol_map = {}, {}
    for date in dates:
        stamp = pd.Timestamp(date)
        past = bear.loc[bear.index <= stamp]
        bear_map[date] = bool(past.iloc[-1]) if len(past) else False
        past_v = highvol.loc[highvol.index <= stamp]
        vol_map[date] = bool(past_v.iloc[-1]) if len(past_v) and not pd.isna(past_v.iloc[-1]) else False
    return bear_map, vol_map


def dispersion_flags(factored: pd.DataFrame, dates) -> dict:
    wide = factored.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    ret21 = wide / wide.shift(21) - 1.0
    dispersion = ret21.std(axis=1, ddof=1)
    trailing_median = dispersion.rolling(252, min_periods=126).median()
    flags = {}
    for date in dates:
        stamp = pd.Timestamp(date)
        past = dispersion.loc[dispersion.index <= stamp]
        med = trailing_median.loc[trailing_median.index <= stamp]
        flags[date] = bool(len(past) and len(med) and not pd.isna(med.iloc[-1])
                           and past.iloc[-1] < med.iloc[-1])
    return flags


def main():
    bundle = load_panel()
    panel, start, end = bundle["panel"], bundle["start"], bundle["end"]
    capital = panel.settings.initial_capital
    market = prepare_market(panel.factored, start=start, end=end)
    signal_days = frozen_signal_days(panel.factored, start, end, panel.settings.rebalance_weekday)
    scored, targets = generate_filing_v7_targets(signal_days, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])
    execution = monthly_weight_reset_targets(targets)
    exec_dates = sorted(execution["Date"].unique())

    ic = weekly_ic(scored, panel.factored)
    bear_map, vol_map = spy_flags(exec_dates)
    disp_map = dispersion_flags(panel.factored, exec_dates)

    detectors = {
        "ic_negative": {d: bool(v) for d, v in freeze_flags(ic, exec_dates).items()},
        "spy_bear": bear_map,
        "spy_highvol": vol_map,
        "low_dispersion": disp_map,
    }

    years = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"),
             "2026_OOS": ("2026-01-02", end)}
    year_markets = {k: prepare_market(panel.factored, start=a, end=b) for k, (a, b) in years.items()}

    base_nav, base_extra = run_with_frozen(execution, set(), panel, start, end, capital, market)
    rows = [{"detector": "baseline (never freeze)", "frozen_weeks": 0,
             **{k: round(v, 2) for k, v in stats(base_nav).items()},
             "turnover": round(base_extra["AnnualizedTurnover"], 1)}]
    for name, (a, b) in years.items():
        piece, _ = run_with_frozen(execution, set(), panel, a, b, capital, year_markets[name])
        rows[0][name] = round(stats(piece)["ROI"], 1)

    rng = np.random.default_rng(SEED)
    controls = {}
    for name, flag_map in detectors.items():
        frozen = {d for d, f in flag_map.items() if f}
        nav, extra = run_with_frozen(execution, frozen, panel, start, end, capital, market)
        row = {"detector": name, "frozen_weeks": len(frozen),
               **{k: round(v, 2) for k, v in stats(nav).items()},
               "turnover": round(extra["AnnualizedTurnover"], 1)}
        for label, (a, b) in years.items():
            piece, _ = run_with_frozen(execution, frozen, panel, a, b, capital, year_markets[label])
            row[label] = round(stats(piece)["ROI"], 1)
        rows.append(row)

        if not frozen:
            controls[name] = {"note": "detector never fired"}
            continue
        sharpes, oos = [], []
        for _ in range(DRAWS):
            draw = set(rng.choice(exec_dates, size=len(frozen), replace=False))
            shuffled, _ = run_with_frozen(execution, draw, panel, start, end, capital, market)
            sharpes.append(stats(shuffled)["Sharpe"])
            piece, _ = run_with_frozen(execution, draw, panel, "2026-01-02", end, capital,
                                       year_markets["2026_OOS"])
            oos.append(stats(piece)["ROI"])
        controls[name] = {
            "frozen_weeks": len(frozen),
            "real_Sharpe": row["Sharpe"],
            "shuffle_Sharpe_mean": round(float(np.mean(sharpes)), 3),
            "real_Sharpe_percentile": round(float((np.array(sharpes) < row["Sharpe"]).mean() * 100), 1),
            "real_2026_ROI": row["2026_OOS"],
            "shuffle_2026_mean": round(float(np.mean(oos)), 2),
            "real_2026_percentile": round(float((np.array(oos) < row["2026_OOS"]).mean() * 100), 1),
        }

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "diag_rotation_detectors.csv", table)
    atomic_json(OUT / "diag_rotation_detectors_control.json", controls)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    print()
    print("=== shuffle controls (%d draws each; a detector must sit in the TOP tail) ===" % DRAWS)
    print(pd.DataFrame(controls).T.to_string())


if __name__ == "__main__":
    main()
