"""Challenger B: buy the dip in strong names, instead of chasing strength.

V7.3 is a pure buy-strength engine.  B keeps the same universe, the same weekly
clock, the same 10bp cost and the same next-open execution, but inverts the
entry: among names that are still in an uptrend and score above the median on
quality+growth, buy the ones that just sold off hardest.

The point is not only "does it win" but "is it uncorrelated" -- a low-correlation
second engine is worth more than a slightly better copy of the first one.

Research only: writes nothing outside research/challenger_race_v1/output.
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

from common import OUT, OOS_START, atomic_csv, atomic_json, frozen_signal_days, load_panel
from engine_bits import (V2_BASELINE, V2_TUNED, backtest_nav, buy_and_hold, qqq_close,
                         spy_close, stats, v2_overlay_nav)
from run_a import daily_returns, sub_windows

DIP_LOOKBACK = 5


def dip_panel(scored: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Attach the short-horizon dip measure to every signal-date row."""
    prices = (1.0 + rets.fillna(0.0)).cumprod()
    short = prices / prices.shift(DIP_LOOKBACK) - 1.0
    stacked = short.stack().rename("ReturnShort").reset_index()
    stacked.columns = ["Date", "Ticker", "ReturnShort"]
    frame = scored.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    stacked["Date"] = pd.to_datetime(stacked["Date"])
    return frame.merge(stacked, on=["Date", "Ticker"], how="left")


def dip_targets(panel: pd.DataFrame, *, top_n: int, dip_column: str, quality_gate: bool,
                trend_gate: bool, min_hold_weeks: int) -> pd.DataFrame:
    """Weekly equal-weight targets for the dip sleeve.

    min_hold_weeks keeps a name for at least that many signal dates before it
    can be replaced, which is the only turnover control applied here."""
    rows = []
    held: dict[str, int] = {}
    for date, group in panel.groupby("Date", sort=True):
        pool = group.loc[group["UniverseMember"].fillna(False) & group["Eligible"].fillna(False)].copy()
        pool = pool.loc[pool[dip_column].notna()]
        if quality_gate and len(pool):
            strength = pool["QualityFactor"].fillna(0.0) + pool["GrowthFactor"].fillna(0.0)
            pool = pool.loc[strength >= strength.median()]
        if trend_gate and len(pool):
            pool = pool.loc[pool["Trend200"].fillna(-1.0) > 0]
        ranked = pool.sort_values(dip_column, ascending=True)["Ticker"].astype(str).tolist()

        # names still inside their minimum hold keep their slot
        keep = [t for t, age in held.items() if age < min_hold_weeks and t in set(group["Ticker"].astype(str))]
        keep = keep[:top_n]
        picks = list(keep)
        for ticker in ranked:
            if len(picks) >= top_n:
                break
            if ticker not in picks:
                picks.append(ticker)

        held = {t: (held.get(t, 0) + 1 if t in held else 1) for t in picks}
        if not picks:
            continue
        weight = 1.0 / len(picks)
        tickers = group["Ticker"].astype(str).to_numpy()
        rows.append(pd.DataFrame({"Date": date, "Ticker": tickers,
                                  "TargetWeight": [weight if t in set(picks) else 0.0 for t in tickers]}))
    frame = pd.concat(rows, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return monthly_weight_reset_targets(frame)


def blend(a: pd.Series, b: pd.Series, share: float = 0.5) -> pd.Series:
    """Daily-rebalanced blend of two NAV curves (blend-level trading cost is
    not modelled, so read this as an upper bound on the blend)."""
    idx = a.index.intersection(b.index)
    ret = share * a.loc[idx].pct_change().fillna(0.0) + (1 - share) * b.loc[idx].pct_change().fillna(0.0)
    return float(a.iloc[0]) * (1 + ret).cumprod()


def main():
    bundle = load_panel()
    panel_obj, start, end = bundle["panel"], bundle["start"], bundle["end"]
    capital = panel_obj.settings.initial_capital
    market = prepare_market(panel_obj.factored, start=start, end=end)
    rets = daily_returns(panel_obj.factored)

    signal_days = frozen_signal_days(panel_obj.factored, start, end, panel_obj.settings.rebalance_weekday)
    scored, targets = generate_filing_v7_targets(signal_days, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])

    champion_base, base_extra = backtest_nav(monthly_weight_reset_targets(targets), panel_obj.factored,
                                             start, end, capital, market)
    champion, _ = v2_overlay_nav(champion_base, V2_TUNED, charge_cost=True)
    champion_orig, _ = v2_overlay_nav(champion_base, V2_BASELINE, charge_cost=True)

    dips = dip_panel(scored, rets)

    variants = {
        "B1 dip5 + quality + trend, n5": dict(top_n=5, dip_column="ReturnShort", quality_gate=True,
                                              trend_gate=True, min_hold_weeks=1),
        "B2 RSI dip + quality + trend, n5": dict(top_n=5, dip_column="RSI14", quality_gate=True,
                                                 trend_gate=True, min_hold_weeks=1),
        "B3 dip5 + quality + trend, n3": dict(top_n=3, dip_column="ReturnShort", quality_gate=True,
                                              trend_gate=True, min_hold_weeks=1),
        "B4 dip5 + quality + trend, n5, hold4w": dict(top_n=5, dip_column="ReturnShort", quality_gate=True,
                                                      trend_gate=True, min_hold_weeks=4),
        "B5 dip5 only (no gates), n5": dict(top_n=5, dip_column="ReturnShort", quality_gate=False,
                                            trend_gate=False, min_hold_weeks=1),
        "B6 dip5 + trend only, n5": dict(top_n=5, dip_column="ReturnShort", quality_gate=False,
                                         trend_gate=True, min_hold_weeks=1),
    }

    curves = {"V7.3 (no overlay)": champion_base,
              "CHAMPION V7.3+V2 tuned": champion,
              "CHAMPION V7.3+V2 original": champion_orig}
    rows = [{"strategy": "V7.3 (no overlay)", **stats(champion_base), **base_extra},
            {"strategy": "CHAMPION V7.3+V2 tuned", **stats(champion), **base_extra},
            {"strategy": "CHAMPION V7.3+V2 original", **stats(champion_orig), **base_extra}]

    for name, kwargs in variants.items():
        nav, extra = backtest_nav(dip_targets(dips, **kwargs), panel_obj.factored,
                                  start, end, capital, market)
        curves[name] = nav
        weekly_b = nav.resample("W-FRI").last().pct_change().dropna()
        weekly_c = champion.resample("W-FRI").last().pct_change().dropna()
        idx = weekly_b.index.intersection(weekly_c.index)
        rows.append({"strategy": name, **stats(nav), **extra,
                     "CorrWithChampion": float(weekly_b.loc[idx].corr(weekly_c.loc[idx]))})

    best_b = max(variants, key=lambda n: stats(curves[n])["Sharpe"])
    curves["BLEND 50/50 champion + %s" % best_b] = blend(champion, curves[best_b])
    rows.append({"strategy": "BLEND 50/50 champion + %s" % best_b,
                 **stats(curves["BLEND 50/50 champion + %s" % best_b])})

    ew = None
    for label, close in (("CONTROL SPY buy&hold", spy_close()), ("CONTROL QQQ buy&hold", qqq_close())):
        nav = buy_and_hold(close, champion_base.index, capital)
        curves[label] = nav
        rows.append({"strategy": label, **stats(nav)})

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "b_results.csv", table)
    frame = pd.DataFrame(curves)
    frame.index.name = "Date"
    atomic_csv(OUT / "b_curves.csv", frame.reset_index())
    atomic_json(OUT / "b_subwindows.json", {n: sub_windows(c) for n, c in curves.items()})

    weekly = frame.resample("W-FRI").last().pct_change().dropna()
    atomic_csv(OUT / "b_correlation.csv", weekly.corr().reset_index())

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(table.round(2).to_string(index=False))
    print()
    print("weekly return correlation")
    print(weekly.corr().round(2).to_string())


if __name__ == "__main__":
    main()
