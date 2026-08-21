"""The same race, widened to 2008-2026.

Data ceiling found first: `SEC Filings/*/filings` only starts in 2019 and
`Financial_Data_real/` is empty, so the filing factor (25% of V7.3) and the
PIT-reconstructed growth/quality factors cannot exist before 2019.  What CAN
run back to 2008 is the price side of the same engine -- momentum, trend,
risk control and the V7.3 MA/MACD/OBV technical slot -- with the fundamental
weights contributing zero.  Call it V7-price.

READ THIS BEFORE READING ANY NUMBER: the universe is today's 31-name dashboard
list run backwards.  Nobody knew in 2008 that NVDA/AVGO/TSLA belonged on it.
Absolute returns here are fiction.  The only readable quantities are the
*differences* between contenders, which all share the identical biased universe:
V7-price vs universe-equal-weight (is selection adding value?), defense on vs
off, and B vs the champion.

Research only.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.portfolio import prepare_market
from stock_research.cross_sectional.signals import (score_panel, generate_rebalance_targets,
                                                    monthly_weight_reset_targets)
from stock_research.cross_sectional.v7_technical import (TECHNICAL_VARIANTS,
                                                         add_v7_technical_factors,
                                                         add_v7_technical_observations,
                                                         scoring_panel_for_variant)
from stock_research.dashboard import engine
from stock_research.dashboard.price_panel import DashboardMember, build_price_only_panel
from stock_research.paths import load_paths

from common import OUT, ROOT, atomic_csv, atomic_json, frozen_signal_days
from engine_bits import (V2_BASELINE, backtest_nav, buy_and_hold, qqq_close, spy_close,
                         stats, v2_weights)
from run_a import basket_vol, daily_returns

START = "2008-01-02"
END = "2026-08-12"
TOP_K = 5
UNIVERSE = ROOT / "config/cross_sectional/dashboard_universe.json"
PANEL_CACHE = OUT / "long_panel.pkl"

WINDOWS = {
    "gfc_2008_2009": ("2008-01-02", "2009-12-31"),
    "recovery_2010_2014": ("2010-01-04", "2014-12-31"),
    "2015_2019": ("2015-01-02", "2019-12-31"),
    "covid_2020": ("2020-01-02", "2020-12-31"),
    "bear_2022": ("2022-01-03", "2022-12-30"),
    "2023_2025": ("2023-01-03", "2025-12-31"),
    "2026_oos": ("2026-01-02", END),
}


def build_long_panel(force: bool = False) -> pd.DataFrame:
    if PANEL_CACHE.exists() and not force:
        return pd.read_pickle(PANEL_CACHE)
    paths = load_paths()
    config = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    members = [DashboardMember(ticker=row["ticker"], company=row["company"],
                               price_path=paths.stock_root / row["price_path"])
               for row in config["tickers"]
               if (paths.stock_root / row["price_path"]).exists()]
    print("members with price files:", len(members))
    settings = engine._settings(START, "2019-12-31", "2020-01-02", END, 8)
    warmup = str((pd.Timestamp(START) - pd.Timedelta(days=engine.WARMUP_BUFFER_DAYS)).date())
    panel = build_price_only_panel(members, settings, warmup_start=warmup)
    panel = add_v7_technical_observations(panel)
    panel = add_v7_technical_factors(panel, settings)
    variant = next(v for v in TECHNICAL_VARIANTS if v.name == engine.TECHNICAL_VARIANT_NAME)
    panel = scoring_panel_for_variant(panel, variant)
    panel["UniverseMember"] = True
    panel.to_pickle(PANEL_CACHE)
    return panel


def v7_price_targets(panel: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V7.3 selection with the fundamental factors absent (they score 0 for
    every name, so the ranking is the pure price side of the same engine)."""
    params = replace(engine._base_params(), top_k=TOP_K, exit_rank=TOP_K + 4,
                     profit_rotation_exit_rank=TOP_K + 4, conviction_exit_rank=TOP_K + 7)
    signal_days = frozen_signal_days(panel, START, END, settings.rebalance_weekday)
    scored = score_panel(signal_days, params)
    targets = generate_rebalance_targets(scored, params)
    targets["Date"] = pd.to_datetime(targets["Date"])
    return scored, targets


def equal_weight_targets(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in scored.groupby("Date", sort=True):
        members = group.loc[group["Eligible"].fillna(False), "Ticker"].astype(str).unique()
        if not len(members):
            continue
        rows.append(pd.DataFrame({"Date": date, "Ticker": members, "TargetWeight": 1.0 / len(members)}))
    frame = pd.concat(rows, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return monthly_weight_reset_targets(frame)


def dip_targets(scored: pd.DataFrame, rets: pd.DataFrame, *, top_n=5, min_hold_weeks=4) -> pd.DataFrame:
    """Challenger B, price-only translation: among names still in an uptrend
    AND stronger than the median on 126-day return (the price stand-in for the
    quality+growth gate), buy the ones that just fell hardest over 5 sessions."""
    prices = (1.0 + rets.fillna(0.0)).cumprod()
    short = (prices / prices.shift(5) - 1.0).stack().rename("ReturnShort").reset_index()
    short.columns = ["Date", "Ticker", "ReturnShort"]
    short["Date"] = pd.to_datetime(short["Date"])
    frame = scored.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.merge(short, on=["Date", "Ticker"], how="left")

    rows, held = [], {}
    for date, group in frame.groupby("Date", sort=True):
        pool = group.loc[group["Eligible"].fillna(False) & group["ReturnShort"].notna()].copy()
        if len(pool):
            pool = pool.loc[pool["Trend200"].fillna(-1.0) > 0]
        if len(pool):
            pool = pool.loc[pool["Return126"] >= pool["Return126"].median()]
        ranked = pool.sort_values("ReturnShort")["Ticker"].astype(str).tolist()
        live = set(group["Ticker"].astype(str))
        keep = [t for t, age in held.items() if age < min_hold_weeks and t in live][:top_n]
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


def defended(nav: pd.Series, mode: str, target_vol: float = 0.20) -> pd.Series:
    """Apply a defense to a finished NAV curve. 10bp charged on every weight
    change, same as the challenger pays inside the execution engine."""
    close = spy_close()
    v2 = pd.Series(v2_weights(close.to_numpy(), V2_BASELINE), index=close.index).reindex(nav.index).ffill().fillna(1.0)
    own = nav.pct_change()
    realised = own.rolling(60).std(ddof=1) * np.sqrt(252)
    vt = (target_vol / realised).clip(upper=1.0).shift(1).ffill().fillna(1.0)
    weights = {"v2": v2, "voltgt": vt, "min": pd.concat([v2, vt], axis=1).min(axis=1)}[mode]
    ret = weights.shift(1).fillna(1.0) * own.fillna(0.0)
    ret = ret - weights.diff().abs().fillna(0.0) * (10.0 / 10_000)
    return float(nav.iloc[0]) * (1 + ret).cumprod()


def window_table(curves: dict) -> pd.DataFrame:
    rows = []
    for name, nav in curves.items():
        row = {"strategy": name, **{k: round(v, 2) for k, v in stats(nav).items()}}
        for label, (a, b) in WINDOWS.items():
            piece = nav.loc[a:b]
            if len(piece) > 30:
                row[label] = round(stats(piece)["ROI"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    settings = engine._settings(START, "2019-12-31", "2020-01-02", END, 8)
    panel = build_long_panel()
    print("panel:", panel["Date"].min().date(), "->", panel["Date"].max().date(),
          panel["Ticker"].nunique(), "tickers")

    scored, targets = v7_price_targets(panel, settings)
    market = prepare_market(panel, start=START, end=END)
    capital = engine.INITIAL_CAPITAL
    rets = daily_returns(panel)

    curves, rows = {}, []
    base_nav, base_extra = backtest_nav(monthly_weight_reset_targets(targets), panel,
                                        START, END, capital, market)
    curves["V7-price top5"] = base_nav
    rows.append({"strategy": "V7-price top5", **base_extra})

    for mode, label in (("v2", "V7-price + V2"), ("voltgt", "V7-price + voltgt20"),
                        ("min", "V7-price + min(V2, voltgt20)")):
        curves[label] = defended(base_nav, mode)

    dip_nav, dip_extra = backtest_nav(dip_targets(scored, rets), panel, START, END, capital, market)
    curves["B dip sleeve (price-only gates)"] = dip_nav
    rows.append({"strategy": "B dip sleeve (price-only gates)", **dip_extra})
    curves["B + min(V2, voltgt20)"] = defended(dip_nav, "min")

    ew_nav, ew_extra = backtest_nav(equal_weight_targets(scored), panel, START, END, capital, market)
    curves["CONTROL universe equal weight"] = ew_nav
    rows.append({"strategy": "CONTROL universe equal weight", **ew_extra})
    for label, close in (("CONTROL SPY buy&hold", spy_close()), ("CONTROL QQQ buy&hold", qqq_close())):
        curves[label] = buy_and_hold(close, base_nav.index, capital)

    table = window_table(curves)
    extras = pd.DataFrame(rows)
    table = table.merge(extras, on="strategy", how="left")
    atomic_csv(OUT / "long_results.csv", table)
    frame = pd.DataFrame(curves)
    frame.index.name = "Date"
    atomic_csv(OUT / "long_curves.csv", frame.reset_index())

    weekly = frame.resample("W-FRI").last().pct_change().dropna()
    atomic_csv(OUT / "long_correlation.csv", weekly.corr().reset_index())
    atomic_json(OUT / "long_windows.json",
                {n: {l: stats(c.loc[a:b]) for l, (a, b) in WINDOWS.items() if len(c.loc[a:b]) > 30}
                 for n, c in curves.items()})

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print()
    print(table.to_string(index=False))
    print()
    print("weekly return correlation vs V7-price top5")
    print(weekly.corr()["V7-price top5"].round(2).to_string())


if __name__ == "__main__":
    main()
