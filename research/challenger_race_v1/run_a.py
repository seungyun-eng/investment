"""Challenger A: volatility-targeted sizing, with NO macro overlay.

Control experiment for the V2 macro alert.  V2's whole documented edge sits in
high-volatility stress windows.  If a signal-free vol target reproduces most of
the benefit, V2 is not contributing market-timing information.

Selection is byte-identical to frozen V7.3.  Only sizing differs.
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

from common import OUT, OOS_START, atomic_csv, atomic_json, load_panel
from engine_bits import (V2_BASELINE, V2_TUNED, backtest_nav, buy_and_hold, qqq_close,
                         spy_close, stats, v2_overlay_nav)

VOL_LOOKBACK = 60
EXPOSURE_CAP = 1.0


def daily_returns(factored: pd.DataFrame) -> pd.DataFrame:
    wide = factored.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index().pct_change()


def basket_vol(rets: pd.DataFrame, weights: pd.Series, asof: pd.Timestamp) -> float:
    """Annualised trailing vol of the target basket, using only data <= asof."""
    window = rets.loc[rets.index <= asof].tail(VOL_LOOKBACK)
    cols = [t for t in weights.index if t in window.columns]
    if not cols or len(window) < VOL_LOOKBACK // 2:
        return float("nan")
    w = weights.reindex(cols).fillna(0.0)
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    port = (window[cols].fillna(0.0) * w).sum(axis=1)
    sd = float(port.std(ddof=1))
    return sd * np.sqrt(252) if sd > 0 else float("nan")


def scaled_targets(targets, rets, target_vol, deadband, inverse_vol=False):
    """Apply exposure scaling (and optionally inverse-vol name weights) to the
    weekly target stream, then keep only the dates that actually need a trade."""
    frame = targets.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    out, diag = [], []
    prev_selected = None
    prev_month = None
    prev_mult = None
    for date, group in frame.groupby("Date", sort=True):
        group = group.copy()
        held = group.loc[pd.to_numeric(group["TargetWeight"], errors="coerce").fillna(0).gt(0)]
        selected = set(held["Ticker"].astype(str))
        weights = pd.Series(held["TargetWeight"].to_numpy(dtype=float), index=held["Ticker"].astype(str))

        if inverse_vol and len(weights):
            window = rets.loc[rets.index <= date].tail(VOL_LOOKBACK)
            vols = {t: float(window[t].std(ddof=1)) for t in weights.index if t in window.columns}
            inv = pd.Series({t: (1.0 / v if v and v > 0 else np.nan) for t, v in vols.items()})
            if len(inv) == len(weights) and inv.notna().all():
                weights = inv / inv.sum() * float(weights.sum())

        vol = basket_vol(rets, weights, date) if len(weights) else float("nan")
        mult = 1.0 if (np.isnan(vol) or target_vol is None) else min(EXPOSURE_CAP, target_vol / vol)
        month = pd.Timestamp(date).to_period("M")

        if prev_selected is None:
            reason = "INITIAL_ALLOCATION"
        elif selected != prev_selected:
            reason = "MEMBERSHIP_CHANGE"
        elif month != prev_month:
            reason = "MONTHLY_WEIGHT_RESET"
        elif prev_mult is not None and deadband > 0 and abs(mult - prev_mult) >= deadband:
            reason = "EXPOSURE_CHANGE"
        elif prev_mult is not None and deadband == 0 and abs(mult - prev_mult) > 1e-9:
            reason = "EXPOSURE_CHANGE"
        else:
            reason = None

        diag.append({"Date": date, "BasketVol": vol, "Multiplier": mult,
                     "Executed": reason is not None, "Reason": reason})
        if reason is not None:
            scaled = group.copy()
            keys = group["Ticker"].astype(str)
            base = pd.Series(0.0, index=keys.to_numpy())
            for ticker, value in weights.items():
                base.loc[ticker] = value
            scaled["TargetWeight"] = base.reindex(keys).to_numpy() * mult
            scaled["ExecutionReason"] = reason
            out.append(scaled)
            prev_mult = mult
        prev_selected, prev_month = selected, month
    executed = pd.concat(out, ignore_index=True) if out else frame.iloc[0:0].copy()
    return executed, pd.DataFrame(diag)


def equal_weight_targets(scored):
    """'Just buy the whole list' control -- equal weight across every universe
    member with a live price, reset monthly."""
    rows = []
    for date, group in scored.groupby("Date", sort=True):
        members = group.loc[group["UniverseMember"].fillna(False), "Ticker"].astype(str).unique()
        if not len(members):
            continue
        rows.append(pd.DataFrame({"Date": date, "Ticker": members, "TargetWeight": 1.0 / len(members)}))
    frame = pd.concat(rows, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return monthly_weight_reset_targets(frame)


def sub_windows(nav):
    marks = {"2024": ("2024-01-02", "2024-12-31"),
             "2025": ("2025-01-02", "2025-12-31"),
             "2026_OOS": (OOS_START, str(nav.index[-1].date())),
             "tariff_2025Q2": ("2025-02-14", "2025-05-14")}
    result = {}
    for label, (a, b) in marks.items():
        piece = nav.loc[str(a):str(b)]
        if len(piece) > 2:
            result[label] = stats(piece)
    return result


def main():
    bundle = load_panel()
    panel, start, end = bundle["panel"], bundle["start"], bundle["end"]
    capital = panel.settings.initial_capital
    market = prepare_market(panel.factored, start=start, end=end)
    rets = daily_returns(panel.factored)

    signal_dates = pd.Index(panel.scored["Date"].unique())
    weekly = panel.factored.loc[panel.factored["Date"].isin(signal_dates)].copy()
    scored, targets = generate_filing_v7_targets(weekly, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])

    curves = {}
    rows = []

    base_nav, base_extra = backtest_nav(monthly_weight_reset_targets(targets), panel.factored,
                                        start, end, capital, market)
    curves["V7.3 (no overlay)"] = base_nav
    rows.append({"strategy": "V7.3 (no overlay)", **stats(base_nav), **base_extra})

    for label, params in (("V7.3 + V2 (original)", V2_BASELINE),
                          ("V7.3 + V2 (tuned)", V2_TUNED)):
        nav, info = v2_overlay_nav(base_nav, params, charge_cost=True)
        curves[label] = nav
        rows.append({"strategy": label, **stats(nav), **base_extra, **info})

    grid = [(tv, db, False) for tv in (0.15, 0.20, 0.25, 0.30) for db in (0.0, 0.10)]
    grid += [(None, 0.0, True), (0.20, 0.10, True), (0.25, 0.10, True)]
    diagnostics = {}
    for target_vol, deadband, inverse_vol in grid:
        if target_vol is None:
            name = "A invvol-only"
        else:
            name = "A voltgt%d%s db%d" % (int(target_vol * 100),
                                          "+invvol" if inverse_vol else "",
                                          int(deadband * 100))
        executed, diag = scaled_targets(targets, rets, target_vol, deadband, inverse_vol)
        nav, extra = backtest_nav(executed, panel.factored, start, end, capital, market)
        curves[name] = nav
        diagnostics[name] = diag
        rows.append({"strategy": name, **stats(nav), **extra,
                     "MeanExposure": float(diag["Multiplier"].mean()),
                     "MinExposure": float(diag["Multiplier"].min())})

    ew_nav, ew_extra = backtest_nav(equal_weight_targets(scored), panel.factored,
                                    start, end, capital, market)
    curves["CONTROL universe equal weight"] = ew_nav
    rows.append({"strategy": "CONTROL universe equal weight", **stats(ew_nav), **ew_extra})
    for label, close in (("CONTROL SPY buy&hold", spy_close()), ("CONTROL QQQ buy&hold", qqq_close())):
        nav = buy_and_hold(close, base_nav.index, capital)
        curves[label] = nav
        rows.append({"strategy": label, **stats(nav)})

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "a_results.csv", table)
    curve_frame = pd.DataFrame(curves)
    curve_frame.index.name = "Date"
    atomic_csv(OUT / "a_curves.csv", curve_frame.reset_index())
    for name, diag in diagnostics.items():
        safe = name.replace(" ", "_").replace("+", "_")
        atomic_csv(OUT / ("a_diag_%s.csv" % safe), diag)
    atomic_json(OUT / "a_subwindows.json", {name: sub_windows(nav) for name, nav in curves.items()})

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
