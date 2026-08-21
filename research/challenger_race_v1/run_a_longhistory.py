"""Challenger A, tested where it can actually be judged: SPY, 1995-2026.

The 2024-2026 registry window contains no sustained bear market, so it cannot
separate "the MA150 alert carries market-timing information" from "any rule
that shrinks in high volatility would have done the same".  This strips both
mechanisms off the stock book and races them on SPY alone across every regime
in the price file, including 2000-02, 2008-09, 2020 and 2022.

Same 10bp cost charged on every weight change, for both mechanisms.
Research only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import OUT, atomic_csv, atomic_json
from engine_bits import V2_BASELINE, V2_TUNED, spy_close, stats, v2_weights

COST = 10.0 / 10_000
VOL_LOOKBACK = 60
WINDOWS = {
    "full": ("1995-01-01", "2026-08-12"),
    "dotcom_2000_2002": ("2000-01-01", "2002-12-31"),
    "gfc_2007_2009": ("2007-01-01", "2009-12-31"),
    "covid_2020": ("2020-01-01", "2020-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "registry_2024_2026": ("2024-01-02", "2026-08-12"),
}


def vol_target_weights(close: pd.Series, target_vol: float, deadband: float) -> pd.Series:
    """Exposure = min(1, target_vol / trailing 60d realised vol), decided on
    t-1 data and applied from t, same one-day lag the V2 rule uses."""
    ret = close.pct_change()
    realised = ret.rolling(VOL_LOOKBACK).std(ddof=1) * np.sqrt(252)
    raw = (target_vol / realised).clip(upper=1.0)
    raw = raw.shift(1)
    out = np.ones(len(close))
    current = 1.0
    values = raw.to_numpy()
    for i in range(len(close)):
        proposed = values[i]
        if not np.isnan(proposed) and abs(proposed - current) >= deadband:
            current = float(proposed)
        out[i] = current
    return pd.Series(out, index=close.index)


def apply_weights(close: pd.Series, weights: pd.Series) -> pd.Series:
    """Both weight sources are already lagged -- v2_weights reads day i-1 and
    sets weight[i], and vol_target_weights shifts its raw ratio before the
    deadband loop -- so no further shift belongs here."""
    ret = close.pct_change().fillna(0.0) * weights
    ret = ret - weights.diff().abs().fillna(0.0) * COST
    return 100_000.0 * (1 + ret).cumprod()


def main():
    close = spy_close()
    # spy_close() starts at 2017 for the overlay work; reload the full file here.
    from engine_bits import SPY_CSV, SPY_TAIL
    spy = pd.read_csv(SPY_CSV, parse_dates=["Date"]).sort_values("Date")
    tail = pd.read_csv(SPY_TAIL, parse_dates=["Date"]).sort_values("Date")
    cutoff = spy["Date"].iloc[-1]
    anchor = tail.loc[tail["Date"] == cutoff, "Close"]
    extra = tail.loc[tail["Date"] > cutoff, ["Date", "Close"]].copy()
    if not anchor.empty and not extra.empty:
        scale = spy["Adj Close"].iloc[-1] / float(anchor.iloc[0])
        extra["Adj Close"] = extra["Close"] * scale
        spy = pd.concat([spy[["Date", "Adj Close"]], extra[["Date", "Adj Close"]]],
                        ignore_index=True).sort_values("Date").reset_index(drop=True)
    close = spy.set_index("Date")["Adj Close"]
    print("SPY history:", close.index[0].date(), "->", close.index[-1].date(), len(close), "sessions")

    strategies = {"buy&hold": pd.Series(1.0, index=close.index),
                  "V2 (original params)": pd.Series(v2_weights(close.to_numpy(), V2_BASELINE), index=close.index),
                  "V2 (tuned on 2024-26)": pd.Series(v2_weights(close.to_numpy(), V2_TUNED), index=close.index)}
    for target_vol in (0.10, 0.12, 0.15, 0.20):
        strategies["A voltgt%d" % int(target_vol * 100)] = vol_target_weights(close, target_vol, 0.10)
    # combine: take the smaller of the two exposures
    strategies["V2 + voltgt15 (min of both)"] = pd.concat(
        [strategies["V2 (original params)"], strategies["A voltgt15"]], axis=1).min(axis=1)

    rows, curves, per_window = [], {}, {}
    for name, weights in strategies.items():
        nav = apply_weights(close, weights)
        curves[name] = nav
        row = {"strategy": name, "MeanExposure": float(weights.mean())}
        for label, (a, b) in WINDOWS.items():
            piece = nav.loc[a:b]
            if len(piece) < 30:
                continue
            piece = piece / piece.iloc[0] * 100_000.0
            s = stats(piece)
            if label == "full":
                row.update({"CAGR": s["CAGR"], "MDD": s["MDD"], "Sharpe": s["Sharpe"], "Calmar": s["Calmar"]})
            row[label + "_ROI"] = s["ROI"]
            row[label + "_MDD"] = s["MDD"]
            per_window.setdefault(name, {})[label] = s
        rows.append(row)

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "a_longhistory_results.csv", table)
    frame = pd.DataFrame(curves)
    frame.index.name = "Date"
    atomic_csv(OUT / "a_longhistory_curves.csv", frame.reset_index())
    atomic_json(OUT / "a_longhistory_windows.json", per_window)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    show = ["strategy", "MeanExposure", "CAGR", "MDD", "Sharpe", "Calmar"] + \
           [c for c in table.columns if c.endswith("_ROI") or (c.endswith("_MDD") and c != "MDD")]
    print(table[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
