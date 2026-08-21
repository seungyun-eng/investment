"""Fair defense race on the 2008-2026 book: V2 vs vol target, matched exposure.

Two corrections to run_long.py:

1.  A 20% vol target on a book whose own volatility averages ~30% is not a
    defense, it is permanent de-risking -- it loses to V2 for a trivial reason.
    Here the target is swept so both mechanisms can be compared at the same
    average exposure.
2.  The engine reports turnover as dollars / *initial* capital, which explodes
    over an 18-year run where NAV grows 100x.  Turnover here is dollars traded
    divided by average equity.

Matched-exposure control: every rule is compared against simply holding its own
mean exposure flat.  Anything a rule earns above that line is what its *timing*
is worth; anything below it means the rule would be better off not timing.

Research only.
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

from common import OUT, atomic_csv
from engine_bits import V2_BASELINE, backtest_nav, stats, v2_weights
from run_long import END, START, WINDOWS, build_long_panel, v7_price_targets

COST = 10.0 / 10_000


def apply_exposure(nav: pd.Series, weights: pd.Series) -> pd.Series:
    """Every weight series passed here is ALREADY lagged by construction --
    v2_weights reads day i-1 and sets weight[i], and the vol-target helpers
    shift their raw ratio before the deadband loop. Shifting again would delay
    both the de-risk and the re-entry by an extra session."""
    own = nav.pct_change().fillna(0.0)
    ret = weights * own - weights.diff().abs().fillna(0.0) * COST
    return float(nav.iloc[0]) * (1 + ret).cumprod()


def vol_target_exposure(nav: pd.Series, target: float, deadband: float = 0.10) -> pd.Series:
    realised = nav.pct_change().rolling(60).std(ddof=1) * np.sqrt(252)
    raw = (target / realised).clip(upper=1.0).shift(1).to_numpy()
    out, current = np.ones(len(nav)), 1.0
    for i, proposed in enumerate(raw):
        if not np.isnan(proposed) and abs(proposed - current) >= deadband:
            current = float(proposed)
        out[i] = current
    return pd.Series(out, index=nav.index)


def main():
    settings = engine._settings(START, "2019-12-31", "2020-01-02", END, 8)
    panel = build_long_panel()
    scored, targets = v7_price_targets(panel, settings)
    market = prepare_market(panel, start=START, end=END)
    nav, _ = backtest_nav(monthly_weight_reset_targets(targets), panel, START, END,
                          engine.INITIAL_CAPITAL, market)

    own_vol = float(nav.pct_change().std(ddof=1) * np.sqrt(252)) * 100
    print("book realised volatility over the full run: %.1f%%" % own_vol)

    from engine_bits import spy_close
    close = spy_close()
    v2 = pd.Series(v2_weights(close.to_numpy(), V2_BASELINE), index=close.index)
    v2 = v2.reindex(nav.index).ffill().fillna(1.0)

    rules = {"V2 (MA150 alert)": v2}
    for target in (0.20, 0.25, 0.30, 0.35, 0.40):
        rules["voltgt%d" % int(target * 100)] = vol_target_exposure(nav, target)
    rules["min(V2, voltgt35)"] = pd.concat([v2, vol_target_exposure(nav, 0.35)], axis=1).min(axis=1)

    rows = [{"rule": "no defense", "MeanExposure": 1.0, **{k: round(v, 3) for k, v in stats(nav).items()}}]
    for name, weights in rules.items():
        defended = apply_exposure(nav, weights)
        flat = apply_exposure(nav, pd.Series(float(weights.mean()), index=nav.index))
        s, f = stats(defended), stats(flat)
        row = {"rule": name, "MeanExposure": round(float(weights.mean()), 3),
               **{k: round(v, 3) for k, v in s.items()},
               "flat_Sharpe": round(f["Sharpe"], 3), "flat_Calmar": round(f["Calmar"], 3),
               "Sharpe_gain_vs_flat": round(s["Sharpe"] - f["Sharpe"], 3),
               "Calmar_gain_vs_flat": round(s["Calmar"] - f["Calmar"], 3)}
        for label, (a, b) in WINDOWS.items():
            piece = defended.loc[a:b]
            if len(piece) > 30:
                row[label] = round(stats(piece)["ROI"], 1)
        rows.append(row)

    for label, (a, b) in WINDOWS.items():
        piece = nav.loc[a:b]
        if len(piece) > 30:
            rows[0][label] = round(stats(piece)["ROI"], 1)

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "long_defense_results.csv", table)
    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 40)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
