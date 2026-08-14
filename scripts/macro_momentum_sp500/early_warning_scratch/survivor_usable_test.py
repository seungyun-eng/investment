from __future__ import annotations
"""Can the surviving feature be turned into a usable defensive rule, and does
it beat a matched random benchmark on the HOLDOUT period only?"""
import numpy as np
import pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.macro_momentum_sp500.portfolio import run_weight_backtest
from stock_research.paths import load_paths

DISCOVERY_END = pd.Timestamp("2017-12-31")
FEATURE = "ContinuingJoblessClaims_Change21_Z252"
EXIT_LAG = 21

config = load_research_config("config/macro_momentum_sp500/research.json")
features = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
dates = pd.to_datetime(features["Date"])
values = pd.to_numeric(features[FEATURE], errors="coerce")
near_peak = pd.to_numeric(features["Drawdown252"], errors="coerce").ge(-0.05)

# Threshold fixed from DISCOVERY only (90th percentile of discovery values).
disc_vals = values[dates.le(DISCOVERY_END) & values.notna()]
threshold = float(np.percentile(disc_vals, 90))
print(f"Threshold from discovery 90th pct: {threshold:.3f}")

alert = (values.ge(threshold) & near_peak).fillna(False)
defended = alert.astype(int).rolling(EXIT_LAG, min_periods=1).max().astype(bool)

def bt(weights_series, label, mask):
    w = features.loc[mask, ["Date", "Open", "Close", "CashRate"]].copy()
    w["TargetWeight"] = weights_series[mask].astype(float).to_numpy()
    w["SignalState"] = label
    r = run_weight_backtest(w.reset_index(drop=True), config, name=label)
    s = r.summary
    return {"label": label, "CAGR_pct": round(s.cagr_percent,2),
            "MDD_pct": round(s.max_drawdown_percent,2), "Sharpe": round(s.sharpe_ratio,3)}

hold_mask = dates.gt(DISCOVERY_END)
ones = pd.Series(1.0, index=features.index)
rows = [bt(ones, "Buy&Hold HOLDOUT(2018+)", hold_mask),
        bt((~defended).astype(float), "ClaimsRule HOLDOUT(2018+)", hold_mask)]
full_mask = pd.Series(True, index=features.index)
rows += [bt(ones, "Buy&Hold FULL(1993-2026)", full_mask),
         bt((~defended).astype(float), "ClaimsRule FULL(1993-2026)", full_mask)]
print(pd.DataFrame(rows).to_string(index=False))

pct_cash = defended[hold_mask].mean()*100
print(f"\nTime in cash on holdout: {pct_cash:.1f}%")

# --- Random benchmark: same number of defensive days, randomly placed in
# blocks on the holdout, 2000 trials. Does the real rule beat them?
print("\n=== Random matched-exposure benchmark on HOLDOUT (2000 trials) ===")
hold_idx = np.flatnonzero(hold_mask.to_numpy())
n_def = int(defended[hold_mask].sum())
block = EXIT_LAG
n_blocks = max(1, round(n_def / block))
rng = np.random.default_rng(7)
real = [r for r in rows if r["label"].startswith("ClaimsRule HOLDOUT")][0]
sharpes, cagrs, mdds = [], [], []
for _ in range(2000):
    starts = rng.choice(hold_idx, n_blocks, replace=True)
    mask = np.zeros(len(features), dtype=bool)
    for s in starts:
        mask[s:s+block] = True
    w = pd.Series((~mask).astype(float), index=features.index)
    res = bt(w, "rand", hold_mask)
    sharpes.append(res["Sharpe"]); cagrs.append(res["CAGR_pct"]); mdds.append(res["MDD_pct"])
sharpes, cagrs, mdds = np.array(sharpes), np.array(cagrs), np.array(mdds)
print(f"Real rule: Sharpe {real['Sharpe']}, CAGR {real['CAGR_pct']}%, MDD {real['MDD_pct']}%")
print(f"Random:    Sharpe median {np.median(sharpes):.3f}, CAGR median {np.median(cagrs):.2f}%, MDD median {np.median(mdds):.2f}%")
print(f"Percentile of real rule vs random --")
print(f"  Sharpe: {(sharpes < real['Sharpe']).mean()*100:.1f}th")
print(f"  CAGR:   {(cagrs < real['CAGR_pct']).mean()*100:.1f}th")
print(f"  MDD:    {(mdds < real['MDD_pct']).mean()*100:.1f}th (higher = better, less negative)")
