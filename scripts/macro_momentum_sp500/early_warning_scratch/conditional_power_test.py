from __future__ import annotations
"""Is there ANY real predictive signal, tested with adequate statistical power?

Reframes the question away from "7 crisis events" (hopelessly underpowered)
to "among near-peak days, does macro stress predict a subsequent large
drawdown?" -- which has thousands of observations.

Discovery = pre-2018 (feature ranking happens here only).
Holdout   = 2018+ (never consulted during ranking).
Block bootstrap accounts for the heavy autocorrelation of daily macro series.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

DISCOVERY_END = pd.Timestamp("2017-12-31")
NEAR_PEAK = -0.05        # within 5% of trailing 252d high
HORIZON = 126            # ~6 months forward
DRAWDOWN = -0.15

config = load_research_config("config/macro_momentum_sp500/research.json")
features = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
dates = pd.to_datetime(features["Date"])
close = pd.to_numeric(features["Close"], errors="coerce")

fmin = pd.concat([close.shift(-o) for o in range(1, HORIZON + 1)], axis=1).min(axis=1)
outcome = (fmin / close - 1).le(DRAWDOWN)

near_peak = pd.to_numeric(features["Drawdown252"], errors="coerce").ge(NEAR_PEAK)
usable = near_peak & fmin.notna()

CANDIDATES = [c for c in features.columns if any(
    k in c for k in ("Z252", "Score", "Breadth", "Momentum_", "TermStructure", "RSI")
) and features[c].notna().sum() > 2000]

disc_mask = usable & dates.le(DISCOVERY_END)
hold_mask = usable & dates.gt(DISCOVERY_END)

print(f"Near-peak usable days -- discovery: {int(disc_mask.sum())}, holdout: {int(hold_mask.sum())}")
print(f"Base rate (15%+ drawdown within {HORIZON}d) -- discovery: {outcome[disc_mask].mean()*100:.1f}%, "
      f"holdout: {outcome[hold_mask].mean()*100:.1f}%")
print(f"Candidate features tested: {len(CANDIDATES)}")

def block_bootstrap_auc_ci(y, x, block=63, n=500, seed=0):
    """Percentile CI for AUC using circular block bootstrap (handles autocorr)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); x = np.asarray(x)
    size = len(y)
    n_blocks = int(np.ceil(size / block))
    out = []
    for _ in range(n):
        starts = rng.integers(0, size, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % size for s in starts])[:size]
        yb, xb = y[idx], x[idx]
        if len(np.unique(yb)) < 2:
            continue
        out.append(roc_auc_score(yb, xb))
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

rows = []
for col in CANDIDATES:
    values = pd.to_numeric(features[col], errors="coerce")
    m = disc_mask & values.notna()
    y = outcome[m].astype(int); x = values[m]
    if y.nunique() < 2 or len(y) < 500:
        continue
    auc = roc_auc_score(y, x)
    rows.append({"feature": col, "discovery_auc": auc, "n": int(len(y))})

disc = pd.DataFrame(rows)
disc["edge"] = (disc["discovery_auc"] - 0.5).abs()
disc = disc.sort_values("edge", ascending=False).reset_index(drop=True)

print("\n=== Top 12 by |AUC-0.5| on DISCOVERY (pre-2018) ===")
print(disc.head(12).to_string(index=False))

print("\n=== Block-bootstrap 95% CI for top 6 (discovery) ===")
top = disc.head(6)
ci_rows = []
for _, r in top.iterrows():
    values = pd.to_numeric(features[r["feature"]], errors="coerce")
    m = disc_mask & values.notna()
    lo, hi = block_bootstrap_auc_ci(outcome[m].astype(int), values[m])
    ci_rows.append({"feature": r["feature"], "discovery_auc": round(r["discovery_auc"], 3),
                    "ci_low": None if lo is None else round(lo, 3),
                    "ci_high": None if hi is None else round(hi, 3),
                    "excludes_0.5": None if lo is None else not (lo <= 0.5 <= hi)})
print(pd.DataFrame(ci_rows).to_string(index=False))

print("\n=== HOLDOUT (2018+) performance of those same top features ===")
hold_rows = []
for _, r in top.iterrows():
    values = pd.to_numeric(features[r["feature"]], errors="coerce")
    m = hold_mask & values.notna()
    y = outcome[m].astype(int); x = values[m]
    if y.nunique() < 2:
        hold_rows.append({"feature": r["feature"], "holdout_auc": None, "n": int(len(y))})
        continue
    hold_rows.append({"feature": r["feature"], "discovery_auc": round(r["discovery_auc"], 3),
                      "holdout_auc": round(roc_auc_score(y, x), 3), "n": int(len(y))})
print(pd.DataFrame(hold_rows).to_string(index=False))

print("\n=== Multiple-comparison context ===")
print(f"Tested {len(disc)} features. Under pure noise, expected max |AUC-0.5| from that many")
print("correlated tests is substantial -- treat any single top result with suspicion.")
