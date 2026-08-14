from __future__ import annotations
"""Rigorous holdout validation of the one feature that survived the
discovery->holdout sign test: ContinuingJoblessClaims_Change21_Z252.

The holdout (2018+) was never used for selection, so a block-bootstrap CI on
holdout AUC is a legitimate test. Also checks decile monotonicity and whether
the signal is LEADING or merely COINCIDENT with the drawdown already underway.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

DISCOVERY_END = pd.Timestamp("2017-12-31")
NEAR_PEAK, HORIZON, DRAWDOWN = -0.05, 126, -0.15
FEATURE = "ContinuingJoblessClaims_Change21_Z252"

config = load_research_config("config/macro_momentum_sp500/research.json")
features = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
dates = pd.to_datetime(features["Date"])
close = pd.to_numeric(features["Close"], errors="coerce")
fmin = pd.concat([close.shift(-o) for o in range(1, HORIZON + 1)], axis=1).min(axis=1)
outcome = (fmin / close - 1).le(DRAWDOWN)
near_peak = pd.to_numeric(features["Drawdown252"], errors="coerce").ge(NEAR_PEAK)
values = pd.to_numeric(features[FEATURE], errors="coerce")
usable = near_peak & fmin.notna() & values.notna()
hold = usable & dates.gt(DISCOVERY_END)

def block_boot_auc(y, x, block=63, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); x = np.asarray(x); size = len(y)
    nb = int(np.ceil(size / block)); out = []
    for _ in range(n):
        starts = rng.integers(0, size, nb)
        idx = np.concatenate([np.arange(s, s + block) % size for s in starts])[:size]
        if len(np.unique(y[idx])) < 2: continue
        out.append(roc_auc_score(y[idx], x[idx]))
    return np.array(out)

y = outcome[hold].astype(int).to_numpy(); x = values[hold].to_numpy()
auc = roc_auc_score(y, x)
boot = block_boot_auc(y, x)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"=== HOLDOUT (2018+) validation of {FEATURE} ===")
print(f"n={len(y)}, positives={int(y.sum())}, base rate={y.mean()*100:.1f}%")
print(f"AUC = {auc:.3f}   block-bootstrap 95% CI = [{lo:.3f}, {hi:.3f}]")
print(f"CI excludes 0.5: {not (lo <= 0.5 <= hi)}")
print(f"P(bootstrap AUC <= 0.5) = {(boot <= 0.5).mean():.3f}")

print("\n=== Decile monotonicity on HOLDOUT ===")
df = pd.DataFrame({"x": x, "y": y})
df["decile"] = pd.qcut(df["x"], 10, labels=False, duplicates="drop")
tab = df.groupby("decile").agg(n=("y", "size"), rate=("y", lambda s: s.mean()*100)).round(1)
print(tab.to_string())
rates = tab["rate"].to_numpy()
print(f"Top decile {rates[-1]:.1f}% vs bottom decile {rates[0]:.1f}%")
print(f"Spearman rank corr (decile vs rate): {pd.Series(rates).corr(pd.Series(range(len(rates))), method='spearman'):.3f}")

print("\n=== LEADING or COINCIDENT? ===")
print("If the signal only fires once the drawdown is already underway it is not")
print("an early warning. Near-peak filter already excludes deep drawdowns, but")
print("check current drawdown at high-signal times:")
dd = pd.to_numeric(features["Drawdown252"], errors="coerce")[hold].to_numpy()
high = x >= np.percentile(x, 90)
print(f"  Mean current drawdown when signal in top decile: {dd[high].mean()*100:.2f}%")
print(f"  Mean current drawdown otherwise:                 {dd[~high].mean()*100:.2f}%")

print("\n=== Where does it fire on holdout? (top-decile date ranges) ===")
d = dates[hold].to_numpy()[high]
s = pd.Series(pd.to_datetime(d))
groups = (s.diff().dt.days.fillna(999) > 30).cumsum()
for _, g in s.groupby(groups):
    print(f"  {g.iloc[0].date()} .. {g.iloc[-1].date()}  ({len(g)} days)")
