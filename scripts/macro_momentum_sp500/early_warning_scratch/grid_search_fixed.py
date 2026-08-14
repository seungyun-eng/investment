from __future__ import annotations
import json
import pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

START_YEAR = 2007
FORWARD_SESSIONS = 63
DRAWDOWN_THRESHOLD = -0.10
CATEGORIES = ("Labor", "Rates", "Inflation", "CreditConditions", "Volatility")

# Bounded grid, declared before inspection. Windows are TRADING DAYS.
# Since long lead time is not required, test shorter (faster-reacting)
# windows too, not just 20-day ones.
GRID = []
for window in (5, 10, 15, 20):
    for min_breadth in (2, 3, 4):
        for frac in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            days = round(window * frac)
            if days < 1 or days > window:
                continue
            GRID.append((min_breadth, days, window))
GRID = sorted(set(GRID))

def build(features):
    frame = pd.DataFrame({"Date": features["Date"], "Close": features["Close"]})
    frame["Labor"] = features["InitialJoblessClaims_Change21_Z252"].ge(1.0) | features["ContinuingJoblessClaims_Change21_Z252"].ge(1.0)
    frame["Rates"] = features["Treasury2Y_Change21_Z252"].ge(1.0) | features["RealYield5Y_Change21_Z252"].ge(1.0)
    frame["Inflation"] = features["CoreCPI_Momentum3M_Z252"].ge(1.0) | features["CorePCE_Momentum3M_Z252"].ge(1.0)
    frame["CreditConditions"] = features["HYYield_Change21_Z252"].ge(1.0) | features["NFCI_Change21_Z252"].ge(1.0)
    frame["Volatility"] = features["VIX_Change21_Z252"].ge(1.0)
    frame["Breadth"] = frame[list(CATEGORIES)].sum(axis=1)
    for min_breadth, days, window in GRID:
        stressed = frame["Breadth"].ge(min_breadth).astype(int)
        frame[f"R_{min_breadth}_{days}_{window}"] = stressed.rolling(window, min_periods=window).sum().ge(days)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    fmin = pd.concat([close.shift(-o) for o in range(1, FORWARD_SESSIONS+1)], axis=1).min(axis=1)
    frame["ForwardDrawdown"] = fmin/close - 1
    frame["LargeDrawdownAhead"] = frame["ForwardDrawdown"].le(DRAWDOWN_THRESHOLD)
    return frame

def weekly(frame):
    w = frame.assign(Week=frame["Date"].dt.to_period("W-FRI")).groupby("Week", sort=True, as_index=False).tail(1).drop(columns="Week")
    return w.loc[w["Date"].dt.year.ge(START_YEAR) & w["ForwardDrawdown"].notna()].reset_index(drop=True)

def metrics(signal, actual):
    s = signal.fillna(False).astype(bool)
    a = actual.astype(bool)
    tp=int((s&a).sum()); fp=int((s&~a).sum()); fn=int((~s&a).sum()); tn=int((~s&~a).sum())
    prec = tp/(tp+fp) if tp+fp else None
    rec = tp/(tp+fn) if tp+fn else None
    fpr = fp/(fp+tn) if fp+tn else None
    return prec, rec, fpr, int(s.sum())

paths = load_paths()
config = load_research_config("config/macro_momentum_sp500/research.json")
features = build_features(load_research_data(paths.macro, config), config)
daily = build(features)
wk = weekly(daily)

rows = []
for min_breadth, days, window in GRID:
    col = f"R_{min_breadth}_{days}_{window}"
    prec, rec, fpr, n = metrics(wk[col], wk["LargeDrawdownAhead"])
    if rec is None or n == 0:
        continue
    rows.append({"min_breadth": min_breadth, "days": days, "window": window,
                 "signals": n, "precision": round(prec*100,1), "recall": round(rec*100,1), "fp_rate": round(fpr*100,1)})

results = pd.DataFrame(rows)
pareto = []
best_fpr_so_far = 200.0
for _, row in results.sort_values("recall", ascending=False).iterrows():
    if row["fp_rate"] < best_fpr_so_far:
        pareto.append(row)
        best_fpr_so_far = row["fp_rate"]
pareto_df = pd.DataFrame(pareto).sort_values("recall", ascending=False)

print("=== Pareto frontier (daily-correct rolling window) ===")
with pd.option_context('display.width', 160, 'display.max_rows', 100):
    print(pareto_df.to_string(index=False))

print()
print("=== B2_15_20 baseline for reference ===")
print(results[(results.min_breadth==2)&(results.days==15)&(results.window==20)].to_string(index=False))

# Save daily+weekly frames for follow-up per-year inspection of Pareto candidates
import pathlib
_here = pathlib.Path(__file__).parent
daily.to_pickle(_here / "daily_grid.pkl")
wk.to_pickle(_here / "weekly_grid.pkl")
