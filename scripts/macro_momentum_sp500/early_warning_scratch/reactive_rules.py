from __future__ import annotations
"""READ-ONLY: reactive (not predictive) overlays on the frozen model.
These require no forecast -- they only react to a decline that has already
started. All rules are causal (trailing-only), so they are implementable live."""
import numpy as np, pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

SERIES = "V7_BASELINE_SAME_ENGINE"
eq = pd.read_csv("artifacts/pit_and_selection_fix/runs/B_PIT_STRICT/equity.csv")
eq = eq[eq.Series == SERIES].copy()
eq["Date"] = pd.to_datetime(eq["Date"]); eq = eq.sort_values("Date").reset_index(drop=True)
model_ret = eq["Equity"].pct_change().fillna(0.0).to_numpy()
equity = eq["Equity"]

config = load_research_config("config/macro_momentum_sp500/research.json")
feat = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
feat["Date"] = pd.to_datetime(feat["Date"])
mm = eq[["Date"]].merge(feat[["Date","CashRate","Close"]], on="Date", how="left").ffill()
cash_daily = (mm["CashRate"].fillna(0)/100/252).to_numpy()
spy = mm["Close"]

def stats(r):
    c = (1+pd.Series(r)).cumprod()
    yrs = (eq.Date.iloc[-1]-eq.Date.iloc[0]).days/365.25
    return {"CAGR_%": round((c.iloc[-1]**(1/yrs)-1)*100,2),
            "MDD_%": round((c/c.cummax()-1).min()*100,2),
            "Sharpe": round(np.mean(r)/np.std(r)*np.sqrt(252),3)}

def run(mask, label):
    # act on NEXT day (signal at close -> trade next session): no look-ahead
    m = np.roll(mask, 1); m[0] = False
    r = np.where(m, cash_daily, model_ret)
    return {"rule": label, **stats(r), "pct_cash": round(m.mean()*100,1)}

rows = [{"rule":"FROZEN MODEL AS-IS", **stats(model_ret), "pct_cash":0.0}]

# --- A. SPY below its N-day moving average
for n in (100, 150, 200):
    ma = spy.rolling(n, min_periods=n).mean()
    rows.append(run((spy < ma).fillna(False).to_numpy(), f"A. SPY < {n}d MA"))

# --- B. Model's own drawdown stop: exit at -X% from running peak,
#        re-enter once it recovers to within -Y% of the peak
for x, y in ((0.10,0.05),(0.15,0.07),(0.20,0.10),(0.25,0.10)):
    peak = equity.cummax()
    dd = (equity/peak - 1).to_numpy()
    mask = np.zeros(len(eq), dtype=bool); out = False
    for i in range(len(eq)):
        if not out and dd[i] <= -x: out = True
        elif out and dd[i] >= -y: out = False
        mask[i] = out
    rows.append(run(mask, f"B. model stop -{int(x*100)}% / back at -{int(y*100)}%"))

# --- C. SPY 12-month (252d) momentum negative
for n in (126, 252):
    mom = spy.pct_change(n)
    rows.append(run((mom < 0).fillna(False).to_numpy(), f"C. SPY {n}d momentum < 0"))

# --- D. combine: model stop AND macro breadth confirmation
CATS=("Labor","Rates","Inflation","CreditConditions","Volatility")
fl = pd.DataFrame({"Date":feat["Date"]})
fl["Labor"]=feat["InitialJoblessClaims_Change21_Z252"].ge(1.0)|feat["ContinuingJoblessClaims_Change21_Z252"].ge(1.0)
fl["Rates"]=feat["Treasury2Y_Change21_Z252"].ge(1.0)|feat["RealYield5Y_Change21_Z252"].ge(1.0)
fl["Inflation"]=feat["CoreCPI_Momentum3M_Z252"].ge(1.0)|feat["CorePCE_Momentum3M_Z252"].ge(1.0)
fl["CreditConditions"]=feat["HYYield_Change21_Z252"].ge(1.0)|feat["NFCI_Change21_Z252"].ge(1.0)
fl["Volatility"]=feat["VIX_Change21_Z252"].ge(1.0)
br = pd.Series(fl[list(CATS)].sum(axis=1).to_numpy(), index=feat["Date"])
br_al = eq["Date"].map(br).ffill().fillna(0).to_numpy()
peak = equity.cummax(); dd = (equity/peak-1).to_numpy()
mask = np.zeros(len(eq),dtype=bool); out=False
for i in range(len(eq)):
    if not out and dd[i] <= -0.15 and br_al[i] >= 2: out=True
    elif out and dd[i] >= -0.07: out=False
    mask[i]=out
rows.append(run(mask, "D. model stop -15% AND breadth>=2"))

print(pd.DataFrame(rows).to_string(index=False))
print()
print("Reference ceilings from the earlier hindsight test:")
print("  perfect timing      CAGR 123.79%  MDD -25.02%  Sharpe 2.600")
print("  hindsight 10d late  CAGR  74.68%  MDD -29.90%  Sharpe 1.764")
