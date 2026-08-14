from __future__ import annotations
"""READ-ONLY research calculation on the frozen model's equity curve.
(a) Partial de-risking instead of all-or-nothing.
(b) Risk-normalised comparison: if the rule frees up drawdown budget, what
    does the return look like at the ORIGINAL risk level?
Leverage here is an analytical device for equal-risk comparison, not advice."""
import numpy as np, pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

SERIES = "V7_BASELINE_SAME_ENGINE"
eq = pd.read_csv("artifacts/pit_and_selection_fix/runs/B_PIT_STRICT/equity.csv")
eq = eq[eq.Series==SERIES].copy(); eq["Date"]=pd.to_datetime(eq["Date"])
eq = eq.sort_values("Date").reset_index(drop=True)
model_ret = eq["Equity"].pct_change().fillna(0.0).to_numpy()

config = load_research_config("config/macro_momentum_sp500/research.json")
feat = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
feat["Date"]=pd.to_datetime(feat["Date"])
mm = eq[["Date"]].merge(feat[["Date","CashRate","Close"]], on="Date", how="left").ffill()
cash_daily = (mm["CashRate"].fillna(0)/100/252).to_numpy()
spy = mm["Close"]
ma150 = spy.rolling(150, min_periods=150).mean()
raw_alert = (spy < ma150).fillna(False).to_numpy()
alert = np.roll(raw_alert,1); alert[0]=False   # trade next session, no look-ahead

def stats(r):
    c=(1+pd.Series(r)).cumprod()
    yrs=(eq.Date.iloc[-1]-eq.Date.iloc[0]).days/365.25
    return {"CAGR_%":round((c.iloc[-1]**(1/yrs)-1)*100,2),
            "MDD_%":round((c/c.cummax()-1).min()*100,2),
            "Sharpe":round(np.mean(r)/np.std(r)*np.sqrt(252),3)}

print("=== (a) PARTIAL de-risking: how much to cut when the rule fires ===")
rows=[{"defensive_weight":"100% invested (do nothing)", **stats(model_ret)}]
for w in (0.75,0.50,0.25,0.0):
    r = np.where(alert, w*model_ret + (1-w)*cash_daily, model_ret)
    rows.append({"defensive_weight":f"cut to {int(w*100)}% during alert", **stats(r)})
print(pd.DataFrame(rows).to_string(index=False))

print()
print("=== (b) RISK-NORMALISED: scale exposure so MDD matches the original -64.33% ===")
print("    (borrowing charged at cash rate + 1.0% spread on the leveraged part)")
SPREAD = 0.01/252
base_mdd = stats(model_ret)["MDD_%"]
out=[]
for w_def in (0.0, 0.25, 0.50):
    # find leverage k such that MDD ~= base_mdd
    lo, hi = 1.0, 3.0
    for _ in range(40):
        k = (lo+hi)/2
        base = np.where(alert, w_def*model_ret + (1-w_def)*cash_daily, model_ret)
        borrow = np.where(k>1, (k-1)*(cash_daily+SPREAD), 0.0)
        r = k*base - borrow
        if stats(r)["MDD_%"] < base_mdd:   # more negative = too risky
            hi = k
        else:
            lo = k
    k = (lo+hi)/2
    base = np.where(alert, w_def*model_ret + (1-w_def)*cash_daily, model_ret)
    r = k*base - np.where(k>1,(k-1)*(cash_daily+SPREAD),0.0)
    out.append({"rule":f"150d MA, cut to {int(w_def*100)}%", "leverage":round(k,2), **stats(r)})
print(pd.DataFrame(
    [{"rule":"do nothing (baseline)","leverage":1.00, **stats(model_ret)}] + out
).to_string(index=False))

print()
print("=== recovery math ===")
for d in (-0.6433, -0.4865, -0.2751):
    print(f"  drawdown {d*100:6.2f}%  ->  needs +{(1/(1+d)-1)*100:6.1f}% to get back to even")
