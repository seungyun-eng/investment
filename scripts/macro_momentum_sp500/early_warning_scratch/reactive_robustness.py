from __future__ import annotations
"""Do the reactive rules hold up outside 2020-2026? Test the SAME rules on SPY
across 33 years and split by decade. If they only work in one window they are
another selection artifact."""
import numpy as np, pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

config = load_research_config("config/macro_momentum_sp500/research.json")
feat = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
feat["Date"] = pd.to_datetime(feat["Date"])
spy = pd.to_numeric(feat["Close"], errors="coerce")
ret = spy.pct_change().fillna(0.0).to_numpy()
cash_daily = (pd.to_numeric(feat["CashRate"], errors="coerce").fillna(0)/100/252).to_numpy()
dates = feat["Date"]

def stats(r, d):
    c = (1+pd.Series(r)).cumprod()
    yrs = (d.iloc[-1]-d.iloc[0]).days/365.25
    sd = np.std(r)
    return {"CAGR_%": round((c.iloc[-1]**(1/yrs)-1)*100,2),
            "MDD_%": round((c/c.cummax()-1).min()*100,2),
            "Sharpe": round(np.mean(r)/sd*np.sqrt(252),3) if sd>0 else 0.0}

def apply(mask):
    m = np.roll(mask,1); m[0]=False
    return np.where(m, cash_daily, ret)

RULES = {}
for n in (100,150,200):
    ma = spy.rolling(n, min_periods=n).mean()
    RULES[f"SPY < {n}d MA"] = (spy < ma).fillna(False).to_numpy()
for n in (126,252):
    RULES[f"SPY {n}d momentum < 0"] = (spy.pct_change(n) < 0).fillna(False).to_numpy()
for x,y in ((0.15,0.07),(0.20,0.10)):
    peak = spy.cummax(); dd=(spy/peak-1).to_numpy()
    mask=np.zeros(len(spy),dtype=bool); out=False
    for i in range(len(spy)):
        if not out and dd[i]<=-x: out=True
        elif out and dd[i]>=-y: out=False
        mask[i]=out
    RULES[f"SPY stop -{int(x*100)}%/back -{int(y*100)}%"]=mask

PERIODS = {
    "FULL 1993-2026": slice(None),
    "1993-1999": (dates.dt.year>=1993)&(dates.dt.year<=1999),
    "2000-2009": (dates.dt.year>=2000)&(dates.dt.year<=2009),
    "2010-2019": (dates.dt.year>=2010)&(dates.dt.year<=2019),
    "2020-2026": (dates.dt.year>=2020),
}

print("Sharpe by period (baseline = buy & hold SPY)")
hdr = f"{'rule':32s}" + "".join(f"{p:>14s}" for p in PERIODS)
print(hdr); print("-"*len(hdr))
base_row = f"{'BUY & HOLD (baseline)':32s}"
for pname, pm in PERIODS.items():
    m = np.ones(len(spy),dtype=bool) if isinstance(pm,slice) else pm.to_numpy()
    base_row += f"{stats(ret[m], dates[m])['Sharpe']:>14.3f}"
print(base_row)
for rname, mask in RULES.items():
    r_all = apply(mask)
    row = f"{rname:32s}"
    for pname, pm in PERIODS.items():
        m = np.ones(len(spy),dtype=bool) if isinstance(pm,slice) else pm.to_numpy()
        row += f"{stats(r_all[m], dates[m])['Sharpe']:>14.3f}"
    print(row)

print()
print("FULL-period detail")
rows=[{"rule":"BUY & HOLD", **stats(ret, dates), "pct_cash":0.0}]
for rname, mask in RULES.items():
    m = np.roll(mask,1); m[0]=False
    rows.append({"rule":rname, **stats(apply(mask), dates), "pct_cash":round(m.mean()*100,1)})
print(pd.DataFrame(rows).to_string(index=False))
