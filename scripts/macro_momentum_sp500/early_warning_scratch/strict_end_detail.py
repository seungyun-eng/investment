from __future__ import annotations
import json
import pandas as pd

import pathlib
_here = pathlib.Path(__file__).parent
daily = pd.read_pickle(_here / "daily_grid.pkl")
wk = pd.read_pickle(_here / "weekly_grid.pkl")

CANDIDATES = {
    "B4_10_20": "R_4_10_20",   # precision 52.4%, FP 1.2%, recall 7.2%, 21 signals
    "B4_14_20": "R_4_14_20",   # precision 61.5%, FP 0.6%, recall 5.3%, 13 signals
    "B4_10_15": "R_4_10_15",   # precision 52.9%, FP 0.9%, recall 5.9%, 17 signals
    "B4_10_10": "R_4_10_10",   # precision 66.7%, FP 0.3%, recall 3.9%, 9 signals
}

def metrics(signal, actual):
    s = signal.fillna(False).astype(bool)
    a = actual.astype(bool)
    tp=int((s&a).sum()); fp=int((s&~a).sum()); fn=int((~s&a).sum()); tn=int((~s&~a).sum())
    prec = tp/(tp+fp) if tp+fp else None
    rec = tp/(tp+fn) if tp+fn else None
    fpr = fp/(fp+tn) if fp+tn else None
    return prec, rec, fpr, int(s.sum())

print("=== Yearly recall for strict candidates, crisis years ===")
for year in [2007,2008,2011,2015,2018,2020,2022,2025]:
    yf = wk.loc[wk["Date"].dt.year.eq(year)]
    if yf.empty:
        continue
    base = yf["LargeDrawdownAhead"].mean()*100
    line = f"{year}(base={base:.1f}%): "
    for name, col in CANDIDATES.items():
        prec, rec, fpr, n = metrics(yf[col], yf["LargeDrawdownAhead"])
        line += f"{name}=R{None if rec is None else round(rec*100,1)} "
    print(line)

print()
print("=== FP counts in quiet years ===")
for year in [2009,2010,2012,2013,2014,2016,2017,2019,2023,2024]:
    yf = wk.loc[wk["Date"].dt.year.eq(year)]
    if yf.empty:
        continue
    line = f"{year}: "
    for name, col in CANDIDATES.items():
        line += f"{name}={int(yf[col].fillna(False).sum())} "
    print(line)

print()
print("=== Onset episodes per candidate (which specific dates fired, hit or not) ===")
for name, col in CANDIDATES.items():
    s = daily[col].fillna(False).astype(bool)
    onset = s & ~s.shift(1, fill_value=False)
    rows = []
    for i in daily.index[onset]:
        row = daily.loc[i]
        rows.append({
            "date": str(row["Date"].date()),
            "fwd63d_pct": None if pd.isna(row["ForwardDrawdown"]) else round(float(row["ForwardDrawdown"]*100),1),
            "hit": bool(row["LargeDrawdownAhead"]),
        })
    print(f"--- {name} ---")
    print(json.dumps(rows, indent=2))
