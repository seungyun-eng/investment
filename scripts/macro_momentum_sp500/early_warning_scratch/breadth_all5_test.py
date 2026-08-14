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

RULES = {
    "B2_15_20": {"min_breadth": 2, "days": 15, "window": 20},
    "B3_15_20": {"min_breadth": 3, "days": 15, "window": 20},
    "B4_15_20": {"min_breadth": 4, "days": 15, "window": 20},
    "B5_10_20": {"min_breadth": 5, "days": 10, "window": 20},
    "B5_15_20": {"min_breadth": 5, "days": 15, "window": 20},
    "B5_5_20":  {"min_breadth": 5, "days": 5,  "window": 20},
}

def build(features):
    frame = pd.DataFrame({"Date": features["Date"], "Close": features["Close"]})
    frame["Labor"] = features["InitialJoblessClaims_Change21_Z252"].ge(1.0) | features["ContinuingJoblessClaims_Change21_Z252"].ge(1.0)
    frame["Rates"] = features["Treasury2Y_Change21_Z252"].ge(1.0) | features["RealYield5Y_Change21_Z252"].ge(1.0)
    frame["Inflation"] = features["CoreCPI_Momentum3M_Z252"].ge(1.0) | features["CorePCE_Momentum3M_Z252"].ge(1.0)
    frame["CreditConditions"] = features["HYYield_Change21_Z252"].ge(1.0) | features["NFCI_Change21_Z252"].ge(1.0)
    frame["Volatility"] = features["VIX_Change21_Z252"].ge(1.0)
    frame["Breadth"] = frame[list(CATEGORIES)].sum(axis=1)
    for name, r in RULES.items():
        stressed = frame["Breadth"].ge(r["min_breadth"]).astype(int)
        frame[name] = stressed.rolling(r["window"], min_periods=r["window"]).sum().ge(r["days"])
    close = pd.to_numeric(frame["Close"], errors="coerce")
    fmin = pd.concat([close.shift(-o) for o in range(1, FORWARD_SESSIONS+1)], axis=1).min(axis=1)
    frame["ForwardDrawdown"] = fmin/close - 1
    frame["LargeDrawdownAhead"] = frame["ForwardDrawdown"].le(DRAWDOWN_THRESHOLD)
    return frame

def weekly(frame):
    w = frame.assign(Week=frame["Date"].dt.to_period("W-FRI")).groupby("Week", sort=True, as_index=False).tail(1).drop(columns="Week")
    return w.loc[w["Date"].dt.year.ge(START_YEAR) & w["ForwardDrawdown"].notna()].reset_index(drop=True)

def metrics(frame, rule):
    s = frame[rule].fillna(False).astype(bool)
    a = frame["LargeDrawdownAhead"].astype(bool)
    tp=int((s&a).sum()); fp=int((s&~a).sum()); fn=int((~s&a).sum()); tn=int((~s&~a).sum())
    prec = tp/(tp+fp) if tp+fp else None
    rec = tp/(tp+fn) if tp+fn else None
    fpr = fp/(fp+tn) if fp+tn else None
    return {"signals": int(s.sum()), "tp": tp, "fp": fp,
            "precision_percent": None if prec is None else round(prec*100,1),
            "recall_percent": None if rec is None else round(rec*100,1),
            "fp_rate_percent": None if fpr is None else round(fpr*100,1)}

def episodes(daily, rule):
    s = daily[rule].fillna(False).astype(bool)
    onset = s & ~s.shift(1, fill_value=False)
    rows=[]
    for i in daily.index[onset]:
        row = daily.loc[i]
        rows.append({"date": str(row["Date"].date()), "fwd63d_pct": None if pd.isna(row["ForwardDrawdown"]) else round(float(row["ForwardDrawdown"]*100),1), "hit": bool(row["LargeDrawdownAhead"])})
    return rows

paths = load_paths()
config = load_research_config("config/macro_momentum_sp500/research.json")
features = build_features(load_research_data(paths.macro, config), config)
daily = build(features)
wk = weekly(daily)

print("=== Aggregate 2007-present ===")
for rule in RULES:
    print(f"{rule:12s} {json.dumps(metrics(wk, rule))}")

print()
print("=== Yearly recall, crisis years ===")
for year in [2007,2008,2011,2015,2018,2022]:
    yf = wk.loc[wk["Date"].dt.year.eq(year)]
    if yf.empty: continue
    base = yf["LargeDrawdownAhead"].mean()*100
    line = f"{year}(base={base:.1f}%): "
    for rule in ["B2_15_20","B4_15_20","B5_10_20","B5_15_20","B5_5_20"]:
        m = metrics(yf, rule)
        line += f"{rule}=R{m['recall_percent']} "
    print(line)

print()
print("=== FP counts, quiet years ===")
for year in [2009,2010,2012,2013,2014,2016,2017,2019,2023,2024]:
    yf = wk.loc[wk["Date"].dt.year.eq(year)]
    if yf.empty: continue
    line = f"{year}: "
    for rule in ["B2_15_20","B4_15_20","B5_10_20","B5_15_20"]:
        line += f"{rule}={int(yf[rule].fillna(False).sum())} "
    print(line)

print()
print("=== Episodes: B5_10_20 (breadth=5, 10 of 20 days) ===")
print(json.dumps(episodes(daily, "B5_10_20"), indent=2))
print()
print("=== Episodes: B5_5_20 (breadth=5, just 5 of 20 days) ===")
print(json.dumps(episodes(daily, "B5_5_20"), indent=2))
