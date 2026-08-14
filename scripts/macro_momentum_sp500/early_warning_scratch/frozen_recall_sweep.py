from __future__ import annotations
"""READ-ONLY: does a HIGHER-RECALL macro rule help the frozen model, even
though it hurt plain SPY? The frozen model falls much harder (-64% vs -34%),
so the payoff from avoiding drawdowns is larger here."""
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

config = load_research_config("config/macro_momentum_sp500/research.json")
feat = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
feat["Date"] = pd.to_datetime(feat["Date"])

CATS = ("Labor","Rates","Inflation","CreditConditions","Volatility")
fl = pd.DataFrame({"Date": feat["Date"]})
fl["Labor"] = feat["InitialJoblessClaims_Change21_Z252"].ge(1.0)|feat["ContinuingJoblessClaims_Change21_Z252"].ge(1.0)
fl["Rates"] = feat["Treasury2Y_Change21_Z252"].ge(1.0)|feat["RealYield5Y_Change21_Z252"].ge(1.0)
fl["Inflation"] = feat["CoreCPI_Momentum3M_Z252"].ge(1.0)|feat["CorePCE_Momentum3M_Z252"].ge(1.0)
fl["CreditConditions"] = feat["HYYield_Change21_Z252"].ge(1.0)|feat["NFCI_Change21_Z252"].ge(1.0)
fl["Volatility"] = feat["VIX_Change21_Z252"].ge(1.0)
fl["Breadth"] = fl[list(CATS)].sum(axis=1)

m = eq[["Date"]].merge(feat[["Date","CashRate"]], on="Date", how="left").ffill()
cash_daily = (m["CashRate"].fillna(0)/100/252).to_numpy()

def stats(r):
    c = (1+pd.Series(r)).cumprod()
    yrs = (eq.Date.iloc[-1]-eq.Date.iloc[0]).days/365.25
    return {"CAGR_%": round((c.iloc[-1]**(1/yrs)-1)*100,2),
            "MDD_%": round((c/c.cummax()-1).min()*100,2),
            "Sharpe": round(np.mean(r)/np.std(r)*np.sqrt(252),3)}

RULES = {
 "B2_4of10  (recall~57%)": (2,4,10), "B2_10of20 (recall~50%)": (2,10,20),
 "B2_7of10  (recall~43%)": (2,7,10), "B2_16of20 (recall~32%)": (2,16,20),
 "B3_8of20  (recall~21%)": (3,8,20), "B3_12of20 (recall~13%)": (3,12,20),
 "B4_10of20 (recall~7%)":  (4,10,20),"B4_14of20 (recall~5%)":  (4,14,20),
}
EXIT_LAG = 21
rows=[{"rule":"FROZEN MODEL AS-IS", **stats(model_ret), "pct_cash":0.0}]
for label,(mb,days,win) in RULES.items():
    alert = fl["Breadth"].ge(mb).astype(int).rolling(win,min_periods=win).sum().ge(days).fillna(False)
    defended = alert.astype(int).rolling(EXIT_LAG,min_periods=1).max().astype(bool)
    dmap = pd.Series(defended.to_numpy(), index=feat["Date"])
    mask = eq["Date"].map(dmap).fillna(False).to_numpy().astype(bool)
    r = np.where(mask, cash_daily, model_ret)
    rows.append({"rule":label, **stats(r), "pct_cash": round(mask.mean()*100,1)})
print(pd.DataFrame(rows).to_string(index=False))
