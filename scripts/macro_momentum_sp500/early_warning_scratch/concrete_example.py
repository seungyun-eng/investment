from __future__ import annotations
"""READ-ONLY: concrete won-denominated walkthrough of the risk-normalised idea."""
import numpy as np, pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

SERIES="V7_BASELINE_SAME_ENGINE"
eq=pd.read_csv("artifacts/pit_and_selection_fix/runs/B_PIT_STRICT/equity.csv")
eq=eq[eq.Series==SERIES].copy(); eq["Date"]=pd.to_datetime(eq["Date"])
eq=eq.sort_values("Date").reset_index(drop=True)
ret=eq["Equity"].pct_change().fillna(0.0).to_numpy()

config=load_research_config("config/macro_momentum_sp500/research.json")
feat=build_features(load_research_data(load_paths().macro,config),config).reset_index(drop=True)
feat["Date"]=pd.to_datetime(feat["Date"])
mm=eq[["Date"]].merge(feat[["Date","CashRate","Close"]],on="Date",how="left").ffill()
cash=(mm["CashRate"].fillna(0)/100/252).to_numpy(); spy=mm["Close"]
alert=np.roll((spy<spy.rolling(150,min_periods=150).mean()).fillna(False).to_numpy(),1); alert[0]=False
SPREAD=0.01/252
START=100_000_000  # 1억원

def path(w_def, k):
    base=np.where(alert, w_def*ret+(1-w_def)*cash, ret)
    r=k*base-np.where(k>1,(k-1)*(cash+SPREAD),0.0)
    return START*(1+pd.Series(r)).cumprod()

scen={
 "A. 아무것도 안 함":            path(1.0,1.00),
 "B. 절반 정리 (배율 없음)":      path(0.5,1.00),
 "C. 전부 정리 (배율 없음)":      path(0.0,1.00),
 "D. 전부 정리 + 1.44배":         path(0.0,1.44),
}
d=eq["Date"]
print("=== 1억원으로 시작했을 때 (2020-01 ~ 2026-07) ===")
for name,p in scen.items():
    dd=(p/p.cummax()-1)
    print(f"{name:26s} 최종 {p.iloc[-1]/1e8:6.2f}억   최저점 {p.min()/1e8:5.2f}억   최대낙폭 {dd.min()*100:6.2f}%")

# The big crash: model peak 2021-02-12 -> trough 2022-12-28
pk=d[d=="2021-02-12"].index[0]; tr=d[d=="2022-12-28"].index[0]
print(f"\n=== 최악의 구간만 확대: {d[pk].date()} (고점) -> {d[tr].date()} (저점) ===")
for name,p in scen.items():
    print(f"{name:26s} {p.iloc[pk]/1e8:6.2f}억 -> {p.iloc[tr]/1e8:5.2f}억  "
          f"({(p.iloc[tr]/p.iloc[pk]-1)*100:6.1f}%)  회복까지 필요한 상승 +{(p.iloc[pk]/p.iloc[tr]-1)*100:.0f}%")

print(f"\n=== 배율 1.44배가 실제로 어떤 상태인지 (D 시나리오) ===")
p=scen["D. 전부 정리 + 1.44배"]
print(f"자산 1억일 때 -> 실제 주식 보유 {1.44:.2f}억, 빌린 돈 {0.44:.2f}억")
print(f"이 경우 주식이 {(1/1.44)*100:.0f}% 넘게 빠지면 자본금 전액 소진 (이론상 -69%)")
print(f"실제로는 그 전에 증거금 부족으로 강제청산됨")
worst=(p/p.cummax()-1).min()
print(f"D 시나리오가 실제 겪은 최대낙폭: {worst*100:.2f}%  -> 청산선까지 여유 {(-69-worst*100):.1f}%p")
