from __future__ import annotations
"""READ-ONLY: show the ACTUAL trade dates the 150d-MA rule would have produced."""
import numpy as np, pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

SERIES="V7_BASELINE_SAME_ENGINE"
eq=pd.read_csv("artifacts/pit_and_selection_fix/runs/B_PIT_STRICT/equity.csv")
eq=eq[eq.Series==SERIES].copy(); eq["Date"]=pd.to_datetime(eq["Date"])
eq=eq.sort_values("Date").reset_index(drop=True)
config=load_research_config("config/macro_momentum_sp500/research.json")
feat=build_features(load_research_data(load_paths().macro,config),config).reset_index(drop=True)
feat["Date"]=pd.to_datetime(feat["Date"])
mm=eq[["Date"]].merge(feat[["Date","Close"]],on="Date",how="left").ffill()
spy=mm["Close"]; ma=spy.rolling(150,min_periods=150).mean()
raw=(spy<ma).fillna(False).to_numpy()
sig=np.roll(raw,1); sig[0]=False          # trade next session
d=eq["Date"]; model=eq["Equity"]

# model peak/trough of the big crash for reference
pk=d[d=="2021-02-12"].index[0]; tr=d[d=="2022-12-28"].index[0]

print("=== 150일선 규칙이 실제로 낸 매매 신호 (2020-2026) ===")
print(f"{'매도일':12s} {'매수일':12s} {'현금보유':>8s}  {'그동안 모델이 움직인 폭':>22s}")
i=0; trades=[]
while i < len(sig):
    if sig[i]:
        s=i
        while i<len(sig) and sig[i]: i+=1
        e=min(i, len(sig)-1)
        move=(model.iloc[e]/model.iloc[s]-1)*100
        trades.append((d[s], d[e], e-s, move))
        print(f"{str(d[s].date()):12s} {str(d[e].date()):12s} {e-s:6d}일  {move:>20.1f}%")
    else: i+=1
print(f"\n총 {len(trades)}번 매매, 현금 보유 총 {sum(t[2] for t in trades)}일 "
      f"(전체의 {sum(t[2] for t in trades)/len(sig)*100:.1f}%)")

print(f"\n=== 2021-2022 대폭락 때 타이밍이 얼마나 늦었나 ===")
print(f"모델 실제 고점 : {d[pk].date()}  (자산 {model.iloc[pk]/model.iloc[0]:.2f}배)")
sell=[t for t in trades if t[0]>d[pk]]
if sell:
    s0=sell[0]
    si=d[d==s0[0]].index[0]
    print(f"규칙이 판 날   : {s0[0].date()}  -> 고점 대비 이미 {(model.iloc[si]/model.iloc[pk]-1)*100:.1f}% 하락한 뒤")
    print(f"                 (고점 후 {si-pk}거래일 늦음)")
    print(f"모델 실제 저점 : {d[tr].date()}  (고점 대비 {(model.iloc[tr]/model.iloc[pk]-1)*100:.1f}%)")
    bi=d[d==s0[1]].index[0]
    print(f"규칙이 산 날   : {s0[1].date()}  -> 저점 대비 이미 {(model.iloc[bi]/model.iloc[tr]-1)*100:+.1f}% 오른 뒤")
    print(f"                 (저점 후 {bi-tr}거래일 늦음)")
