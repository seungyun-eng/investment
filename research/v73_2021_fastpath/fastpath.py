#!/usr/bin/env python3
"""Fast-path planner/merger for V7.3 2021 PIT recovery.

This code changes throughput only. It never changes Frozen V7.3 logic or
promotes a row without a worker-supplied source-qualified delta.
"""
from __future__ import annotations
import argparse, gzip, io, json, re, tempfile, zipfile
from pathlib import Path
import pandas as pd

READY_RE = re.compile(r"^READY(?:_|$)", re.I)
GATES = {
    "membership":"MembershipGateStatus","identity":"IdentityGateStatus",
    "financial":"FinancialFactorGateStatus","price":"PriceGateStatus",
    "basis":"AdjustedBasisGateStatus","action":"CorporateActionGateStatus",
}

def ready(s):
    return s.fillna("").astype(str).str.match(READY_RE)

def boolcol(df, c):
    if c not in df:
        return pd.Series(False, index=df.index)
    s=df[c]
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin({"1","true","t","yes","y"})

def csv_bytes(name, b):
    if name.endswith(".gz"):
        b=gzip.decompress(b)
    return pd.read_csv(io.BytesIO(b), low_memory=False)

def load(path):
    p=Path(path)
    if p.suffix.lower()==".zip":
        with zipfile.ZipFile(p) as z:
            names=z.namelist()
            pref=[n for n in names if n.endswith("V73_2021_REBUILT_STATE.csv.gz")]
            if not pref:
                pref=[n for n in names if n.lower().endswith("statekey_delta.csv.gz")]
            if not pref:
                pref=[n for n in names if n.lower().endswith("statekey_delta.csv")]
            if not pref:
                pref=[n for n in names if n.lower().endswith((".csv.gz",".csv"))]
            if not pref:
                raise ValueError(f"No CSV in {p}")
            return csv_bytes(pref[0], z.read(pref[0]))
    if p.name.endswith(".csv.gz"):
        return pd.read_csv(p, compression="gzip", low_memory=False)
    if p.suffix.lower()==".csv":
        return pd.read_csv(p, low_memory=False)
    if p.suffix.lower()==".parquet":
        return pd.read_parquet(p)
    raise ValueError(path)

def canonical(df):
    df=df.copy()
    if "StateKey" not in df:
        df["StateKey"]=(
            pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")+"|"+
            df["HistoricalTicker2021"].astype(str).str.upper()+"|"+
            df["ShareClassKey"].fillna("COMMON_UNSPECIFIED").astype(str))
    if df.StateKey.duplicated().any():
        raise ValueError("duplicate StateKey")
    return df

def priority(df):
    df=canonical(df)
    mi=ready(df[GATES["membership"]]) & ready(df[GATES["identity"]])
    p=ready(df[GATES["price"]]); b=ready(df[GATES["basis"]]); a=ready(df[GATES["action"]])
    sg=boolcol(df,"ShareGrowthStrictSourceQualified")
    ff=boolcol(df,"FinancialStrictSourceQualifiedReady")
    x=pd.DataFrame({
        "ticker":df.HistoricalTicker2021.astype(str).str.upper(),"rows":1,
        "mi":mi.astype(int),"price":p.astype(int),"basis":b.astype(int),
        "action":a.astype(int),"pba":(p&b&a).astype(int),
        "sg":sg.astype(int),"fin":ff.astype(int),
        "numeric9":boolcol(df,"FinancialEvidenceNumeric9OtherComplete").astype(int),
        "textflag":boolcol(df,"FinancialEvidenceTextFlagComplete").astype(int),
        "fresh":boolcol(df,"FinancialEvidenceFresh450").astype(int),
        "filing":boolcol(df,"FinancialEvidenceHasFiling").astype(int),
        "cik":pd.to_numeric(df.get("CanonicalCIK"),errors="coerce").notna().astype(int),
        "needs_v21":boolcol(df,"NeedsV21ContinuityJoin").astype(int),
    }).groupby("ticker",sort=True).sum(numeric_only=True)
    x["sharegrowth_score"]=20*x.mi+20*x.pba+12*x.numeric9+8*x.textflag+4*x.fresh+2*x.filing-30*x.sg
    x["identity_score"]=8*(x.rows-x.mi)+2*x.cik+x.needs_v21
    x["pba_score"]=12*x.mi-5*x.pba+3*x.price
    x["financial_score"]=20*x.sg+10*x.numeric9+8*x.textflag+4*x.fresh-30*x.fin
    return x.reset_index()

def shard(items, score, n, batch):
    items=items.sort_values([score,"rows","ticker"],ascending=[False,False,True]).head(n*batch)
    bins=[[] for _ in range(n)]; loads=[0.0]*n
    for r in items.to_dict("records"):
        ok=[i for i in range(n) if len(bins[i])<batch]
        if not ok: break
        i=min(ok,key=lambda k:(loads[k],len(bins[k]),k))
        bins[i].append(r); loads[i]+=max(1,float(r[score]))+.05*float(r["rows"])
    return bins

def plan(df,n=8,batch=8):
    q=priority(df)
    specs={
        "sharegrowth":(q[(q.mi>0)&(q.sg<q.mi)],"sharegrowth_score"),
        "identity":(q[q.mi<q.rows],"identity_score"),
        "pba":(q[(q.mi>0)&(q.pba<q.mi)],"pba_score"),
        "financial":(q[(q.sg>0)&(q.fin<q.sg)],"financial_score"),
    }
    queues={k:shard(v,s,n,batch) for k,(v,s) in specs.items()}
    return {
        "summary":{"rows":int(len(df)),"tickers":int(q.ticker.nunique()),"shards":n,
                   "batch_size":batch,"capacity_per_domain_per_cycle":n*batch,
                   "candidate_tickers":{k:int(len(v)) for k,(v,_) in specs.items()}},
        "queues":queues}

def write_plan(obj,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/"fastpath_plan.json").write_text(json.dumps(obj,indent=2),encoding="utf-8")
    for domain,ss in obj["queues"].items():
        for i,rows in enumerate(ss):
            pd.DataFrame(rows).to_csv(out/f"{domain}_shard_{i:02d}.csv",index=False)

def delta_table(path):
    p=Path(path)
    if p.suffix.lower()!=".zip":
        return load(p),p.name
    with zipfile.ZipFile(p) as z:
        ns=z.namelist()
        c=[n for n in ns if n.lower().endswith("statekey_delta.csv.gz")]
        if not c: c=[n for n in ns if n.lower().endswith("statekey_delta.csv")]
        if not c: c=[n for n in ns if "changed_statekey_delta" in n.lower() and n.lower().endswith((".csv",".csv.gz"))]
        if not c: raise ValueError(f"No statekey delta in {p}")
        return csv_bytes(c[0],z.read(c[0])),f"{p.name}:{c[0]}"

def promote(base,new):
    br=ready(base); nr=ready(new); out=base.copy()
    out.loc[nr]=new.loc[nr]
    info=(~br)&(~nr)&new.notna()&(new.astype(str)!="")
    out.loc[info]=new.loc[info]
    return out

def merge_one(base,d,name):
    d=d.drop_duplicates("StateKey",keep="last").set_index("StateKey",drop=False)
    x=base.set_index("StateKey",drop=False); keys=x.index.intersection(d.index)
    rec={"source":name,"overlap":int(len(keys)),"changes":{}}
    sm={
      "MembershipGateStatusAfter":"MembershipGateStatus",
      "IdentityGateStatusAfter":"IdentityGateStatus",
      "PriceGateStatusAfter":"PriceGateStatus",
      "AdjustedBasisGateStatusAfter":"AdjustedBasisGateStatus",
      "CorporateActionGateStatusAfter":"CorporateActionGateStatus"}
    for dc,bc in sm.items():
        if dc in d and bc in x:
            before=int(ready(x.loc[keys,bc]).sum())
            x.loc[keys,bc]=promote(x.loc[keys,bc],d.loc[keys,dc])
            rec["changes"][bc]=int(ready(x.loc[keys,bc]).sum())-before
    bm=["ShareGrowthStrictSourceQualified","ShareGrowthFrozenCompatible",
        "FinancialStrictSourceQualifiedReady","V37StrictRowDiagnosticReady",
        "FrozenEntryFinancialReady"]
    for c in bm:
        if c in d:
            if c not in x: x[c]=False
            before=int(boolcol(x.loc[keys],c).sum())
            x.loc[keys,c]=(boolcol(x.loc[keys],c)|boolcol(d.loc[keys],c)).values
            rec["changes"][c]=int(boolcol(x.loc[keys],c).sum())-before
    return x.reset_index(drop=True),rec

def recount(df):
    out={k:int(ready(df[c]).sum()) for k,c in GATES.items()}
    strict=ready(df[GATES["membership"]])&ready(df[GATES["identity"]])&ready(df[GATES["financial"]])&ready(df[GATES["price"]])&ready(df[GATES["basis"]])&ready(df[GATES["action"]])
    prior=boolcol(df,"CertifiedRebuilt")
    df["CertifiedRebuilt"]=prior|strict
    fc=ready(df[GATES["membership"]])&ready(df[GATES["identity"]])&ready(df[GATES["price"]])&ready(df[GATES["basis"]])&ready(df[GATES["action"]])&boolcol(df,"FrozenEntryFinancialReady")
    df["FrozenV73InputReadyCandidate"]=boolcol(df,"FrozenV73InputReadyCandidate")|fc
    out.update(certified=int(boolcol(df,"CertifiedRebuilt").sum()),
               frozen_candidate=int(boolcol(df,"FrozenV73InputReadyCandidate").sum()),
               sharegrowth_strict=int(boolcol(df,"ShareGrowthStrictSourceQualified").sum()),
               financial_strict=int(boolcol(df,"FinancialStrictSourceQualifiedReady").sum()))
    return df,out

def write_zip(df,path,summary):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); f=td/"V73_2021_REBUILT_STATE.csv.gz"
        df.to_csv(f,index=False,compression="gzip")
        (td/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            z.write(f,f.name); z.write(td/"summary.json","summary.json")

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("plan"); p.add_argument("--central",required=True); p.add_argument("--out",required=True)
    p.add_argument("--shards",type=int,default=8); p.add_argument("--batch-size",type=int,default=8)
    m=sp.add_parser("merge"); m.add_argument("--central",required=True); m.add_argument("--delta",action="append",required=True); m.add_argument("--out",required=True)
    a=ap.parse_args()
    if a.cmd=="plan":
        obj=plan(load(a.central),a.shards,a.batch_size); write_plan(obj,a.out); print(json.dumps(obj["summary"],indent=2)); return
    base=canonical(load(a.central)); _,before=recount(base.copy()); receipts=[]
    for pth in a.delta:
        d,n=delta_table(pth); base,r=merge_one(base,d,n); receipts.append(r)
    base,after=recount(base)
    summary={"counts_before":before,"counts_after":after,"deltas":receipts,
             "guardrails":["Frozen V7.3 logic unchanged","StateKey-only merge","READY monotonic","no current-ticker backcast","no forward fill","no cross-share-class substitution"]}
    write_zip(base,a.out,summary); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
