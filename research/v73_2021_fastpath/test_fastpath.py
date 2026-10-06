import importlib.util
from pathlib import Path
import pandas as pd

P=Path(__file__).with_name("fastpath.py")
spec=importlib.util.spec_from_file_location("fastpath",P)
fp=importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)

def sample():
    return pd.DataFrame([
      dict(StateKey="2021-01-04|AAA|C",Date="2021-01-04",HistoricalTicker2021="AAA",ShareClassKey="C",
           MembershipGateStatus="READY_X",IdentityGateStatus="READY_X",FinancialFactorGateStatus="UNKNOWN",
           PriceGateStatus="UNKNOWN",AdjustedBasisGateStatus="UNKNOWN",CorporateActionGateStatus="UNKNOWN",
           ShareGrowthStrictSourceQualified=False,ShareGrowthFrozenCompatible=False,FinancialStrictSourceQualifiedReady=False,
           FrozenEntryFinancialReady=False,FinancialEvidenceNumeric9OtherComplete=True,FinancialEvidenceTextFlagComplete=True,
           FinancialEvidenceFresh450=True,FinancialEvidenceHasFiling=True,CanonicalCIK=1,NeedsV21ContinuityJoin=False),
      dict(StateKey="2021-01-05|AAA|C",Date="2021-01-05",HistoricalTicker2021="AAA",ShareClassKey="C",
           MembershipGateStatus="READY_X",IdentityGateStatus="READY_X",FinancialFactorGateStatus="UNKNOWN",
           PriceGateStatus="READY_X",AdjustedBasisGateStatus="READY_X",CorporateActionGateStatus="READY_X",
           ShareGrowthStrictSourceQualified=False,ShareGrowthFrozenCompatible=False,FinancialStrictSourceQualifiedReady=False,
           FrozenEntryFinancialReady=False,FinancialEvidenceNumeric9OtherComplete=True,FinancialEvidenceTextFlagComplete=True,
           FinancialEvidenceFresh450=True,FinancialEvidenceHasFiling=True,CanonicalCIK=1,NeedsV21ContinuityJoin=False),
      dict(StateKey="2021-01-04|BBB|C",Date="2021-01-04",HistoricalTicker2021="BBB",ShareClassKey="C",
           MembershipGateStatus="WAITING",IdentityGateStatus="WAITING",FinancialFactorGateStatus="UNKNOWN",
           PriceGateStatus="UNKNOWN",AdjustedBasisGateStatus="UNKNOWN",CorporateActionGateStatus="UNKNOWN",
           ShareGrowthStrictSourceQualified=False,ShareGrowthFrozenCompatible=False,FinancialStrictSourceQualifiedReady=False,
           FrozenEntryFinancialReady=False,FinancialEvidenceNumeric9OtherComplete=False,FinancialEvidenceTextFlagComplete=False,
           FinancialEvidenceFresh450=False,FinancialEvidenceHasFiling=False,CanonicalCIK=2,NeedsV21ContinuityJoin=True),
    ])

def test_plan_disjoint_shards():
    obj=fp.plan(sample(),2,2)
    for shards in obj["queues"].values():
        seen=[x["ticker"] for s in shards for x in s]
        assert len(seen)==len(set(seen))

def test_ready_never_downgrades():
    d=pd.DataFrame([{"StateKey":"2021-01-05|AAA|C","PriceGateStatusAfter":"BLOCKED"}])
    out,rec=fp.merge_one(sample(),d,"x")
    assert out.loc[out.StateKey.eq("2021-01-05|AAA|C"),"PriceGateStatus"].iloc[0]=="READY_X"

def test_ready_promotion():
    d=pd.DataFrame([{"StateKey":"2021-01-04|AAA|C","PriceGateStatusAfter":"READY_EXACT"}])
    out,rec=fp.merge_one(sample(),d,"x")
    assert out.loc[out.StateKey.eq("2021-01-04|AAA|C"),"PriceGateStatus"].iloc[0]=="READY_EXACT"
    assert rec["changes"]["PriceGateStatus"]==1
