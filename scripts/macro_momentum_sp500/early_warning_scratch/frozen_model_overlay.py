from __future__ import annotations
"""READ-ONLY counterfactual on the frozen model's equity curve.
Does NOT modify the frozen model or its artifacts."""
import numpy as np
import pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.macro_momentum_sp500.event_warning import (
    build_two_stage_signal, collapse_alert_episodes, WarningRule)
from stock_research.paths import load_paths

SERIES = "V7_BASELINE_SAME_ENGINE"
eq = pd.read_csv("artifacts/pit_and_selection_fix/runs/B_PIT_STRICT/equity.csv")
eq = eq[eq.Series == SERIES].copy()
eq["Date"] = pd.to_datetime(eq["Date"])
eq = eq.sort_values("Date").reset_index(drop=True)
model_ret = eq["Equity"].pct_change().fillna(0.0)

# cash rate from macro data (annual %), aligned
config = load_research_config("config/macro_momentum_sp500/research.json")
feat = build_features(load_research_data(load_paths().macro, config), config).reset_index(drop=True)
feat["Date"] = pd.to_datetime(feat["Date"])
merged = eq[["Date"]].merge(feat[["Date", "CashRate", "Drawdown252",
                                  "MacroConfirmationScore", "EarlyWarningBreadth",
                                  "Momentum_5", "Momentum_21"]], on="Date", how="left").ffill()
cash_daily = (merged["CashRate"].fillna(0) / 100 / 252).to_numpy()

def stats(returns, dates):
    curve = (1 + pd.Series(returns)).cumprod()
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (curve.iloc[-1]) ** (1/years) - 1
    dd = (curve / curve.cummax() - 1).min()
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    return {"CAGR_%": round(cagr*100, 2), "MDD_%": round(dd*100, 2), "Sharpe": round(sharpe, 3)}

def apply_overlay(in_cash_mask):
    r = np.where(in_cash_mask, cash_daily, model_ret.to_numpy())
    return r

rows = [{"scenario": "0_FROZEN_MODEL_AS_IS", **stats(model_ret.to_numpy(), eq["Date"]),
         "pct_cash": 0.0}]

# --- Detect the frozen model's OWN drawdowns >= 15%
curve = eq["Equity"]
peak = curve.cummax()
dd = curve / peak - 1
events = []
in_ev = False
for i in range(len(dd)):
    if not in_ev and dd.iloc[i] <= -0.15:
        # find preceding peak
        pk = int(np.argmax(curve[:i+1].to_numpy() == peak.iloc[i]))
        in_ev = True; start = pk; low = i
    elif in_ev:
        if curve.iloc[i] < curve.iloc[low]: low = i
        if curve.iloc[i] >= peak.iloc[start]:
            events.append((start, low, i)); in_ev = False
if in_ev: events.append((start, low, None))
print(f"Frozen-model own 15%+ drawdowns: {len(events)}개")
for s, l, r_ in events:
    print(f"  peak {eq.Date[s].date()} -> trough {eq.Date[l].date()} "
          f"({dd.iloc[l]*100:.1f}%) -> recovery {eq.Date[r_].date() if r_ else 'not-recovered'}")

# --- Scenario 1: PERFECT hindsight (exit exactly at peak, buy back at trough)
mask = np.zeros(len(eq), dtype=bool)
for s, l, _ in events: mask[s:l+1] = True
rows.append({"scenario": "1_PERFECT_HINDSIGHT",
             **stats(apply_overlay(mask), eq["Date"]),
             "pct_cash": round(mask.mean()*100, 1)})

# --- Scenario 2: realistic-ish hindsight, 10 days late both ends
mask2 = np.zeros(len(eq), dtype=bool)
for s, l, _ in events: mask2[min(s+10, len(eq)-1):min(l+10, len(eq))] = True
rows.append({"scenario": "2_HINDSIGHT_BUT_10D_LATE",
             **stats(apply_overlay(mask2), eq["Date"]),
             "pct_cash": round(mask2.mean()*100, 1)})

# --- Scenario 3: ACTUAL Codex event-warning rule
rule = WarningRule(vulnerability_mode="both", macro_threshold=0.65, breadth_threshold=3,
                   vulnerability_lookback=21, trigger_z=2.0, momentum_5_threshold=-0.02,
                   minimum_trigger_components=3, trigger_confirmation_days=3,
                   maximum_pre_alert_drawdown=-0.05)
sig = build_two_stage_signal(feat, rule)
eps = collapse_alert_episodes(sig, rule.episode_merge_gap, rule.cooldown_days)
alert_dates = [e.start_date for e in eps]
print(f"\nCodex rule alert dates in this period: {[str(d.date()) for d in alert_dates if d >= eq.Date.min()]}")
# exit on alert, re-enter 63 sessions after the model's next trough (mechanical)
mask3 = np.zeros(len(eq), dtype=bool)
for d in alert_dates:
    if d < eq.Date.min() or d > eq.Date.max(): continue
    i = int(eq.index[eq.Date >= d][0])
    # hold cash until model recovers to its pre-alert level, cap 252 days
    lvl = curve.iloc[i]
    j = i
    while j < len(eq)-1 and j < i+252 and curve.iloc[j] < lvl: j += 1
    mask3[i:j+1] = True
rows.append({"scenario": "3_ACTUAL_CODEX_SIGNAL",
             **stats(apply_overlay(mask3), eq["Date"]),
             "pct_cash": round(mask3.mean()*100, 1)})

# --- Scenario 4: signal fires but timing is WRONG (fires at random times, same exposure)
rng = np.random.default_rng(0)
sims = []
n_days = int(mask3.sum()); block = 40
for _ in range(1000):
    m = np.zeros(len(eq), dtype=bool)
    for _ in range(max(1, n_days // block)):
        s0 = rng.integers(0, len(eq))
        m[s0:s0+block] = True
    sims.append(stats(apply_overlay(m), eq["Date"])["CAGR_%"])
sims = np.array(sims)
print(f"\n4_RANDOM_TIMING same cash exposure, CAGR distribution:")
print(f"   median {np.median(sims):.2f}%,  5~95% range [{np.percentile(sims,5):.2f}%, {np.percentile(sims,95):.2f}%]")

print()
print(pd.DataFrame(rows).to_string(index=False))
