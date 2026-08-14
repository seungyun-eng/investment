from __future__ import annotations
import json
import numpy as np
import pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.event_warning import (
    build_two_stage_signal, collapse_alert_episodes, detect_drawdown_events,
    evaluate_alert_episodes, warning_rule_candidates,
)
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

DISCOVERY_END = pd.Timestamp("2017-12-31")
HOLDOUT_START = pd.Timestamp("2018-01-01")
MINIMUM_LEAD, MAXIMUM_LEAD = 7, 63

config = load_research_config("config/macro_momentum_sp500/research.json")
data = load_research_data(load_paths().macro, config)
features = build_features(data, config).reset_index(drop=True)
events = detect_drawdown_events(features, -0.15)
data_start = pd.Timestamp(features["Date"].min())
data_end = pd.Timestamp(features["Date"].max())
dates = pd.to_datetime(features["Date"])

# --- Part 1: instrument the FULL candidate sweep (not just the winner) to see
# the distribution of discovery-episode-counts across all 5,346 real rules.
rows = []
for rule in warning_rule_candidates():
    signal = build_two_stage_signal(features, rule)
    episodes = collapse_alert_episodes(signal, rule.episode_merge_gap, rule.cooldown_days)
    d = evaluate_alert_episodes(episodes, events, data_start, DISCOVERY_END, MINIMUM_LEAD, MAXIMUM_LEAD)
    rows.append({
        "episodes": d["alert_episodes"], "fp": d["false_positive_episodes"],
        "captured": d["captured_events"],
    })
candidates = pd.DataFrame(rows)
print("=== Distribution of discovery alert-episode counts across all 5,346 real candidates ===")
print(candidates["episodes"].value_counts().sort_index().to_string())

exactly_one = candidates[candidates["episodes"] == 1]
print(f"\nCandidates producing exactly 1 discovery episode: {len(exactly_one)}")
print(f"Of those, FP==0 and captured>=1: {((exactly_one['fp']==0)&(exactly_one['captured']>=1)).sum()}")
print(f"  -> real-rule conditional success rate given exactly-1-episode: "
      f"{((exactly_one['fp']==0)&(exactly_one['captured']>=1)).mean()*100:.1f}%")

# --- Part 2: Monte Carlo random single-date placement, same discovery period,
# same evaluate_alert_episodes / FP-capture definition, for a true random
# baseline at matched "1 episode" frequency.
discovery_mask = dates.le(DISCOVERY_END)
eligible_indices = np.flatnonzero(discovery_mask.to_numpy())
rng = np.random.default_rng(42)
n_trials = 20000
successes = 0
captures = 0
for _ in range(n_trials):
    idx = int(rng.choice(eligible_indices))
    from stock_research.macro_momentum_sp500.event_warning import AlertEpisode
    ep = AlertEpisode(idx, idx, dates.iloc[idx], dates.iloc[idx])
    d = evaluate_alert_episodes([ep], events, data_start, DISCOVERY_END, MINIMUM_LEAD, MAXIMUM_LEAD)
    if d["false_positive_episodes"] == 0 and d["captured_events"] >= 1:
        successes += 1
    if d["captured_events"] >= 1:
        captures += 1

p_random = successes / n_trials
print(f"\n=== Random single-date placement Monte Carlo (n={n_trials}) ===")
print(f"P(random single alert has FP=0 AND captures an event) = {p_random*100:.2f}%")
print(f"P(random single alert captures an event at all, regardless of FP) = {captures/n_trials*100:.2f}%")

print(f"\n=== Comparison ===")
print(f"Real candidates, exactly-1-episode, success rate: "
      f"{((exactly_one['fp']==0)&(exactly_one['captured']>=1)).mean()*100:.2f}%  (n={len(exactly_one)})")
print(f"Random single-date placement, success rate:        {p_random*100:.2f}%  (n={n_trials})")

# --- Part 3: binomial check -- given p_random, what's the probability of
# getting >=9 successes out of 5346 trials by pure chance (matching the
# actual "9 eligible zero-FP candidates" count)?
from scipy import stats
prob_9_or_more = 1 - stats.binom.cdf(8, 5346, p_random)
print(f"\nP(>= 9 successes out of 5346 independent random trials | p={p_random:.5f}) = {prob_9_or_more:.6f}")
print("NOTE: the 5,346 real candidates are NOT independent (highly correlated, same underlying series),")
print("so this binomial figure is illustrative only, not a rigorous p-value.")
