# V7-3 SEC filing ablation and optimization

This experiment keeps the V7-3 weekly signal and next-session-open execution
engine fixed while testing how much accession-specific SEC filing evidence adds.

The output compares the same-engine V7 baseline, 5%, 10%, and 15% SEC score
blends, an explicit SEC red-flag veto, a 100% SEC-ranked portfolio, and a
combined optimized policy. The pure SEC rank uses only reported growth,
margins, free cash flow, leverage, dilution, and disclosure evidence. Price
trend filters and stop rules remain V7 risk controls; therefore “100% SEC”
describes company ranking, not the absence of technical execution rules.

The reported pure-SEC portfolio additionally requires all six core filing
coverage fields and a non-negative cross-sectional filing quality score.

Candidate selection uses only 2020-2024. The already-observed 2025-2026 period
is generated after selection and is report-only. A strict pass requires at
least 35% training CAGR, no negative training calendar year, drawdown no worse
than -35%, and four of five years above SPY. If no candidate meets every rule,
the result is explicitly labeled `BEST_TRAIN_ONLY_NO_STRICT_PASS` and selected
by the smallest combined constraint shortfall before return is considered.

Run:

```powershell
python scripts/cross_sectional/run_filing_v7_optimization.py
```

Results are written atomically under the sibling OneDrive
`Results/Cross_Sectional/filing_v7_optimization` directory.

## Cash-gated known16 comparison variant

The base `known_2020_growth_16_v7.json` universe never applies the SPY
200/50-session market cash gate that the live Top-10-plus-watchlist model
uses, so the two configurations were not risk-comparable. A second universe
config, `known_2020_growth_16_v7_cash_gate.json`, is identical to the base
known16 config except it adds the same `market_cash_gate` block used by
`live_top10_watchlist.json` (`slow_sessions: 200`, `fast_sessions: 50`,
`band: 0.01`). No `base_v7_params`, no `optimization_grid`, and no
`selection_constraints` were changed; the 672-candidate grid search
(including both `hard_stop_returns` of -0.35 and -0.25) is re-run unmodified
under the gated environment so the candidate selection procedure stays
identical to how the live model's policy was chosen. `run_filing_v7_optimization.py`
now forwards `universe_config["market_cash_gate"]` into
`run_filing_v7_optimization()`; when a universe config omits the key (as the
base known16 config does), the call passes an empty dict and behavior is
unchanged from before this change.

Run:

```powershell
python scripts/cross_sectional/run_filing_v7_optimization.py `
  --universe-config config/cross_sectional/known_2020_growth_16_v7_cash_gate.json
```
