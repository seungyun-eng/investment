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
