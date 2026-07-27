# Macro + momentum SP500 research

This workflow is intentionally separate from `macro_sp500` V1/V2. Its first
question is whether point-in-time macro and price features predict future SPY
return or drawdown risk. Portfolio allocation is a secondary test.

## Information timing

- SPY and daily market features use the current close and may trade only at the
  next trading day's open.
- Monthly macro observations are made available 45 calendar days after their
  observation date. This is conservative and avoids treating revised monthly
  values as if they were known on the first day of the month.
- NFCI receives a seven-day availability lag. Monthly GS10 and TB3MS series in
  the mixed-frequency FRED file receive the same 45-day conservative lag as
  other monthly observations.
- Training samples are purged when their future target window overlaps an inner
  or outer validation period.
- SPY uses adjusted prices so dividends and splits are represented.

## Credit-data correction

The existing sibling file named `HY_Spread.csv` was downloaded with FRED series
`BAMLH0A3HYCEY`. That series is the ICE BofA CCC & Lower US High Yield Index
Effective Yield, not an option-adjusted spread. This workflow leaves the source
file unchanged but exposes it as `HYYield`. When GS10 exists,
`HYExcessYieldProxy = HYYield - GS10` is explicitly labelled as a proxy.

Official broad high-yield OAS (`BAMLH0A0HYM2`) is downloaded as `HYOAS` by the
supplemental updater. The currently returned FRED history begins in 2023, so it
cannot support the full long-history walk-forward test and must not be presented
as such.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml,dev]"
.\.venv\Scripts\python.exe scripts\macro_momentum_sp500\update_research_data.py
.\.venv\Scripts\python.exe scripts\macro_momentum_sp500\run_research.py
```

Generated data and research results remain in sibling OneDrive folders. No raw
data or generated report is committed to the repository.

The full report compares the new strict-OOS allocation with Buy & Hold, static
70% and 76% SPY portfolios, and the previously generated V2 strict-OOS daily
result rebased to the same start date. The V2 row is a prior-result comparison,
not a re-optimization inside this workflow.

`StatefulMacro` is the primary allocation challenger. It preserves the original
daily `MacroMomentum` rule as a baseline but replaces daily threshold resets
with weekly state transitions, hysteresis, minimum holding and recovery periods,
a normal-state cooldown, and a stronger gate before reducing exposure below the
most recent normal-state reference price. Its allocation uses the same
point-in-time OOS model predictions and signal function in both the main
simulation and sensitivity analysis.

The stateful rule does not use the unstable expected-return regression to trade.
It combines smoothed 63/126-day drawdown probabilities with an explicit macro
confirmation score covering volatility, credit, NFCI financial conditions,
labor, and the yield curve. Moderate allocation changes require agreement
between the model and macro data, while unusually strong and still-rising macro
stress can independently enter caution or defensive states. Falling macro
stress plus recovered 63-day price momentum and the 126-day trend can begin a
measured re-entry. The exact formulas and thresholds are documented in
`METHODOLOGY.md`; changing them is a new strategy experiment, not a silent
continuation of the original rule.
