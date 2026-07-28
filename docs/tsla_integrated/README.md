# TSLA integrated financial, macro, and technical signal

This independent workflow combines:

- TSLA price-derived technical features;
- quarterly Tesla financial statement features;
- strict out-of-sample SP500 macro-risk predictions.

The portfolio can be LONG, CASH, or SHORT. It begins LONG for a fair comparison
with Buy & Hold. Independent 10-200-session entry and exit trend windows are
optimized to leave confirmed declines and re-enter recoveries. Primary LONG
and SHORT signals also use the financial, macro, technical, and strict-OOS
downside composite. Short borrow cost, trading cost, slippage, minimum holding
period, and separate fixed and trailing stops are included.

Financial observations become available only after a configurable 45-calendar-
day reporting lag. Daily signals use closing information and execute at the
next session's open. Optimization and final simulation both call
`generate_integrated_signals`.

The default strategy-development and optimization period is 2019-01-01 through
2025-12-31. Parameters are frozen before the 2026-01-01 onward final
real-world holdout is evaluated. Earlier observations may provide rolling-
feature warm-up history, but they are excluded from optimization performance.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\tsla_integrated\run_research.py
```

Outputs are atomically written to the sibling
`Results/Tesla/integrated_signal` folder.

## Objective

For the multi-equity alpha workflow, a candidate must complete a real trade and
produce full-period 2019-2025 ROI above both zero and same-period Buy & Hold.
Chronological fold alpha determines robustness priority. Maximum drawdown and
the configured -40% reference are reported, but drawdown is not a hard return
gate. The selection score combines full-period, mean-fold, and worst-fold
log-alpha plus drawdown improvement.
ROI is net return:

`(final_value / total_injected - 1) * 100`.

## Limitations

- The workbook contains period-end dates, not actual SEC filing timestamps.
  The 45-day lag is a conservative approximation and should eventually be
  replaced with exact filing acceptance dates.
- The SP500 macro model is a market-regime input, not a Tesla-specific model.
- Taxes, partial fills, bid-ask spread variation, and market impact are absent.
- Holdout results must never be used to choose parameters.
