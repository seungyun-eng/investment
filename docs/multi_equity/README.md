# Fifteen-equity Buy & Hold alpha optimization

This workflow applies the Tesla integrated financial, macro, technical, and
strict-OOS downside signal to exactly these equities:

`NVDA, AAPL, GOOG, MSFT, TSM, AVGO, META, TSLA, COST, UNH, CVX, PLTR, NVO,
APP, VRT`.

## Research split

- Parameter selection: 2019-01-01 through 2025-12-31.
- Development folds: 2019-2021, 2022-2023, and 2024-2025.
- Final frozen diagnostic: 2026-01-01 onward.
- PLTR and APP begin at their IPO dates because pre-IPO prices do not exist.
- Quarterly financial rows become available after a fixed 45-calendar-day lag.
- Signals use closing information and execute at the next session's open.
- Transaction costs, slippage, idle-cash interest, and short borrow costs are
  included.
- ROI is net return:
  `(final_value / total_injected - 1) * 100`.

Every eligible candidate must have positive full-period 2019-2025 ROI, beat
same-period Buy & Hold, and complete a real LONG or SHORT cycle. Maximum
drawdown remains a reported risk diagnostic, but it is not allowed to hide a
return-qualified candidate.

Candidate priority is based only on development data. The robustness order is:
beat Buy & Hold in all three chronological folds, then the latest two folds,
then the latest fold, then only the full development period. Full-period and
fold alpha rank candidates within each tier.

Entry and exit trend windows are independently optimized from 10 to 200
sessions. This allows a position to start LONG like Buy & Hold, exit a confirmed
decline, and re-enter without waiting for the stricter financial/macro entry.
Financial, macro, technical, and strict-OOS downside features still drive the
composite, primary entry, and SHORT decision. The trend exit/re-entry prevents
those slower inputs from missing most of a strong bull market.

The ensemble search tests one or more top eligible candidates and 50%-100%
entry consensus. The selected configuration must itself beat Buy & Hold with
positive development ROI and at least one completed trade. This preserves a
valid single-candidate solution when a larger consensus dilutes its alpha.
The selected parameters and consensus are frozen before 2026 evaluation.
Optimization and simulation both call
`generate_integrated_signals` through the same `generate_consensus_signals`
function.

## Run

```powershell
python scripts/multi_equity/run_research.py --candidates 2000 --workers 4
```

The generated per-ticker audit files and aggregate ROI summary are written
atomically to the sibling `Results/Multi_Equity/integrated_signal` folder.

The latest audited outcome is documented in
[`BUY_HOLD_ALPHA_RESULTS.md`](BUY_HOLD_ALPHA_RESULTS.md). The older
cash-consensus result is retained as a superseded audit only.
