# Fifteen-equity Buy & Hold alpha result

Run date: 2026-07-27

Official run timestamp: `20260727_215458_230107`

Price evaluation end: 2026-07-24

## Selection rule

- Generate and select parameters only from 2019-01-01 through 2025-12-31.
- Evaluate 2,000 candidates independently for each ticker.
- Start LONG for a fair comparison with Buy & Hold.
- Require full-development net ROI above both zero and same-period Buy & Hold.
- Require at least one completed LONG or SHORT cycle.
- Prioritize candidates that beat Buy & Hold in all three development folds,
  then the latest two folds, then the latest fold, then only the full period.
- Optimize independent 10-200-session exit and re-entry trends.
- Use financial, macro, technical, and strict-OOS downside inputs for the
  composite, primary LONG entry, and SHORT decision.
- Generate each signal after the close and execute at the next open.
- Include transaction cost, slippage, idle-cash interest, and short borrow cost.
- Calculate ROI as `(final_value / total_injected - 1) * 100`.

## Results

| Ticker | 2019-25 strategy | 2019-25 Hold | Excess | MDD | Trades | 2026 strategy | 2026 Hold | Excess | Trades | Strict pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| NVDA | 7,584.17% | 5,471.39% | 2,112.78%p | -37.51% | 7 | 7.34% | 7.34% | 0.00%p | 0 | No |
| AAPL | 782.83% | 654.66% | 128.17%p | -32.82% | 94 | 21.50% | 22.77% | -1.26%p | 10 | No |
| GOOG | 679.01% | 502.39% | 176.61%p | -42.53% | 13 | -1.95% | 0.37% | -2.32%p | 1 | No |
| MSFT | 453.43% | 382.75% | 70.68%p | -31.45% | 8 | -21.29% | -19.17% | -2.11%p | 2 | No |
| TSM | 1,202.45% | 759.21% | 443.25%p | -28.36% | 3 | 21.91% | 21.91% | 0.00%p | 0 | No |
| AVGO | 1,568.40% | 1,314.05% | 254.35%p | -36.66% | 2 | 7.24% | 7.24% | 0.00%p | 0 | No |
| META | 873.53% | 389.68% | 483.84%p | -27.32% | 10 | -14.46% | -7.83% | -6.62%p | 2 | No |
| TSLA | 4,955.45% | 2,095.55% | 2,859.90%p | -46.06% | 12 | -33.59% | -31.10% | -2.49%p | 2 | No |
| COST | 383.37% | 323.00% | 60.37%p | -20.63% | 8 | 6.57% | 7.97% | -1.40%p | 1 | No |
| UNH | 127.60% | 35.47% | 92.13%p | -37.11% | 5 | 13.45% | 24.99% | -11.54%p | 1 | No |
| CVX | 183.51% | 37.12% | 146.39%p | -21.59% | 7 | 18.27% | 17.58% | 0.69%p | 1 | Yes |
| PLTR | 4,639.77% | 1,732.55% | 2,907.23%p | -52.90% | 1 | -32.02% | -29.34% | -2.68%p | 2 | No |
| NVO | 560.85% | 115.42% | 445.43%p | -19.50% | 11 | -5.61% | -8.95% | 3.34%p | 1 | No |
| APP | 5,392.66% | 935.62% | 4,457.04%p | -50.19% | 5 | -47.55% | -36.41% | -11.13%p | 3 | No |
| VRT | 3,788.07% | 1,560.30% | 2,227.77%p | -31.58% | 10 | 45.40% | 58.99% | -13.59%p | 1 | No |

All 15 development strategies are positive and beat Buy & Hold. In the
observed 2026 window, eight are positive, two beat Buy & Hold, and only CVX is
both positive and strictly above Buy & Hold. NVDA, TSM, and AVGO equal Buy &
Hold because they complete no tactical 2026 trade; equality is not counted as
a pass. NVO beats a negative Buy & Hold result but is also negative, so it is
not counted as a pass.

The equal-weight mean 2026 standalone strategy ROI is -0.99%, versus +2.42%
for Buy & Hold. The mean excess is -3.41 percentage points.

## Conclusion

The revised optimizer now enforces the requested Buy & Hold alpha condition in
the 2019-2025 development sample and prevents a cash-only result from passing.
It does not validate the stronger claim that the signals can guarantee positive
future alpha. The 2026 diagnostic rejects that claim for 14 of 15 tickers.

The exact 2026 low and high are knowable only after the fact. Selecting new
parameters from these 2026 outcomes would be look-ahead leakage, not a tradable
signal. Because 2026 has already been observed during this work, any later
revision must be labeled a diagnostic recheck; the next clean validation is
prospective data after the strategy is frozen.
