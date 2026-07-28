# Fifteen-equity absolute-positive consensus result

> Superseded audit result. This cash-heavy run does not satisfy the later
> requirement that every strategy beat Buy & Hold with real trades. It is kept
> only for reproducibility; do not treat it as the current strategy result.

Run date: 2026-07-27

Official run timestamp: `20260727_201030_296820`

## Rule

- Candidate generation and selection use 2019-01-01 through 2025-12-31 only.
- Each ticker evaluates 2,000 parameter candidates.
- Candidate ranking uses absolute net ROI and absolute MDD. Buy & Hold excess
  return is not part of the score.
- Up to the top 50 eligible candidates vote on LONG and SHORT entries.
- Entry consensus starts at 70%. If any independent development fold has
  non-positive ROI, consensus rises in fixed 5% steps until all folds are
  positive.
- Exit consensus is 50%.
- Ambiguous entry signals remain in interest-bearing cash.
- Signals use prior-close information and execute at the next open with costs,
  slippage, and short borrow expense.
- ROI is `(final_value / total_injected - 1) * 100`.

MSFT and META required 75% entry consensus. All other tickers passed the
development gate at 70%.

## Absolute ROI

| Ticker | Consensus | Development ROI | 2019-21 | 2022-23 | 2024-25 | 2026 ROI | 2026 completed trades | End state |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NVDA | 70% | 964.69% | 116.05% | 79.26% | 198.54% | 2.04% | 0 | CASH |
| AAPL | 70% | 376.73% | 244.31% | 39.23% | 6.64% | 2.04% | 0 | CASH |
| GOOG | 70% | 331.76% | 68.84% | 40.03% | 85.58% | 2.04% | 0 | CASH |
| MSFT | 75% | 132.71% | 46.59% | 48.16% | 4.46% | 2.04% | 0 | CASH |
| TSM | 70% | 202.38% | 22.98% | 3.53% | 118.23% | 1.43% | 1 | LONG |
| AVGO | 70% | 130.76% | 44.18% | 1.80% | 59.51% | 2.04% | 0 | CASH |
| META | 75% | 170.60% | 18.33% | 35.82% | 72.66% | 2.04% | 0 | CASH |
| TSLA | 70% | 20.45% | 2.64% | 6.91% | 9.77% | 2.04% | 0 | CASH |
| COST | 70% | 164.01% | 61.07% | 16.87% | 26.31% | 2.04% | 0 | CASH |
| UNH | 70% | 30.48% | 2.64% | 10.88% | 12.23% | 2.04% | 0 | CASH |
| CVX | 70% | 83.89% | 20.72% | 35.63% | 9.77% | 2.04% | 0 | CASH |
| PLTR | 70% | 861.99% | 0.10% | 100.41% | 379.53% | 2.04% | 0 | CASH |
| NVO | 70% | 77.53% | 4.33% | 28.70% | 11.87% | 2.04% | 0 | CASH |
| APP | 70% | 410.54% | 0.06% | 58.13% | 222.68% | 2.04% | 0 | CASH |
| VRT | 70% | 616.02% | 19.96% | 206.98% | 89.37% | 2.04% | 0 | CASH |

All fifteen 2026 ROIs are positive. Their mean is 2.00%.

## Interpretation

This result meets the requested absolute-positive condition for the observed
2026 window, but the source of return matters. Fourteen tickers did not complete
a 2026 trade. Their approximately 2.04% ROI came from remaining in cash and
earning the point-in-time cash rate. TSM was the only ticker to complete a trade;
it ended with another LONG open and a 1.43% ROI.

No 2026 SHORT signal reached the required consensus. That is a valid abstention,
not evidence that the strategy can predict every decline. A system cannot
guarantee positive future ROI without either abstaining, accepting risk, or
using future information.

The first single-member run had already exposed 2026 results. This consensus
result is therefore a diagnostic recheck, not a pristine holdout. The consensus
threshold itself is selected only from 2019-2025 folds, but the next genuinely
unseen validation must be prospective data after the freeze date.
