# Fifteen-equity first frozen results

Run date: 2026-07-27

## Selection rule

- Exactly 2,000 candidates were evaluated independently for each ticker.
- Candidate selection used only observations dated from 2019-01-01 through
  2025-12-31.
- The absolute-return-first score weights continuous development log return at
  55%, mean fold log return at 25%, worst fold log return at 15%, and mean
  drawdown improvement at 5%.
- A candidate was ineligible unless every development fold had positive net
  ROI, at least two folds traded, and continuous and worst-fold MDD were no
  worse than -40%.
- The frozen parameters were then evaluated from 2026-01-01 through the latest
  common price input, 2026-07-24.

ROI includes the configured transaction costs, slippage, short borrow costs,
and point-in-time idle-cash interest. Completed trades count closed LONG or
SHORT cycles; a position still open on 2026-07-24 is reflected in ROI but not
in that count.

## Results

| Ticker | Development ROI | 2019-21 | 2022-23 | 2024-25 | 2026 ROI | 2026 B&H | Excess | 2026 MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVDA | 7,736.60% | 589.10% | 195.04% | 270.55% | 8.89% | 7.34% | 1.55% | -18.33% |
| AAPL | 496.36% | 264.27% | 11.75% | 59.68% | 8.12% | 22.77% | -14.65% | -12.71% |
| GOOG | 588.34% | 135.68% | 34.60% | 120.51% | 0.76% | 0.37% | 0.39% | -7.08% |
| MSFT | 269.15% | 168.89% | 53.93% | 1.27% | 2.04% | -19.17% | 21.21% | 0.00% |
| TSM | 839.44% | 192.32% | 9.92% | 168.88% | 21.91% | 21.91% | 0.00% | -18.37% |
| AVGO | 459.93% | 41.86% | 69.47% | 113.39% | -8.57% | 7.24% | -15.81% | -17.03% |
| META | 263.35% | 37.85% | 43.60% | 77.49% | 4.68% | -7.83% | 12.52% | -13.93% |
| TSLA | 608.86% | 370.00% | 2.56% | 55.81% | -19.11% | -31.10% | 11.99% | -30.63% |
| COST | 289.81% | 165.67% | 16.87% | 20.01% | 2.04% | 7.97% | -5.93% | 0.00% |
| UNH | 97.88% | 77.01% | 4.52% | 4.07% | 2.04% | 24.99% | -22.95% | 0.00% |
| CVX | 122.89% | 37.06% | 35.63% | 17.19% | 1.00% | 17.58% | -16.58% | -8.02% |
| PLTR | 1,816.09% | 0.10% | 99.72% | 661.61% | 2.04% | -29.34% | 31.38% | 0.00% |
| NVO | 248.81% | 67.96% | 49.85% | 18.37% | 2.04% | -8.95% | 10.99% | 0.00% |
| APP | 2,248.55% | 0.06% | 58.13% | 1,384.34% | -19.47% | -36.41% | 16.95% | -32.79% |
| VRT | 1,512.43% | 42.83% | 185.13% | 285.61% | 49.60% | 58.99% | -9.39% | -25.32% |

Twelve of fifteen frozen strategies had positive 2026 ROI. The equal-weight
average of the fifteen standalone strategy ROIs was 3.87%, compared with 2.42%
for the equal-weight average of their Buy & Hold ROIs.

## Interpretation

The absolute-return-first objective produced high development returns, but it
did not make future positive return certain. AVGO, TSLA, and APP were negative
in the frozen 2026 diagnostic. AVGO lost on a SHORT cycle, TSLA held an open
LONG through the end date, and APP lost on a completed LONG cycle. Replacing
only those candidates after observing 2026 would turn the final diagnostic into
holdout selection and is therefore not reported as a valid fix.

The financial workbooks for all tickers except TSLA end in 2025. Their 2026
signals therefore use the latest lagged 2025 financial row available in the
local source data. PLTR and APP also lack pre-IPO price observations, so their
development samples begin on 2020-09-30 and 2021-04-15 respectively.

This is the first frozen multi-equity run, not a guarantee of positive future
return. Because 2026 results have now been observed, further design changes
must be evaluated prospectively or with an additional later holdout.
