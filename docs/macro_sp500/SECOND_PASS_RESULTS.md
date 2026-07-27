# Macro SP500 second-pass result

Run date: 2026-07-25

## Formula changes from the first pass

These changes are intentional and the first-pass files remain available:

- SPY adjusted open and adjusted close replace the unadjusted legacy price.
- Monthly effective Fed Funds observations provide the cash return.
- VIX uses a trailing five-year percentile that excludes the current day.
- Stage 1 requires VIX stress plus the first drawdown threshold.
- Stage 2 requires the second drawdown threshold.
- Stage 3 requires the third drawdown threshold plus a 3% or 5% rebound and
  either a 20-day moving-average recovery or a 20% VIX decline from its
  20-day peak.
- Tactical exposure returns to the core only after the minimum hold, VIX
  normalization, a 200-day moving-average recovery, and five confirmation
  days.
- Trades execute at the next trading day's open.
- Rebalancing occurs only after a target change or a 3%, 5%, or 7% allocation
  drift. Orders below 2% of portfolio value are skipped.
- OOS cash, shares, and crisis state continue across calendar-year folds.
- ROI remains `(final_value / total_injected - 1) * 100`.

## Method

- Common adjusted SPY, VIX, and Fed Funds dates: 1993-01-29 through 2026-07-23
- Candidate grid: 972 combinations
- Walk-forward: trailing 10 calendar years train, next calendar year test
- OOS folds: 24, covering 2003 through 2026
- Selection constraint: at least 90% of Buy & Hold training CAGR and no more
  than 85% of Buy & Hold training MDD
- If no candidate satisfies both constraints, the candidate with the smallest
  normalized constraint violation is selected, followed by Calmar and CAGR.

## OOS result

| Metric | Macro V2 | Buy & Hold | Static 70/30 | Static 76/24 |
|---|---:|---:|---:|---:|
| Final value | 805,619 | 1,274,027 | 745,144 | 830,865 |
| ROI | 705.62% | 1,174.03% | 645.14% | 730.86% |
| CAGR | 9.26% | 11.41% | 8.90% | 9.41% |
| MDD | -49.05% | -55.19% | -41.33% | -44.70% |
| Calmar | 0.189 | 0.207 | 0.215 | 0.210 |

Macro V2 averaged 75.88% equity exposure. The exposure-matched static 76/24
portfolio exceeded V2 return while also producing a smaller drawdown. This is
evidence that the dynamic VIX/drawdown timing did not add value over a simple
static allocation in this test.

V2 did improve implementation quality:

- Rebalances fell from 4,387 in V1 to 24.
- Turnover fell from 69.25 to 5.49 times initial capital.
- It beat Static 70/30 in total return.
- It reduced losses relative to Buy & Hold in 2008, 2018, 2020, and 2022.

Only 6 of 24 folds had at least one candidate satisfying both training
constraints; 18 folds used the documented closest-constraint fallback.

## Latest selected parameters

| Parameter | Value |
|---|---:|
| Core weight | 80% |
| VIX entry percentile | 85% |
| Drawdown profile | -10%, -20%, -30% |
| Target profile | 80%, 90%, 100% |
| Rebound threshold | 3% |
| Rebalance band | 7% |
| Minimum hold | 60 trading days |

## Interpretation

V2 is a cleaner and more realistic implementation, but it is still not a
production strategy. The crisis rules reduced individual crisis-year losses,
yet the strategy did not improve full-period risk-adjusted performance versus
an exposure-matched static portfolio.

The next research question should therefore not be a finer VIX grid. It should
test whether the tactical reserve is useful only for rare, deep drawdowns, or
whether a static allocation should remain the macro baseline.

## Generated artifacts

The parameter record is index 1 in
`Results/Parameters/macro_sp500_v2_parameters.xlsx`. Detailed folds, daily OOS
values, rebalances, and the HTML report are under
`Results/SP500/macro_sp500_v2`.
