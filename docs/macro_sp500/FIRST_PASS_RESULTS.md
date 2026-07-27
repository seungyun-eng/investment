# Macro SP500 first-pass result

Run date: 2026-07-25

## Method

- Common S&P proxy and VIX dates: 1993-01-29 through 2025-11-03
- Candidate grid: 972 combinations
- Walk-forward design: trailing 10 calendar years train, next calendar year test
- Out-of-sample folds: 23, covering 2003 through 2025
- Initial and total injected capital: 100,000
- Transaction cost: 5 bps
- Slippage: 5 bps
- Signal timing: close signal, next trading day's open execution

## Out-of-sample result

| Metric | Macro SP500 | Buy & Hold |
|---|---:|---:|
| Final value | 492,638.26 | 697,154.10 |
| ROI | 392.64% | 597.15% |
| CAGR | 7.23% | 8.88% |
| Maximum drawdown | -56.10% | -56.75% |
| Calmar | 0.129 | 0.156 |

The strategy outperformed Buy & Hold in 10 of 23 test years. Its average equity
exposure was 74.20%, but it still experienced almost the same maximum drawdown.
It generated 4,387 rebalance records and 69.25 times initial-capital turnover.

## Latest selected parameters

These are the parameters selected using 2015-2024 training data for the partial
2025 test fold:

| Parameter | Value |
|---|---:|
| VIX lookback | 3 years |
| Core S&P weight | 50% |
| Warning score minimum | 2 |
| Warning addition | 0% |
| VIX entry percentile | 95% |
| VIX exit percentile | 40% |
| Minimum hold | 126 trading days |

Across all folds, a 50% core was selected 22 times, no warning addition was
selected 22 times, the 40th-percentile exit was selected 19 times, and the
126-day hold was selected 13 times.

## Interpretation

This is a valid first baseline, not a production strategy. It underperformed
Buy & Hold while providing almost no maximum-drawdown improvement. The first
warning tranche was effectively rejected by the optimizer in 22 of 23 folds.
Exact target-weight restoration also caused excessive small rebalances.

The next pass should test a rebalance band or signal-change-only execution,
explicit dividend-adjusted total-return data, and continuous portfolio state
across test-year boundaries before narrowing the percentile grid.

## Generated artifacts

The parameter record is index 1 in
`Results/Parameters/macro_sp500_parameters.xlsx`. Detailed folds, daily OOS
values, rebalances, and the interactive HTML report are under
`Results/SP500/macro_sp500`.
