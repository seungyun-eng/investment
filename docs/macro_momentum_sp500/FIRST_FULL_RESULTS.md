# First full macro + momentum results

Run date: 2026-07-25  
Strict OOS period: 2007-01-03 through 2026-07-23  
Adjusted SPY observations: 8,427 total; 4,919 OOS

These results are intentionally reported even though they are negative. No
allocation threshold or prediction was inverted after observing OOS outcomes.

## Predictive results

| Model design | 126d risk AUC | 126d risk Brier | 126d return Spearman | 126d quintile spread |
|---|---:|---:|---:|---:|
| Rolling 10y, daily training | 0.3866 | 0.3518 | -0.0548 | -2.46% |
| Expanding history, every 5th training day | 0.5119 | 0.2481 | -0.2576 | -8.86% |

The rolling model failed outright. Its lowest predicted-risk decile experienced
a 42.7% realized event rate, while its highest decile experienced only 7.5%.
The expanding/weekly challenger fixed the extreme inversion but remained poorly
calibrated and barely exceeded random discrimination.

The return regressions did not demonstrate a stable edge. The challenger was
particularly unstable across regimes: 126-day Spearman was negative from
2007–2011, improved in several post-2012 years, and was -0.258 in aggregate.

## Portfolio results

All portfolios start with 100,000, use the same 2007–2026 dates, and apply the
configured costs where applicable.

| Portfolio | CAGR | MDD | Sharpe | Average exposure |
|---|---:|---:|---:|---:|
| Rolling/daily MacroMomentum | 9.47% | -55.19% | 0.567 | 94.73% |
| Expanding/weekly MacroMomentum | 10.64% | -50.76% | 0.633 | 97.80% |
| Buy & Hold | 10.79% | -55.19% | 0.623 | 100.00% |
| Static 70% SPY | 8.39% | -41.32% | 0.656 | 71.37% |
| Static 76% SPY | 9.00% | -44.24% | 0.654 | 77.81% |
| Prior V2 strict-OOS, rebased | 8.98% | -49.05% | 0.611 | 77.08% |

The expanding/weekly model reduced MDD by 4.43 percentage points and slightly
improved Sharpe, but did not match Buy & Hold CAGR. Its 21-day block-bootstrap
annualized excess-return estimate was -0.28%, with a 90% interval of -1.71% to
+1.25% and only a 37.3% probability of positive excess return.

The most favorable sensitivity row used a 0.70 risk threshold, 25% defensive
weight, and zero costs. It produced 10.83% CAGR, only 0.04 percentage point
above Buy & Hold. At configured costs it fell to 10.77%. This is a post-hoc
sensitivity result and not evidence of an edge.

## What the experiment learned

1. Coincident stress is not the same as advance warning. The rolling model
   assigned near-zero average risk in 2007 and 2008, then high risk after the
   crisis. It reacted to the state rather than forecasting its onset.
2. Ten years is too little history for a rare-event model. Expanding training
   improved risk AUC from 0.387 to 0.512 and reduced turnover, but not enough.
3. Daily overlapping 126-day labels greatly overstate the effective sample
   size. Weekly subsampling improved stability but did not create return
   predictability.
4. The most recurrent risk features were long realized/downside volatility,
   Fed Funds change, BAA-10y spread changes, high-yield effective yield, and
   rates. Their importance varied sharply by fold, so they are diagnostic, not
   a stable formula.
5. Allocation cannot rescue a weak signal. The rolling strategy traded 79
   times with turnover equal to 122.8 times initial capital, underperformed,
   and did not reduce MDD. More threshold search would be curve-fitting.

## Decision

Neither model is suitable for deployment. The expanding/weekly version is the
better research baseline because it retains rare crises and reduces duplicated
labels, but it remains a no-trade result.

The next scientifically defensible iteration is not another allocation grid.
It should split the problem into:

- a low-frequency crisis-onset model evaluated only while SPY is near its
  trailing high;
- a separate stress/recovery model for contrarian entry after a drawdown;
- regime-aware or expanding training with calibrated probabilities;
- event-level evaluation so one six-month episode does not count as roughly
  126 independent successes.

Any next model should be rejected unless risk AUC is consistently above 0.55,
calibration is monotonic, 126-day return quintile spread is positive across
most folds, and cost-adjusted bootstrap excess return is credible.

## Data limitations

- Monthly release lags are conservative approximations, not true vintage
  ALFRED data. Revisions can still create residual point-in-time bias.
- Official broad HYOAS history returned by the current FRED endpoint begins in
  2023. The long-history file named `HY_Spread.csv` is actually
  `BAMLH0A3HYCEY` effective yield and is labelled `HYYield` in this workflow.
- The prior V2 row loads its previously generated strict-OOS daily file and
  rebases it to the comparison start; V2 was not re-optimized in this run.
- This is a new SPY research workflow. Outputs were not compared for parity
  with the legacy notebooks, and no parity claim is made.

## Stateful macro follow-up

Run timestamp: `20260725_143133_110892`  
Report timestamp: `20260725_143133_593563`

This follow-up kept the same strict OOS dates and added an explicit,
point-in-time macro stress composite. The composite separates level and trend
stress across volatility, credit, NFCI financial conditions, labor, and the
yield curve. The primary allocation is a pre-specified state machine with
weekly confirmation, entry/exit hysteresis, a 20-session minimum hold, a
10-session recovery period, a loss-sale gate, and macro-plus-momentum recovery.
The expected-return regression is retained as a diagnostic but is not allowed
to trade the stateful portfolio.

### Updated predictive and portfolio results

The added macro features improved drawdown-event discrimination, but the return
model remained unusable:

| Horizon | Risk AUC | Risk Brier | Return Spearman | Return quintile spread |
|---|---:|---:|---:|---:|
| 21 days | 0.6447 | 0.0719 | 0.0003 | -0.18% |
| 63 days | 0.6541 | 0.1585 | -0.0687 | -2.17% |
| 126 days | 0.5931 | 0.2244 | -0.2269 | -8.26% |
| 252 days | n/a | n/a | -0.2631 | -11.95% |

| Portfolio | Final value | CAGR | MDD | Sharpe | Average exposure | Trades |
|---|---:|---:|---:|---:|---:|---:|
| StatefulMacro, two confirmations | 620,815.51 | 9.79% | -32.24% | 0.732 | 84.09% | 34 |
| Enhanced stateless MacroMomentum | 734,497.38 | 10.74% | -50.76% | 0.637 | 97.90% | 14 |
| Buy & Hold | 741,260.49 | 10.79% | -55.19% | 0.623 | 100.00% | 1 |
| Three-confirmation sensitivity candidate | 808,700.87 | 11.28% | -33.72% | 0.777 | 88.81% | 20 |

The primary stateful rule materially reduced drawdown and increased Sharpe, but
gave up 1.00 percentage point of CAGR versus Buy & Hold. Its 21-day block
bootstrap estimated -1.82% annualized arithmetic excess return, with a 90%
interval of -4.76% to +1.35% and only a 16.6% probability of positive excess
return.

The three-confirmation row is an exploratory sensitivity result, not a new
strict-OOS result. It was selected after observing the full OOS table. A
2007-2016/2017-2026 audit showed why it must not be promoted:

| Period | Three-confirmation CAGR | Buy & Hold CAGR | Three-confirmation MDD | Buy & Hold MDD |
|---|---:|---:|---:|---:|
| 2007-2016 discovery half | 8.79% | 6.90% | -20.45% | -55.19% |
| 2017-2026 validation half | 13.97% | 15.02% | -33.72% | -33.72% |

The candidate's full-period arithmetic excess-return bootstrap was still
negative (-0.30%) with a 41.95% probability of being positive. In the validation
half, the estimate was -1.13% and the probability of a positive value was only
14.45%.

### Trade-timing diagnosis

The state machine fixed the original rapid-whipsaw symptom:

- Stateless paired cycles had a 7.5-day median duration; 6 of 10 were no longer
  than 10 calendar days.
- Stateful paired cycles had a 219.5-day median duration; none of 22 were no
  longer than 20 calendar days.

It did not eliminate economically adverse timing:

- 4 of 11 normal-to-defensive cycles sold below the preceding return-to-normal
  execution price.
- 9 of 11 defensive-to-normal cycles repurchased above the preceding defensive
  sale price.
- In 2020, the strategy reduced exposure at 219.73 after the crash had already
  begun and returned to full exposure at 294.96, 34.24% higher. The strong
  macro override correctly bypassed the weak-loss-sale gate, but the macro data
  were coincident or lagging rather than advance warning.
- The same mechanism helped during the long 2007-2009 and 2022 risk regimes,
  showing the trade-off: stronger confirmation avoids quick churn but can miss
  fast V-shaped recoveries.

The paired-cycle adverse flag is a state-transition diagnostic, not tax-lot
realized-P&L accounting.

### Follow-up decision

The stateful rule is useful as a drawdown-control experiment, not as evidence of
a deployable excess-return strategy. More threshold search on these same OOS
years would be curve fitting. The next defensible model should separate
crisis-onset detection from post-drawdown recovery and lock all allocation
rules before evaluating a new untouched period or different broad-market
sample.
