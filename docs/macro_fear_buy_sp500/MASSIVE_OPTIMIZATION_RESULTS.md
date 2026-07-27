# 100,000-candidate fear-buy optimization

Run date: 2026-07-26

## Research protocol

- Initial capital: $40,000.
- Monthly contribution: $4,000, accumulated as cash until deployment.
- Transaction fee: 5 bps.
- Slippage: 5 bps.
- Development selection: data through 2016-12-30 only.
- Untouched holdout: 2017-01-03 onward.
- ROI: `(final value / total injected - 1) * 100`.
- Candidate seed: 20260726.
- Exact development simulations: 100,000.
- Multi-fold refinements: 2,196 candidates.
- Development folds: 2007-2011 crisis and 2012-2016 bull regime.
- Final holdout results did not affect parameter selection.

Every candidate used the same point-in-time feature builder, canonical signal
function, and contribution portfolio simulator used by the final frozen
evaluation.

## Frozen winners

The three candidates represent different development-only objectives rather
than three post-hoc holdout winners.

| Strategy | Holdout final | Holdout ROI | Holdout XIRR | Holdout MDD | Holdout profit/BH | Full ROI | Full MDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Return | $1,123,026 | 126.42% | 15.28% | -31.33% | 0.981x | 363.18% | -51.93% |
| Balanced | $1,129,163 | 127.65% | 15.38% | -32.55% | 0.990x | 375.77% | -51.92% |
| Safety | $1,035,318 | 108.73% | 13.79% | -28.42% | 0.844x | 327.14% | -32.82% |
| Monthly Buy & Hold | $1,135,309 | 128.89% | 15.47% | -33.73% | 1.000x | 365.93% | -55.08% |

The Balanced candidate produced 1.027x Buy & Hold profit on the full-history
diagnostic, but only 0.990x on untouched holdout. It is therefore not evidence
of repeatable excess return. The Safety candidate materially reduced full
history MDD, but paid for it with lower exposure and lower profit.

### Return candidate

- Candidate ID: 64247.
- Core allocation: 77.8%; tranche: 15.9 percentage points.
- Fear-score weights: VIX percentile 44.6%, drawdown 23.9%, macro 12.2%,
  downside momentum 10.6%, model risk 8.6%.
- VIX thresholds: 87.7%, 93.0%, 94.0% percentiles.
- Drawdown overrides: -4.0%, -13.7%, -23.1%.
- Accumulated-cash deployment: 75%, 75%, 100%.
- Minimum hold: 63 sessions; profit buffer: 10.5%.

### Balanced candidate

- Candidate ID: 95145.
- Core allocation: 77.3%; tranche: 15.9 percentage points.
- Fear-score weights: macro 54.4%, drawdown 18.4%, VIX percentile 14.2%,
  model risk 7.6%, downside momentum 5.3%.
- VIX thresholds: 88.7%, 94.2%, 98.5% percentiles.
- Drawdown overrides: -4.1%, -21.1%, -26.3%.
- Accumulated-cash deployment: 75%, 75%, 100%.
- Minimum hold: 42 sessions; profit buffer: 6.7%.

### Safety candidate

- Candidate ID: 73044.
- Core allocation: 74.2%; tranche: 16.9 percentage points.
- Fear-score weights: VIX percentile 51.1%, model risk 28.4%, drawdown 10.0%,
  downside momentum 9.5%, macro 1.1%.
- VIX thresholds: 75.5%, 81.2%, 93.3% percentiles.
- Drawdown overrides: -5.7%, -11.4%, -37.9%.
- Accumulated-cash deployment: 25% at every fear level.
- Contribution deployment cooldown: 126 sessions.
- Minimum hold: 42 sessions; profit buffer: 12.2%.

## Holdout uncertainty

A 21-session block bootstrap used flow-adjusted daily excess returns, so monthly
deposits were not counted as investment returns.

| Strategy | Annual excess estimate | 5%-95% interval | Probability excess > 0 |
|---|---:|---:|---:|
| Return | -0.804% | -2.002% to 0.377% | 12.9% |
| Balanced | -0.764% | -1.788% to 0.278% | 11.9% |
| Safety | -2.505% | -4.134% to -0.843% | 0.8% |

None of the three candidates established reliable holdout alpha. Return and
Balanced reduced holdout MDD slightly while nearly matching Buy & Hold profit.
Safety reduced holdout MDD by 5.30 percentage points, but lost 15.6% of Buy &
Hold profit.

## Daily-reset 2x long and -2x short tests

The exact full-instrument variants used synthetic daily-reset prices with:

- +2x or -2x the daily SPY move;
- 0.95% annual product expense;
- point-in-time cash rate plus a 1% financing spread;
- the same 5 bps fee and 5 bps slippage;
- unchanged original SPY signals.

Tactical 2x and conditional -2x hedge rows are flow-adjusted overlay
approximations. They are not ETF execution backtests.

### Untouched holdout

| Strategy | Variant | ROI | XIRR | MDD | Profit/base |
|---|---|---:|---:|---:|---:|
| Return | Base 1x | 126.42% | 15.28% | -31.33% | 1.000x |
| Return | Tactical 2x long | 170.16% | 18.49% | -40.87% | 1.346x |
| Return | Full 2x long | 243.97% | 22.87% | -55.32% | 1.930x |
| Return | Conditional -2x hedge | 125.12% | 15.17% | -31.33% | 0.990x |
| Return | Full -2x short stress | -78.43% | -35.69% | -97.47% | negative |
| Balanced | Base 1x | 127.65% | 15.38% | -32.55% | 1.000x |
| Balanced | Tactical 2x long | 200.08% | 20.40% | -51.89% | 1.567x |
| Balanced | Full 2x long | 243.47% | 22.84% | -57.08% | 1.907x |
| Balanced | Conditional -2x hedge | 126.78% | 15.31% | -32.55% | 0.993x |
| Balanced | Full -2x short stress | -78.63% | -35.96% | -97.41% | negative |
| Safety | Base 1x | 108.73% | 13.79% | -28.42% | 1.000x |
| Safety | Tactical 2x long | 175.29% | 18.83% | -44.85% | 1.612x |
| Safety | Full 2x long | 199.98% | 20.39% | -50.76% | 1.839x |
| Safety | Conditional -2x hedge | 92.97% | 12.35% | -28.42% | 0.855x |
| Safety | Full -2x short stress | -65.11% | -22.70% | -91.27% | negative |

Full 2x long increased holdout profit by 1.84x to 1.93x, but roughly doubled
the economically important drawdown. It did not reach 2x holdout net profit for
any frozen winner after daily reset, financing, expenses, fees, and slippage.

The development-selected conditional -2x hedge used at most 20% of otherwise
unused cash when prior-close model risk and macro stress were high. It did not
improve holdout MDD for any winner and reduced holdout return. The full -2x
short stress test was catastrophic because a long-run rising index and daily
inverse compounding overwhelm the strategy.

## Conclusion

- Best near-Buy-&-Hold compromise: Balanced, but its holdout profit was still
  0.990x Buy & Hold.
- Best safety result: Safety, with holdout MDD -28.42% versus -33.73%, but much
  lower profit.
- Best aggressive result: Return with full 2x long, but holdout MDD worsened
  from -31.33% to -55.32%.
- The tested -2x hedge did not add safety out of sample.
- No unlevered candidate demonstrated stable excess return, and leverage
  amplified exposure rather than improving signal quality.

## Reproducible outputs

The sibling Results folder contains:

- `development_screening_20260726_120459_592450.csv`;
- `development_candidates_20260726_120459_592450.csv`;
- `development_winners_20260726_120459_592450.csv`;
- `frozen_evaluation_20260726_120459_592450.csv`;
- `manifest_20260726_120459_592450.json`;
- `leverage_comparison_20260726_122332_613505.csv`;
- `leverage_manifest_20260726_122332_613505.json`;
- `frozen_bootstrap_20260726_122545_424066.csv`;
- `validation_manifest_20260726_122545_424066.json`.

The leverage study uses a synthetic product rather than historical leveraged
ETF prices. Taxes, borrow availability, margin liquidation, fund tracking
error, bid-ask widening, and investor-specific constraints remain unresolved.
