# Two-times Buy & Hold optimization audit

Run date: 2026-07-25

## Objective and guardrails

The requested target was interpreted as:

`strategy net profit / Monthly Buy & Hold net profit >= 2.0`.

The audit kept the following rules:

- initial capital: $40,000;
- $4,000 enters cash on the first available session of each later month;
- no leverage and no short selling;
- signals use close information and execute no earlier than the next open;
- 5 bps fee and 5 bps slippage;
- parameter selection uses data through 2016-12-30 only;
- 2017-01-03 onward is not used to select a parameter;
- ROI remains `(final value / total injected - 1) * 100`.

For the full 2007-2026 history, Buy & Hold net profit was $3,571,480.
The requested two-times-profit threshold therefore requires:

- net profit: at least $7,142,961;
- final value: at least $8,118,961.

## Signal reliability

Escalation events were counted only when weekly fear severity increased.
The small and regime-dependent samples do not support calling the signal
high-confidence.

| Level | Development events | Development 126d mean | Holdout events | Holdout 126d mean |
|---|---:|---:|---:|---:|
| MILD_FEAR | 15 | 5.41% | 17 | 12.23% |
| FEAR | 9 | 5.14% | 12 | 10.21% |
| PANIC | 3 | -13.08% | 4 | 34.24% |

The PANIC result changes sign across the two periods and has only seven total
escalation events. FEAR also changes from a 44% positive 126-day rate in
development to 92% in holdout. These differences are too large for a stable
position-size estimate.

The separate strict-OOS macro model has modest drawdown ranking power:

| Horizon | Development AUC | Holdout AUC |
|---|---:|---:|
| 21 sessions | 0.656 | 0.611 |
| 63 sessions | 0.668 | 0.631 |
| 126 sessions | 0.582 | 0.600 |

Its expected-return prediction is not stable. The 126-session Spearman
correlation was -0.572 in development and +0.201 in holdout. A fixed direction
or threshold therefore cannot be justified from both periods.

## Searches performed

More than 2,400 documented long-only combinations were checked across:

- mild/fear/panic cash-deployment fractions and deployment cooldowns;
- VIX percentiles, fear-score thresholds, panic thresholds, holding periods,
  profit buffers, and euphoria exits;
- fear plus five-day price and VIX-reversal confirmation;
- existing stateful macro-risk parameters;
- 100/150/200/250/300-session moving-average defense;
- rolling risk-percentile plus trend defense.

The production policy optimizer evaluated 108 contribution policies with the
signal parameters frozen before looking at holdout.

## Reproducible selected policy

Development-period net profit selected:

- MILD_FEAR deployment: 0%;
- FEAR deployment: 100%;
- PANIC deployment: 100%;
- contribution-deployment cooldown: 21 sessions.

| Period | Strategy final value | Buy & Hold | Profit ratio | Strategy MDD | Buy & Hold MDD |
|---|---:|---:|---:|---:|---:|
| Development | $910,803 | $925,517 | 0.964 | -49.75% | -55.08% |
| Untouched holdout | $1,155,279 | $1,135,309 | 1.031 | -27.23% | -33.73% |
| Full diagnostic | $4,570,383 | $4,547,480 | 1.006 | -49.75% | -55.08% |

The ending-value improvement on holdout is caused by money-weighted deposit
timing. A 21-session block bootstrap of flow-adjusted daily excess return gave:

| Period | Annualized excess estimate | 5%-95% interval | Probability positive |
|---|---:|---:|---:|
| Full | 0.173% | -0.785% to 1.234% | 59.9% |
| Holdout | -0.629% | -2.422% to 1.356% | 28.5% |

The interval crosses zero widely. The selected policy is therefore a useful
drawdown-reduction candidate, not a statistically established alpha signal.

## Hindsight upper bound

As a diagnostic ceiling, every monthly contribution was allowed to know its
exact lowest future SPY price before buying. This impossible oracle excludes
fees and never mistimes a contribution.

| Maximum wait | Full-history profit versus Buy & Hold | Holdout profit versus Buy & Hold |
|---|---:|---:|
| 3 months | 1.08x | 1.08x |
| 12 months | 1.19x | 1.16x |
| Unlimited | 1.27x | 1.19x |

Even perfect future knowledge of contribution purchase dates cannot reach 2x
profit. Contribution timing alone is therefore mathematically insufficient.

## Conclusion

No honest, untouched-holdout parameter set reached two times Buy & Hold profit.
The best reproducible contribution policy reached 1.031x holdout profit and
1.006x on the full diagnostic history while reducing drawdown.

A reported 2x result under these constraints would require at least one of:

- selecting parameters after observing holdout;
- using future information;
- ignoring costs or external cash flows;
- leverage, short exposure, derivatives, or a broader return source.

Those changes were intentionally not made. The next valid research step is to
add independent point-in-time predictors and require their benefit to repeat
across multiple crisis and non-crisis regimes before considering leverage.
