# V7-3 daily-reset 2X product overlay

## Purpose

This experiment tests whether a cash-funded daily-reset 2X product sleeve can
raise the V7-3 return while limiting drawdown with a causal risk regime. It does
not use a margin balance, negative cash, or a personal loan.

The portfolio always satisfies:

```text
V7 1X weight + 2X product weight + cash weight = 100% of capital
```

Its approximate gross market exposure is:

```text
V7 1X weight + 2 * product weight
```

For example, 50% in V7 1X and 50% in a 2X product is 150% gross exposure, but
the cash paid at purchase is still only 100% of account equity.

## Two product models must not be confused

- `SYNTHETIC_SPY_2X` is an implementable *concept*: retain part of V7-3 and buy
  a daily-reset S&P 500 2X product with the rest. The backtest is still a
  synthetic fee-and-financing proxy, not an actual ETF price history.
- `SYNTHETIC_V7_2X` applies daily 2X returns to the changing V7-3 basket. This
  is a hypothetical research wrapper. The project does not assume that one
  exchange-traded product exists for this rotating basket.

Consequently, a high-return V7 2X result is a strategy-design target, not an
immediately tradable ETF result. The SPY 2X result is the more realistic product
comparison.

## Daily-reset return and costs

For each trading day, the synthetic 2X product return is:

```text
2 * underlying daily return
- (expense ratio + point-in-time Fed funds rate + financing spread) / 252
```

The default assumptions are a 0.95% annual expense ratio, Fed funds plus a 1%
financing spread, and 10 bps whenever capital switches among V7, the product,
and cash. Daily resetting means a multi-day return is not two times the
underlying multi-day return. Volatility drag can be substantial.

## Causal risk regime

The prior close determines the next session's allocation. The grid varies:

- SPY price above its 100-, 150-, 200-, or 250-day moving average;
- an optional clearance buffer above that moving average in refined runs;
- optional positive 63-day moving-average slope;
- VIX ceilings of 20, 25, 30, or effectively disabled;
- V7 drawdown floors of -6%, -10%, -15%, or effectively disabled;
- risk-on 2X product weights from 25% through 100%;
- risk-off 2X weights of 0%, 10%, or 20%;
- risk-off cash weights from 0% through 100%.

The cash dimension is essential for testing a lower MDD. Turning off only the
2X sleeve returns the account to V7 1X and cannot reliably reduce the original
V7 drawdown.

## Selection and interpretation

The primary selection uses only the 2020-2024 training interval and three
training folds. Full-period labels such as `TARGET_40_45_POST_HOC` and
`BEST_MDD_AT_35_POST_HOC` deliberately use the already observed 2025-2026
period and are research diagnostics, not fresh validation.

All candidates use the same simulation and signal functions. The run aborts if
the no-overlay candidate does not reproduce the accepted V7-3 risk-weighted
baseline within the configured tolerance.

### Annual-consistency selection

`v7_3_product_overlay_annual_consistency.json` adds a training-only objective
for investors who care about the distribution of individual calendar-year
returns rather than full-period CAGR alone. Every candidate records:

- the worst, mean, and standard deviation of 2020-2024 calendar returns;
- the RMS shortfall below 35% and excess above 45%;
- the count of years inside or above the target band; and
- the worst trailing 252-session return whose ending date is in training.

Selection is deliberately worst-year first. It sorts candidates by whether all
training years meet the floor, then worst calendar return, target shortfall,
calendar-return standard deviation, and finally the configured composite
score. Observed 2025 and partial 2026 returns are report-only and never enter
this selection rule.

The manifest field `annual_target_feasible_on_training_candidates` must be
checked before interpreting the selected row. A false value means the search
space did not contain a model meeting the requested annual floor; the closest
candidate is an honest feasibility diagnostic, not a claim that the target was
achieved or can be guaranteed.

### SPY-relative selection

The optional `benchmark_consistency` block changes the primary question from
"is CAGR high?" to "did the candidate beat SPY in every training calendar
year?" Candidate diagnostics include annual percentage-point alpha, the worst
and mean annual alpha, RMS benchmark shortfall, and the worst trailing
252-session alpha. SPY and the strategy both start with a zero return on the
first simulation observation so the comparison uses the same capital start.

The refined SPY-relative configuration first removes candidates whose training
MDD is worse than SPY by the configured allowance. It then prioritizes:

1. beating SPY in every 2020-2024 calendar year;
2. the worst annual percentage-point alpha;
3. the worst absolute calendar return; and
4. benchmark shortfall, MDD, rolling alpha, and alpha dispersion.

Calendar-year dominance does not imply dominance over every possible holding
window. `TrainBenchmarkRollingWorstAlpha` must be inspected separately; a
negative value means at least one trailing 252-session window lagged SPY.

Important remaining limitations are actual product tracking error and spreads,
taxes, delisting and liquidity constraints, the lack of an exchange-traded V7
2X wrapper, and restated rather than true point-in-time financial history.

## Run

Use `scripts/cross_sectional/run_v7_product_overlay.py` with the accepted V7-3
equity CSV plus SPY, VIX, and Fed-funds inputs. Outputs are written atomically
to the sibling `Results/Cross_Sectional/v7_product_overlay` directory unless an
explicit output directory is supplied.
