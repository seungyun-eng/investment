# SPY tactical-defense baseline reconciliation, cost, and OOS check

Date: 2026-08-14

Status: **research-only; do not promote**

This note reconciles the two reported V1 baselines and validates the scratch
quick-recovery rule without changing `stateful_macro_target_weights`, the
frozen model, or any production strategy code. The reproducible analysis is
`scripts/macro_momentum_sp500/analysis/spy_tactical_defense_validation.py`.

## Executive conclusion

1. The V1 discrepancy is fully explained. The 185.9% run slices the SPY data
   at 2020 first and then calculates MA150/RSI. The 195.6% run calculates the
   indicators on the full pre-2020 history and slices the evaluation window
   afterward. Missing the first 149 sessions of MA150 in the cold-start run
   misses the COVID defensive episode. This is not rounding.
2. Under the scratch script's original close-to-close convention,
   quick-recovery still beats V1 after 5 bp commission/fee plus 5-10 bp
   slippage. However, that convention is not execution-safe: some decisions
   using today's close receive today's entire close-to-close return.
3. Under a causal close-signal -> next adjusted-open execution simulation with
   the repository-standard 5 bp fee plus 5 bp slippage, quick-recovery loses to
   V1 in 2020-2026: 174.3% vs 185.9% total return and 1.015 vs 1.057 Sharpe.
4. The `-3%` depth and `3-day` window were selected by inspecting the same
   2020-2026 period used for the headline evaluation. There is no untouched
   forward holdout. In a historical 2018-2019 holdout using the project's
   established discovery-through-2017 split, quick-recovery also loses to V1.
5. Therefore checks 1-3 do **not** collectively hold. No integration sketch
   for `stateful_macro_target_weights` was written, and promotion should stop.

## 1. Exact baseline reconciliation

### Reproduction

Both rows below use the scratch script's exact state machine and its original
close-to-close weight convention. The only changed variable is when the
indicators are calculated.

| Indicator construction | Total return | CAGR | MDD | Sharpe | Sells |
|---|---:|---:|---:|---:|---:|
| Slice at 2020, then calculate indicators | 185.850% | 17.317% | -33.717% | 0.976 | 9 |
| Calculate on full history, then slice at 2020 | 195.634% | 17.919% | -27.224% | 1.080 | 10 |

The second row exactly reproduces the earlier ad-hoc result
`195.6% / 17.9% / -27.2% / 1.08`. The first row exactly reproduces the current
scratch script and handoff result `185.9% / 17.3% / -33.7% / 0.98`.

### Exact code divergence

In `spy_tactical_defense_v1_quickrecovery.py`:

- `load_spy()` filters to `Date >= 2020-01-01` first.
- `compute_indicators()` subsequently calls `rolling(150)` on that shortened
  frame.
- MA150 is consequently unavailable for approximately the first 149 trading
  sessions of 2020. The strategy remains invested during that interval and
  cannot react to the COVID decline.

Moving indicator construction before the evaluation slice adds one defensive
episode, changes sell count from 9 to 10, and changes MDD by 6.49 percentage
points. No strategy threshold needs to change to reproduce the earlier result.

### `reactive_rules.py` and `actual_trades.py` are not the V1 baseline

The two earlier files are the conceptual origin of the idea, but their
backtest is materially different from the V1 scratch state machine.

| Behavior | `reactive_rules.py` / `actual_trades.py` | V1 scratch script |
|---|---|---|
| Return stream | Frozen model equity return | SPY return |
| Exit test | Every day SPY is below MA150 | Three consecutive confirmed days |
| Re-entry | Every day SPY is no longer below MA150 | RSI tiers plus three-day MA fallback |
| Execution | Signal shifted one session | Mixed; see timing issue below |
| While defensive | Cash yield in `reactive_rules.py` | Zero cash yield |
| Stateful tiers | None | 30% at RSI<35, 80% at RSI<30 |

Thus the 195.6% baseline does not come from matching `reactive_rules.py` more
closely. It comes solely from giving the V1 state machine a valid pre-2020
indicator warm-up.

### Additional scratch semantics found line-by-line

- The docstring says all triggers use day `i-1`, but quick-recovery compares
  the current close `px` with the sell price, and capitulation also uses the
  current close.
- The new weight is stored in `weight[i]`, then multiplied by the full return
  from close `i-1` to close `i`. A decision made with close `i` therefore
  receives or avoids a return that was already realized. This is a timing
  inconsistency, not merely a conservative one-day delay.
- The sell decision uses prior closes, but records `sell_px = close[i]`.
  Quick-recovery may then buy on any of days 1, 2, or 3 after that sell if the
  current close is above `sell_px`; it does not specifically wait for the
  third-day close.
- `below_streak` is not reset on re-entry. This is part of the reproduced
  scratch behavior and can make a new below-MA exit arrive quickly.
- `use_fb5` is not read by the state machine. Only a non-zero `fb_pct` changes
  the fallback threshold.

## 2. Transaction-cost check

### Assumption

The macro-momentum configuration and portfolio engine use:

- transaction fee/cost: 5 bp one way;
- slippage: 5 bp one way;
- stress case here: 5 bp fee plus 10 bp slippage one way;
- fixed dollar commission: $0, because no per-ticket commission model was
  found in this research line.

The first table deliberately preserves the original scratch convention so it
answers the direct question about the reported 9 versus 17 sells. Cost is
applied to absolute portfolio-weight turnover.

| Original scratch convention | Total return | CAGR | MDD | Sharpe | Sells | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| V1, no cost | 185.85% | 17.32% | -33.72% | 0.976 | 9 | 19.0x |
| Quick, no cost | 208.59% | 18.69% | -33.72% | 1.035 | 17 | 35.0x |
| V1, 5+5 bp | 180.48% | 16.98% | -33.72% | 0.960 | 9 | 19.0x |
| Quick, 5+5 bp | 198.00% | 18.06% | -33.72% | 1.006 | 17 | 35.0x |
| V1, 5+10 bp | 177.83% | 16.81% | -33.72% | 0.952 | 9 | 19.0x |
| Quick, 5+10 bp | 192.85% | 17.75% | -33.72% | 0.991 | 17 | 35.0x |

On this convention, quick-recovery remains above V1 and buy-and-hold after
costs. The two rejected rules remain rejected: at 5+5 bp, the 5%-fallback
variant returns 175.76% with 0.945 Sharpe, and the capitulation variant returns
133.96% with 0.827 Sharpe.

That result is not sufficient for promotion because cost did not repair the
timing inconsistency. A second simulation therefore executes every close-based
decision at the next adjusted open and debits fees and slippage from cash.

| Causal next-open, 5+5 bp, 2020-2026 | Total return | CAGR | MDD | Sharpe | Sells |
|---|---:|---:|---:|---:|---:|
| V1 | 185.93% | 17.32% | -25.32% | 1.057 | 10 |
| V1 + quick-recovery | 174.28% | 16.58% | -25.32% | 1.015 | 14 |
| 5%-fallback rejected | 121.71% | 12.87% | -25.32% | 0.824 | 41 |
| Capitulation rejected | 109.80% | 11.93% | -25.61% | 0.794 | 45 |
| Buy and hold | 153.05% | 15.16% | -33.72% | 0.801 | 0 |

Quick-recovery still beats buy-and-hold on total return and Sharpe, but it no
longer beats V1. Its claimed incremental edge is therefore sensitive to the
execution convention.

## 3. Parameter tuning and out-of-sample check

The handoff explicitly states that the depth threshold, recovery window, and
related variants were tuned interactively while inspecting the 2020-2026
result. Therefore the headline window is in-sample. Splitting it after the
fact cannot manufacture a clean forward holdout, especially because 2025 was
one of the inspected cases.

The project's macro early-warning discipline uses discovery through 2017 and
holdout from 2018 onward. Applying that calendar split with the already-fixed
quick parameters gives the following causal, net-of-5+5-bp results:

| Period | Rule | Total return | CAGR | MDD | Sharpe |
|---|---|---:|---:|---:|---:|
| 1994-2017 discovery-style history | V1 | 536.52% | 8.02% | -42.70% | 0.574 |
| 1994-2017 discovery-style history | Quick | 640.61% | 8.71% | -43.02% | 0.613 |
| 2018-2019 historical holdout | V1 | 29.94% | 14.04% | -13.14% | 1.039 |
| 2018-2019 historical holdout | Quick | 27.32% | 12.88% | -14.89% | 0.957 |

The long pre-2018 history shows a modest quick-recovery advantage over V1,
but that advantage reverses in the 2018-2019 holdout. This is also only a
historical robustness test: because the rule was invented later using
2020-2026, it is not a genuine forward deployment record.

For additional diagnosis, independent-start subperiods show:

- 2020-2023: V1 74.90% / 0.882 Sharpe versus Quick 67.78% / 0.820;
- 2024-2026: the two rules are identical at 64.41% / 1.419, but this period is
  contaminated by the interactive tuning and cannot be called OOS.

The incremental quick-recovery edge therefore does **not** survive the
available holdout and causal-execution checks.

## 4. Decision

Do not integrate this rule into `stateful_macro_target_weights` yet. Per the
requested gate, an integration sketch is warranted only if baseline,
transaction-cost, and OOS checks all hold. The cost-only comparison holds, but
causal execution and historical holdout do not.

The 5%-below-MA fallback and 4-day capitulation buy remain rejected. They were
not reintroduced or retuned.

## Reproduction and scope

Run:

```powershell
$env:PYTHONPATH='src'
python scripts/macro_momentum_sp500/analysis/spy_tactical_defense_validation.py
```

Data used: the repository-configured sibling OneDrive macro data file
`SPY Adjusted Historical Data.csv`, through 2026-07-31. No result CSV or model
artifact is written. The script reads paths through `load_paths()` and does
not modify production code.
