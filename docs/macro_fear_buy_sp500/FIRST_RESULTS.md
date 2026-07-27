# First strict-OOS results

Run date: 2026-07-25

Prediction history: 2007-01-03 through 2026-07-23. The prediction inputs are
the previously generated strict-OOS annual walk-forward predictions from the
`macro_momentum_sp500` workflow.

## Selected parameters

The development-only grid selected:

- 80% initial core and two 10% tactical tranches;
- mild fear score 0.50;
- fear score 0.70;
- euphoria trim score 0.60;
- 189-session minimum hold;
- 5% signal-reference profit buffer.

The selection cutoff was 2016-12-30. The 2017-01-03 onward period was not used
to rank or select candidates.

## Portfolio comparison

| Period | Portfolio | CAGR | MDD | Sharpe | Average exposure |
|---|---:|---:|---:|---:|---:|
| Full OOS | Macro Fear Buy | 10.86% | -53.08% | 0.646 | 93.30% |
| Full OOS | Buy & Hold | 10.79% | -55.19% | 0.623 | 100.00% |
| 2017+ holdout | Macro Fear Buy | 15.23% | -33.66% | 0.891 | 95.62% |
| 2017+ holdout | Buy & Hold | 15.02% | -33.72% | 0.861 | 100.00% |

Full-period ending values from one initial $100,000 injection were
$749,933 for Macro Fear Buy and $741,260 for Buy & Hold.

This is a narrow observed edge, not strong evidence of persistent alpha. The
21-session block bootstrap estimated:

- full-OOS annualized arithmetic excess return: -0.11 percentage points;
- probability of positive full-OOS excess: 37.3%;
- 2017+ annualized arithmetic excess return: +0.08 percentage points;
- probability of positive 2017+ excess: 62.0%.

The compounded CAGR can be slightly better while average arithmetic daily
excess is slightly worse because the strategy changes exposure and return
sequence.

## Actual behavior

The first run executed:

- 12 tactical buys;
- 10 tactical trims;
- zero realized tactical loss trims;
- all tactical trims left the core sleeve untouched.

It bought in the 2008, 2011, 2015–2016, 2018, 2022, 2023, and 2025 fear
episodes. It was already fully allocated after the late-2018 fear buys, so it
did not have spare tactical cash to deploy during the March 2020 crash.

That 2020 behavior is the main unresolved design issue. A post-hoc
maximum-holding-period experiment rebuilt cash before 2020 and reduced the
holdout drawdown, but the holdout had already been inspected. It was therefore
not promoted into the production rule or described as untouched evidence.

## Interpretation

The directional correction worked: fear now increases exposure and euphoria can
reduce only tactical exposure. It also removed the old behavior of selling
during panic and buying back after recovery.

The first result does not prove that the strategy will beat Buy & Hold. Its most
credible benefit so far is disciplined reserve deployment with a modestly
better observed drawdown and Sharpe ratio. More independent data or a later
forward period is required before treating the return edge as reliable.
