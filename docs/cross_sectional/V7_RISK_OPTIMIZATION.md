# V7-3 risk optimization

This experiment leaves V7-3 stock selection, financial formulas, technical
factors, exit conditions, and next-session-open execution unchanged. It only
changes position weights and total long exposure after the existing signal
functions have selected the five names.

## Allocation

For each selected name, the unnormalized allocation is:

`exp(score_strength * selected-name AlphaScore z-score) *
annualized_volatility ** (-inverse_volatility_strength)`

The optional concentration control limits a name to 30% and a mapped current
GICS sector to 40% of gross long exposure. If a five-name selection cannot
satisfy those caps, the smallest feasible cap is recorded and used. Missing
sector classifications are treated as ticker-specific unmapped groups.

## Exposure

The covariance estimate uses the trailing 63 sessions through the signal
close, requires 42 observations, and shrinks the sample covariance 50% toward
its diagonal. Desired gross exposure is target volatility divided by forecast
portfolio volatility and is capped by the candidate maximum. If SPY closes
below its trailing 200-session average, candidates with a risk-off gate reduce
gross exposure to their configured cap. Trades still execute at the next
available session open.

Negative cash is charged the configured annual funding rate by calendar day.
Reported ROI remains net return:

`(final_value / total_injected - 1) * 100`

The fine search also evaluates fixed gross exposure from 1.00x through 1.60x.
This is modeled notional leverage, not a specific leveraged ETF. The expanded
search deliberately permits a training drawdown as low as -55% instead of the
earlier -45% research constraint; this change is explicit because a 35% CAGR
target cannot be evaluated honestly without exposing the associated drawdown.

## Input and baseline parity gates

The optimizer requires a membership history with at least 50 distinct change
snapshots. The eight annual snapshots in `sp500_membership.csv` are rejected;
the run must use `sp500_membership_changes.csv` so mid-year additions and
deletions affect the first eligible weekly signal.

Before any candidate is evaluated, the unlevered equal-weight baseline must
reproduce the frozen continuous V7-3 result within the configured tolerance:

- ROI: 344.0391775367818%
- CAGR: 25.466975056825515%
- maximum drawdown: -36.47016472092517%
- Sharpe: 1.0292186650376662

If parity fails, the optimizer aborts instead of ranking candidates against a
different strategy path.

## Selection and interpretation

Candidates are selected only with 2020-2024 and the configured training folds.
The robust score combines training CAGR, median fold CAGR, fold dispersion,
Sharpe, drawdown, and turnover. A candidate must also pass the declared
drawdown, positive-fold, worst-fold, and no-ruin constraints.

The 2025 and 2026 periods have already been observed and are report-only, not
fresh validation. Macrotrends financial history is restated rather than true
point-in-time data, the 575-name panel omits unavailable acquired or delisted
constituents, and the available sector map is current rather than historical.
The output must therefore remain labelled `POST_HOC_EXPERIMENT`.

Every candidate's full-period metrics are written with `ReportOnly` in the
column name. A separate 35% target diagnostic chooses the lowest-drawdown
candidate that reaches 35% over the already-observed full period. It is never
used as the training-selected candidate and is explicitly labelled `POST_HOC`.
