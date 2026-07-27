# Methodology

## Signal timing

All feature values are measured at a session close. A target change executes at
the next available session open. No future price or macro observation is used by
the signal function.

## Fear score

The production fear score is a weighted average of:

- trailing five-year VIX percentile: 30%;
- point-in-time macro confirmation: 20%;
- trailing percentile of the strict-OOS 63/126-session drawdown-risk model: 20%;
- current 252-session price drawdown intensity: 20%;
- negative 63-session momentum intensity: 10%.

VIX and model-risk percentiles require 252 observations before trading. The
five-year rolling window makes `high VIX` relative to the market regime instead
of relying on a permanent VIX 25 threshold.

## Buying

The system reviews signals on the final trading session of each `W-FRI` week.
It adds at most one 10-percentage-point tactical tranche per review:

- mild fear: VIX percentile at least 80% with score confirmation, or drawdown
  at least 8%;
- fear: VIX percentile at least 90% with stronger confirmation, or drawdown
  at least 15%;
- panic: VIX percentile at least 97% with stronger confirmation, or drawdown
  at least 25%.

Deeper fear can therefore deploy more capital, while a sudden first signal
cannot spend the entire reserve at one price.

## Trimming

Fear relief alone is not a sell signal. Tactical exposure is reduced by one
tranche only when all conditions hold:

- the minimum holding period has elapsed since the latest tactical buy;
- fear and VIX percentile have returned below their exit ceilings;
- the independent euphoria score is high;
- SPY is above the tactical signal-reference cost by the configured profit
  buffer.

The core sleeve is never sold.

## Optimization

Optimization and final simulation both call
`generate_fear_buy_signals` and `run_signal_backtest`. Candidate ranking uses
development-period CAGR, with Sharpe as the tie-breaker. The 2017+ holdout never
participates in selection.

## Accounting

Transaction costs and slippage are applied at execution. Idle cash earns the
point-in-time cash rate. ROI is:

`(final value / total injected - 1) * 100`.
