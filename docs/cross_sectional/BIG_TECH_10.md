# Big Tech 10 financial-technical experiment

## Universe

The fixed universe contains GOOGL, META, AMZN, TSLA, NVDA, AAPL, NFLX,
MSFT, AVGO, and AMD. All ten have local price history before 2020 and local
quarterly financial workbooks. This is a hindsight-selected current-winner
universe, not a point-in-time reconstruction of a published Big Tech index.

## Signals and execution

The experiment reuses the existing cross-sectional financial growth, quality,
momentum, and risk-control factors. Its technical trend component is the V7-3
average of MA, normalized MACD, and 21/63-session OBV flow. The optimizer varies
the factor weights, top-k count, rank exits, and entry thresholds while keeping
the standard weekly close signal and next-session-open execution. Transaction
cost is 10 basis points.

Candidate selection uses only 2020-2024 and the three declared training folds.
The 2025 and partial 2026 results are report-only. Outputs also compare weekly
equal weight, equal-weight buy-and-hold, each individual stock, and SPY.

## Important limitations

- Fixing these ten successful companies with hindsight creates major
  survivorship and winner-selection bias.
- Quarterly financial history is current restated history with an assumed
  45-day release lag, not true point-in-time statements.
- Local prices are split-normalized closes without dividend reinvestment.
- Two thousand parameter candidates create material multiple-testing risk.
- A high backtest CAGR is not a claim of repeatable future performance.

The experiment is therefore labelled `POST_HOC_EXPERIMENT` and must not be
treated as production-ready until it passes winner-exclusion, walk-forward,
and actual point-in-time data tests.
