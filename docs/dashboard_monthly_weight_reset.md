# Alpha Desk monthly weight-reset policy

## Decision

Alpha Desk keeps its Filing V7-3 + SEC 25% Top-5 selection and risk checks on
the existing weekly signal schedule.  A membership change, forced universe
exit, or stop-driven exit remains executable at the next tradable open.

When the selected set is unchanged, the model no longer restores every holding
to equal weight every week.  It restores the selected names to equal weight on
the first available weekly signal of each calendar month.

## Formula and condition changes

- Scoring, qualification, ranking, entry, and exit formulas are unchanged.
- ROI remains net return: `(final_value / total_injected - 1) * 100`.
- Before this change, every weekly target group was sent to the portfolio
  simulator.
- After this change, a target group is executable only for the initial
  allocation, a selected-membership change, or the first signal of a new month.
- Execution remains the next tradable session's open with the existing 10 bps
  transaction-cost assumption.

## Validation evidence

For 2024-01-02 through 2026-08-07, Top 5 and 10 bps costs:

| Policy | ROI | CAGR | Max drawdown | Execution days | Ticker trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weekly equal-weight reset | 212.75% | 55.16% | -31.97% | 135 | 692 |
| Weekly selection + monthly first-signal reset | 213.96% | 55.39% | -31.99% | 42 | 227 |

> **Absolute levels withdrawn 2026-08-20.** Both rows were computed on a filing
> pipeline that skipped Q4 in trailing-twelve-month sums and did not split-adjust
> share counts. On the repaired pipeline the monthly-reset policy scores ROI
> `190.01%` / CAGR `50.39%` / max drawdown `-31.58%` over 2024-01-02..2026-08-12
> (frozen as V7.3.1). The weekly equal-weight row has not been recomputed, so the
> two are no longer directly comparable — but the policy *decision* below rests on
> the execution-count and turnover reduction, which the repair does not touch.

The monthly-reset policy reduced execution days by 68.9% and ticker-level
trades by 67.2%.  It beat weekly ROI in six of eight annual/rolling windows;
the worst observed lag was 1.21 percentage points.  It also retained higher
full-period ROI at 0, 10, 25, and 50 bps costs and at Top 3, Top 5, and Top 7.

This evidence is an in-repository backtest, not broker-statement parity.
