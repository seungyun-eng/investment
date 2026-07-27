# Monthly contribution scenario

> Historical result: this file describes the earlier target-rebalancing
> interpretation of monthly deposits. The user later clarified that each
> $4,000 deposit must accumulate in cash until a buy signal. See
> `TWO_X_OPTIMIZATION_AUDIT.md` for the corrected rule and latest results.

Run date: 2026-07-25

Assumptions:

- initial lump sum: $40,000;
- $4,000 deposited before the first available open of every later calendar
  month;
- Macro Fear Buy invests toward the current 80%/90%/100% target and leaves the
  remainder in cash;
- Monthly Buy & Hold invests all available cash;
- 5 bps transaction cost and 5 bps slippage;
- deposits are excluded from investment profit.

## Same full OOS history

Period: 2007-01-03 through 2026-07-23.

Total cash injected was $976,000: the initial $40,000 plus 234 monthly deposits.

| Portfolio | Final value | Net profit | ROI | TWR CAGR | XIRR |
|---|---:|---:|---:|---:|---:|
| Macro Fear Buy | $4,628,227 | $3,652,227 | 374.20% | 10.98% | 13.58% |
| Monthly Buy & Hold | $4,547,480 | $3,571,480 | 365.93% | 10.78% | 13.44% |
| Initial 80% contribution target | $3,525,779 | $2,549,779 | 261.25% | 9.19% | 11.37% |

Macro Fear Buy finished $80,747 above Monthly Buy & Hold in this history.

## Fresh 2017 holdout start

This simulation resets the portfolio to $40,000 on 2017-01-03 while retaining
only precomputed point-in-time feature history. It then makes 114 monthly
deposits. Total cash injected was $496,000.

| Portfolio | Final value | Net profit | ROI | TWR CAGR | XIRR |
|---|---:|---:|---:|---:|---:|
| Macro Fear Buy | $1,145,556 | $649,556 | 130.96% | 14.75% | 15.64% |
| Monthly Buy & Hold | $1,135,309 | $639,309 | 128.89% | 14.96% | 15.47% |
| Initial 80% contribution target | $999,071 | $503,071 | 101.43% | 12.61% | 13.14% |

Macro Fear Buy finished $10,247 above Monthly Buy & Hold. Its time-weighted
return was slightly lower, but its money-weighted return and ending value were
higher because the timing of later deposits was better.

## Accounting interpretation

`ROI` is `(final value / total injected - 1) * 100`.

TWR CAGR removes external deposits and measures the strategy's underlying
compounded return. XIRR measures the investor's annualized return while
respecting the exact dates and sizes of deposits. For an accumulation plan,
ending value, net profit, and XIRR are more directly relevant than ordinary
CAGR.
