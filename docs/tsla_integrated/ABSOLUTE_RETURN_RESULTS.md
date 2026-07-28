# TSLA absolute-return LONG/CASH/SHORT result

Run date: 2026-07-27

## Research rule

- Development and optimization: 2019-01-01 through 2025-12-31.
- Development folds: 2019-2021, 2022-2023, and 2024-2025.
- Final diagnostic: 2026-01-01 onward.
- Every development fold must have positive ROI.
- Continuous development and every fold must have MDD no worse than -40%.
- Quarterly financial information uses a 45-calendar-day availability lag.
- Signals use prior-close information and execute at the next open.
- Idle cash earns the point-in-time cash rate.
- Short exposure pays a 3% annual borrow cost plus normal fees and slippage.

The separate TSLA downside classifier produces expanding-window strict-OOS
probabilities. A training row is used only after its full 21-session outcome is
known. The downside probability blocks LONG entry when risk is high and permits
SHORT entry only with bearish technical and macro confirmation.

## Frozen candidate

Candidate 926 was selected without using its 2026 result.

| Period | ROI | MDD | Completed trades |
|---|---:|---:|---:|
| 2019-2021 | 283.00% | -17.60% | 6 |
| 2022-2023 | 28.55% | -18.03% | 3 |
| 2024-2025 | 28.98% | -11.29% | 4 |
| Continuous 2019-2025 | 350.04% | -35.87% | 14 |

## 2026 diagnostic

| Portfolio | Final value | ROI | MDD |
|---|---:|---:|---:|
| Integrated strategy | $46,019 | 15.05% | -2.60% |
| Buy & Hold | $27,561 | -31.10% | -31.61% |

The strategy bought at the 2026-05-06 open and sold at the 2026-05-13 open.
It then remained in cash through the July decline. No SHORT trade occurred.
Most of the gain came from the completed LONG trade; idle-cash interest supplied
the remainder.

## Honest interpretation

This run meets the requested positive-ROI condition and its Buy/Sell decisions
were economically useful in 2026. It does not establish a guarantee of positive
future return. Multiple earlier development runs displayed 2026 results, so
2026 is no longer a pristine never-observed holdout. Candidate 926 was ranked
using 2019-2025 only, but the next genuinely untouched evidence must come from
forward paper trading after the frozen run date.
