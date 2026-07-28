# TSLA integrated signal: first multi-fold results

Run date: 2026-07-27

## Protocol

- Feature warm-up may use observations before 2019.
- Strategy development and optimization use 2019-01-01 through 2025-12-31.
- The chronological development folds are 2019-2021, 2022-2023, and
  2024-2025.
- The selection score emphasizes worst-fold excess ROI versus Buy & Hold,
  followed by mean-fold excess ROI and mean drawdown improvement.
- The 2,000 candidates use the same canonical signal and next-open portfolio
  simulator as final evaluation.
- Quarterly financial observations become usable 45 calendar days after their
  reported period end.
- The 2026 evaluation does not affect candidate ranking.

## Frozen development winner

Candidate 1875:

- technical weight: 27.60%;
- financial weight: 52.18%;
- macro weight: 20.22%;
- buy threshold: 0.5528;
- sell threshold: 0.4387;
- stop loss: 22.68%;
- minimum hold: 63 sessions.

| Development period | Excess ROI vs Buy & Hold | MDD improvement | Completed trades |
|---|---:|---:|---:|
| 2019-2021 | 20.31% | 0.00% | 4 |
| 2022-2023 | 0.13% | 0.21% | 3 |
| 2024-2025 | 23.68% | 0.06% | 5 |

The worst development fold remained positive, but the full 2019-2025 strategy
ROI was 1,977.45% versus a higher Buy & Hold result. Fold resets and one
continuous full-period portfolio answer different path-dependent questions, so
both must remain visible.

## 2026 evaluation

| Portfolio | Final value | ROI | MDD | Completed sells |
|---|---:|---:|---:|---:|
| Integrated signal | $25,493 | -36.27% | -36.63% | 1 |
| Buy & Hold | $27,553 | -31.12% | -31.61% | 0 |

The frozen integrated signal underperformed Buy & Hold by 5.15 percentage
points and had a 5.02 percentage-point worse drawdown. It therefore failed the
2026 generalization test.

## Interpretation and limitations

- The initial objective incorrectly allowed full-period compounded ROI to
  overwhelm a weak fold. That objective was rejected using development-only
  diagnostics and replaced by the robust fold objective reported here.
- Earlier smoke runs already displayed 2026 outcomes while the workflow was
  being debugged. Although the final candidate ranking uses only 2019-2025,
  2026 can no longer be described as a pristine never-observed holdout.
- Exact SEC filing acceptance timestamps are not yet present. The 45-day
  availability lag is an approximation.
- The current signal does not establish reliable alpha. The honest next
  untouched test is forward paper trading after the frozen run date or a later
  calendar period that has not been inspected during development.
