# V6-B Task 3: DSR and PBO

## Status and scope

- Model status: `POST_HOC_DIAGNOSTIC_PASS`
- `validation_is_fresh=false`
- V6-B weights, signals, execution, and portfolio rules were not changed.
- The result remains exposed to survivorship bias, 2,000-trial selection bias,
  and post-hoc observation of 2025/2026.
- The methodology follows Bailey and López de Prado's Deflated Sharpe Ratio and
  Bailey et al.'s CSCV/PBO framework.

## Inputs

Candidate summary:

`Results/Cross_Sectional/rank_signals/20260729_055627_508765_v6_b_replacement_hurdle/optimization_candidates.csv`

Frozen selection record:

`Results/Cross_Sectional/rank_signals/20260729_055627_508765_v6_b_replacement_hurdle/selected_strategy.json`

Selected-strategy training equity proxy:

`Results/Cross_Sectional/pit_diagnostic/20260729_181622_210555_v6_b_task2/scenario_equity.csv`

The candidate file contains 2,000 unique nine-dimensional tuples:

1. `momentum_weight`
2. `trend_weight`
3. `growth_weight`
4. `quality_weight`
5. `risk_control_weight`
6. `top_k`
7. `exit_rank`
8. `trend_floor`
9. `momentum_floor`

It also contains `TrainROI`, `TrainCAGR`, `TrainSharpe`, three training-fold
excess CAGRs, and contaminated 2025 selection ROI/CAGR/Sharpe. Two deterministic
seed candidates contain `exit_rank=6`, and one contains
`momentum_floor=-0.10`; these are present in addition to the random support.

## DSR procedure

1. Use candidate 1931's recorded annualized training Sharpe as the observed
   Sharpe.
2. Estimate the cross-trial Sharpe variance from all 2,000 candidate rows.
3. Derive daily returns from the stored 2020–2024 selected-strategy equity and
   calculate unbiased sample skewness and Pearson kurtosis.
4. Calculate the expected maximum Sharpe under a zero-Sharpe null using the
   Euler–Mascheroni extreme-value approximation.
5. Calculate DSR for nominal `N=2,000`.
6. Estimate a nine-dimension-aware effective-trial proxy from the average
   candidate performance correlation across the three stored training folds. Apply Bailey
   and López de Prado's interpolation
   `N_eff = rho + (1-rho) * N`.

The effective-trial estimate is a proxy because candidate-by-date return
histories were not saved. The paper warns that correlation estimates are
ill-conditioned when the number of trials exceeds the time dimension; here
there are only four aggregate blocks.

## DSR result

All values below remain potentially inflated by survivorship, multiple
testing, and post-hoc contamination.

| Trial assumption | N | Expected max Sharpe under zero null | DSR | Verdict |
| --- | ---: | ---: | ---: | --- |
| Nominal | 2,000.00 | 0.7044 | 0.9784 | Strong |
| Three-training-fold nine-dimensional proxy | 940.19 | 0.6615 | 0.9828 | Strong |

Additional inputs:

- Candidate 1931 recorded training Sharpe: `1.621589`
- Equity-proxy Sharpe: `1.622049`
- Difference: `0.000460`
- Cross-trial Sharpe variance: `0.041750`
- Return observations: `1,257`
- Skewness: `-0.119351`
- Pearson kurtosis: `6.317162`

The DSR result rejects a zero-Sharpe null after the specified multiplicity
correction. It does not establish benchmark-relative alpha, remove survivor
bias, or show that candidate 1931 was the best training candidate. Candidate
1931 was only at the 79.69th training-Sharpe percentile; candidate 139 had the
maximum training Sharpe of 2.04995.

## CSCV/PBO procedure and result

The stored file does not contain the required candidate-by-date return matrix
for full CSCV. A coarse `S=4` diagnostic uses:

1. 2020–2021 excess CAGR
2. 2022 excess CAGR
3. 2023–2024 excess CAGR
4. contaminated 2025 selection excess CAGR

All six symmetric two-block/two-block splits are evaluated, with two-year
blocks receiving twice the weight of one-year blocks. None of the six
in-sample winners fell below the complementary out-of-sample median:

- Coarse PBO: `0.000`
- Resolution: `1/6 = 0.1667`
- Pre-registered classification: low
- Exact daily-return CSCV PBO: **not identified**

This low coarse PBO cannot be treated as a clean validation result because one
of only four blocks is the contaminated 2025 selection period and the
candidate-by-date matrix is absent.

## IS versus 2025 relationship

- Spearman correlation, training Sharpe vs 2025 selection Sharpe:
  `rho=0.196537`, `p=7.28e-19`
- Classification by effect size: very weak positive rank relationship
- Candidate 139, the maximum training-Sharpe candidate, ranked at the 71.69th
  percentile on 2025 selection Sharpe.
- Candidate 1931 ranked at the 97.95th percentile on 2025 selection Sharpe and
  first on 2025 selection excess CAGR.

The tiny p-value reflects 2,000 observations and does not make the relationship
strong. The effect size shows that training Sharpe only weakly predicts 2025
selection Sharpe.

## Task 1 cross-interpretation

The 2025 period was directly used to choose candidate 1931, and its exceptional
performance was materially linked to the same WDC exposure that dominates the
later result. A high rank across stored blocks can therefore reflect repeated
exposure to one winner rather than a factor model that generalizes across
independent opportunities. The DSR and coarse PBO test candidate-level
statistics; neither removes WDC path dependence or current-survivor universe
contamination.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/analyze_overfitting_dsr_pbo.py `
  --candidates ../Results/Cross_Sectional/rank_signals/<frozen-v6-b-run>/optimization_candidates.csv `
  --selected-strategy ../Results/Cross_Sectional/rank_signals/<frozen-v6-b-run>/selected_strategy.json `
  --selected-equity ../Results/Cross_Sectional/pit_diagnostic/<task2-run>/scenario_equity.csv
```

## Live-capital implication

The formal DSR result is strong and the coarse four-block PBO does not identify
overfitting, but the weak IS–2025 rank correlation, contaminated selection
period, WDC dependence, and missing full CSCV matrix prevent this from
authorizing live deployment. Exact PBO requires a `date × 2,000 candidates`
return matrix saved from the original optimization run; it cannot be recovered
from the summary CSV alone.
