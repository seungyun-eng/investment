# V6-B Task 2: Point-in-Time Universe Validation

## Status

- Model state: `POST_HOC_DIAGNOSTIC_PASS`
- `validation_is_fresh=false`
- V6-B weights, score function, target generation, execution delay, and 10 bps
  turnover cost remain frozen.
- A true point-in-time (PIT), delisting-inclusive run was **not identified** from
  local data. The executed annual reconstruction is explicitly labelled
  `LOCAL_ANNUAL_SURVIVOR_PROXY_CAUSAL`, not PIT evidence.
- All reported returns can be inflated by survivorship bias, the 2,000-candidate
  multiple search, and post-hoc observation of 2025/2026.

## Input schemas

### Historical membership snapshots

CSV columns:

| Column | Type | Required | Meaning |
| --- | --- | --- | --- |
| `AsOfDate` | ISO date | yes | Date when this universe became knowable |
| `Ticker` | string | yes | Security ticker used by the market panel |
| `Rank` | integer | no | Vendor liquidity/market-cap rank |
| `Selected` | boolean | no | Membership flag; if omitted, every row is a member |

The file must contain the selected top 200 at each snapshot, or a broader
ranked cross-section with `Selected=true` only for the top 200. It must not be
restricted to names that survive today. Annual snapshots are the minimum
acceptable frequency; weekly/monthly snapshots are preferred.

### Delisting events

CSV columns:

| Column | Type | Required | Meaning |
| --- | --- | --- | --- |
| `Ticker` | string | yes | Delisted security |
| `EffectiveDate` | ISO date | yes | First date the security cannot be traded |
| `DelistingReturn` | decimal | conditional | Vendor total delisting return |
| `DelistingCategory` | string | no | Normalized `BANKRUPTCY`, `LIQUIDATION`, or `WORTHLESS` |
| `Exchange` | string | no | Used only by an imputation sensitivity |

Observed delisting returns are used first. Bankruptcy, liquidation, or worthless
codes force `-1.00`. Missing returns must be run as sensitivities: `-1.00`
conservative stress, Nasdaq `-0.55`/other exchange `-0.30` imputation, and `0.00`
optimistic upper bound. Missing returns must never be silently replaced.

### Existing market and financial inputs

The existing V6-B loaders supply adjusted daily OHLCV and current-restated
Macrotrends financial histories. Macrotrends is not PIT: dates are fiscal
quarter-ends, restatements can be reflected retroactively, and the 45-day lag is
only an approximation.

## Procedure

1. Acquire an annual-or-better historical membership file and a
   delisting-inclusive price/return file.
2. Load all securities that appear in any membership snapshot, including names
   absent from today's data folder.
3. Resolve delisting returns under each documented missing-value scenario.
4. Choose each completed week's final session from a reference exchange
   calendar. Do not inspect the rest of the week's future coverage.
5. Apply the latest membership snapshot known on each causal signal date.
6. Recompute the existing cross-sectional factors and call the frozen V6-B
   score and target functions.
7. Execute at the next market session's open, charge 10 bps on turnover, and
   force a delisted position to cash before any rebalance on or after its
   effective date.
8. Report training 2020–2024, 2025, and 2026 separately. Compare excess-return
   retention with the frozen current-survivor baseline.

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/analyze_pit_universe.py `
  --strategy ../Results/Cross_Sectional/rank_signals/<frozen-run>/selected_strategy.json `
  --ticker-config ../Results/Cross_Sectional/backfill_runs/<top200-run>/automatic_tickers.json `
  --supplemental-ticker-config config/tickers.json `
  --pit-membership path/to/membership.csv `
  --delistings path/to/delistings.csv `
  --missing-delisting-return-policy ERROR
```

Repeat the run with `total_loss`, `exchange_haircut`, and `zero` only as explicit
sensitivity scenarios when observed delisting returns are missing.

## Data-source routes

1. **CRSP through WRDS**: preferred research-grade route. CRSP provides
   delisting dates and returns; WRDS and CRSP access are institution/licence
   based and quoted rather than a public retail price. Add point-in-time
   Compustat or filing vintages for financial factors.
2. **Sharadar through Nasdaq Data Link**: premium `SEP`, `SF1`, and ticker/event
   tables can cover active and delisted securities. Full-history pricing is
   account/licence based; the free entry dataset is not a US top-200 PIT
   substitute.
3. **Norgate Data Platinum/Diamond**: a practical retail route for delisted US
   prices and historical index constituents. It does not by itself provide
   point-in-time fundamentals, so SEC filing vintages are still needed.
4. **Free fallback**: archive historical exchange lists, SEC submissions/XBRL
   facts by filing accession and first-seen date, manually add delisted names,
   and cross-check corporate actions. This is auditable but does not provide a
   complete free delisting-return database, so remaining selection and
   delisting bias must be disclosed.

## Executed local diagnostic

Output directory:

`Results/Cross_Sectional/pit_diagnostic/20260729_200447_896222_v6_b_task2`

The raw-folder audit found 228 folders. It excluded `Bitcoin` and `Xrp` as
cryptocurrencies, `S&P500` as an index/ETF benchmark, and the empty `Spy`
folder, leaving 224 individual equities. Historical snapshots additionally
require at least 42 of the last 63 sessions, price >= $5, market cap >= $1
billion with no missing-value exception, median dollar volume >= $10 million,
and at least two years since the first local price observation.

The following are contaminated diagnostic values, not unbiased expected
returns:

| Scenario | Period | Strategy ROI | Benchmark ROI | Excess | Baseline excess retained |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen current snapshot, retrospective date rule | 2020–2024 | 1,671.37% | 219.03% | 1,452.34%p | 100.00% |
| current snapshot, causal date rule | 2020–2024 | 1,671.37% | 219.03% | 1,452.34%p | 100.00% |
| local survivor proxy, causal date rule | 2020–2024 | 722.70% | 312.63% | 410.06%p | 28.24% |
| frozen current snapshot, retrospective date rule | 2025 | 282.76% | 31.97% | 250.79%p | 100.00% |
| current snapshot, causal date rule | 2025 | 282.76% | 31.97% | 250.79%p | 100.00% |
| local survivor proxy, causal date rule | 2025 | 49.03% | 30.49% | 18.54%p | 7.39% |
| frozen current snapshot, retrospective date rule | 2026 YTD | 60.32% | 19.03% | 41.30%p | 100.00% |
| current snapshot, causal date rule | 2026 YTD | 52.90% | 17.33% | 35.57%p | 86.14% |
| local survivor proxy, causal date rule | 2026 YTD | 16.94% | 13.35% | 3.59%p | 8.70% |

The annual snapshot counts were 22, 24, 27, 31, 39, 200, and 200 for
2020 through 2026. The 2020–2022 snapshots have fewer than 30 names, and even
2023–2024 remain very small. The 28.24% training retention therefore meets the
pre-registered "primary source" numerical threshold, but it cannot isolate
survivorship/look-ahead from severe early-universe undercoverage.

The refreshed-data rerun of the current-200 scenario did not reproduce the
prompt's frozen baseline (795.76%, 69.37%, and 12.73% strategy ROI). The
requested retention denominator therefore remains the pre-registered baseline,
while the refreshed rerun is preserved separately in `scenario_summary.csv`.

## WDC answers

1. True PIT membership on 2025-01-01: **not identified** without the missing
   historical cross-section and delisted names.
2. Local survivor proxy: WDC was included at rank 165 in 2025. The
   frozen signal first selected it on 2025-07-11 and executed on 2025-07-14,
   the same dates as the current-snapshot run.
3. True PIT performance reductions for 2020–2024, 2025, and 2026:
   **not identified**. The table above is only a data-readiness and sensitivity
   result.

## Pre-registered interpretation

- If a genuine PIT run retains under 50% of frozen-baseline training excess,
  classify survivorship bias as a primary performance source.
- Retention of 50%–75% means material survivorship dependence.
- Retention above 75% means this test does not identify survivorship as the
  primary source, but does not remove multiple-testing, financial-restatement,
  or post-hoc contamination.
- WDC must be checked independently: membership, selection date, holding
  episode, and contribution. A changed WDC outcome explains whether the
  universe correction attacks the already identified winner dependence.

## Deployment implication

The diagnostic is not sufficient to authorize live capital: the exact answers
to all three WDC/PIT questions remain unidentified. It does establish the
executable gate for the next data acquisition—no PIT result should be accepted
unless the run reports `TruePITExecuted=true`, contains delisted histories, and
passes the frozen-strategy checks.
