# V7 point-in-time market-cap universe: source sample

## Scope

This milestone builds and audits only the 2019 and 2020 source samples. It
does not run a V6-B or V7 backtest, and it does not change any V6-B strategy
code or parameter.

## Source decision

The final research-grade source should be CRSP or a delisting-inclusive
Sharadar/Nasdaq Data Link subscription. Those sources were not configured in
the local environment during this run.

The free sample therefore keeps two sources separate:

1. TradeFomo publishes a January 1 historical market-cap ranking, but its
   public page exposes only 50 rows. It also displays the current ticker
   `META` in the 2019 snapshot, when the contemporaneous ticker was `FB`.
   The sample preserves `PublishedTicker=META`, corrects the known historical
   identifier to `Ticker=FB`, and marks `TickerHistoryStatus`. This one
   correction is not a comprehensive historical security master, so the
   direct table is useful as a top-50 cross-check, not as the final PIT
   top-100 security master.
2. The free top-100 proxy uses the latest historical S&P 500 membership
   snapshot on or before January 1, then computes market capitalization from
   Yahoo's historical close and shares outstanding. Historical closes are
   converted back to their raw price basis for any later stock splits before
   multiplication by historical shares.

The S&P proxy is not the actual whole-US-market top 100. It can omit large
US-listed securities outside the S&P 500, and Yahoo does not guarantee that
its historical shares are first-seen point-in-time values.

## Reproduction

```powershell
python scripts/cross_sectional/build_pit_universe_source_sample.py `
  --years 2019 2020
```

The script resolves the sibling OneDrive data folders through
`stock_research.paths`; it does not hardcode a drive, username, or OneDrive
path. The Yahoo security cache makes the step independently resumable.

## Sample result

Run:

`20260729_134455_747330_v7_pit_2019_2020_sample`

| As of | Dataset | Rows | Local price rows | Local coverage | Missing local |
|---|---|---:|---:|---:|---:|
| 2019-01-01 | TradeFomo direct published | 50 | 48 | 96% | 2 |
| 2019-01-01 | Historical S&P market-cap proxy | 100 | 85 | 85% | 15 |
| 2019-01-01 | Direct-50 plus proxy fill sample | 100 | 86 | 86% | 14 |
| 2020-01-01 | TradeFomo direct published | 50 | 49 | 98% | 1 |
| 2020-01-01 | Historical S&P market-cap proxy | 100 | 84 | 84% | 16 |
| 2020-01-01 | Direct-50 plus proxy fill sample | 100 | 85 | 85% | 15 |

For the S&P proxy inputs, 425 of 497 historical constituents were rank
eligible in 2019 and 438 of 499 were rank eligible in 2020. Across the
two-year union, Yahoo returned 452 successful securities, 11 securities with
no eligible snapshot, and 62 failures. The failures are disproportionately
delisted, acquired, or subsequently renamed securities, which is direct
evidence that the free source does not close the delisting gap.

The hybrid sample is an audit convenience only. Ranks 1-50 are directly
published; ranks after 50 are explicitly labeled
`SP500_PROXY_FILL_NOT_ACTUAL_WHOLE_MARKET_RANK`.

## Outputs

- `tradefomo_direct_rankings.csv`: directly published top 50
- `sp500_proxy_candidates.csv`: all historical membership inputs and data
  quality flags
- `sp500_proxy_top100.csv`: S&P-only market-cap proxy
- `hybrid_top100_sample.csv`: direct top 50 plus clearly labeled proxy fill
- `source_comparison.csv`: direct/proxy overlap
- `sample_coverage.csv`: local coverage by snapshot
- `missing_local_sample.csv`: sample names without local price files
- `fetch_log.csv`: source and ticker-level failures
- `manifest.json`: settings, source choices, and limitations

## Decision before the full 2019-2026 build

The free pipeline is suitable for diagnosing coverage and preparing a
backfill queue. It is not sufficient to certify an exact, delisting-inclusive
whole-market top-100 universe. A full V7 truth-measurement dataset requires
either CRSP/Sharadar access or an explicitly accepted S&P-only proxy with the
remaining missing delisted histories documented.
