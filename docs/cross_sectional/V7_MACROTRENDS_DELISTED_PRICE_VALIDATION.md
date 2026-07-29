# V7 Macrotrends delisted-price validation

## Decision

The small-sample gate failed on 2026-07-29. Macrotrends exposes daily OHLCV
through the chart iframe for an active control (WDC), but did not expose the
same data for CELG, RTN, or TIF:

| Ticker | Case | Public price page | Daily chart endpoint | Daily OHLCV |
|---|---|---:|---:|---:|
| WDC | active control | available | HTTP 200 | 3,771 rows |
| CELG | acquired/delisted | 404 | HTTP 500 | unavailable |
| RTN | merger/old ticker | 404 | HTTP 500 | unavailable |
| TIF | acquired/delisted | 404 | HTTP 500 | unavailable |

The 62-name expansion was therefore not run. This is a failed data-source
validation, not evidence that every one of the 62 names was individually
queried at Macrotrends.

## Page structure

The public price-history URL is:

```text
https://www.macrotrends.net/stocks/charts/{ticker}/{slug}/stock-price-history
```

For an active name, the page embeds this chart endpoint:

```text
https://www.macrotrends.net/production/stocks/desktop/PRODUCTION/stock_price_history.php?t={ticker}&yb=15
```

The endpoint contains a JavaScript `dataDaily` array with fields `d`, `o`,
`h`, `l`, `c`, and `v`. WDC volume values matched the local Back Test volume
after interpreting `v` as millions of shares.

## Adjustment compatibility

Macrotrends describes its price history as adjusted for splits and dividends.
The WDC control comparison showed close differences that were larger in older
history while volumes matched closely. Representative observations:

| Date | Local close | Macrotrends close | Local volume | Macrotrends volume |
|---|---:|---:|---:|---:|
| 2019-01-02 | 28.9191 | 27.3220 | 8.45M | 8.448M |
| 2019-07-29 | 42.1693 | 40.7462 | 5.73M | 5.729M |
| 2024-12-31 | 45.0718 | 44.8664 | 5.21M | 5.206M |
| 2025-12-31 | 172.2700 | 172.1430 | 3.55M | 3.552M |
| 2026-07-28 | 463.5100 | 463.5100 | 10.51M | 10.512M |

The schemas are mechanically convertible, but historical adjusted close values
are not identical. Macrotrends prices must not be silently appended to the
local series without an explicit normalization and overlap policy.

## Reproduction

```powershell
python scripts/cross_sectional/validate_macrotrends_delisted_prices.py `
  --target CELG:celgene:Celgene `
  --target RTN:raytheon:Raytheon `
  --target TIF:tiffany:Tiffany
```

The script uses a browser User-Agent, short-lived requests, exponential
backoff, and at least 0.5 seconds between requests. Outputs are written
atomically under the sibling OneDrive `Results` folder.

Macrotrends data is a current-site reconstruction and does not provide complete
point-in-time provenance. Even a successful price fetch would not by itself
make the V7 data point-in-time.
