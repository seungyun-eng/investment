# SEC filing pipeline

This pipeline downloads public SEC EDGAR metadata, standardized Company Facts,
and primary periodic filing documents for a configured universe. Raw documents
are cached outside the repository under the sibling `SEC Filings` folder.
Derived point-in-time features are written to `Processed Data/SEC Filings`.

SEC requires automated downloaders to declare a descriptive User-Agent with a
contact email. Keep that value outside source control:

```powershell
$env:SEC_USER_AGENT = "Personal Investment Research your-email@example.com"
python scripts/cross_sectional/sync_sec_filings.py `
  --config config/cross_sectional/sec_filings_known_2020_growth_16.json `
  --universe-config config/cross_sectional/known_2020_growth_16_v7.json
```

The default 0.2-second interval limits this process to at most five sequential
requests per second, below the SEC's published ten-request-per-second ceiling.
Existing raw files are reused. Use `--refresh-metadata` to update submissions and
Company Facts, and `--refresh-documents` only when primary documents must be
downloaded again.

Features become eligible on the day after SEC acceptance. The extra day is a
conservative rule that prevents a filing accepted after the market close from
being used in the same day's trading signal.

The text output contains auditable phrase counts and explicit red flags. It does
not label management as good or bad. Management execution requires a separate
history of guidance versus subsequent realized results.

After the filing sync completes, run the fixed core/tactical prototype:

```powershell
python scripts/cross_sectional/run_filing_hybrid.py
```

The prototype holds three 70%-total core positions and up to two 15% tactical
positions. Core ranking is 80% true-value evidence and 20% price-risk evidence.
Tactical ranking is led by MA/MACD/OBV and momentum, with filing durability and
balance-sheet guards. These fixed rules were designed after observing the test
period and therefore remain post-hoc until a future untouched period is seen.

The checked-in hybrid configuration records the nearest training-period
candidate from a 288-candidate risk-rule grid. No candidate passed all requested
training constraints. The stored candidate is marked
`RELAXED_TRAIN_SELECTION_NO_STRICT_PASS`; it must not be presented as a validated
35%-45% annual-return strategy. The 2025-2026 period remains report-only and was
not used to select that configuration.

The SEC-derived columns are accession-specific and point-in-time. The current
hybrid still inherits the earlier V7 `GrowthFactor` and `QualityFactor`, whose
Macrotrends histories can contain later restatements. It is therefore a mixed
PIT prototype, not a fully filing-native historical valuation model.
