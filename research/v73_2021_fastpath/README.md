# V7.3 2021 FastPath

Goal: cut the 2021 PIT recovery wall-clock time by replacing the one-ticker-per-cycle workflow with deterministic batch shards and same-cycle StateKey merges.

## What changes
- Builds independent priority queues for ShareGrowth, Identity, Price/Basis/CorporateAction, and Financial.
- Default capacity is 8 shards × 8 tickers = 64 tickers/domain/cycle.
- Uses greedy load balancing so high-value tickers are spread across shards.
- Merges multiple worker deltas in one pass by StateKey.
- READY statuses are monotonic: a later blocked/unknown delta cannot downgrade an already READY row.
- Recomputes only readiness/candidate booleans from already-authoritative gates.

## What does not change
- Frozen V7.3 signal, entry, ranking, sizing, execution, universe, or NAV logic.
- No current-ticker backcast.
- No price forward-fill.
- No cross-share-class substitution.
- No inferred CERTIFIED row without all authoritative gates.

## Measured on 2026-10-06 central state
Input: 127,009 StateKeys / 523 tickers.

Planner with 8 shards × 8 tickers:
- ShareGrowth candidates: 115 tickers
- Identity candidates: 396 tickers
- P/B/A candidates: 34 tickers
- Financial candidates: 13 tickers
- Capacity: 64 tickers/domain/cycle

Local tests: 3 passed.

## Usage
```bash
python research/v73_2021_fastpath/fastpath.py plan \
  --central V73_2021_Rebuilt_Central_State_Auto_*.zip \
  --out fastpath_plan --shards 8 --batch-size 8

python research/v73_2021_fastpath/fastpath.py merge \
  --central CENTRAL.zip \
  --delta SHARE_SHARD0.zip --delta SHARE_SHARD1.zip \
  --delta ID_SHARD0.zip --delta PBA_SHARD0.zip \
  --out CENTRAL_FASTPATH.zip
```

The planner only assigns work; each shard must still produce provenance-qualified evidence and StateKey deltas.
