# Macro SP500 workflow

This workflow is intentionally isolated from the existing VIX and Technical
strategies. It uses an investable S&P 500 proxy, actual daily VIX, a trailing
VIX percentile, S&P drawdown, and a fixed warning score to produce target
portfolio weights.

## First-pass model

- Core S&P exposure remains invested.
- A warning score may add a small early tranche.
- VIX percentiles or S&P drawdowns raise the target exposure in three stages.
- The tactical allocation returns to the core weight after a minimum holding
  period and confirmed VIX normalization.
- Signals use close-of-day information and execute at the next trading day's
  open.
- The optimization and simulation entry points call the same feature, signal,
  and target-weight functions.

## Commands

```powershell
python scripts/macro_sp500/optimize.py
python scripts/macro_sp500/simulate.py 1 --start 2019-01-01 --end 2025-11-03
```

Parameters are stored in the sibling OneDrive `Results/Parameters` folder.
Generated CSV and HTML files are stored under
`Results/SP500/macro_sp500`. Outputs use atomic replacement.

The measured first-pass result and selected parameters are recorded in
`FIRST_PASS_RESULTS.md`.

The second-pass method and measured result are recorded in
`SECOND_PASS_RESULTS.md`.

## V2 commands

```powershell
python scripts/macro_sp500/update_adjusted_spy.py
python scripts/macro_sp500/optimize_v2.py
```

## Important limitation

The current legacy S&P CSV must be audited before production use. If it lacks an
adjusted close or explicit dividend cash flows, strategy and Buy & Hold returns
exclude dividends. Walk-forward folds also reset portfolio state at each
test-year boundary in this first-pass optimizer. The current portfolio engine
also restores the exact target weight whenever price drift creates a difference,
which can generate excessive small rebalances. A practical rebalance band should
be tested in the next pass.
