# Macro Fear Buy SP500

This workflow is deliberately separate from `macro_momentum_sp500`.
The older workflow remains a defensive benchmark. This workflow implements the
opposite investment thesis:

- hold a long-term SPY core;
- keep a tactical cash reserve during ordinary or optimistic markets;
- deploy that reserve in weekly tranches as VIX, macro stress, model risk,
  drawdown, and downside momentum confirm fear;
- never sell the core sleeve;
- trim only the tactical sleeve after a minimum holding period, a profit gate,
  and a separate euphoria signal.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\macro_fear_buy_sp500\run_research.py
```

The runner uses the newest strict-OOS prediction file produced by the
`macro_momentum_sp500` workflow. It does not rerun every other project workflow.
Generated data and the HTML report are written atomically to the sibling
`Results/SP500/macro_fear_buy_sp500` folder.

## Research split

Parameter selection uses observations through 2016-12-30 only. Parameters are
then frozen. Results beginning 2017-01-03 are reported as an untouched holdout.
The full history remains useful context, but it is not an unbiased estimate of
parameter-selection performance.

## Important limitation

With one initial capital injection and no leverage, keeping tactical cash creates
cash drag in long bull markets. The strategy can improve the timing of reserve
deployment without necessarily beating 100% Buy & Hold.
