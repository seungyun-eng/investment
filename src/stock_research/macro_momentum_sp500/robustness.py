from __future__ import annotations

import numpy as np
import pandas as pd


def yearly_portfolio_metrics(daily_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, daily in daily_frames.items():
        frame = daily.copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["DailyReturn"] = frame["TotalValue"].pct_change()
        for year, part in frame.groupby(frame["Date"].dt.year):
            returns = part["DailyReturn"].dropna()
            if returns.empty:
                continue
            compounded = (1 + returns).prod() - 1
            volatility = returns.std(ddof=1) * np.sqrt(252)
            rows.append(
                {
                    "Portfolio": name,
                    "Year": int(year),
                    "Return(%)": compounded * 100,
                    "Volatility(%)": volatility * 100,
                    "Positive": compounded > 0,
                }
            )
    return pd.DataFrame(rows)


def block_bootstrap_excess_return(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    block_size: int = 21,
    samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    merged = strategy[["Date", "TotalValue"]].merge(
        benchmark[["Date", "TotalValue"]],
        on="Date",
        suffixes=("_Strategy", "_Benchmark"),
        validate="one_to_one",
    )
    strategy_return = merged["TotalValue_Strategy"].pct_change()
    benchmark_return = merged["TotalValue_Benchmark"].pct_change()
    excess = (strategy_return - benchmark_return).dropna().to_numpy(dtype=float)
    if len(excess) < block_size:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(excess) - block_size + 1)
    observations: list[float] = []
    blocks_needed = int(np.ceil(len(excess) / block_size))
    for _ in range(samples):
        pieces = [
            excess[start : start + block_size]
            for start in rng.choice(starts, size=blocks_needed, replace=True)
        ]
        draw = np.concatenate(pieces)[: len(excess)]
        observations.append(float(draw.mean() * 252 * 100))
    values = np.asarray(observations)
    return pd.DataFrame(
        [
            {
                "Metric": "Annualized arithmetic excess return (%)",
                "Estimate": float(excess.mean() * 252 * 100),
                "BootstrapP05": float(np.quantile(values, 0.05)),
                "BootstrapMedian": float(np.quantile(values, 0.50)),
                "BootstrapP95": float(np.quantile(values, 0.95)),
                "ProbabilityPositive": float(np.mean(values > 0)),
                "BlockDays": block_size,
                "Samples": samples,
            }
        ]
    )
