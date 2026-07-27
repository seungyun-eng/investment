from __future__ import annotations

import numpy as np
import pandas as pd


def flow_adjusted_block_bootstrap(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    block_size: int = 21,
    samples: int = 5_000,
    seed: int = 20260726,
) -> pd.DataFrame:
    """Bootstrap flow-adjusted daily excess returns in contiguous blocks."""

    merged = strategy[["Date", "FlowAdjustedReturn"]].merge(
        benchmark[["Date", "FlowAdjustedReturn"]],
        on="Date",
        suffixes=("_Strategy", "_Benchmark"),
        validate="one_to_one",
    )
    excess = (
        pd.to_numeric(
            merged["FlowAdjustedReturn_Strategy"],
            errors="coerce",
        )
        - pd.to_numeric(
            merged["FlowAdjustedReturn_Benchmark"],
            errors="coerce",
        )
    ).dropna().to_numpy(dtype=float)
    if len(excess) < block_size:
        raise ValueError("Not enough observations for the requested block size.")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(excess) - block_size + 1)
    blocks_needed = int(np.ceil(len(excess) / block_size))
    draws = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        draw = np.concatenate(
            [excess[start : start + block_size] for start in selected]
        )[: len(excess)]
        draws[sample] = float(draw.mean() * 252.0 * 100.0)
    return pd.DataFrame(
        [
            {
                "Metric": "Annualized arithmetic excess return (%)",
                "Estimate": float(excess.mean() * 252.0 * 100.0),
                "BootstrapP05": float(np.quantile(draws, 0.05)),
                "BootstrapMedian": float(np.quantile(draws, 0.50)),
                "BootstrapP95": float(np.quantile(draws, 0.95)),
                "ProbabilityPositive": float(np.mean(draws > 0.0)),
                "BlockDays": block_size,
                "Samples": samples,
            }
        ]
    )


def yearly_flow_adjusted_comparison(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Compare calendar-year returns without treating deposits as returns."""

    merged = strategy[["Date", "FlowAdjustedReturn"]].merge(
        benchmark[["Date", "FlowAdjustedReturn"]],
        on="Date",
        suffixes=("_Strategy", "_Benchmark"),
        validate="one_to_one",
    )
    merged["Date"] = pd.to_datetime(merged["Date"])
    rows: list[dict[str, object]] = []
    for year, part in merged.groupby(merged["Date"].dt.year):
        strategy_return = (
            1.0
            + pd.to_numeric(
                part["FlowAdjustedReturn_Strategy"],
                errors="coerce",
            ).fillna(0.0)
        ).prod() - 1.0
        benchmark_return = (
            1.0
            + pd.to_numeric(
                part["FlowAdjustedReturn_Benchmark"],
                errors="coerce",
            ).fillna(0.0)
        ).prod() - 1.0
        rows.append(
            {
                "Year": int(year),
                "StrategyReturn(%)": float(strategy_return * 100.0),
                "BuyHoldReturn(%)": float(benchmark_return * 100.0),
                "ExcessReturn(pp)": float(
                    (strategy_return - benchmark_return) * 100.0
                ),
                "StrategyWon": bool(strategy_return > benchmark_return),
            }
        )
    return pd.DataFrame(rows)
