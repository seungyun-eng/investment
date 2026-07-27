from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.macro_momentum_sp500.robustness import (
    block_bootstrap_excess_return,
)


def test_block_bootstrap_is_deterministic_and_detects_positive_excess() -> None:
    dates = pd.bdate_range("2020-01-01", periods=300)
    strategy = pd.DataFrame(
        {"Date": dates, "TotalValue": 100 * np.cumprod(np.repeat(1.001, 300))}
    )
    benchmark = pd.DataFrame(
        {"Date": dates, "TotalValue": 100 * np.cumprod(np.repeat(1.0005, 300))}
    )

    first = block_bootstrap_excess_return(
        strategy, benchmark, block_size=21, samples=100, seed=7
    )
    second = block_bootstrap_excess_return(
        strategy, benchmark, block_size=21, samples=100, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "ProbabilityPositive"] == 1.0
