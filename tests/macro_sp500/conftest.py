from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stock_research.macro_sp500.config import (
    MacroSp500Params,
    MacroSp500Settings,
)
from stock_research.macro_sp500.config_v2 import MacroSp500V2Settings


@pytest.fixture
def macro_settings() -> MacroSp500Settings:
    return MacroSp500Settings(
        initial_capital=100_000.0,
        warning_lookback_days=5,
        drawdown_lookback_days=20,
        panic_confirmation_window=10,
        exit_confirmation_days=2,
        stage_2_quantile_step=0.05,
        stage_3_quantile=0.99,
        drawdown_levels=(-0.05, -0.10, -0.15),
        stage_target_weights=(0.80, 0.90, 1.00),
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        cash_annual_rate=0.0,
        training_years=1,
        first_test_year=2020,
        minimum_vix_observations=20,
        minimum_cagr_fraction_of_buy_hold=0.0,
    )


@pytest.fixture
def macro_params() -> MacroSp500Params:
    return MacroSp500Params(
        vix_lookback_years=3,
        core_weight=0.70,
        warning_score_min=2,
        warning_addition=0.10,
        vix_entry_quantile=0.90,
        exit_vix_quantile=0.50,
        minimum_hold_days=1,
    )


@pytest.fixture
def synthetic_macro_data() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", "2021-12-31")
    index = np.arange(len(dates), dtype=float)
    close = 100.0 * np.exp(index * 0.0003 + np.sin(index / 35.0) * 0.03)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * (1.0 + np.sin(index / 11.0) * 0.001),
            "Close": close,
            "Volume": 1_000_000.0 + (index % 20) * 10_000.0,
            "VIX": 18.0 + np.sin(index / 17.0) * 5.0 + (index % 113 == 0) * 15.0,
        }
    )


@pytest.fixture
def one_year_settings(macro_settings: MacroSp500Settings) -> MacroSp500Settings:
    return replace(macro_settings, training_years=1)


@pytest.fixture
def v2_settings() -> MacroSp500V2Settings:
    return MacroSp500V2Settings(
        initial_capital=100_000.0,
        vix_lookback_years=5,
        warning_lookback_days=5,
        drawdown_lookback_days=20,
        reversal_lookback_days=5,
        recovery_sma_days=10,
        exit_vix_quantile=0.50,
        exit_confirmation_days=2,
        vix_decline_from_peak=0.20,
        minimum_trade_fraction=0.02,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        fallback_cash_annual_rate=0.0,
        training_years=1,
        first_test_year=2020,
        minimum_vix_observations=10,
        minimum_cagr_fraction_of_buy_hold=0.90,
        maximum_mdd_fraction_of_buy_hold=0.85,
        static_benchmark_weight=0.70,
    )
