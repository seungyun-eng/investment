from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.macro_fear_buy_sp500.config import FearBuyParams
from stock_research.macro_fear_buy_sp500.contributions import (
    ContributionConfig,
    ContributionDeploymentPolicy,
    run_contribution_backtest,
)


def _signals() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-01",
            "2024-02-29",
            "2024-03-01",
            "2024-03-29",
        ]
    )
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 100.0, 105.0, 105.0, 110.0],
            "Close": [100.0, 100.0, 105.0, 105.0, 110.0],
            "CashRate": 0.0,
            "TargetWeight": [0.80, 0.80, 0.90, 0.90, 0.80],
            "SignalState": ["CORE", "CORE", "FEAR", "FEAR", "CORE"],
            "TransitionReason": ["", "", "FEAR_BUY", "", "EUPHORIA_TRIM"],
        }
    )


def test_monthly_contributions_are_not_counted_as_profit() -> None:
    params = FearBuyParams(core_weight=0.80, tranche_weight=0.10)
    config = ContributionConfig(
        initial_lump_sum=40_000.0,
        monthly_contribution=4_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    result = run_contribution_backtest(
        _signals(),
        params,
        config,
        name="test",
    )

    assert result.summary.contribution_count == 2
    assert result.summary.total_injected == 48_000.0
    assert np.isclose(
        result.summary.roi_percent,
        (result.summary.final_value / 48_000.0 - 1.0) * 100.0,
    )
    assert result.daily["Contribution"].tolist() == [
        0.0,
        4_000.0,
        0.0,
        4_000.0,
        0.0,
    ]


def test_contribution_rebalancing_never_sells_core() -> None:
    params = FearBuyParams(core_weight=0.80, tranche_weight=0.10)
    config = ContributionConfig(
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    result = run_contribution_backtest(
        _signals(),
        params,
        config,
        name="test",
    )

    sells = result.trades[result.trades["Action"] == "SELL"]
    assert sells["Sleeve"].eq("TACTICAL").all()
    assert result.daily["CoreShares"].diff().fillna(0).ge(-1e-12).all()
    assert result.daily["Cash"].ge(-1e-12).all()


def test_monthly_contributions_accumulate_without_a_fear_signal() -> None:
    signals = _signals().copy()
    signals["TargetWeight"] = 0.80
    signals["DecisionDay"] = False
    signals["DesiredTargetWeight"] = 0.80
    params = FearBuyParams(core_weight=0.80, tranche_weight=0.10)
    config = ContributionConfig(
        initial_lump_sum=40_000.0,
        monthly_contribution=4_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    result = run_contribution_backtest(signals, params, config, name="test")

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["Reason"] == "INITIAL_LUMP_SUM"
    assert result.daily.iloc[-1]["Cash"] == 16_000.0


def test_accumulated_cash_deploys_on_an_active_fear_signal() -> None:
    dates = pd.to_datetime(
        ["2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01", "2024-03-04"]
    )
    signals = pd.DataFrame(
        {
            "Date": dates,
            "Open": 100.0,
            "Close": 100.0,
            "CashRate": 0.0,
            "TargetWeight": [1.0] * len(dates),
            "DesiredTargetWeight": [1.0] * len(dates),
            "DecisionDay": [True, False, False, True, False],
            "TriggerLevel": ["PANIC"] * len(dates),
            "SignalState": ["PANIC_ALLOCATED"] * len(dates),
            "TransitionReason": [""] * len(dates),
        }
    )
    params = FearBuyParams(core_weight=0.80, tranche_weight=0.10)
    config = ContributionConfig(
        initial_lump_sum=40_000.0,
        monthly_contribution=4_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    result = run_contribution_backtest(signals, params, config, name="test")

    march_deposit = result.daily.loc[
        result.daily["Date"] == pd.Timestamp("2024-03-01")
    ].iloc[0]
    assert march_deposit["Contribution"] == 4_000.0
    assert march_deposit["Cash"] == 4_000.0
    march_buy = result.trades.loc[
        result.trades["Date"] == pd.Timestamp("2024-03-04")
    ]
    assert np.isclose(march_buy["Notional"].sum(), 4_000.0)
    assert march_buy["Reason"].eq("PANIC_ACCUMULATED_CASH_BUY").all()
    assert result.daily.iloc[-1]["Cash"] == 0.0


def test_staged_policy_preserves_cash_for_deeper_fear() -> None:
    dates = pd.to_datetime(
        ["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"]
    )
    signals = pd.DataFrame(
        {
            "Date": dates,
            "Open": 100.0,
            "Close": 100.0,
            "CashRate": 0.0,
            "TargetWeight": [0.80, 0.80, 0.90, 1.00],
            "DesiredTargetWeight": [0.80, 0.90, 1.00, 1.00],
            "DecisionDay": [False, True, True, False],
            "TriggerLevel": [
                "NO_FEAR",
                "MILD_FEAR",
                "PANIC",
                "PANIC",
            ],
            "SignalState": ["CORE", "MILD", "PANIC", "PANIC"],
            "TransitionReason": ["", "", "", ""],
        }
    )
    params = FearBuyParams(core_weight=0.80, tranche_weight=0.10)
    config = ContributionConfig(
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    policy = ContributionDeploymentPolicy(
        mild_fraction=0.25,
        fear_fraction=0.50,
        panic_fraction=1.0,
        cooldown_sessions=21,
    )

    result = run_contribution_backtest(
        signals,
        params,
        config,
        name="test",
        deployment_policy=policy,
    )

    mild_buy = result.trades[
        result.trades["Reason"] == "MILD_FEAR_ACCUMULATED_CASH_BUY"
    ]
    assert np.isclose(mild_buy["Notional"].sum(), 1_000.0)
    assert np.isclose(
        result.daily.loc[
            result.daily["Date"] == pd.Timestamp("2024-02-02"),
            "PendingContributionCash",
        ].iloc[0],
        3_000.0,
    )
    panic_buy = result.trades[
        result.trades["Reason"] == "PANIC_ACCUMULATED_CASH_BUY"
    ]
    assert np.isclose(panic_buy["Notional"].sum(), 3_000.0)
    assert result.daily.iloc[-1]["PendingContributionCash"] == 0.0
