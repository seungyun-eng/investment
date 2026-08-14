from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.macro_momentum_sp500.config import ResearchConfig
from stock_research.macro_momentum_sp500.evaluation import _sample_rows, nested_walk_forward
from stock_research.macro_momentum_sp500.models import CandidateSpec, candidate_specs
from stock_research.macro_momentum_sp500.portfolio import (
    load_prior_v2_oos_benchmark,
    prediction_target_weights,
    run_weight_backtest,
    stateful_macro_target_weights,
    trade_cycle_diagnostics,
)
from stock_research.macro_momentum_sp500.targets import build_targets


def test_candidate_budget_covers_linear_tree_and_feature_groups() -> None:
    config = ResearchConfig(search_budget_per_task=24)
    classification = candidate_specs("classification", config)
    regression = candidate_specs("regression", config)

    assert len(classification) == 24
    assert len(regression) == 24
    assert {item.family for item in classification} == {
        "logistic",
        "hist_gradient_boosting",
    }
    assert {item.family for item in regression} == {
        "ridge",
        "hist_gradient_boosting",
    }
    assert {item.feature_group for item in classification} == set(config.feature_groups)


def test_training_stride_keeps_time_ordered_observations() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-05", "2024-01-01", "2024-01-04", "2024-01-02", "2024-01-03"]
            ),
            "Value": range(5),
        }
    )

    sampled = _sample_rows(frame, 2)

    assert sampled["Date"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])
    )


def test_allocation_requires_joint_risk_and_negative_return() -> None:
    config = ResearchConfig()
    predictions = pd.DataFrame(
        {
            "RiskProbability_21": [0.8, 0.8, 0.4],
            "RiskProbability_63": [0.8, 0.8, 0.4],
            "RiskProbability_126": [0.8, 0.8, 0.4],
            "PredictedExcessReturn_63": [-0.1, 0.1, -0.1],
            "PredictedExcessReturn_126": [-0.1, 0.1, -0.1],
        }
    )
    weights = prediction_target_weights(predictions, config)

    assert weights["TargetWeight"].tolist() == [0.25, 1.0, 1.0]


def _state_predictions(
    *,
    periods: int = 30,
    close: float = 100.0,
    risk: float = 0.40,
    macro: float = 0.40,
) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "Close": close,
            "CashRate": 0.0,
            "MacroConfirmationScore": macro,
            "RiskProbability_63": risk,
            "RiskProbability_126": risk,
            "PredictedExcessReturn_63": -0.01,
            "PredictedExcessReturn_126": -0.01,
        }
    )


def test_stateful_macro_uses_weekly_confirmation_and_minimum_hold() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=1,
        state_entry_confirmations=1,
        state_exit_confirmations=1,
        state_normal_cooldown_days=1,
        state_min_hold_days=10,
        state_recovery_days=5,
    )
    predictions = _state_predictions()
    predictions.loc[4:, ["RiskProbability_63", "RiskProbability_126"]] = 0.60
    predictions.loc[4:, "MacroConfirmationScore"] = 0.60
    predictions.loc[9:, ["RiskProbability_63", "RiskProbability_126"]] = 0.40
    predictions.loc[9:, "MacroConfirmationScore"] = 0.40

    signals = stateful_macro_target_weights(predictions, config)

    assert signals.loc[4, "SignalState"] == "CAUTION"
    assert signals.loc[9, "SignalState"] == "CAUTION"
    assert signals.loc[14, "SignalState"] == "RECOVERY"
    assert signals.loc[19, "SignalState"] == "NORMAL"


def test_stateful_macro_blocks_weak_loss_sale_but_allows_emergency() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=1,
        state_entry_confirmations=1,
        state_normal_cooldown_days=1,
    )
    predictions = _state_predictions(periods=10)
    predictions.loc[4, "Close"] = 95.0
    predictions.loc[4, ["RiskProbability_63", "RiskProbability_126"]] = 0.56
    predictions.loc[4, "MacroConfirmationScore"] = 0.56

    blocked = stateful_macro_target_weights(predictions, config)

    assert blocked.loc[4, "SignalState"] == "NORMAL"
    assert bool(blocked.loc[4, "LossGateBlocked"])

    predictions.loc[4, ["RiskProbability_63", "RiskProbability_126"]] = 0.85
    predictions.loc[4, "MacroConfirmationScore"] = 0.70
    emergency = stateful_macro_target_weights(predictions, config)

    assert emergency.loc[4, "SignalState"] == "DEFENSIVE"
    assert emergency.loc[4, "TransitionReason"] == "EMERGENCY_RISK_AND_MACRO"


def test_stateful_macro_ignores_midweek_threshold_spike() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=1,
        state_entry_confirmations=1,
        state_normal_cooldown_days=1,
    )
    predictions = _state_predictions(periods=10)
    predictions.loc[2, ["RiskProbability_63", "RiskProbability_126"]] = 0.75
    predictions.loc[2, "MacroConfirmationScore"] = 0.62

    signals = stateful_macro_target_weights(predictions, config)

    assert signals["SignalState"].eq("NORMAL").all()


def test_stateful_macro_early_warning_gate_can_trigger_independently_of_macro_and_risk() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=1,
        state_entry_confirmations=1,
        state_normal_cooldown_days=1,
    )
    predictions = _state_predictions(periods=15, risk=0.10, macro=0.40)
    early_warning = pd.DataFrame(
        {
            "Date": predictions["Date"],
            "EarlyWarningCaution": [False] * 4 + [True] * 11,
            "EarlyWarningDefensive": False,
        }
    )

    without_gate = stateful_macro_target_weights(predictions, config)
    with_gate = stateful_macro_target_weights(predictions, config, early_warning=early_warning)

    assert without_gate["SignalState"].eq("NORMAL").all()
    assert with_gate.loc[4, "SignalState"] == "CAUTION"
    assert with_gate.loc[4, "TransitionReason"] == "CONFIRMED_EARLY_WARNING_CAUTION"


def test_stateful_macro_early_warning_gate_defaults_to_unchanged_behaviour() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=1,
        state_entry_confirmations=1,
        state_normal_cooldown_days=1,
    )
    predictions = _state_predictions(periods=15, risk=0.10, macro=0.40)
    predictions.loc[4:, "MacroConfirmationScore"] = 0.76

    without_param = stateful_macro_target_weights(predictions, config)
    with_empty_gate = stateful_macro_target_weights(
        predictions,
        config,
        early_warning=pd.DataFrame(
            {
                "Date": predictions["Date"],
                "EarlyWarningCaution": False,
                "EarlyWarningDefensive": False,
            }
        ),
    )

    pd.testing.assert_frame_equal(without_param, with_empty_gate)


def test_stateful_macro_can_defend_without_model_risk_confirmation() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=1,
        state_entry_confirmations=1,
        state_normal_cooldown_days=1,
    )
    predictions = _state_predictions(periods=15, risk=0.10, macro=0.40)
    predictions.loc[4:, "MacroConfirmationScore"] = 0.76

    signals = stateful_macro_target_weights(predictions, config)

    assert signals.loc[4, "SignalState"] == "DEFENSIVE"
    assert (
        signals.loc[4, "TransitionReason"]
        == "CONFIRMED_MACRO_ONLY_DEFENSIVE"
    )


def test_stateful_macro_only_emergency_can_trigger_midweek() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=1,
    )
    predictions = _state_predictions(periods=10, risk=0.10, macro=0.40)
    predictions.loc[2, "MacroConfirmationScore"] = 0.86

    signals = stateful_macro_target_weights(predictions, config)

    assert signals.loc[2, "SignalState"] == "DEFENSIVE"
    assert signals.loc[2, "TransitionReason"] == "EMERGENCY_MACRO_SHOCK"


def test_stateful_macro_uses_momentum_to_begin_recovery() -> None:
    config = ResearchConfig(
        state_risk_smoothing_days=1,
        state_macro_smoothing_days=2,
        state_entry_confirmations=1,
        state_exit_confirmations=1,
        state_normal_cooldown_days=1,
        state_min_hold_days=3,
        state_momentum_days=2,
        state_trend_sma_days=3,
    )
    predictions = _state_predictions(periods=15, risk=0.40, macro=0.40)
    predictions["Open"] = np.arange(100.0, 115.0)
    predictions["Close"] = predictions["Open"]
    predictions.loc[4, ["RiskProbability_63", "RiskProbability_126"]] = 0.60
    predictions.loc[4, "MacroConfirmationScore"] = 0.80
    predictions.loc[5:, "MacroConfirmationScore"] = np.linspace(0.59, 0.53, 10)

    signals = stateful_macro_target_weights(predictions, config)

    assert signals.loc[4, "SignalState"] == "CAUTION"
    assert signals.loc[9, "MacroScore"] > config.state_exit_macro
    assert bool(signals.loc[9, "MomentumRecovery"])
    assert signals.loc[9, "SignalState"] == "RECOVERY"


def test_portfolio_executes_previous_close_signal_at_next_open() -> None:
    config = ResearchConfig(
        transaction_cost_bps=0,
        slippage_bps=0,
        rebalance_band=0.01,
    )
    dates = pd.bdate_range("2024-01-01", periods=3)
    signals = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 110.0, 120.0],
            "Close": [100.0, 110.0, 120.0],
            "CashRate": [0.0, 0.0, 0.0],
            "TargetWeight": [0.0, 1.0, 1.0],
            "SignalState": ["DEFENSIVE", "NORMAL", "NORMAL"],
        }
    )
    result = run_weight_backtest(signals, config, name="test")

    assert result.trades.loc[0, "Date"] == dates[0]
    assert result.trades.loc[0, "Action"] == "BUY"
    assert result.trades.loc[1, "Date"] == dates[1]
    assert result.trades.loc[1, "Action"] == "SELL"
    assert result.trades.loc[1, "SignalDate"] == dates[0]
    assert np.isclose(
        result.summary.roi_percent,
        (result.summary.final_value / result.summary.total_injected - 1) * 100,
    )


def test_trade_cycle_diagnostics_pairs_partial_exposure_states() -> None:
    config = ResearchConfig(
        transaction_cost_bps=0,
        slippage_bps=0,
        rebalance_band=0.01,
    )
    dates = pd.bdate_range("2024-01-01", periods=5)
    signals = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 98.0, 97.0, 101.0, 102.0],
            "Close": [100.0, 98.0, 97.0, 101.0, 102.0],
            "CashRate": 0.0,
            "TargetWeight": [0.70, 0.70, 1.0, 1.0, 1.0],
            "SignalState": ["CAUTION", "CAUTION", "NORMAL", "NORMAL", "NORMAL"],
        }
    )
    result = run_weight_backtest(signals, config, name="test")

    cycles = trade_cycle_diagnostics(result)

    assert cycles["CycleType"].tolist() == [
        "NORMAL_TO_DEFENSIVE",
        "DEFENSIVE_TO_NORMAL",
    ]
    assert bool(cycles.loc[0, "Adverse"])
    assert bool(cycles.loc[1, "Adverse"])


def test_outer_model_selection_receives_only_purged_labels(monkeypatch) -> None:
    config = ResearchConfig(
        training_years=3,
        first_test_year=2003,
        inner_validation_years=2,
        return_horizons=(2,),
        risk_horizons=(2,),
        primary_return_horizon=2,
        primary_risk_horizon=2,
        minimum_train_rows=20,
        search_budget_per_task=1,
    )
    dates = pd.bdate_range("2000-01-03", "2004-12-31")
    close = 100 + np.arange(len(dates)) * 0.02 + np.sin(np.arange(len(dates)) / 20)
    features = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "Close": close,
            "Volume": 1_000_000,
            "CashRate": 0.0,
            "VIX": 20.0,
            "Drawdown252": 0.0,
            "Momentum_5": pd.Series(close).pct_change(5),
        }
    )
    targets = build_targets(features, config)
    selection_calls: list[int] = []

    def fake_select(
        train,
        _features,
        _target,
        target_end,
        task,
        outer_year,
        _config,
    ):
        assert train[target_end].max() < pd.Timestamp(outer_year, 1, 1)
        selection_calls.append(outer_year)
        family = "logistic" if task == "classification" else "ridge"
        params = (("C", 1.0), ("balanced", False)) if task == "classification" else (("alpha", 1.0),)
        return CandidateSpec(task, family, "price", params), []

    def fake_fit_predict(
        _train,
        test,
        _features,
        _target,
        _target_end,
        candidate,
        _test_start,
        _config,
        *,
        collect_importance=False,
    ):
        value = 0.25 if candidate.task == "classification" else 0.01
        predicted = value + np.linspace(0, 0.001, len(test))
        importance = (
            pd.DataFrame(
                {
                    "Feature": ["Momentum_5"],
                    "ImportanceMagnitude": [1.0],
                    "SignedEffect": [1.0],
                    "NormalizedImportance": [1.0],
                }
            )
            if collect_importance
            else pd.DataFrame()
        )
        return predicted, ["Momentum_5"], len(_train), importance

    monkeypatch.setattr(
        "stock_research.macro_momentum_sp500.evaluation._select_candidate",
        fake_select,
    )
    monkeypatch.setattr(
        "stock_research.macro_momentum_sp500.evaluation._fit_predict_target",
        fake_fit_predict,
    )

    result = nested_walk_forward(features, targets, config)

    assert selection_calls
    assert not result.predictions.empty
    assert set(result.feature_importance["Feature"]) == {"Momentum_5"}


def test_prior_v2_is_rebased_to_same_comparison_capital(tmp_path) -> None:
    folder = tmp_path / "macro_sp500_v2"
    folder.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=4)
    pd.DataFrame(
        {
            "Date": dates,
            "TotalValue": [200.0, 220.0, 210.0, 240.0],
            "ActualWeight": [0.8] * 4,
            "Cash": [40.0] * 4,
        }
    ).to_csv(folder / "1_oos_daily_20200101.csv", index=False)
    pd.DataFrame(
        {
            "Date": [dates[1]],
            "Action": ["BUY"],
            "Notional": [20.0],
            "Fee": [0.1],
        }
    ).to_csv(folder / "1_oos_rebalances_20200101.csv", index=False)

    result = load_prior_v2_oos_benchmark(
        folder,
        start=dates[0],
        end=dates[-1],
        config=ResearchConfig(),
    )

    assert result is not None
    assert result.daily["TotalValue"].iloc[0] == 100_000
    assert result.daily["TotalValue"].iloc[-1] == 120_000
    assert np.isclose(result.summary.roi_percent, 20.0)
    assert "strict-OOS" in result.source
