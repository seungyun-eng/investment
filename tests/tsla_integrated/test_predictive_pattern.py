from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.tsla_integrated.predictive_pattern import (
    add_pattern_phase_flags,
    add_two_month_excursion_targets,
    classify_predictive_pattern,
    build_pattern_execution_signals,
    prequential_binary_probability,
    prepare_news_features,
)


def test_two_month_targets_use_first_session_on_or_after_anniversary() -> None:
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-02", "2025-02-03", "2025-03-03", "2025-03-04"]
            ),
            "AdjClose": [100.0, 120.0, 90.0, 150.0],
        }
    )
    result = add_two_month_excursion_targets(prices)
    assert result.loc[0, "TargetEndDate"] == pd.Timestamp("2025-03-03")
    assert result.loc[0, "ForwardMaxGain2M"] == pytest.approx(0.20)
    assert result.loc[0, "ForwardMaxDrawdown2M"] == pytest.approx(-0.10)
    assert result.loc[0, "ForwardTerminalReturn2M"] == pytest.approx(-0.10)


def _prequential_frame(rows: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=rows, freq="B")
    feature = np.sin(np.arange(rows) / 20.0)
    target = np.roll(feature, -10)
    target[-10:] = np.nan
    return pd.DataFrame(
        {
            "Date": dates,
            "TargetEndDate": dates + pd.offsets.BDay(10),
            "Feature": feature,
            "Target": target,
        }
    )


def test_prequential_training_excludes_unrealized_labels() -> None:
    frame = _prequential_frame()
    result = prequential_binary_probability(
        frame,
        feature_columns=["Feature"],
        target_column="Target",
        positive_threshold=0.0,
        positive_when_above=True,
        c_value=0.1,
        first_prediction_date="2020-01-01",
        first_training_date="2018-01-01",
        minimum_training_rows=300,
    )
    first = result.loc[result["Probability"].notna()].iloc[0]
    prediction_date = pd.Timestamp(first["Date"])
    expected = (
        frame["TargetEndDate"].lt(prediction_date) & frame["Target"].notna()
    ).sum()
    assert first["TrainingRows"] == expected


def test_news_features_are_neutral_when_missing() -> None:
    result = prepare_news_features(pd.DataFrame({"Date": ["2025-01-01"]}))
    assert result.loc[0, "NewsDirection"] == 0.0
    assert result.loc[0, "NewsSignalActiveNumeric"] == 0.0
    assert result.loc[0, "EventClusters"] == 0.0


def test_predictive_pattern_keeps_conflicts_explicit() -> None:
    frame = pd.DataFrame(
        {
            "OpportunityPercentile": [0.9, 0.2, 0.9],
            "RiskPercentile": [0.2, 0.9, 0.9],
            "MOMENTUM_CONTINUATION": [0.8, 0.2, 0.8],
            "WASHOUT_REVERSAL": [0.3, 0.1, 0.3],
            "CATALYST_RERATING": [0.1, 0.0, 0.1],
            "BLOWOFF_RISK": [0.2, 0.3, 0.2],
            "BEAR_CONTINUATION_RISK": [0.3, 0.8, 0.3],
        }
    )
    result = classify_predictive_pattern(frame, include_catalyst=False)
    assert result.loc[0, "PredictivePattern"] == "MOMENTUM_CONTINUATION"
    assert result.loc[1, "PredictivePattern"] == "BEAR_CONTINUATION_RISK"
    assert result.loc[2, "PredictivePattern"] == "CONFLICT_HIGH_BOTH"


def test_catalyst_is_out_of_scope_by_default() -> None:
    frame = pd.DataFrame(
        {
            "OpportunityPercentile": [0.9],
            "RiskPercentile": [0.1],
            "MOMENTUM_CONTINUATION": [0.2],
            "WASHOUT_REVERSAL": [0.3],
            "CATALYST_RERATING": [0.95],
            "BLOWOFF_RISK": [0.1],
            "BEAR_CONTINUATION_RISK": [0.1],
        }
    )
    result = classify_predictive_pattern(frame)
    assert result.loc[0, "PredictivePattern"] == "WASHOUT_REVERSAL"


def test_phase_flags_distinguish_washout_setup_from_confirmed_reversal() -> None:
    frame = pd.DataFrame(
        {
            "Return5D": [-0.05, 0.08],
            "Return60D": [-0.40, -0.40],
            "RSI14": [25.0, 31.0],
            "DistanceFromSMA200": [-0.25, -0.25],
            "WASHOUT_REVERSAL": [0.40, 0.70],
            "CATALYST_RERATING": [0.0, 0.0],
            "BLOWOFF_RISK": [0.0, 0.0],
            "OpportunityPercentile": [0.50, 0.80],
            "RiskPercentile": [0.90, 0.20],
        }
    )
    result = add_pattern_phase_flags(frame)
    assert result.loc[0, "DetailedPatternPhase"] == "WASHOUT_FALLING_KNIFE_RISK"
    assert result.loc[1, "DetailedPatternPhase"] == "WASHOUT_REVERSAL_HIGH_EDGE"


def test_execution_signals_use_only_selected_non_catalyst_policy() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=3),
            "AdjOpen": [100.0, 90.0, 95.0],
            "AdjClose": [100.0, 88.0, 96.0],
            "FedFundsRate": [4.0, 4.0, 4.0],
            "WashoutSetup": [False, True, True],
            "WashoutReversalConfirmed": [False, False, True],
            "CatalystConfirmed": [False, True, False],
        }
    )
    setup = build_pattern_execution_signals(frame, scale_in_policy="washout_setup")
    confirmed = build_pattern_execution_signals(frame, scale_in_policy="confirmed")
    assert setup["BearRegime"].tolist() == [False, True, True]
    assert confirmed["BearRegime"].tolist() == [False, False, True]
    assert setup.loc[1, "CompositeScore"] == 0.0
