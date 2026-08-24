from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tsla_integrated.pattern_regime import (
    NEUTRAL_PATTERN,
    PatternRegimeConfig,
    build_pattern_regime_model,
    calculate_pattern_scores,
    classify_pattern_regimes,
    summarize_forward_outcomes,
)


def _features(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=rows, freq="B"),
            "Return5D": [0.03] * rows,
            "Return20D": [0.15] * rows,
            "Return60D": [0.40] * rows,
            "RealizedVol21": [0.55] * rows,
            "VolumeSurprise21": [0.10] * rows,
            "MACDSpread": [0.01] * rows,
            "RSI14": [60.0] * rows,
            "DistanceFromSMA50": [0.15] * rows,
            "DistanceFromSMA200": [0.30] * rows,
            "MarginTrend": [0.04] * rows,
            "GrowthComposite": [0.30] * rows,
            "ShareGrowthYoYFiled": [0.02] * rows,
            "Treasury10Y": [3.0] * rows,
            "YieldCurve10Y2Y": [0.50] * rows,
            "VIX_Level": [17.0] * rows,
        }
    )


def test_scores_are_bounded_and_use_causal_news() -> None:
    features = _features()
    news = pd.DataFrame(
        {
            "Date": features["Date"],
            "NewsScore": [0.5] * 7 + [1.0],
            "NewsSignalActive": [False] * 7 + [True],
        }
    )
    scored = calculate_pattern_scores(features, news)
    pattern_columns = [
        "MOMENTUM_CONTINUATION",
        "WASHOUT_REVERSAL",
        "CATALYST_RERATING",
        "BLOWOFF_RISK",
        "BEAR_CONTINUATION_RISK",
    ]
    assert scored[pattern_columns].min().min() >= 0.0
    assert scored[pattern_columns].max().max() <= 1.0
    assert scored.loc[0, "CATALYST_RERATING"] == 0.0
    assert scored.loc[7, "CATALYST_RERATING"] > 0.6


def test_washout_requires_reversal_confirmation() -> None:
    falling = _features(10)
    falling["Return5D"] = -0.15
    falling["Return20D"] = -0.35
    falling["Return60D"] = -0.55
    falling["DistanceFromSMA50"] = -0.30
    falling["DistanceFromSMA200"] = -0.35
    falling["RSI14"] = list(range(34, 24, -1))
    falling["MACDSpread"] = [-0.02 - index * 0.002 for index in range(10)]
    reversing = falling.copy()
    reversing.loc[5:, "Return5D"] = 0.08
    reversing.loc[5:, "RSI14"] = [30, 34, 38, 42, 46]
    reversing.loc[5:, "MACDSpread"] = [-0.025, -0.020, -0.014, -0.008, -0.002]
    assert (
        calculate_pattern_scores(reversing).iloc[-1]["WASHOUT_REVERSAL"]
        > calculate_pattern_scores(falling).iloc[-1]["WASHOUT_REVERSAL"]
    )


def test_state_confirmation_and_neutral_margin() -> None:
    scores = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=5),
            "MOMENTUM_CONTINUATION": [0.70, 0.70, 0.70, 0.59, 0.59],
            "WASHOUT_REVERSAL": [0.20, 0.20, 0.20, 0.56, 0.56],
            "CATALYST_RERATING": [0.10] * 5,
            "BLOWOFF_RISK": [0.15] * 5,
            "BEAR_CONTINUATION_RISK": [0.25] * 5,
        }
    )
    result = classify_pattern_regimes(
        scores, PatternRegimeConfig(confirmation_sessions=3)
    )
    assert result.loc[0, "StablePattern"] == NEUTRAL_PATTERN
    assert result.loc[2, "StablePattern"] == "MOMENTUM_CONTINUATION"
    assert result.loc[3, "RawPattern"] == NEUTRAL_PATTERN


def test_model_does_not_mutate_input_and_summary_uses_net_rates() -> None:
    features = _features()
    original = features.copy(deep=True)
    result = build_pattern_regime_model(features)
    pd.testing.assert_frame_equal(features, original)
    result["ForwardReturnPct"] = [60, 40, -50, 0, 55, -10, -45, 5]
    summary = summarize_forward_outcomes(result)
    assert summary["Observations"].sum() == 8
    assert summary["Up50Count"].sum() == 2
    assert summary["Down40Count"].sum() == 2


def test_configuration_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        classify_pattern_regimes(
            calculate_pattern_scores(_features()),
            PatternRegimeConfig(confirmation_sessions=0),
        )
