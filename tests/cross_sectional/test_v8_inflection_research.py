from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_research.cross_sectional.v8_inflection_research import (
    V81InflectionConfig,
    V82InflectionConfig,
    add_forward_multibagger_labels,
    add_inflection_observations,
    add_v81_scores,
    add_v82_scores,
    generate_v81_targets,
)


def test_forward_labels_use_only_strictly_future_sessions() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=6),
            "Ticker": ["A"] * 6,
            "Close": [1.0, 2.0, 4.0, 3.0, 5.0, 6.0],
        }
    )
    labeled = add_forward_multibagger_labels(
        frame, horizon_24m=2, horizon_36m=3
    )
    first = labeled.iloc[0]
    assert first["ForwardReturn24m"] == 3.0
    assert first["ForwardMaxReturn24m"] == 3.0
    assert bool(first["Label4x24m"])
    assert np.isnan(labeled.iloc[-2]["ForwardMaxReturn24m"])


def test_inflection_observations_change_only_on_new_financial_period() -> None:
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["A"] * len(dates),
            "Close": np.linspace(100.0, 160.0, len(dates)),
            "Volume": np.linspace(1_000_000, 1_400_000, len(dates)),
            "FinancialPeriodEnd": pd.Timestamp("2023-12-31"),
            "RevenueGrowthYoY": 0.20,
            "OperatingMargin": 0.10,
            "EpsTtm": 2.0,
            "EbitdaTtm": 100.0,
        }
    )
    frame.loc[frame.index >= 200, "FinancialPeriodEnd"] = pd.Timestamp(
        "2024-03-31"
    )
    frame.loc[frame.index >= 200, "RevenueGrowthYoY"] = 0.35
    frame.loc[frame.index >= 200, "OperatingMargin"] = 0.14
    observed = add_inflection_observations(frame)
    last = observed.iloc[-1]
    assert last["RevenueGrowthAcceleration"] == pytest.approx(0.15)
    assert last["OperatingMarginSequentialChange"] == pytest.approx(0.04)
    assert observed["Return252"].notna().sum() == 8


def test_v81_separates_secular_and_cyclical_recovery() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-28"] * 3),
            "Ticker": ["SEC", "CYC", "NOISE"],
            "Eligible": [True, True, True],
            "Close": [100.0, 50.0, 20.0],
            "Shares": [2_000.0, 1_000.0, 1_000.0],
            "GrowthFactor": [0.4, 0.2, 0.4],
            "QualityFactor": [0.4, 0.1, 0.2],
            "MAFactor": [0.3, 0.3, 0.3],
            "MACDFactor": [0.3, 0.3, 0.3],
            "OBVFactor": [0.3, 0.3, 0.3],
            "MomentumFactor": [0.3, 0.3, 0.3],
            "RiskControlFactor": [0.1, 0.1, 0.1],
            "Trend200": [0.2, 0.2, 0.2],
            "Return126": [0.3, 0.3, 0.3],
            "Return252": [0.5, 0.5, 0.5],
            "HighProximity252": [-0.05, -0.05, -0.05],
            "Volume20To126": [1.2, 1.2, 1.2],
            "MACDLineNormalized": [0.02, 0.02, 0.02],
            "OBVFlow63": [0.2, 0.2, 0.2],
            "EpsTtm": [3.0, 0.5, 0.01],
            "PriorEpsTtm": [2.0, -0.5, 0.001],
            "EbitdaTtm": [200.0, 50.0, 2.0],
            "PriorEbitdaTtm": [150.0, -20.0, 1.0],
            "EpsTtmSequentialChange": [1.0, 1.0, 0.009],
            "EbitdaTtmSequentialChange": [50.0, 70.0, 1.0],
            "EpsTtmGrowthYoY": [0.5, 0.4, 9.0],
            "EpsTtmGrowthAcceleration": [0.1, 0.2, 8.0],
            "EbitdaTtmGrowthYoY": [0.4, 1.2, 1.0],
            "EbitdaTtmGrowthAcceleration": [0.1, 0.8, 0.9],
            "DcfPriceGrowthYoY": [0.3, 0.4, 0.5],
            "RevenueGrowthYoY": [0.30, 0.15, -0.05],
            "RevenueGrowthAcceleration": [0.05, 0.20, 1.0],
            "OperatingMarginChangeYoY": [0.03, 0.08, -0.02],
            "OperatingMarginSequentialChange": [0.01, 0.04, 0.10],
            "FreeCashFlowMargin": [0.20, -0.05, 0.10],
        }
    )
    scored = add_v81_scores(frame, V81InflectionConfig())
    values = scored.set_index("Ticker")
    assert bool(values.loc["SEC", "SecularAccelerationQualified"])
    assert bool(values.loc["CYC", "CyclicalRecoveryQualified"])
    assert not bool(values.loc["NOISE", "InflectionQualifiedV81"])
    v82 = add_v82_scores(frame, V82InflectionConfig()).set_index("Ticker")
    assert bool(v82.loc["NOISE", "MarginPriceBreakoutQualified"])
    assert (
        v82.loc["NOISE", "SignalArchetype"]
        == "MARGIN_PRICE_BREAKOUT"
    )


def test_v81_scales_only_after_distinct_new_quarters() -> None:
    config = V81InflectionConfig(
        core_slots=1,
        core_total_weight=0.70,
        inflection_slots=1,
        inflection_scout_weight=0.10,
        inflection_confirm_weight=0.20,
        inflection_max_weight=0.30,
    )
    dates = pd.to_datetime(
        [
            "2025-01-03",
            "2025-01-10",
            "2025-01-17",
            "2025-04-04",
            "2025-07-04",
        ]
    )
    periods = pd.to_datetime(
        [
            "2024-12-31",
            "2024-12-31",
            "2024-12-31",
            "2025-03-31",
            "2025-06-30",
        ]
    )
    rows = []
    for date, period in zip(dates, periods, strict=True):
        rows.append(
            {
                "Date": date,
                "Ticker": "FAST",
                "Company": "Fast",
                "Close": 100.0,
                "CoreQualified": False,
                "CoreRank": np.nan,
                "CoreScore": 0.2,
                "GrowthFactor": 0.3,
                "QualityFactor": 0.3,
                "InflectionQualifiedV81": True,
                "InflectionRankV81": 1.0,
                "InflectionScoreV81": 0.4,
                "SignalArchetype": "SECULAR_ACCELERATION",
                "FinancialPeriodEnd": period,
                "RevenueGrowthYoY": 0.3,
                "OperatingMarginChangeYoY": 0.03,
                "EpsTtmGrowthYoY": 0.4,
                "EbitdaTtmGrowthYoY": 0.4,
                "Trend200": 0.2,
                "Return126": 0.3,
                "PriceVolumeEvidenceScore": 0.2,
            }
        )
    targets = generate_v81_targets(pd.DataFrame(rows), config)
    fast = targets.loc[targets["Ticker"].eq("FAST")]
    assert list(fast["TradeAction"]) == ["BUY", "SCALE_UP", "SCALE_UP"]
    assert list(fast["TargetWeight"]) == [0.1, 0.2, 0.3]
    assert list(fast["ConfirmedNewQuarters"]) == [0, 1, 2]
