from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.filing_signals import (
    FilingHybridConfig,
    add_filing_factors,
    add_filing_hybrid_scores,
    add_market_regime,
    generate_filing_hybrid_targets,
    merge_filing_features,
)


def test_merge_filing_features_prevents_future_information() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-04-30", "2025-05-02"]),
            "Ticker": ["AAA", "AAA"],
            "Eligible": [True, True],
        }
    )
    filings = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "AvailableDate": pd.to_datetime(["2025-05-01"]),
            "AccessionNumber": ["0001"],
            "Form": ["10-Q"],
            "RevenueGrowthYoYFiled": [0.25],
        }
    )
    merged = merge_filing_features(panel, filings)
    assert pd.isna(merged.iloc[0]["FiledRevenueGrowthYoY"])
    assert merged.iloc[1]["FiledRevenueGrowthYoY"] == pytest.approx(0.25)


def test_filing_factor_rewards_durability_and_low_leverage() -> None:
    rows = []
    for index, ticker in enumerate(("A", "B", "C", "D")):
        strength = 4 - index
        rows.append(
            {
                "Date": pd.Timestamp("2025-05-02"),
                "Ticker": ticker,
                "Eligible": True,
                "FiledRevenueGrowthYoY": strength / 10,
                "FiledNetIncomeGrowthYoY": strength / 10,
                "FiledGrossMargin": strength / 10,
                "FiledOperatingMargin": strength / 10,
                "FiledFreeCashFlowMargin": strength / 10,
                "FiledInterestCoverage": strength * 2,
                "FiledNetDebtToAssets": index / 10,
                "FiledDebtToEquity": index / 10,
                "FiledShareGrowthYoY": index / 100,
                "FiledStockCompensationToRevenue": index / 100,
                "FiledRiskTermsPer10kWords": index + 1,
                "FiledLeverageTermsPer10kWords": index + 1,
                "FiledDilutionTermsPer10kWords": index + 1,
            }
        )
    scored = add_filing_factors(
        pd.DataFrame(rows), minimum_cross_section_size=4
    ).set_index("Ticker")
    assert scored.loc["A", "FilingFundamentalFactor"] > scored.loc[
        "D", "FilingFundamentalFactor"
    ]


def test_hybrid_score_separates_core_and_tactical() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-05-02", "2025-05-02"]),
            "Ticker": ["CORE", "FAST"],
            "Eligible": [True, True],
            "GrowthFactor": [0.4, -0.1],
            "QualityFactor": [0.4, -0.1],
            "TrendFactor": [0.2, 0.4],
            "RiskControlFactor": [0.1, -0.1],
            "MAFactor": [0.2, 0.5],
            "MACDFactor": [0.2, 0.5],
            "OBVFactor": [0.2, 0.5],
            "MomentumFactor": [0.2, 0.5],
            "Trend200": [0.1, 0.3],
            "Return126": [0.2, 0.6],
            "FilingDurabilityFactor": [0.4, 0.0],
            "FilingBalanceSheetFactor": [0.4, 0.0],
            "FilingDisclosureSafetyFactor": [0.2, 0.0],
            "FilingFundamentalFactor": [0.4, 0.0],
            "FiledGoingConcernFlag": [False, False],
            "FiledMaterialWeaknessFlag": [False, False],
            "FiledRestatementFlag": [False, False],
        }
    )
    scored = add_filing_hybrid_scores(frame).set_index("Ticker")
    assert bool(scored.loc["CORE", "FilingCoreQualified"])
    assert bool(scored.loc["FAST", "FilingTacticalQualified"])
    assert scored.loc["CORE", "FilingCoreRank"] == pytest.approx(1)
    assert scored.loc["FAST", "FilingTacticalRank"] == pytest.approx(1)


def test_tactical_hard_stop_replaces_failed_position() -> None:
    config = FilingHybridConfig(
        core_slots=1,
        core_total_weight=0.70,
        tactical_slots=1,
        tactical_position_weight=0.15,
    )
    dates = pd.to_datetime(["2025-01-03", "2025-01-10"])
    rows = []
    for date_index, date in enumerate(dates):
        for ticker in ("CORE", "FAST", "NEXT"):
            fast_close = 100.0 if date_index == 0 else 80.0
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Close": fast_close if ticker == "FAST" else 100.0,
                    "FilingCoreQualified": ticker == "CORE",
                    "FilingCoreRank": 1.0 if ticker == "CORE" else 3.0,
                    "FilingTacticalQualified": ticker in {"FAST", "NEXT"},
                    "FilingTacticalRank": 1.0 if ticker == "FAST" else 2.0,
                    "GrowthFactor": 0.2,
                    "QualityFactor": 0.2,
                    "FilingFundamentalFactor": 0.2,
                    "MomentumFactor": 0.2,
                    "TrendFactor": 0.2,
                    "FiledGoingConcernFlag": False,
                    "FiledMaterialWeaknessFlag": False,
                    "FiledRestatementFlag": False,
                }
            )
    targets = generate_filing_hybrid_targets(pd.DataFrame(rows), config)
    fast = targets.loc[targets["Ticker"].eq("FAST")]
    assert fast.iloc[-1]["TradeAction"] == "SELL"
    assert fast.iloc[-1]["Reason"] == "TACTICAL_HARD_STOP"
    latest = targets.loc[targets["Date"].eq(dates[-1])]
    assert latest.loc[latest["Ticker"].eq("NEXT"), "TradeAction"].iloc[0] == "BUY"


def test_market_regime_uses_hysteresis_without_future_prices() -> None:
    dates = pd.date_range("2024-01-01", periods=205, freq="B")
    close = [100.0] * 200 + [98.0, 99.5, 100.0, 101.5, 100.0]
    spy = pd.DataFrame({"Date": dates, "Close": close})
    panel = pd.DataFrame({"Date": dates, "Ticker": "AAA"})
    result = add_market_regime(
        panel,
        spy,
        FilingHybridConfig(
            market_regime_band=0.01,
            market_regime_fast_sessions=1,
        ),
    )
    assert not bool(result.loc[result["Date"].eq(dates[200]), "MarketRiskOn"].iloc[0])
    assert not bool(result.loc[result["Date"].eq(dates[202]), "MarketRiskOn"].iloc[0])
    assert bool(result.loc[result["Date"].eq(dates[203]), "MarketRiskOn"].iloc[0])


def test_market_risk_off_exits_tactical_and_scales_core() -> None:
    config = FilingHybridConfig(
        core_slots=1,
        core_total_weight=0.70,
        risk_off_core_total_weight=0.40,
        tactical_slots=1,
        tactical_position_weight=0.15,
    )
    rows = []
    for date, risk_on in (
        (pd.Timestamp("2025-01-03"), True),
        (pd.Timestamp("2025-01-10"), False),
    ):
        for ticker in ("CORE", "FAST"):
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Close": 100.0,
                    "MarketRiskOn": risk_on,
                    "FilingCoreQualified": ticker == "CORE",
                    "FilingCoreRank": 1.0,
                    "FilingTacticalQualified": ticker == "FAST",
                    "FilingTacticalRank": 1.0,
                    "GrowthFactor": 0.2,
                    "QualityFactor": 0.2,
                    "FilingFundamentalFactor": 0.2,
                    "MomentumFactor": 0.2,
                    "TrendFactor": 0.2,
                    "FiledGoingConcernFlag": False,
                    "FiledMaterialWeaknessFlag": False,
                    "FiledRestatementFlag": False,
                }
            )
    targets = generate_filing_hybrid_targets(pd.DataFrame(rows), config)
    risk_off = targets.loc[targets["Date"].eq(pd.Timestamp("2025-01-10"))]
    assert risk_off.loc[risk_off["Ticker"].eq("CORE"), "TargetWeight"].iloc[0] == pytest.approx(0.40)
    assert risk_off.loc[risk_off["Ticker"].eq("FAST"), "TargetWeight"].iloc[0] == pytest.approx(0.0)
    assert risk_off.loc[risk_off["Ticker"].eq("FAST"), "Reason"].iloc[0] == "MARKET_RISK_OFF"


def test_core_replacement_requires_two_confirmations() -> None:
    config = FilingHybridConfig(
        core_slots=1,
        core_total_weight=0.70,
        core_minimum_hold_weeks=0,
        core_replacement_confirm_weeks=2,
        tactical_slots=1,
        tactical_position_weight=0.15,
    )
    rows = []
    dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
    for position, date in enumerate(dates):
        for ticker in ("OLD", "NEW"):
            new_is_best = position > 0 and ticker == "NEW"
            old_is_initial_best = position == 0 and ticker == "OLD"
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Close": 100.0,
                    "MarketRiskOn": True,
                    "FilingCoreQualified": True,
                    "FilingCoreRank": (
                        1.0 if new_is_best or old_is_initial_best else 2.0
                    ),
                    "FilingCoreScore": (
                        0.5 if new_is_best else 0.3 if old_is_initial_best else 0.2
                    ),
                    "FilingTacticalQualified": False,
                    "FilingTacticalRank": float("nan"),
                    "GrowthFactor": 0.2,
                    "QualityFactor": 0.2,
                    "FilingFundamentalFactor": 0.2,
                    "MomentumFactor": 0.2,
                    "TrendFactor": 0.2,
                    "FiledGoingConcernFlag": False,
                    "FiledMaterialWeaknessFlag": False,
                    "FiledRestatementFlag": False,
                }
            )
    targets = generate_filing_hybrid_targets(pd.DataFrame(rows), config)
    changes = targets.loc[targets["TradeAction"].isin(["BUY", "SELL"])]
    replacement = changes.loc[changes["Date"].eq(dates[-1])]
    assert (
        replacement.set_index("Ticker").loc["OLD", "Reason"]
        == "CORE_STRONGER_CHALLENGER"
    )
    assert (
        replacement.set_index("Ticker").loc["NEW", "Reason"]
        == "CORE_CONFIRMED_STRONGER_VALUE"
    )
