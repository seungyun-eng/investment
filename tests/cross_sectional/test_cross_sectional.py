from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.config import (
    ResearchSettings,
    StrategyParams,
)
from stock_research.cross_sectional.features import (
    _build_available_financials,
    add_cross_sectional_factors,
    build_equity_features,
)
from stock_research.cross_sectional.optimization import _candidate_parameters
from stock_research.cross_sectional.portfolio import run_portfolio_backtest
from stock_research.cross_sectional.signals import (
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)
from stock_research.cross_sectional.v6_reporting import (
    build_position_ledger,
    frozen_v6_variants,
)


def _settings() -> ResearchSettings:
    return ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        minimum_price_history_sessions=3,
        minimum_cross_section_size=1,
    )


def _params() -> StrategyParams:
    return StrategyParams(
        momentum_weight=1.0,
        trend_weight=0.0,
        growth_weight=0.0,
        quality_weight=0.0,
        risk_control_weight=0.0,
        top_k=1,
        exit_rank=2,
        trend_floor=-1.0,
        momentum_floor=-1.0,
    )


def test_validation_must_not_overlap_training() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        ResearchSettings(
            train_start="2020-01-01",
            train_end="2024-12-31",
            validation_periods={"bad": ("2024-12-31", "2025-01-02")},
        )


def test_selection_label_must_name_a_validation_period() -> None:
    with pytest.raises(ValueError, match="selection_validation_label"):
        ResearchSettings(
            train_start="2020-01-01",
            train_end="2024-12-31",
            validation_periods={"2025": ("2025-01-01", "2025-12-31")},
            selection_validation_label="2026",
        )


def test_financial_weight_constraint_applies_to_every_candidate() -> None:
    settings = ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        candidate_count=25,
        minimum_financial_weight=0.30,
    )
    candidates = _candidate_parameters(settings)
    assert len(candidates) == 25
    assert all(candidate.financial_weight >= 0.30 for candidate in candidates)


def test_financial_features_obey_release_lag_and_staleness() -> None:
    dates = pd.date_range("2020-01-01", periods=230, freq="B")
    prices = pd.DataFrame(
        {
            "Date": dates,
            "Open": range(1, 231),
            "Close": range(1, 231),
            "High": range(1, 231),
            "Low": range(1, 231),
        }
    )
    financials = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2018-12-31",
                    "2019-03-31",
                    "2019-06-30",
                    "2019-09-30",
                    "2019-12-31",
                ]
            ),
            "Revenue": [10, 11, 12, 13, 20],
            "Operating Income": [1, 1, 1, 1, 4],
            "Cash Flow From Operating Activities": [2, 2, 2, 2, 4],
            "Net Change In Property, Plant, And Equipment": [-1, -1, -1, -1, -1],
            "Total Assets": [20, 20, 20, 20, 25],
            "Cash On Hand": [5, 5, 5, 5, 8],
            "Long Term Debt": [2, 2, 2, 2, 2],
        }
    )
    features = build_equity_features(
        prices,
        financials,
        ticker="TEST",
        company="Test",
        settings=_settings(),
    )
    before = features.loc[features["Date"].eq("2020-02-13")].iloc[0]
    after = features.loc[features["Date"].eq("2020-02-14")].iloc[0]
    assert pd.isna(before["RevenueGrowthYoY"])
    assert after["RevenueGrowthYoY"] == pytest.approx(1.0)
    stale = features.loc[features["Date"].eq("2020-09-01")].iloc[0]
    assert stale["FinancialStale"]
    assert pd.isna(stale["RevenueGrowthYoY"])


def test_ttm_financial_momentum_builds_growth_and_valuation_inputs() -> None:
    dates = pd.date_range("2018-03-31", periods=9, freq="Q")
    financials = pd.DataFrame(
        {
            "Date": dates,
            "Revenue": [100.0] * 4 + [150.0] * 5,
            "Net Income": [10.0] * 4 + [15.0] * 5,
            "EBIT": [16.0] * 4 + [24.0] * 5,
            "EBITDA": [20.0] * 4 + [30.0] * 5,
            "Shares Outstanding": [10.0] * 9,
            "Total Depreciation And Amortization - Cash Flow": [4.0] * 9,
            "Net Change In Property, Plant, And Equipment": [-2.0] * 9,
            "Total Change In Assets/Liabilities": [0.0] * 9,
            "Total Liabilities": [10.0] * 9,
            "Cash On Hand": [10.0] * 9,
            "Total Assets": [100.0] * 9,
            "Long Term Debt": [0.0] * 9,
            "Cash Flow From Operating Activities": [15.0] * 9,
        }
    )
    settings = ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        financial_feature_mode="ttm_value_momentum",
    )
    available = _build_available_financials(financials, settings)
    row = available.iloc[7]
    assert row["EpsTtm"] == pytest.approx(6.0)
    assert row["EpsTtmGrowthYoY"] == pytest.approx(0.5)
    assert row["EbitdaTtm"] == pytest.approx(120.0)
    assert row["EbitdaTtmGrowthYoY"] == pytest.approx(0.5)
    assert row["DcfPrice"] > 0
    expected_dcf_growth = row["DcfPrice"] / available.iloc[3]["DcfPrice"] - 1
    assert row["DcfPriceGrowthYoY"] == pytest.approx(expected_dcf_growth)
    assert row["DcfPriceGrowthYoY"] > 0
    assert row["FinancialAvailableDate"] == row[
        "FinancialPeriodEnd"
    ] + pd.Timedelta(days=45)


def test_sparse_financial_cross_section_is_neutral_not_artificial_first() -> None:
    settings = ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        minimum_cross_section_size=8,
        financial_feature_mode="ttm_value_momentum",
    )
    tickers = [f"T{i}" for i in range(8)]
    panel = pd.DataFrame(
        {
            "Date": pd.Timestamp("2026-07-24"),
            "Ticker": tickers,
            "Eligible": True,
            "Return21": range(8),
            "Return63": range(8),
            "Return126": range(8),
            "Trend50": range(8),
            "Trend200": range(8),
            "Volatility63": range(1, 9),
            "Drawdown126": range(8),
            "RevenueGrowthYoY": [None] * 8,
            "EpsGrowthYoY": [None] * 8,
            "OperatingMarginChangeYoY": [None] * 8,
            "OperatingMargin": [0.2] + [None] * 7,
            "FreeCashFlowMargin": [0.1] + [None] * 7,
            "ReturnOnInvestment": [None] * 8,
            "NetCashToAssets": [None] * 8,
            "EpsTtmGrowthYoY": [-0.2] + [None] * 7,
            "EpsTtmGrowthAcceleration": [0.1] + [None] * 7,
            "DcfPriceGrowthYoY": [-0.3] + [None] * 7,
            "EbitdaTtmGrowthYoY": [-0.1] + [None] * 7,
            "EbitdaTtmGrowthAcceleration": [0.1] + [None] * 7,
            "DcfUpside": [-0.8] + [None] * 7,
            "GrowthAdjustedPe": [None] * 8,
            "GrowthAdjustedEvEbitda": [None] * 8,
        }
    )
    ranked = add_cross_sectional_factors(panel, settings)
    assert ranked["GrowthFactor"].isna().all()
    assert ranked["QualityFactor"].isna().all()


def test_negative_ttm_growth_cannot_receive_positive_factor_score() -> None:
    settings = ResearchSettings(
        train_start="2020-01-01",
        train_end="2024-12-31",
        validation_periods={"2025": ("2025-01-01", "2025-12-31")},
        minimum_cross_section_size=8,
        financial_feature_mode="ttm_value_momentum",
    )
    panel = pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-12-31"),
            "Ticker": [f"T{i}" for i in range(8)],
            "Eligible": True,
            "Return21": range(8),
            "Return63": range(8),
            "Return126": range(8),
            "Trend50": range(8),
            "Trend200": range(8),
            "Volatility63": range(1, 9),
            "Drawdown126": range(8),
            "RevenueGrowthYoY": [None] * 8,
            "EpsGrowthYoY": [None] * 8,
            "OperatingMarginChangeYoY": [None] * 8,
            "OperatingMargin": [None] * 8,
            "FreeCashFlowMargin": [None] * 8,
            "ReturnOnInvestment": [None] * 8,
            "NetCashToAssets": [None] * 8,
            "EpsTtmGrowthYoY": [-0.1 * (index + 1) for index in range(8)],
            "EpsTtmGrowthAcceleration": [None] * 8,
            "DcfPriceGrowthYoY": [None] * 8,
            "EbitdaTtmGrowthYoY": [None] * 8,
            "EbitdaTtmGrowthAcceleration": [None] * 8,
            "DcfUpside": [None] * 8,
            "GrowthAdjustedPe": [None] * 8,
            "GrowthAdjustedEvEbitda": [None] * 8,
        }
    )
    ranked = add_cross_sectional_factors(panel, settings)
    assert ranked["GrowthFactor"].notna().all()
    assert ranked["GrowthFactor"].le(0).all()


def test_exit_rank_band_retains_existing_holding() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-05"] * 3 + ["2024-01-12"] * 3
            ),
            "Ticker": ["A", "B", "C"] * 2,
            "Eligible": True,
            "Trend200": 0.1,
            "Return126": 0.1,
            "MomentumFactor": [0.5, 0.3, 0.1, 0.3, 0.5, 0.1],
            "TrendFactor": 0.0,
            "GrowthFactor": 0.0,
            "QualityFactor": 0.0,
            "RiskControlFactor": 0.0,
        }
    )
    targets = generate_rebalance_targets(score_panel(panel, _params()), _params())
    second = targets.loc[targets["Date"].eq("2024-01-12")].set_index("Ticker")
    assert bool(second.loc["A", "ModelSelected"])
    assert second.loc["A", "TradeAction"] == "HOLD"
    assert not bool(second.loc["B", "ModelSelected"])


def test_signal_day_prefers_complete_cross_section_over_later_date() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-07-23",
                    "2026-07-23",
                    "2026-07-23",
                    "2026-07-24",
                    "2026-07-24",
                    "2026-07-27",
                    "2026-07-27",
                    "2026-07-27",
                    "2026-07-28",
                    "2026-07-28",
                ]
            ),
            "Ticker": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "A",
                "B",
                "C",
                "A",
                "B",
            ],
        }
    )

    result = signal_day_panel(
        panel,
        "2026-07-20",
        "2026-07-31",
    )

    assert list(result["Date"].drop_duplicates()) == [
        pd.Timestamp("2026-07-23"),
        pd.Timestamp("2026-07-27"),
    ]


def test_compact_targets_match_full_target_weights() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-05"] * 3 + ["2024-01-12"] * 3
            ),
            "Ticker": ["A", "B", "C"] * 2,
            "Eligible": True,
            "Trend200": 0.1,
            "Return126": 0.1,
            "MomentumFactor": [0.5, 0.3, 0.1, 0.3, 0.5, 0.1],
            "TrendFactor": 0.0,
            "GrowthFactor": 0.0,
            "QualityFactor": 0.0,
            "RiskControlFactor": 0.0,
        }
    )
    scored = score_panel(panel, _params())
    full = generate_rebalance_targets(scored, _params())
    compact_scored = score_panel(panel, _params(), compact=True)
    compact = generate_rebalance_targets(
        compact_scored,
        _params(),
        compact=True,
    )

    expected = (
        full.loc[full["TargetWeight"].gt(0), ["Date", "Ticker", "TargetWeight"]]
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    actual = (
        compact.loc[
            compact["TargetWeight"].gt(0),
            ["Date", "Ticker", "TargetWeight"],
        ]
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_loss_aware_exit_retains_loser_until_conviction_breakdown() -> None:
    params = StrategyParams(
        momentum_weight=1.0,
        trend_weight=0.0,
        growth_weight=0.0,
        quality_weight=0.0,
        risk_control_weight=0.0,
        top_k=1,
        exit_rank=2,
        trend_floor=-1.0,
        momentum_floor=-1.0,
        loss_aware_exit_enabled=True,
        minimum_exit_gain=0.01,
        conviction_exit_rank=3,
        conviction_trend_floor=-0.10,
        conviction_momentum_floor=-0.20,
        hard_stop_return=-0.35,
        minimum_hold_rebalances=1,
    )
    dates = pd.to_datetime(
        ["2025-01-03"] * 4
        + ["2025-01-10"] * 4
        + ["2025-01-17"] * 4
    )
    scored = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["A", "B", "C", "D"] * 3,
            "Qualified": True,
            "Rank": [1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3],
            "Close": [100, 100, 100, 100, 95, 100, 100, 100, 90, 100, 100, 100],
            "Trend200": [0.1] * 8 + [-0.20, 0.1, 0.1, 0.1],
            "Return126": [0.1] * 8 + [-0.30, 0.1, 0.1, 0.1],
        }
    )
    targets = generate_rebalance_targets(scored, params)

    protected = targets.loc[
        targets["Date"].eq("2025-01-10")
        & targets["Ticker"].eq("A")
    ].iloc[0]
    conviction_exit = targets.loc[
        targets["Date"].eq("2025-01-17")
        & targets["Ticker"].eq("A")
    ].iloc[0]
    replacement = targets.loc[
        targets["Date"].eq("2025-01-17")
        & targets["Ticker"].eq("B")
    ].iloc[0]

    assert protected["TradeAction"] == "HOLD"
    assert protected["SignalReferenceReturn"] == pytest.approx(-0.05)
    assert conviction_exit["TradeAction"] == "SELL"
    assert conviction_exit["ExitReason"] == "CONVICTION_BREAKDOWN"
    assert replacement["TradeAction"] == "BUY"


def test_loss_aware_exit_allows_profitable_rotation() -> None:
    params = StrategyParams(
        momentum_weight=1.0,
        trend_weight=0.0,
        growth_weight=0.0,
        quality_weight=0.0,
        risk_control_weight=0.0,
        top_k=1,
        exit_rank=2,
        trend_floor=-1.0,
        momentum_floor=-1.0,
        loss_aware_exit_enabled=True,
        minimum_exit_gain=0.01,
        conviction_exit_rank=3,
        conviction_trend_floor=-0.10,
        conviction_momentum_floor=-0.20,
        hard_stop_return=-0.35,
        minimum_hold_rebalances=4,
    )
    scored = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-03"] * 4 + ["2025-01-10"] * 4
            ),
            "Ticker": ["A", "B", "C", "D"] * 2,
            "Qualified": True,
            "Rank": [1, 2, 3, 4, 4, 1, 2, 3],
            "Close": [100, 100, 100, 100, 102, 100, 100, 100],
            "Trend200": 0.1,
            "Return126": 0.1,
        }
    )
    targets = generate_rebalance_targets(scored, params)
    sold = targets.loc[
        targets["Date"].eq("2025-01-10")
        & targets["Ticker"].eq("A")
    ].iloc[0]

    assert sold["TradeAction"] == "SELL"
    assert sold["ExitReason"] == "PROFITABLE_ROTATION"


def test_v6_winner_retention_requires_wider_rank_and_confirmation() -> None:
    params = StrategyParams(
        momentum_weight=1.0,
        trend_weight=0.0,
        growth_weight=0.0,
        quality_weight=0.0,
        risk_control_weight=0.0,
        top_k=1,
        exit_rank=2,
        trend_floor=-1.0,
        momentum_floor=-1.0,
        loss_aware_exit_enabled=True,
        conviction_exit_rank=5,
        profit_rotation_exit_rank=3,
        profit_rotation_confirmation_rebalances=2,
    )
    scored = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-03"] * 4
                + ["2025-01-10"] * 4
                + ["2025-01-17"] * 4
            ),
            "Ticker": ["A", "B", "C", "D"] * 3,
            "Qualified": True,
            "Rank": [1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3],
            "AlphaScore": [0.4, 0.3, 0.2, 0.1] * 3,
            "Close": [100, 100, 100, 100, 110, 100, 100, 100, 112, 100, 100, 100],
            "Trend200": 0.1,
            "Return126": 0.1,
        }
    )
    targets = generate_rebalance_targets(scored, params)
    first_warning = targets.loc[
        targets["Date"].eq("2025-01-10")
        & targets["Ticker"].eq("A")
    ].iloc[0]
    confirmed = targets.loc[
        targets["Date"].eq("2025-01-17")
        & targets["Ticker"].eq("A")
    ].iloc[0]

    assert first_warning["TradeAction"] == "HOLD"
    assert first_warning["ProfitExitStreak"] == 1
    assert confirmed["TradeAction"] == "SELL"
    assert confirmed["ExitReason"] == "PROFITABLE_ROTATION"


def test_v6_replacement_hurdle_blocks_small_score_advantage() -> None:
    params = StrategyParams(
        momentum_weight=1.0,
        trend_weight=0.0,
        growth_weight=0.0,
        quality_weight=0.0,
        risk_control_weight=0.0,
        top_k=1,
        exit_rank=2,
        trend_floor=-1.0,
        momentum_floor=-1.0,
        loss_aware_exit_enabled=True,
        conviction_exit_rank=5,
        replacement_score_advantage=0.05,
    )
    scored = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-03"] * 3
                + ["2025-01-10"] * 3
                + ["2025-01-17"] * 3
            ),
            "Ticker": ["A", "B", "C"] * 3,
            "Qualified": True,
            "Rank": [1, 2, 3, 3, 1, 2, 3, 1, 2],
            "AlphaScore": [
                0.30,
                0.20,
                0.10,
                0.10,
                0.14,
                0.12,
                0.10,
                0.20,
                0.12,
            ],
            "Close": [100, 100, 100, 110, 100, 100, 112, 100, 100],
            "Trend200": 0.1,
            "Return126": 0.1,
        }
    )
    targets = generate_rebalance_targets(scored, params)
    blocked = targets.loc[
        targets["Date"].eq("2025-01-10")
        & targets["Ticker"].eq("A")
    ].iloc[0]
    allowed = targets.loc[
        targets["Date"].eq("2025-01-17")
        & targets["Ticker"].eq("A")
    ].iloc[0]

    assert blocked["TradeAction"] == "HOLD"
    assert blocked["ReplacementScoreAdvantage"] == pytest.approx(0.04)
    assert allowed["TradeAction"] == "SELL"
    assert allowed["ReplacementScoreAdvantage"] == pytest.approx(0.10)


def test_v6_overheated_entry_guard_skips_top_ranked_candidate() -> None:
    params = StrategyParams(
        momentum_weight=1.0,
        trend_weight=0.0,
        growth_weight=0.0,
        quality_weight=0.0,
        risk_control_weight=0.0,
        top_k=1,
        exit_rank=2,
        trend_floor=-1.0,
        momentum_floor=-1.0,
        overheated_entry_enabled=True,
        overheated_return126=1.0,
        overheated_trend200=0.50,
        overheated_drawdown126_floor=-0.03,
    )
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-03"] * 2),
            "Ticker": ["A", "B"],
            "Eligible": True,
            "Close": [200.0, 100.0],
            "Trend200": [0.60, 0.20],
            "Return126": [1.20, 0.30],
            "Drawdown126": [-0.01, -0.10],
            "MomentumFactor": [0.5, 0.3],
            "TrendFactor": 0.0,
            "GrowthFactor": 0.0,
            "QualityFactor": 0.0,
            "RiskControlFactor": 0.0,
        }
    )
    scored = score_panel(panel, params)
    targets = generate_rebalance_targets(scored, params)
    selected = targets.loc[targets["ModelSelected"]]
    blocked = targets.loc[targets["Ticker"].eq("A")].iloc[0]

    assert selected["Ticker"].tolist() == ["B"]
    assert bool(blocked["EntryBlocked"])
    assert blocked["EntryBlockReason"] == "OVERHEATED_ENTRY"


def test_v6_trailing_stop_uses_peak_after_activation() -> None:
    params = StrategyParams(
        momentum_weight=1.0,
        trend_weight=0.0,
        growth_weight=0.0,
        quality_weight=0.0,
        risk_control_weight=0.0,
        top_k=1,
        exit_rank=2,
        trend_floor=-1.0,
        momentum_floor=-1.0,
        loss_aware_exit_enabled=True,
        conviction_exit_rank=3,
        trailing_stop_enabled=True,
        trailing_stop_activation_gain=0.20,
        trailing_stop_drawdown=-0.15,
    )
    scored = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-03"] * 2
                + ["2025-01-10"] * 2
                + ["2025-01-17"] * 2
            ),
            "Ticker": ["A", "B"] * 3,
            "Qualified": True,
            "Rank": [1, 2, 1, 2, 1, 2],
            "AlphaScore": [0.3, 0.2] * 3,
            "Close": [100, 100, 130, 100, 110, 100],
            "Trend200": 0.1,
            "Return126": 0.1,
        }
    )
    targets = generate_rebalance_targets(scored, params)
    sold = targets.loc[
        targets["Date"].eq("2025-01-17")
        & targets["Ticker"].eq("A")
    ].iloc[0]

    assert sold["TradeAction"] == "SELL"
    assert sold["ExitReason"] == "TRAILING_STOP"
    assert sold["PeakReferenceReturn"] == pytest.approx(0.30)
    assert sold["TrailingDrawdown"] == pytest.approx(110 / 130 - 1)


def test_v6_frozen_ablation_preserves_v5_factor_weights() -> None:
    base = _params()
    variants = frozen_v6_variants(base)

    assert list(variants) == [
        "V5",
        "V6-A",
        "V6-B",
        "V6-C",
        "V6-D",
        "V6-BC",
        "V6-ALL",
    ]
    assert all(
        params.factor_weights == base.factor_weights
        for params in variants.values()
    )
    assert variants["V6-A"].profit_rotation_exit_rank == 10
    assert variants["V6-B"].replacement_score_advantage == pytest.approx(0.05)
    assert variants["V6-C"].overheated_entry_enabled
    assert variants["V6-D"].hard_stop_return == pytest.approx(-0.20)
    assert variants["V6-BC"].replacement_score_advantage == pytest.approx(0.05)
    assert variants["V6-BC"].overheated_entry_enabled
    assert variants["V6-ALL"].trailing_stop_enabled


def test_v6_position_ledger_uses_next_open_execution() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06"]
            ),
            "Ticker": ["A", "A", "A"],
            "Open": [10.0, 11.0, 12.0],
            "Close": [10.5, 11.5, 12.5],
        }
    )
    targets = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "Ticker": ["A", "A"],
            "TradeAction": ["BUY", "SELL"],
            "Rank": [1.0, 3.0],
            "AlphaScore": [0.3, 0.1],
            "ExitReason": [None, "PROFITABLE_ROTATION"],
        }
    )
    ledger = build_position_ledger(
        panel,
        targets,
        variant="V5",
        latest_date=pd.Timestamp("2025-01-06"),
    )
    position = ledger.iloc[0]

    assert position["EntryExecutionDate"] == pd.Timestamp("2025-01-03")
    assert position["EntryExecutionPrice"] == pytest.approx(11.0)
    assert position["ExitExecutionDate"] == pd.Timestamp("2025-01-06")
    assert position["ExitExecutionPrice"] == pytest.approx(12.0)
    assert position["ExecutionPriceReturn"] == pytest.approx(12 / 11 - 1)


def test_portfolio_executes_signal_at_next_open_and_uses_net_roi() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06"]
            ),
            "Ticker": ["A", "A", "A"],
            "Open": [10.0, 10.0, 11.0],
            "Close": [10.0, 11.0, 12.0],
        }
    )
    targets = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02"]),
            "Ticker": ["A"],
            "TargetWeight": [1.0],
        }
    )
    result = run_portfolio_backtest(
        panel,
        targets,
        start="2025-01-02",
        end="2025-01-06",
        initial_capital=100.0,
        transaction_cost_bps=0.0,
    )
    assert result.executions.iloc[0]["ExecutionDate"] == pd.Timestamp("2025-01-03")
    assert result.summary.final_value == pytest.approx(120.0)
    assert result.summary.roi_percent == pytest.approx(20.0)
