from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths

from .config import ResearchSettings
from .data import discover_universe
from .pit_validation import apply_membership_to_panel, causal_signal_day_panel
from .portfolio import PortfolioResult, run_portfolio_backtest
from .v7_pit_evaluation import (
    build_v7_source_panel,
    load_ready_tickers,
    normalize_change_membership,
)
from .v7_slot_sweep import spy_buy_and_hold
from .v7_technical import (
    add_v7_technical_factors,
    add_v7_technical_observations,
)
from .v7_trade_report import build_execution_ledger
from .v8_hybrid import (
    V8HybridConfig,
    V8Scenario,
    add_v8_scores,
    build_period_summary,
    build_v8_position_ledger,
    build_v8_trade_events,
)
from .winner_attribution import summarize_ticker_contributions


@dataclass(frozen=True)
class V81InflectionConfig:
    """Pre-registered V8.1 rules; V8.0 remains unchanged."""

    core_slots: int = 7
    core_total_weight: float = 0.70
    core_market_cap_rank: int = 100
    core_minimum_market_cap: float = 10_000_000_000
    core_minimum_hold_quarters: int = 4
    inflection_slots: int = 2
    inflection_scout_weight: float = 0.03
    inflection_confirm_weight: float = 0.08
    inflection_max_weight: float = 0.15
    inflection_minimum_hold_weeks: int = 13
    inflection_market_cap_rank: int = 300
    inflection_minimum_market_cap: float = 2_000_000_000
    hard_stop_return: float = -0.35
    secular_revenue_growth: float = 0.15
    secular_eps_growth: float = 0.20
    secular_ebitda_growth: float = 0.15
    cyclical_revenue_growth: float = 0.05
    cyclical_revenue_acceleration: float = 0.10
    cyclical_margin_improvement: float = 0.03
    minimum_return126: float = 0.10
    minimum_volume_ratio: float = 0.90
    minimum_high_proximity252: float = -0.20

    @property
    def core_position_weight(self) -> float:
        return self.core_total_weight / self.core_slots

    def __post_init__(self) -> None:
        if self.core_slots < 1 or self.inflection_slots < 1:
            raise ValueError("V8.1 sleeves require positive slot counts")
        maximum = (
            self.core_total_weight
            + self.inflection_slots * self.inflection_max_weight
        )
        if maximum > 1 + 1e-9:
            raise ValueError("V8.1 maximum target weights exceed 100%")
        if not -1 < self.hard_stop_return < 0:
            raise ValueError("hard_stop_return must be between -1 and zero")


@dataclass(frozen=True)
class V82InflectionConfig(V81InflectionConfig):
    """Add a price-led margin breakout without modifying V8.1."""

    margin_breakout_improvement: float = 0.08
    margin_breakout_return126: float = 0.15
    margin_breakout_high_proximity252: float = -0.05


@dataclass(frozen=True)
class V81Artifacts:
    output_dir: Path
    report_html: Path
    summary_csv: Path
    period_summary_csv: Path
    event_study_csv: Path
    label_summary_csv: Path
    winner_episodes_csv: Path
    named_trajectories_csv: Path
    rule_diagnostics_csv: Path
    trade_events_csv: Path
    position_ledger_csv: Path
    contributions_csv: Path
    targets_csv: Path
    equity_csv: Path
    data_audit_csv: Path
    manifest_json: Path


EVENT_FEATURES = (
    "RevenueGrowthYoY",
    "RevenueGrowthAcceleration",
    "EpsTtmGrowthYoY",
    "EbitdaTtmGrowthYoY",
    "OperatingMarginChangeYoY",
    "OperatingMarginSequentialChange",
    "FreeCashFlowMargin",
    "Return126",
    "Return252",
    "Trend200",
    "HighProximity252",
    "Volume20To126",
    "MACDLineNormalized",
    "OBVFlow63",
)


def add_inflection_observations(panel: pd.DataFrame) -> pd.DataFrame:
    """Add causal price-volume and quarter-to-quarter observations."""

    frame = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    price_outputs = {
        name: pd.Series(np.nan, index=frame.index, dtype=float)
        for name in (
            "Return5",
            "Return252",
            "HighProximity252",
            "Volume20To126",
        )
    }
    for indexes in frame.groupby("Ticker", sort=False).groups.values():
        positions = np.asarray(indexes, dtype=int)
        close = pd.to_numeric(
            frame.loc[positions, "Close"], errors="coerce"
        ).reset_index(drop=True)
        volume = pd.to_numeric(
            frame.loc[positions, "Volume"], errors="coerce"
        ).reset_index(drop=True)
        volume20 = volume.rolling(20, min_periods=15).mean()
        volume126 = volume.rolling(126, min_periods=84).mean()
        price_outputs["Return5"].iloc[positions] = close.pct_change(
            5, fill_method=None
        )
        price_outputs["Return252"].iloc[positions] = close.pct_change(
            252, fill_method=None
        )
        price_outputs["HighProximity252"].iloc[positions] = (
            close / close.rolling(252, min_periods=168).max() - 1
        )
        price_outputs["Volume20To126"].iloc[positions] = (
            volume20 / volume126.replace(0, np.nan)
        )
    for column, values in price_outputs.items():
        frame[column] = values

    quarter_columns = [
        "Ticker",
        "FinancialPeriodEnd",
        "RevenueGrowthYoY",
        "OperatingMargin",
        "EpsTtm",
        "EbitdaTtm",
    ]
    quarters = (
        frame.loc[frame["FinancialPeriodEnd"].notna(), quarter_columns]
        .drop_duplicates(["Ticker", "FinancialPeriodEnd"], keep="last")
        .sort_values(["Ticker", "FinancialPeriodEnd"])
        .reset_index(drop=True)
    )
    grouped = quarters.groupby("Ticker", sort=False)
    quarters["RevenueGrowthAcceleration"] = grouped[
        "RevenueGrowthYoY"
    ].diff()
    quarters["OperatingMarginSequentialChange"] = grouped[
        "OperatingMargin"
    ].diff()
    quarters["PriorEpsTtm"] = grouped["EpsTtm"].shift()
    quarters["PriorEbitdaTtm"] = grouped["EbitdaTtm"].shift()
    quarters["EpsTtmSequentialChange"] = (
        quarters["EpsTtm"] - quarters["PriorEpsTtm"]
    )
    quarters["EbitdaTtmSequentialChange"] = (
        quarters["EbitdaTtm"] - quarters["PriorEbitdaTtm"]
    )
    derived = [
        "Ticker",
        "FinancialPeriodEnd",
        "RevenueGrowthAcceleration",
        "OperatingMarginSequentialChange",
        "PriorEpsTtm",
        "PriorEbitdaTtm",
        "EpsTtmSequentialChange",
        "EbitdaTtmSequentialChange",
    ]
    frame = frame.merge(
        quarters[derived],
        on=["Ticker", "FinancialPeriodEnd"],
        how="left",
        validate="many_to_one",
    )
    return frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def add_forward_multibagger_labels(
    panel: pd.DataFrame,
    *,
    horizon_24m: int = 504,
    horizon_36m: int = 756,
) -> pd.DataFrame:
    """Attach research-only forward labels without using them in signals."""

    frame = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    outputs = {
        name: pd.Series(np.nan, index=frame.index, dtype=float)
        for name in (
            "ForwardReturn24m",
            "ForwardMaxReturn24m",
            "ForwardReturn36m",
            "ForwardMaxReturn36m",
        )
    }
    for indexes in frame.groupby("Ticker", sort=False).groups.values():
        positions = np.asarray(indexes, dtype=int)
        close = pd.to_numeric(
            frame.loc[positions, "Close"], errors="coerce"
        ).reset_index(drop=True)
        for horizon, suffix in (
            (horizon_24m, "24m"),
            (horizon_36m, "36m"),
        ):
            indexer = FixedForwardWindowIndexer(window_size=horizon)
            future_max = (
                close.shift(-1)
                .rolling(window=indexer, min_periods=horizon)
                .max()
            )
            outputs[f"ForwardReturn{suffix}"].iloc[positions] = (
                close.shift(-horizon) / close - 1
            )
            outputs[f"ForwardMaxReturn{suffix}"].iloc[positions] = (
                future_max / close - 1
            )
    for column, values in outputs.items():
        frame[column] = values
    frame["Label4x24m"] = frame["ForwardMaxReturn24m"].ge(3.0).where(
        frame["ForwardMaxReturn24m"].notna()
    )
    frame["Label10x36m"] = frame["ForwardMaxReturn36m"].ge(9.0).where(
        frame["ForwardMaxReturn36m"].notna()
    )
    return frame


def monthly_event_sample(
    labeled_panel: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Select one common final market session per month."""

    frame = labeled_panel.loc[
        labeled_panel["Date"].between(start, end)
    ].copy()
    month_end_dates = (
        frame[["Date"]]
        .drop_duplicates()
        .assign(Month=lambda value: value["Date"].dt.to_period("M"))
        .groupby("Month")["Date"]
        .max()
    )
    sample = frame.loc[
        frame["Date"].isin(month_end_dates.to_numpy())
        & frame["Eligible"].fillna(False)
        & frame["UniverseMember"].fillna(False)
    ].copy()
    valid_forward = sample["ForwardReturn24m"].notna()
    sample["LabelTopDecile24m"] = pd.Series(
        pd.NA, index=sample.index, dtype="boolean"
    )
    sample.loc[valid_forward, "LabelTopDecile24m"] = (
        sample.loc[valid_forward]
        .groupby("Date")["ForwardReturn24m"]
        .rank(pct=True, method="average")
        .ge(0.90)
    )
    return sample.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def summarize_multibagger_labels(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = {
        "4X_WITHIN_24M": ("Label4x24m", "ForwardMaxReturn24m"),
        "10X_WITHIN_36M": ("Label10x36m", "ForwardMaxReturn36m"),
        "TOP_DECILE_24M": ("LabelTopDecile24m", "ForwardReturn24m"),
    }
    for label, (column, forward_column) in definitions.items():
        valid = sample[column].notna()
        positive = sample[column].eq(True)  # noqa: E712
        rows.append(
            {
                "Label": label,
                "ValidMonthlyObservations": int(valid.sum()),
                "PositiveObservations": int((valid & positive).sum()),
                "ValidTickers": int(sample.loc[valid, "Ticker"].nunique()),
                "PositiveTickers": int(
                    sample.loc[valid & positive, "Ticker"].nunique()
                ),
                "BaseRate": float(
                    positive.loc[valid].astype(float).mean()
                ),
                "MedianForwardReturnPositive": float(
                    pd.to_numeric(
                        sample.loc[valid & positive, forward_column],
                        errors="coerce",
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_event_study(sample: pd.DataFrame) -> pd.DataFrame:
    """Measure top-quintile feature lift without fitting a classifier."""

    labels = {
        "4X_WITHIN_24M": "Label4x24m",
        "10X_WITHIN_36M": "Label10x36m",
        "TOP_DECILE_24M": "LabelTopDecile24m",
    }
    rows: list[dict[str, object]] = []
    for label_name, label_column in labels.items():
        for feature in EVENT_FEATURES:
            valid = sample[label_column].notna() & sample[feature].notna()
            data = sample.loc[
                valid, ["Date", label_column, feature]
            ].copy()
            if data.empty:
                continue
            data["FeaturePercentile"] = data.groupby("Date")[feature].rank(
                pct=True, method="average"
            )
            top = data["FeaturePercentile"].ge(0.80)
            bottom = data["FeaturePercentile"].le(0.20)
            base_rate = data[label_column].astype(float).mean()
            top_rate = data.loc[top, label_column].astype(float).mean()
            bottom_rate = (
                data.loc[bottom, label_column].astype(float).mean()
            )
            top_lift = (
                top_rate / base_rate
                if base_rate > 0 and pd.notna(top_rate)
                else np.nan
            )
            bottom_lift = (
                bottom_rate / base_rate
                if base_rate > 0 and pd.notna(bottom_rate)
                else np.nan
            )
            rows.append(
                {
                    "Label": label_name,
                    "Feature": feature,
                    "Observations": len(data),
                    "PositiveObservations": int(
                        data[label_column].eq(True).sum()  # noqa: E712
                    ),
                    "BaseRate": base_rate,
                    "TopQuintileRate": top_rate,
                    "TopQuintileLift": top_lift,
                    "BottomQuintileRate": bottom_rate,
                    "BottomQuintileLift": bottom_lift,
                    "BestTail": (
                        "HIGH"
                        if np.nan_to_num(top_lift, nan=-np.inf)
                        >= np.nan_to_num(bottom_lift, nan=-np.inf)
                        else "LOW"
                    ),
                    "BestTailLift": np.nanmax(
                        np.asarray([top_lift, bottom_lift], dtype=float)
                    ),
                    "Spearman": data[feature].corr(
                        data[label_column].astype(float),
                        method="spearman",
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Label", "BestTailLift"], ascending=[True, False]
    ).reset_index(drop=True)


def winner_episode_summary(sample: pd.DataFrame) -> pd.DataFrame:
    """Return the earliest eligible anchor for each labeled winner."""

    rows: list[pd.Series] = []
    definitions = {
        "4X_WITHIN_24M": ("Label4x24m", "ForwardMaxReturn24m"),
        "10X_WITHIN_36M": ("Label10x36m", "ForwardMaxReturn36m"),
        "TOP_DECILE_24M": ("LabelTopDecile24m", "ForwardReturn24m"),
    }
    preferred = [
        "Label",
        "Date",
        "Ticker",
        "Company",
        "FinancialPeriodEnd",
        "ForwardOutcome",
        *EVENT_FEATURES,
    ]
    for label, (column, outcome) in definitions.items():
        positives = sample.loc[sample[column].eq(True)].copy()  # noqa: E712
        earliest = (
            positives.sort_values(["Ticker", "Date"])
            .groupby("Ticker", as_index=False)
            .head(1)
        )
        for _, row in earliest.iterrows():
            result = row.copy()
            result["Label"] = label
            result["ForwardOutcome"] = row[outcome]
            rows.append(result)
    if not rows:
        return pd.DataFrame(columns=preferred)
    result = pd.DataFrame(rows)
    return result.loc[
        :, [column for column in preferred if column in result]
    ].sort_values(["Label", "ForwardOutcome"], ascending=[True, False])


def named_company_trajectories(
    sample: pd.DataFrame,
    tickers: tuple[str, ...] = ("WDC", "NVDA"),
) -> pd.DataFrame:
    columns = [
        "Date",
        "Ticker",
        "Company",
        "FinancialPeriodEnd",
        "Label4x24m",
        "Label10x36m",
        "ForwardMaxReturn24m",
        "ForwardMaxReturn36m",
        *EVENT_FEATURES,
    ]
    return sample.loc[
        sample["Ticker"].isin(tickers),
        [column for column in columns if column in sample],
    ].reset_index(drop=True)


def add_v81_scores(
    panel: pd.DataFrame,
    config: V81InflectionConfig,
) -> pd.DataFrame:
    """Add separate secular-growth and cyclical-recovery evidence gates."""

    v8_config = V8HybridConfig(
        core_slots=config.core_slots,
        core_total_weight=config.core_total_weight,
        core_market_cap_rank=config.core_market_cap_rank,
        core_minimum_market_cap=config.core_minimum_market_cap,
        core_minimum_hold_quarters=config.core_minimum_hold_quarters,
        inflection_slots=config.inflection_slots,
        inflection_scout_weight=config.inflection_scout_weight,
        inflection_confirm_weight=config.inflection_confirm_weight,
        inflection_max_weight=config.inflection_max_weight,
        inflection_minimum_hold_weeks=config.inflection_minimum_hold_weeks,
        inflection_market_cap_rank=config.inflection_market_cap_rank,
        inflection_minimum_market_cap=(
            config.inflection_minimum_market_cap
        ),
    )
    frame = add_v8_scores(panel, v8_config)
    eligible = frame["Eligible"].fillna(False)
    eps_growth = pd.to_numeric(
        frame["EpsTtmGrowthYoY"], errors="coerce"
    ).clip(-2.0, 3.0)
    ebitda_growth = pd.to_numeric(
        frame["EbitdaTtmGrowthYoY"], errors="coerce"
    ).clip(-2.0, 3.0)
    evidence_inputs = (
        "RevenueGrowthYoY",
        "RevenueGrowthAcceleration",
        "OperatingMarginChangeYoY",
        "OperatingMarginSequentialChange",
        "FreeCashFlowMargin",
    )
    ranked = {
        column: _centered_rank(frame, column, eligible)
        for column in evidence_inputs
    }
    ranked["RobustEpsGrowth"] = _centered_rank_values(
        frame, eps_growth, eligible
    )
    ranked["RobustEbitdaGrowth"] = _centered_rank_values(
        frame, ebitda_growth, eligible
    )
    frame["RobustEpsGrowth"] = eps_growth
    frame["RobustEbitdaGrowth"] = ebitda_growth
    frame["FundamentalEvidenceScore"] = pd.concat(
        list(ranked.values()), axis=1
    ).mean(axis=1)
    frame["PriceVolumeEvidenceScore"] = frame[
        ["MAFactor", "MACDFactor", "OBVFactor", "MomentumFactor"]
    ].mean(axis=1)

    common = (
        eligible
        & frame["MarketCapRank"].le(config.inflection_market_cap_rank)
        & frame["MarketCap"].ge(config.inflection_minimum_market_cap)
        & frame["Trend200"].ge(0)
        & frame["Return126"].ge(config.minimum_return126)
        & frame["HighProximity252"].ge(
            config.minimum_high_proximity252
        )
        & frame["Volume20To126"].ge(config.minimum_volume_ratio)
        & frame["MACDLineNormalized"].gt(0)
        & frame["OBVFlow63"].gt(0)
    )
    profitable = frame["EpsTtm"].gt(0) & frame["EbitdaTtm"].gt(0)
    frame["SecularAccelerationQualified"] = (
        common
        & profitable
        & frame["RevenueGrowthYoY"].ge(config.secular_revenue_growth)
        & eps_growth.ge(config.secular_eps_growth)
        & ebitda_growth.ge(config.secular_ebitda_growth)
        & frame["OperatingMarginChangeYoY"].ge(0)
        & frame["FreeCashFlowMargin"].ge(0)
    )
    cyclical_profit_recovery = (
        frame["PriorEbitdaTtm"].le(0) & frame["EbitdaTtm"].gt(0)
    ) | (
        frame["EbitdaTtm"].gt(0)
        & ebitda_growth.ge(0.30)
        & frame["EbitdaTtmSequentialChange"].gt(0)
    )
    frame["CyclicalRecoveryQualified"] = (
        common
        & frame["RevenueGrowthYoY"].ge(config.cyclical_revenue_growth)
        & frame["RevenueGrowthAcceleration"].ge(
            config.cyclical_revenue_acceleration
        )
        & frame["OperatingMarginChangeYoY"].ge(
            config.cyclical_margin_improvement
        )
        & frame["OperatingMarginSequentialChange"].gt(0)
        & cyclical_profit_recovery
    )
    frame["InflectionQualifiedV81"] = (
        frame["SecularAccelerationQualified"]
        | frame["CyclicalRecoveryQualified"]
    )
    frame["SignalArchetype"] = np.select(
        [
            frame["SecularAccelerationQualified"],
            frame["CyclicalRecoveryQualified"],
        ],
        ["SECULAR_ACCELERATION", "CYCLICAL_RECOVERY"],
        default="",
    )
    frame["InflectionScoreV81"] = (
        0.60 * frame["FundamentalEvidenceScore"]
        + 0.25 * frame["PriceVolumeEvidenceScore"]
        + 0.15 * pd.to_numeric(frame["QualityFactor"], errors="coerce")
    )
    frame["InflectionRankV81"] = (
        frame["InflectionScoreV81"]
        .where(frame["InflectionQualifiedV81"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    return frame


def add_v82_scores(
    panel: pd.DataFrame,
    config: V82InflectionConfig,
) -> pd.DataFrame:
    """Add the price-led margin-breakout archetype found in event review."""

    frame = add_v81_scores(panel, config)
    eligible = frame["Eligible"].fillna(False)
    common_size = (
        eligible
        & frame["MarketCapRank"].le(config.inflection_market_cap_rank)
        & frame["MarketCap"].ge(config.inflection_minimum_market_cap)
    )
    frame["MarginPriceBreakoutQualified"] = (
        common_size
        & frame["OperatingMarginSequentialChange"].ge(
            config.margin_breakout_improvement
        )
        & frame["FreeCashFlowMargin"].gt(0)
        & frame["Return126"].ge(config.margin_breakout_return126)
        & frame["Trend200"].gt(0)
        & frame["HighProximity252"].ge(
            config.margin_breakout_high_proximity252
        )
        & frame["MACDLineNormalized"].gt(0)
        & frame["OBVFlow63"].gt(0)
    )
    frame["InflectionQualifiedV81"] |= frame[
        "MarginPriceBreakoutQualified"
    ]
    frame["SignalArchetype"] = np.select(
        [
            frame["SecularAccelerationQualified"],
            frame["CyclicalRecoveryQualified"],
            frame["MarginPriceBreakoutQualified"],
        ],
        [
            "SECULAR_ACCELERATION",
            "CYCLICAL_RECOVERY",
            "MARGIN_PRICE_BREAKOUT",
        ],
        default="",
    )
    frame["InflectionRankV81"] = (
        frame["InflectionScoreV81"]
        .where(frame["InflectionQualifiedV81"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    return frame


def summarize_rule_diagnostics(
    scored_sample: pd.DataFrame,
) -> pd.DataFrame:
    labels = {
        "4X_WITHIN_24M": "Label4x24m",
        "10X_WITHIN_36M": "Label10x36m",
        "TOP_DECILE_24M": "LabelTopDecile24m",
    }
    rules = {
        "SECULAR_ACCELERATION": "SecularAccelerationQualified",
        "CYCLICAL_RECOVERY": "CyclicalRecoveryQualified",
        "COMBINED_V81": "InflectionQualifiedV81",
    }
    if "MarginPriceBreakoutQualified" in scored_sample:
        rules = {
            "SECULAR_ACCELERATION": "SecularAccelerationQualified",
            "CYCLICAL_RECOVERY": "CyclicalRecoveryQualified",
            "MARGIN_PRICE_BREAKOUT": "MarginPriceBreakoutQualified",
            "COMBINED_V82": "InflectionQualifiedV81",
        }
    rows: list[dict[str, object]] = []
    for label_name, label_column in labels.items():
        for rule_name, rule_column in rules.items():
            valid = scored_sample[label_column].notna()
            positive = scored_sample[label_column].eq(True)  # noqa: E712
            selected = scored_sample[rule_column].fillna(False)
            base_rate = positive.loc[valid].astype(float).mean()
            selected_valid = valid & selected
            precision = positive.loc[selected_valid].astype(float).mean()
            positive_count = int((valid & positive).sum())
            rows.append(
                {
                    "Label": label_name,
                    "Rule": rule_name,
                    "ValidObservations": int(valid.sum()),
                    "SelectedObservations": int(selected_valid.sum()),
                    "BaseRate": base_rate,
                    "SelectedWinRate": precision,
                    "Lift": (
                        precision / base_rate
                        if base_rate > 0 and pd.notna(precision)
                        else np.nan
                    ),
                    "WinnerRecall": (
                        int((valid & selected & positive).sum())
                        / positive_count
                        if positive_count
                        else np.nan
                    ),
                    "SelectedTickers": int(
                        scored_sample.loc[selected_valid, "Ticker"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def generate_v81_targets(
    signal_days: pd.DataFrame,
    config: V81InflectionConfig,
) -> pd.DataFrame:
    """Generate V8.1 targets with independent-quarter scale confirmation."""

    frame = signal_days.sort_values(["Date", "Ticker"]).copy()
    signal_dates = pd.DatetimeIndex(
        frame["Date"].drop_duplicates().sort_values()
    )
    quarter_review_dates = set(
        pd.Series(signal_dates, index=signal_dates)
        .groupby(signal_dates.to_period("Q"))
        .max()
        .tolist()
    )
    core: dict[str, dict[str, object]] = {}
    inflection: dict[str, dict[str, object]] = {}
    last_weights: dict[str, float] = {}
    records: list[pd.DataFrame] = []

    for date, group in frame.groupby("Date", sort=True):
        date = pd.Timestamp(date)
        indexed = group.set_index("Ticker", drop=False)
        actions: dict[str, str] = {}
        reasons: dict[str, str] = {}
        exit_reasons: dict[str, str] = {}
        prior_sleeves = {
            **{ticker: "CORE" for ticker in core},
            **{ticker: "INFLECTION" for ticker in inflection},
        }

        core_review = not core or date in quarter_review_dates
        if core_review:
            for ticker in list(core):
                state = core[ticker]
                state["QuartersHeld"] = int(
                    state.get("QuartersHeld", 0)
                ) + 1
                row = _ticker_row(indexed, ticker)
                reason = _core_exit_reason(row, state, config)
                if reason:
                    core.pop(ticker)
                    actions[ticker] = "SELL"
                    exit_reasons[ticker] = reason
                    reasons[ticker] = _core_sell_reason(row, reason)
            candidates = group.loc[group["CoreQualified"]].sort_values(
                ["CoreRank", "Ticker"]
            )
            for row in candidates.itertuples(index=False):
                ticker = str(row.Ticker)
                if len(core) >= config.core_slots:
                    break
                if ticker in core or ticker in inflection:
                    continue
                values = pd.Series(row._asdict())
                core[ticker] = {
                    "EntryDate": date,
                    "QuartersHeld": 1,
                }
                actions[ticker] = "BUY"
                reasons[ticker] = _core_buy_reason(values)

        for ticker in list(inflection):
            state = inflection[ticker]
            state["WeeksHeld"] = int(state.get("WeeksHeld", 0)) + 1
            row = _ticker_row(indexed, ticker)
            period = _row_timestamp(row, "FinancialPeriodEnd")
            last_period = _state_timestamp(
                state.get("LastFinancialPeriod")
            )
            new_financial_quarter = (
                period is not None
                and (last_period is None or period > last_period)
            )
            if new_financial_quarter:
                state["LastFinancialPeriod"] = period
                if _row_bool(row, "InflectionQualifiedV81"):
                    state["ConfirmedNewQuarters"] = int(
                        state.get("ConfirmedNewQuarters", 0)
                    ) + 1
                    state["FailedNewQuarters"] = 0
                else:
                    state["FailedNewQuarters"] = int(
                        state.get("FailedNewQuarters", 0)
                    ) + 1

            exit_reason = _inflection_exit_reason(row, state, config)
            if exit_reason:
                inflection.pop(ticker)
                actions[ticker] = "SELL"
                exit_reasons[ticker] = exit_reason
                reasons[ticker] = _inflection_sell_reason(
                    row, exit_reason, state
                )
                continue

            old_stage = int(state.get("Stage", 1))
            confirmed = int(state.get("ConfirmedNewQuarters", 0))
            new_stage = min(3, 1 + confirmed)
            if new_stage > old_stage:
                state["Stage"] = new_stage
                actions[ticker] = "SCALE_UP"
                reasons[ticker] = _scale_reason(
                    row, new_stage, confirmed
                )

        candidates = group.loc[
            group["InflectionQualifiedV81"]
        ].sort_values(["InflectionRankV81", "Ticker"])
        for row in candidates.itertuples(index=False):
            ticker = str(row.Ticker)
            if len(inflection) >= config.inflection_slots:
                break
            if ticker in inflection or ticker in core:
                continue
            values = pd.Series(row._asdict())
            period = _row_timestamp(values, "FinancialPeriodEnd")
            inflection[ticker] = {
                "EntryDate": date,
                "EntryClose": _row_number(values, "Close"),
                "WeeksHeld": 1,
                "LastFinancialPeriod": period,
                "ConfirmedNewQuarters": 0,
                "FailedNewQuarters": 0,
                "Stage": 1,
                "Archetype": str(values.get("SignalArchetype", "")),
            }
            actions[ticker] = "BUY"
            reasons[ticker] = _inflection_buy_reason(values)

        weights = {
            ticker: config.core_position_weight for ticker in core
        }
        stage_weights = {
            1: config.inflection_scout_weight,
            2: config.inflection_confirm_weight,
            3: config.inflection_max_weight,
        }
        weights.update(
            {
                ticker: stage_weights[int(state["Stage"])]
                for ticker, state in inflection.items()
            }
        )
        if sum(weights.values()) > 1 + 1e-9:
            raise RuntimeError("V8.1 targets exceed 100%")
        changed = set(weights) != set(last_weights) or any(
            abs(weight - last_weights.get(ticker, 0.0)) > 1e-12
            for ticker, weight in weights.items()
        )
        if not changed:
            continue

        output_rows: list[dict[str, object]] = []
        for ticker in sorted(set(weights) | set(last_weights)):
            row = _ticker_row(indexed, ticker)
            base = (
                row.to_dict()
                if row is not None
                else {"Date": date, "Ticker": ticker}
            )
            sleeve = (
                "CORE"
                if ticker in core
                else "INFLECTION"
                if ticker in inflection
                else prior_sleeves.get(ticker, "UNKNOWN")
            )
            state = inflection.get(ticker, {})
            base.update(
                {
                    "Date": date,
                    "SignalDate": date,
                    "Ticker": ticker,
                    "Sleeve": sleeve,
                    "TargetWeight": weights.get(ticker, 0.0),
                    "TradeAction": actions.get(ticker, "HOLD"),
                    "Reason": reasons.get(
                        ticker, "기존 장기 투자 논리가 유지되어 보유."
                    ),
                    "ExitReason": exit_reasons.get(ticker, ""),
                    "CoreQuartersHeld": core.get(ticker, {}).get(
                        "QuartersHeld"
                    ),
                    "InflectionWeeksHeld": state.get("WeeksHeld"),
                    "InflectionStage": state.get("Stage"),
                    "ConfirmedNewQuarters": state.get(
                        "ConfirmedNewQuarters"
                    ),
                }
            )
            output_rows.append(base)
        records.append(pd.DataFrame(output_rows))
        last_weights = weights

    if not records:
        return pd.DataFrame(
            columns=["Date", "Ticker", "TargetWeight", "TradeAction"]
        )
    return pd.concat(records, ignore_index=True)


def run_v81_scenario(
    panel: pd.DataFrame,
    settings: ResearchSettings,
    config: V81InflectionConfig,
    *,
    start: str,
    end: str,
    reference_dates: pd.DatetimeIndex,
    label: str,
    excluded_tickers: tuple[str, ...] = (),
    score_builder: Callable[
        [pd.DataFrame, V81InflectionConfig], pd.DataFrame
    ] = add_v81_scores,
) -> V8Scenario:
    scenario_panel = panel.copy()
    if excluded_tickers:
        scenario_panel["Eligible"] &= ~scenario_panel["Ticker"].isin(
            excluded_tickers
        )
    scored = score_builder(scenario_panel, config)
    signal_days = causal_signal_day_panel(
        scored,
        start=start,
        end=end,
        reference_market_dates=reference_dates,
        rebalance_weekday=settings.rebalance_weekday,
    )
    targets = generate_v81_targets(signal_days, config)
    result = run_portfolio_backtest(
        scored,
        targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        record_attribution=True,
    )
    executions, _ = build_execution_ledger(
        scored,
        targets,
        result,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    events = build_v8_trade_events(targets, executions)
    event_extras = [
        "SignalArchetype",
        "InflectionScoreV81",
        "InflectionRankV81",
        "FundamentalEvidenceScore",
        "PriceVolumeEvidenceScore",
        "ConfirmedNewQuarters",
        "FinancialPeriodEnd",
        "RevenueGrowthYoY",
        "RevenueGrowthAcceleration",
        "OperatingMarginChangeYoY",
        "OperatingMarginSequentialChange",
        "FreeCashFlowMargin",
        "Return252",
        "HighProximity252",
        "Volume20To126",
    ]
    available_extras = [
        column
        for column in event_extras
        if column in targets and column not in events
    ]
    if available_extras:
        extras = targets.loc[
            targets["TradeAction"].isin(["BUY", "SELL", "SCALE_UP"]),
            ["Date", "Ticker", *available_extras],
        ]
        events = events.merge(
            extras,
            on=["Date", "Ticker"],
            how="left",
            validate="one_to_one",
        )
    positions = build_v8_position_ledger(
        events,
        executions,
        result,
        scored,
        end_date=end,
    )
    contributions = summarize_ticker_contributions(
        result,
        {label: (start, end)},
    )
    return V8Scenario(
        label=label,
        excluded_tickers=excluded_tickers,
        result=result,
        targets=targets,
        execution_ledger=executions,
        trade_events=events,
        positions=positions,
        contributions=contributions,
    )


def run_v81_inflection_research(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    spy_path: str | Path,
    frozen_strategy_path: str | Path,
    config: V81InflectionConfig | V82InflectionConfig | None = None,
    version: str = "V8_1",
    expected_ready_count: int = 575,
    v8_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> V81Artifacts:
    normalized_version = version.upper().replace(".", "_")
    if normalized_version not in {"V8_1", "V8_2"}:
        raise ValueError("version must be V8_1 or V8_2")
    if config is None:
        config = (
            V82InflectionConfig()
            if normalized_version == "V8_2"
            else V81InflectionConfig()
        )
    score_builder: Callable[
        [pd.DataFrame, V81InflectionConfig], pd.DataFrame
    ]
    score_builder = (
        add_v82_scores
        if normalized_version == "V8_2"
        else add_v81_scores
    )
    ready_tickers = load_ready_tickers(
        backfill_status_path, expected_count=expected_ready_count
    )
    members, _ = discover_universe(paths, ticker_config_path)
    missing = sorted(
        ready_tickers - {member.ticker for member in members}
    )
    if missing:
        raise ValueError(
            "Ready tickers are not discoverable: " + ", ".join(missing)
        )
    warmup_start = str(
        (
            pd.Timestamp(settings.train_start) - pd.DateOffset(years=2)
        ).date()
    )
    base_panel, data_audit = build_v7_source_panel(
        paths,
        members,
        settings,
        ready_tickers=ready_tickers,
        warmup_start=warmup_start,
    )
    observed = add_v7_technical_observations(base_panel)
    membership = normalize_change_membership(pd.read_csv(membership_path))
    pit_panel = apply_membership_to_panel(observed, membership, settings)
    technical = add_v7_technical_factors(pit_panel, settings)
    enhanced = add_inflection_observations(technical)
    latest_end = str(pd.Timestamp(enhanced["Date"].max()).date())
    reference_dates = pd.DatetimeIndex(
        enhanced["Date"].drop_duplicates().sort_values()
    )

    labeled = add_forward_multibagger_labels(enhanced)
    event_sample = monthly_event_sample(
        labeled, start=settings.train_start, end=latest_end
    )
    label_summary = summarize_multibagger_labels(event_sample)
    event_study = feature_event_study(event_sample)
    episodes = winner_episode_summary(event_sample)
    trajectories = named_company_trajectories(event_sample)
    scored_sample = score_builder(event_sample, config)
    rule_diagnostics = summarize_rule_diagnostics(scored_sample)

    base_label = f"{normalized_version}_QUARTER_CONFIRMED"
    base = run_v81_scenario(
        enhanced,
        settings,
        config,
        start=settings.train_start,
        end=latest_end,
        reference_dates=reference_dates,
        label=base_label,
        score_builder=score_builder,
    )
    top_winner = str(
        base.contributions.sort_values(
            "NetPnL", ascending=False
        ).iloc[0]["Ticker"]
    )
    ex_wdc = run_v81_scenario(
        enhanced,
        settings,
        config,
        start=settings.train_start,
        end=latest_end,
        reference_dates=reference_dates,
        label=f"{normalized_version}_EX_WDC",
        excluded_tickers=("WDC",),
        score_builder=score_builder,
    )
    if top_winner == "WDC":
        ex_top = V8Scenario(
            label=f"{normalized_version}_EX_TOP_WINNER_WDC",
            excluded_tickers=ex_wdc.excluded_tickers,
            result=ex_wdc.result,
            targets=ex_wdc.targets,
            execution_ledger=ex_wdc.execution_ledger,
            trade_events=ex_wdc.trade_events,
            positions=ex_wdc.positions,
            contributions=ex_wdc.contributions.assign(
                Period=f"{normalized_version}_EX_TOP_WINNER_WDC"
            ),
        )
    else:
        ex_top = run_v81_scenario(
            enhanced,
            settings,
            config,
            start=settings.train_start,
            end=latest_end,
            reference_dates=reference_dates,
            label=f"{normalized_version}_EX_TOP_WINNER_{top_winner}",
            excluded_tickers=(top_winner,),
            score_builder=score_builder,
        )
    scenarios = (base, ex_wdc, ex_top)

    spy = load_sp500_proxy(spy_path)
    spy_summary, spy_equity = spy_buy_and_hold(
        spy,
        start=settings.train_start,
        end=latest_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    summary = _build_summary(
        scenarios,
        spy_summary,
        v8_summary_path=v8_summary_path,
    )
    periods = {
        "TRAIN_2020_2024": (
            settings.train_start,
            min(settings.train_end, latest_end),
        ),
        **{
            label: (start, min(end, latest_end))
            for label, (start, end) in settings.validation_periods.items()
            if start <= latest_end
        },
        "FULL_2020_2026": (settings.train_start, latest_end),
    }
    period_summary = build_period_summary(
        scenarios,
        spy_equity,
        periods,
        initial_capital=settings.initial_capital,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v8_inflection_research"
            / (
                f"{timestamp}_{normalized_version.lower()}"
                "_quarter_confirmed"
            )
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "report_html": (
            destination
            / f"{normalized_version.lower()}_inflection_report.html"
        ),
        "summary_csv": destination / "summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "event_study_csv": destination / "feature_event_study.csv",
        "label_summary_csv": destination / "label_summary.csv",
        "winner_episodes_csv": destination / "winner_episodes.csv",
        "named_trajectories_csv": (
            destination / "wdc_nvda_trajectories.csv"
        ),
        "rule_diagnostics_csv": destination / "rule_diagnostics.csv",
        "trade_events_csv": destination / "trade_events.csv",
        "position_ledger_csv": destination / "position_ledger.csv",
        "contributions_csv": destination / "contributions.csv",
        "targets_csv": destination / "targets.csv",
        "equity_csv": destination / "equity.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "manifest_json": destination / "manifest.json",
    }
    simple_frames = {
        "summary_csv": summary,
        "period_summary_csv": period_summary,
        "event_study_csv": event_study,
        "label_summary_csv": label_summary,
        "winner_episodes_csv": episodes,
        "named_trajectories_csv": trajectories,
        "rule_diagnostics_csv": rule_diagnostics,
        "data_audit_csv": data_audit,
    }
    for key, value in simple_frames.items():
        atomic_to_csv(value, outputs[key], index=False)

    scenario_outputs: dict[str, list[pd.DataFrame]] = {
        "trade_events_csv": [],
        "position_ledger_csv": [],
        "contributions_csv": [],
        "targets_csv": [],
        "equity_csv": [],
    }
    for scenario in scenarios:
        for key, value in (
            ("trade_events_csv", scenario.trade_events),
            ("position_ledger_csv", scenario.positions),
            ("contributions_csv", scenario.contributions),
            ("targets_csv", scenario.targets),
            ("equity_csv", scenario.result.daily),
        ):
            labeled_frame = value.copy()
            labeled_frame.insert(0, "Scenario", scenario.label)
            scenario_outputs[key].append(labeled_frame)
    spy_labeled = spy_equity.copy()
    spy_labeled.insert(0, "Scenario", "SPY_BUY_HOLD")
    scenario_outputs["equity_csv"].append(spy_labeled)
    for key, frames in scenario_outputs.items():
        atomic_to_csv(
            pd.concat(frames, ignore_index=True),
            outputs[key],
            index=False,
        )

    _atomic_text(
        outputs["report_html"],
        _render_report(
            summary,
            period_summary,
            label_summary,
            event_study,
            rule_diagnostics,
            base,
            top_winner=top_winner,
            version=normalized_version,
        ),
    )
    _atomic_json(
        outputs["manifest_json"],
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": (
                f"{normalized_version} multi-bagger event study and "
                "independent-quarter inflection diagnostic"
            ),
            "model_status": "POST_HOC_DIAGNOSTIC_PROTOTYPE",
            "validation_is_fresh": False,
            "optimization_performed": False,
            "config": asdict(config),
            "version": normalized_version,
            "event_labels": {
                "Label4x24m": "maximum next-504-session return >= 300%",
                "Label10x36m": "maximum next-756-session return >= 900%",
                "LabelTopDecile24m": (
                    "same-date top decile of next-504-session end return"
                ),
            },
            "staging_rule": (
                "3% scout; 8% only after one newly reported qualifying "
                "financial quarter; 15% only after a second new quarter"
            ),
            "execution_rule": (
                "causal final W-FRI session close signal, next session open"
            ),
            "transaction_cost_bps": settings.transaction_cost_bps,
            "top_winner": top_winner,
            "financial_point_in_time": False,
            "frozen_v6_v7_v8_unchanged": True,
            "frozen_strategy_sha256": _sha256(
                Path(frozen_strategy_path)
            ),
            "backfill_status_sha256": _sha256(
                Path(backfill_status_path)
            ),
            "membership_sha256": _sha256(Path(membership_path)),
            "caveats": [
                (
                    "Macrotrends financials are restated current history "
                    "with an approximate 45-day availability lag, not true "
                    "point-in-time statements."
                ),
                (
                    "The 575-name S&P membership proxy omits unavailable "
                    "acquired and delisted constituents."
                ),
                (
                    "Monthly observations overlap in their forward-return "
                    "windows and therefore are not independent samples."
                ),
                (
                    "All 2025 and 2026 outcomes were already observed; no "
                    "reported period is fresh out-of-sample validation."
                ),
            ],
            "outputs": {
                name: str(path) for name, path in outputs.items()
            },
        },
    )
    return V81Artifacts(output_dir=destination, **outputs)


def _build_summary(
    scenarios: tuple[V8Scenario, ...],
    spy_summary: dict[str, object],
    *,
    v8_summary_path: str | Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        result = scenario.result.summary
        closed = scenario.positions.loc[
            scenario.positions["Status"].eq("CLOSED")
        ]
        rows.append(
            {
                "Strategy": scenario.label,
                "StartDate": result.start_date,
                "EndDate": result.end_date,
                "FinalValue": result.final_value,
                "ROI": result.roi_percent,
                "CAGR": result.cagr_percent,
                "Sharpe": result.sharpe_ratio,
                "MaxDrawdown": result.max_drawdown_percent,
                "AnnualizedTurnover": result.annualized_turnover,
                "ModelBuys": int(
                    scenario.trade_events["TradeAction"].eq("BUY").sum()
                ),
                "ModelSells": int(
                    scenario.trade_events["TradeAction"].eq("SELL").sum()
                ),
                "ScaleUps": int(
                    scenario.trade_events["TradeAction"]
                    .eq("SCALE_UP")
                    .sum()
                ),
                "DistinctTickers": int(
                    scenario.positions["Ticker"].nunique()
                ),
                "MedianClosedHoldingDays": (
                    float(closed["HoldingCalendarDays"].median())
                    if not closed.empty
                    else np.nan
                ),
                "ClosedLossRate": (
                    float(closed["NetPnL"].lt(0).mean() * 100)
                    if not closed.empty
                    else np.nan
                ),
                "ExcludedTickers": ",".join(
                    scenario.excluded_tickers
                ),
            }
        )
    rows.append(
        {
            "Strategy": "SPY_BUY_HOLD",
            "StartDate": spy_summary["StartDate"],
            "EndDate": spy_summary["EndDate"],
            "FinalValue": spy_summary["FinalValue"],
            "ROI": spy_summary["ROI"],
            "CAGR": spy_summary["CAGR"],
            "Sharpe": spy_summary["Sharpe"],
            "MaxDrawdown": spy_summary["MaxDrawdown"],
            "AnnualizedTurnover": 0.0,
            "ModelBuys": 1,
            "ModelSells": 0,
            "ScaleUps": 0,
            "DistinctTickers": 1,
            "MedianClosedHoldingDays": np.nan,
            "ClosedLossRate": np.nan,
            "ExcludedTickers": "",
        }
    )
    summary = pd.DataFrame(rows)
    if v8_summary_path is not None and Path(v8_summary_path).exists():
        previous = pd.read_csv(v8_summary_path)
        selected = previous.loc[
            previous["Strategy"].eq("V8_HYBRID")
        ].copy()
        if len(selected) == 1:
            selected["Strategy"] = "V8_0_REFERENCE"
            summary = pd.concat(
                [
                    summary,
                    selected.reindex(columns=summary.columns),
                ],
                ignore_index=True,
            )
    return summary


def _centered_rank(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
) -> pd.Series:
    return _centered_rank_values(
        frame,
        pd.to_numeric(frame[column], errors="coerce"),
        eligible,
    )


def _centered_rank_values(
    frame: pd.DataFrame,
    values: pd.Series,
    eligible: pd.Series,
) -> pd.Series:
    return (
        values.where(eligible)
        .groupby(frame["Date"])
        .rank(pct=True, method="average")
        - 0.5
    )


def _ticker_row(
    indexed: pd.DataFrame,
    ticker: str,
) -> pd.Series | None:
    if ticker not in indexed.index:
        return None
    row = indexed.loc[ticker]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def _row_number(row: pd.Series | None, column: str) -> float:
    if row is None:
        return np.nan
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def _row_bool(row: pd.Series | None, column: str) -> bool:
    return bool(row is not None and bool(row.get(column, False)))


def _row_timestamp(
    row: pd.Series | None,
    column: str,
) -> pd.Timestamp | None:
    if row is None:
        return None
    value = pd.to_datetime(row.get(column), errors="coerce")
    return pd.Timestamp(value) if pd.notna(value) else None


def _state_timestamp(value: object) -> pd.Timestamp | None:
    converted = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(converted) if pd.notna(converted) else None


def _core_exit_reason(
    row: pd.Series | None,
    state: dict[str, object],
    config: V81InflectionConfig,
) -> str | None:
    severe = bool(
        _row_number(row, "EpsTtmGrowthYoY") < -0.20
        and _row_number(row, "EbitdaTtmGrowthYoY") < -0.20
        and _row_number(row, "Trend200") < -0.15
    )
    if severe:
        return "CORE_SEVERE_THESIS_BREAK"
    if int(state.get("QuartersHeld", 0)) < config.core_minimum_hold_quarters:
        return None
    persistent = bool(
        _row_number(row, "GrowthFactor") < -0.20
        and _row_number(row, "QualityFactor") < -0.15
    )
    return "CORE_QUALITY_GROWTH_BREAK" if persistent else None


def _inflection_exit_reason(
    row: pd.Series | None,
    state: dict[str, object],
    config: V81InflectionConfig,
) -> str | None:
    entry_close = pd.to_numeric(
        state.get("EntryClose"), errors="coerce"
    )
    current_close = _row_number(row, "Close")
    since_entry = (
        current_close / float(entry_close) - 1
        if pd.notna(entry_close) and float(entry_close) > 0
        else np.nan
    )
    if pd.notna(since_entry) and since_entry <= config.hard_stop_return:
        return "INFLECTION_HARD_STOP"
    severe_fundamental = bool(
        _row_number(row, "RevenueGrowthYoY") < 0
        and _row_number(row, "EbitdaTtm") <= 0
        and _row_number(row, "Trend200") < -0.15
    )
    severe_deterioration = bool(
        _row_number(row, "EpsTtmGrowthYoY") < 0
        and _row_number(row, "EbitdaTtmGrowthYoY") < 0
        and _row_number(row, "OperatingMarginChangeYoY") < -0.05
        and _row_number(row, "Trend200") < -0.15
    )
    if severe_fundamental or severe_deterioration:
        return "INFLECTION_SEVERE_THESIS_BREAK"
    if int(state.get("WeeksHeld", 0)) < config.inflection_minimum_hold_weeks:
        return None
    persistent = bool(
        int(state.get("FailedNewQuarters", 0)) >= 2
        and _row_number(row, "PriceVolumeEvidenceScore") < 0
    )
    return (
        "INFLECTION_TWO_QUARTER_BREAKDOWN"
        if persistent
        else None
    )


def _core_buy_reason(row: pd.Series) -> str:
    return (
        "대형 복리주 매수: "
        f"시총 순위 {_fmt_rank(row.get('MarketCapRank'))}, "
        f"CoreScore {_fmt(row.get('CoreScore'))}, "
        f"성장 {_fmt(row.get('GrowthFactor'))}, "
        f"품질 {_fmt(row.get('QualityFactor'))}."
    )


def _core_sell_reason(row: pd.Series | None, code: str) -> str:
    return (
        f"장기 투자 논리 훼손({code}): "
        f"EPS 성장 {_pct(_row_number(row, 'EpsTtmGrowthYoY'))}, "
        f"EBITDA 성장 {_pct(_row_number(row, 'EbitdaTtmGrowthYoY'))}, "
        f"200일 추세 {_pct(_row_number(row, 'Trend200'))}."
    )


def _inflection_buy_reason(row: pd.Series) -> str:
    return (
        f"{row.get('SignalArchetype', 'INFLECTION')} 탐색 매수: "
        f"순위 {_fmt_rank(row.get('InflectionRankV81'))}, "
        f"점수 {_fmt(row.get('InflectionScoreV81'))}, "
        f"매출 성장 {_pct(row.get('RevenueGrowthYoY'))}, "
        f"영업마진 YoY 변화 {_pct(row.get('OperatingMarginChangeYoY'))}. "
        "3%만 진입하며 새 재무분기가 확인될 때만 확대."
    )


def _scale_reason(
    row: pd.Series | None,
    stage: int,
    confirmations: int,
) -> str:
    target = {2: "8%", 3: "15%"}[stage]
    return (
        f"진입 후 서로 다른 새 재무분기 {confirmations}개가 "
        f"성장·수익성·가격 조건을 재확인해 {target}로 확대. "
        f"최근 분기말 {_row_timestamp(row, 'FinancialPeriodEnd')}, "
        f"V8.1 점수 {_fmt(_row_number(row, 'InflectionScoreV81'))}."
    )


def _inflection_sell_reason(
    row: pd.Series | None,
    code: str,
    state: dict[str, object],
) -> str:
    return (
        f"변곡점 투자 논리 훼손({code}): "
        f"실패한 새 분기 {int(state.get('FailedNewQuarters', 0))}개, "
        f"매출 성장 {_pct(_row_number(row, 'RevenueGrowthYoY'))}, "
        f"마진 변화 {_pct(_row_number(row, 'OperatingMarginChangeYoY'))}, "
        f"126일 수익률 {_pct(_row_number(row, 'Return126'))}."
    )


def _render_report(
    summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
    event_study: pd.DataFrame,
    rule_diagnostics: pd.DataFrame,
    base: V8Scenario,
    *,
    top_winner: str,
    version: str,
) -> str:
    top_features = (
        event_study.sort_values(
            ["Label", "BestTailLift"], ascending=[True, False]
        )
        .groupby("Label", as_index=False)
        .head(6)
    )
    trades = base.trade_events.loc[
        base.trade_events["Sleeve"].eq("INFLECTION")
        & base.trade_events["TradeAction"].isin(
            ["BUY", "SCALE_UP", "SELL"]
        )
    ].copy()
    trade_columns = [
        "ExecutionDate",
        "Ticker",
        "TradeAction",
        "TargetWeight",
        "SignalArchetype",
        "ConfirmedNewQuarters",
        "FinancialPeriodEnd",
        "Reason",
    ]
    caveat = (
        "주의: 이 결과는 Macrotrends의 현재 재작성 재무와 불완전한 "
        "575종목 과거 유니버스를 사용한 사후 진단이다. 2025·2026은 "
        "이미 관찰됐으므로 독립 OOS가 아니며, 수익률에는 생존편향·"
        "다중검정·사후관찰 오염 가능성이 있다."
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{version} Inflection Research</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;
padding:0 20px;color:#202124}} table{{border-collapse:collapse;width:100%;
font-size:13px;margin:10px 0 28px}} th,td{{border:1px solid #ddd;
padding:7px;text-align:right}} th{{background:#f3f5f7}}
td:first-child,td:nth-child(2){{text-align:left}} .warning{{background:#fff3cd;
padding:12px;border-left:5px solid #d39e00}} code{{background:#f1f3f4;
padding:2px 4px}}</style></head><body>
<h1>{version} 멀티배거 변곡점 진단</h1>
<p class="warning">{caveat}</p>
<h2>무엇을 고쳤나</h2>
<p>V8.0의 4주·8주 반복 확인을 폐기했다. {version}은 3% 탐색 진입 후
서로 다른 새 재무분기 하나가 다시 조건을 만족해야 8%, 두 번째 새
분기가 확인돼야 15%가 된다. WDC형 순환 회복과 NVDA형 구조적 성장을
서로 다른 게이트로 분리했다. V8_2는 여기에 영업마진의 큰 순차 개선과
52주 고점 복귀가 함께 나타나는 가격 선행형 신호를 추가한다.</p>
<h2>성과 비교</h2>{summary.to_html(index=False, border=0)}
<h2>구간별 비교</h2>{period_summary.to_html(index=False, border=0)}
<h2>멀티배거 표본 수</h2>{label_summary.to_html(index=False, border=0)}
<h2>상위 특성</h2>{top_features.to_html(index=False, border=0)}
<h2>고정 규칙 진단</h2>{rule_diagnostics.to_html(index=False, border=0)}
<h2>Inflection 매매</h2>{trades[
        [column for column in trade_columns if column in trades]
    ].to_html(index=False, border=0)}
<p>최대 기여 종목: <strong>{top_winner}</strong>. 이 종목 제외 결과는
성과표에 별도로 표시했다.</p>
</body></html>"""


def _fmt(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):.3f}" if pd.notna(numeric) else "N/A"


def _fmt_rank(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{int(round(float(numeric)))}위" if pd.notna(numeric) else "N/A"


def _pct(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):+.1%}" if pd.notna(numeric) else "N/A"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )
