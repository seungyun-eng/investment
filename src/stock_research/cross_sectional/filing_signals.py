from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FILING_FEATURE_COLUMNS = {
    "RevenueGrowthYoYFiled": "FiledRevenueGrowthYoY",
    "NetIncomeGrowthYoYFiled": "FiledNetIncomeGrowthYoY",
    "GrossMargin": "FiledGrossMargin",
    "OperatingMargin": "FiledOperatingMargin",
    "FreeCashFlowMargin": "FiledFreeCashFlowMargin",
    "NetDebtToAssets": "FiledNetDebtToAssets",
    "DebtToEquity": "FiledDebtToEquity",
    "InterestCoverage": "FiledInterestCoverage",
    "ShareGrowthYoYFiled": "FiledShareGrowthYoY",
    "StockCompensationToRevenue": "FiledStockCompensationToRevenue",
    "ResearchAndDevelopmentToRevenue": "FiledResearchAndDevelopmentToRevenue",
    "RiskTermsPer10kWords": "FiledRiskTermsPer10kWords",
    "LeverageTermsPer10kWords": "FiledLeverageTermsPer10kWords",
    "DilutionTermsPer10kWords": "FiledDilutionTermsPer10kWords",
    "CompetitionTermsPer10kWords": "FiledCompetitionTermsPer10kWords",
    "GoingConcernFlag": "FiledGoingConcernFlag",
    "MaterialWeaknessFlag": "FiledMaterialWeaknessFlag",
    "RestatementFlag": "FiledRestatementFlag",
}

FINANCIAL_FILING_COLUMNS = tuple(
    target
    for source, target in FILING_FEATURE_COLUMNS.items()
    if source
    not in {
        "RiskTermsPer10kWords",
        "LeverageTermsPer10kWords",
        "DilutionTermsPer10kWords",
        "CompetitionTermsPer10kWords",
        "GoingConcernFlag",
        "MaterialWeaknessFlag",
        "RestatementFlag",
    }
)


@dataclass(frozen=True)
class FilingHybridConfig:
    core_slots: int = 3
    core_total_weight: float = 0.70
    core_minimum_hold_weeks: int = 13
    core_hard_stop_return: float = -0.25
    core_replacement_score_advantage: float = 0.10
    core_replacement_confirm_weeks: int = 2
    tactical_slots: int = 2
    tactical_position_weight: float = 0.15
    tactical_hard_stop_return: float = -0.15
    tactical_time_stop_weeks: int = 8
    tactical_required_peak_return: float = 0.10
    tactical_trailing_activation_return: float = 0.15
    tactical_trailing_drawdown: float = -0.10
    thesis_break_confirm_weeks: int = 2
    market_regime_enabled: bool = True
    market_regime_sma_sessions: int = 200
    market_regime_fast_sessions: int = 50
    market_regime_band: float = 0.01
    risk_off_core_total_weight: float = 0.40

    @property
    def core_position_weight(self) -> float:
        return self.core_total_weight / self.core_slots

    def __post_init__(self) -> None:
        maximum = (
            self.core_total_weight
            + self.tactical_slots * self.tactical_position_weight
        )
        if self.core_slots < 1 or self.tactical_slots < 1:
            raise ValueError("core and tactical slots must be positive")
        if maximum > 1 + 1e-9:
            raise ValueError("maximum filing hybrid weights exceed 100%")
        if not 0 <= self.risk_off_core_total_weight <= self.core_total_weight:
            raise ValueError(
                "risk_off_core_total_weight must be between zero and core total"
            )
        if not 1 <= self.market_regime_fast_sessions < self.market_regime_sma_sessions:
            raise ValueError(
                "market_regime_fast_sessions must be below the slow window"
            )
        if not -1 < self.core_hard_stop_return < 0:
            raise ValueError("core_hard_stop_return must be between -1 and 0")
        if not -1 < self.tactical_hard_stop_return < 0:
            raise ValueError(
                "tactical_hard_stop_return must be between -1 and 0"
            )


def merge_filing_features(
    panel: pd.DataFrame,
    filing_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only filings available by each market date, per ticker."""

    if filing_features.empty:
        result = panel.copy()
        return _add_empty_filing_columns(result)
    required = {"Ticker", "AvailableDate"}
    missing = required - set(filing_features)
    if missing:
        raise ValueError(f"Filing features missing columns: {sorted(missing)}")
    available_sources = [
        column for column in FILING_FEATURE_COLUMNS if column in filing_features
    ]
    right = filing_features[
        ["Ticker", "AvailableDate", "AccessionNumber", "Form", *available_sources]
    ].copy()
    right["Ticker"] = right["Ticker"].astype(str).str.upper()
    right["AvailableDate"] = pd.to_datetime(
        right["AvailableDate"], errors="coerce"
    ).dt.tz_localize(None)
    right = right.rename(columns=FILING_FEATURE_COLUMNS)
    for column in FINANCIAL_FILING_COLUMNS:
        if column in right:
            right[column] = right.groupby("Ticker", sort=False)[column].ffill()
    right = right.dropna(subset=["AvailableDate"])
    right = right.sort_values(["Ticker", "AvailableDate"]).drop_duplicates(
        ["Ticker", "AvailableDate"], keep="last"
    )

    left = panel.copy()
    left["Ticker"] = left["Ticker"].astype(str).str.upper()
    left["Date"] = pd.to_datetime(left["Date"], errors="raise").dt.tz_localize(
        None
    )
    outputs: list[pd.DataFrame] = []
    for ticker, ticker_panel in left.groupby("Ticker", sort=False):
        ticker_filings = right.loc[right["Ticker"].eq(ticker)].drop(
            columns="Ticker"
        )
        ordered = ticker_panel.sort_values("Date")
        if ticker_filings.empty:
            outputs.append(_add_empty_filing_columns(ordered))
            continue
        merged = pd.merge_asof(
            ordered,
            ticker_filings.sort_values("AvailableDate"),
            left_on="Date",
            right_on="AvailableDate",
            direction="backward",
            allow_exact_matches=True,
        )
        outputs.append(merged)
    result = pd.concat(outputs, ignore_index=True)
    result["FilingAgeDays"] = (
        result["Date"] - result["AvailableDate"]
    ).dt.days
    if result["FilingAgeDays"].dropna().lt(0).any():
        raise RuntimeError("Future SEC filing leaked into the market panel")
    return result.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def add_market_regime(
    panel: pd.DataFrame,
    spy_prices: pd.DataFrame,
    config: FilingHybridConfig,
) -> pd.DataFrame:
    """Add a causal SPY moving-average regime with a no-trade hysteresis band."""

    frame = panel.copy()
    if not config.market_regime_enabled:
        frame["MarketRiskOn"] = True
        frame["SPYTrendRegime"] = np.nan
        return frame
    required = {"Date", "Close"}
    missing = required - set(spy_prices)
    if missing:
        raise ValueError(f"SPY prices missing columns: {sorted(missing)}")
    spy = spy_prices[["Date", "Close"]].copy()
    spy["Date"] = pd.to_datetime(spy["Date"], errors="raise")
    spy = spy.sort_values("Date").drop_duplicates("Date", keep="last")
    close = pd.to_numeric(spy["Close"], errors="coerce")
    average = close.rolling(
        config.market_regime_sma_sessions,
        min_periods=config.market_regime_sma_sessions,
    ).mean()
    fast_average = close.rolling(
        config.market_regime_fast_sessions,
        min_periods=config.market_regime_fast_sessions,
    ).mean()
    spy["SPYTrendRegime"] = close / average - 1
    spy["SPYFastVsSlowRegime"] = fast_average / average - 1
    state = True
    states: list[bool] = []
    for trend, fast_vs_slow in zip(
        spy["SPYTrendRegime"],
        spy["SPYFastVsSlowRegime"],
        strict=True,
    ):
        if pd.notna(trend) and pd.notna(fast_vs_slow):
            if trend > config.market_regime_band and fast_vs_slow > 0:
                state = True
            elif trend < -config.market_regime_band or fast_vs_slow < 0:
                state = False
        states.append(state)
    spy["MarketRiskOn"] = states
    result = frame.merge(
        spy[
            [
                "Date",
                "SPYTrendRegime",
                "SPYFastVsSlowRegime",
                "MarketRiskOn",
            ]
        ],
        on="Date",
        how="left",
        validate="many_to_one",
    )
    result["MarketRiskOn"] = (
        result["MarketRiskOn"].astype("boolean").ffill().fillna(True).astype(bool)
    )
    return result


def add_filing_factors(
    panel: pd.DataFrame,
    *,
    minimum_cross_section_size: int = 4,
) -> pd.DataFrame:
    frame = _add_empty_filing_columns(panel.copy())
    eligible = frame["Eligible"].fillna(False)
    positive = {
        "FiledRevenueGrowthYoY": _rank(
            frame, "FiledRevenueGrowthYoY", eligible, minimum_cross_section_size
        ),
        "FiledNetIncomeGrowthYoY": _rank(
            frame,
            "FiledNetIncomeGrowthYoY",
            eligible,
            minimum_cross_section_size,
        ),
        "FiledGrossMargin": _rank(
            frame, "FiledGrossMargin", eligible, minimum_cross_section_size
        ),
        "FiledOperatingMargin": _rank(
            frame,
            "FiledOperatingMargin",
            eligible,
            minimum_cross_section_size,
        ),
        "FiledFreeCashFlowMargin": _rank(
            frame,
            "FiledFreeCashFlowMargin",
            eligible,
            minimum_cross_section_size,
        ),
        "FiledInterestCoverage": _rank(
            frame,
            "FiledInterestCoverage",
            eligible,
            minimum_cross_section_size,
        ),
        "LowNetDebt": -_rank(
            frame,
            "FiledNetDebtToAssets",
            eligible,
            minimum_cross_section_size,
        ),
        "LowDebtToEquity": -_rank(
            frame,
            "FiledDebtToEquity",
            eligible,
            minimum_cross_section_size,
        ),
        "LowShareDilution": -_rank(
            frame,
            "FiledShareGrowthYoY",
            eligible,
            minimum_cross_section_size,
        ),
        "LowStockCompensation": -_rank(
            frame,
            "FiledStockCompensationToRevenue",
            eligible,
            minimum_cross_section_size,
        ),
    }
    frame["FilingDurabilityFactor"] = pd.concat(
        [
            positive["FiledRevenueGrowthYoY"],
            positive["FiledNetIncomeGrowthYoY"],
            positive["FiledGrossMargin"],
            positive["FiledOperatingMargin"],
            positive["FiledFreeCashFlowMargin"],
        ],
        axis=1,
    ).mean(axis=1)
    frame["FilingBalanceSheetFactor"] = pd.concat(
        [
            positive["FiledInterestCoverage"],
            positive["LowNetDebt"],
            positive["LowDebtToEquity"],
            positive["LowShareDilution"],
            positive["LowStockCompensation"],
        ],
        axis=1,
    ).mean(axis=1)
    disclosure_risk = pd.concat(
        [
            _rank(
                frame,
                "FiledRiskTermsPer10kWords",
                eligible,
                minimum_cross_section_size,
            ),
            _rank(
                frame,
                "FiledLeverageTermsPer10kWords",
                eligible,
                minimum_cross_section_size,
            ),
            _rank(
                frame,
                "FiledDilutionTermsPer10kWords",
                eligible,
                minimum_cross_section_size,
            ),
        ],
        axis=1,
    ).mean(axis=1)
    red_flags = sum(
        frame[column].eq(True).astype(int)
        for column in (
            "FiledGoingConcernFlag",
            "FiledMaterialWeaknessFlag",
            "FiledRestatementFlag",
        )
    )
    frame["FilingDisclosureSafetyFactor"] = (
        -disclosure_risk - 0.25 * red_flags
    ).clip(-0.75, 0.50)
    frame["FilingFundamentalFactor"] = (
        0.50 * frame["FilingDurabilityFactor"]
        + 0.35 * frame["FilingBalanceSheetFactor"]
        + 0.15 * frame["FilingDisclosureSafetyFactor"]
    )
    coverage_columns = [
        "FiledRevenueGrowthYoY",
        "FiledOperatingMargin",
        "FiledFreeCashFlowMargin",
        "FiledNetDebtToAssets",
        "FiledInterestCoverage",
        "FiledShareGrowthYoY",
    ]
    frame["FilingCoverageCount"] = frame[coverage_columns].notna().sum(axis=1)
    frame.loc[
        ~eligible,
        [
            "FilingDurabilityFactor",
            "FilingBalanceSheetFactor",
            "FilingDisclosureSafetyFactor",
            "FilingFundamentalFactor",
        ],
    ] = np.nan
    return frame


def add_filing_hybrid_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Create fixed core and tactical scores without optimizing on outcomes."""

    frame = panel.copy()
    filing_available = frame["FilingFundamentalFactor"].notna()
    base_true_value = (
        0.55 * pd.to_numeric(frame["GrowthFactor"], errors="coerce")
        + 0.45 * pd.to_numeric(frame["QualityFactor"], errors="coerce")
    )
    filing_true_value = (
        0.30 * pd.to_numeric(frame["GrowthFactor"], errors="coerce")
        + 0.25 * pd.to_numeric(frame["QualityFactor"], errors="coerce")
        + 0.20
        * pd.to_numeric(frame["FilingDurabilityFactor"], errors="coerce")
        + 0.15
        * pd.to_numeric(
            frame["FilingBalanceSheetFactor"], errors="coerce"
        )
        + 0.10
        * pd.to_numeric(
            frame["FilingDisclosureSafetyFactor"], errors="coerce"
        )
    )
    frame["TrueValueFactor"] = filing_true_value.where(
        filing_available, base_true_value
    )
    technical = frame[
        [column for column in ("MAFactor", "MACDFactor", "OBVFactor") if column in frame]
    ].mean(axis=1)
    frame["FilingCoreScore"] = (
        0.80 * frame["TrueValueFactor"]
        + 0.10 * pd.to_numeric(frame["TrendFactor"], errors="coerce")
        + 0.10 * pd.to_numeric(frame["RiskControlFactor"], errors="coerce")
    )
    frame["FilingTacticalScore"] = (
        0.45 * technical
        + 0.30 * pd.to_numeric(frame["MomentumFactor"], errors="coerce")
        + 0.15 * pd.to_numeric(frame["GrowthFactor"], errors="coerce")
        + 0.05
        * pd.to_numeric(frame["FilingDurabilityFactor"], errors="coerce").fillna(0)
        + 0.05
        * pd.to_numeric(
            frame["FilingBalanceSheetFactor"], errors="coerce"
        ).fillna(0)
    )
    critical = (
        frame["FiledGoingConcernFlag"].eq(True)
        | frame["FiledMaterialWeaknessFlag"].eq(True)
        | frame["FiledRestatementFlag"].eq(True)
    )
    eligible = frame["Eligible"].fillna(False)
    frame["FilingCoreQualified"] = (
        eligible
        & ~critical
        & frame["TrueValueFactor"].ge(0)
        & frame["FilingFundamentalFactor"].fillna(0).ge(-0.10)
        & frame["Trend200"].ge(-0.10)
    )
    frame["FilingTacticalQualified"] = (
        eligible
        & ~critical
        & frame["Trend200"].ge(0)
        & frame["Return126"].ge(0.10)
        & frame["GrowthFactor"].ge(-0.20)
        & frame["FilingBalanceSheetFactor"].fillna(0).ge(-0.40)
    )
    frame["FilingCoreRank"] = (
        frame["FilingCoreScore"]
        .where(frame["FilingCoreQualified"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    frame["FilingTacticalRank"] = (
        frame["FilingTacticalScore"]
        .where(frame["FilingTacticalQualified"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    return frame


def generate_filing_hybrid_targets(
    signal_days: pd.DataFrame,
    config: FilingHybridConfig,
) -> pd.DataFrame:
    """Hold three core names and two tightly risk-controlled tactical names."""

    frame = signal_days.sort_values(["Date", "Ticker"]).copy()
    core: dict[str, dict[str, float | int]] = {}
    tactical: dict[str, dict[str, float | int]] = {}
    last_weights: dict[str, float] = {}
    records: list[pd.DataFrame] = []
    challenger_pair: tuple[str, str] | None = None
    challenger_weeks = 0

    for date, group in frame.groupby("Date", sort=True):
        indexed = group.set_index("Ticker", drop=False)
        market_risk_on = bool(group["MarketRiskOn"].iloc[0]) if "MarketRiskOn" in group else True
        actions: dict[str, str] = {}
        reasons: dict[str, str] = {}
        prior_sleeves = {
            **{ticker: "CORE" for ticker in core},
            **{ticker: "TACTICAL" for ticker in tactical},
        }

        for ticker in list(core):
            state = core[ticker]
            state["WeeksHeld"] = int(state["WeeksHeld"]) + 1
            row = _row(indexed, ticker)
            close = _number(row, "Close")
            entry = float(state["EntryClose"])
            position_return = close / entry - 1 if close > 0 and entry > 0 else np.nan
            critical = _critical_filing_flag(row)
            thesis_break = bool(
                _number(row, "GrowthFactor") < -0.20
                and _number(row, "QualityFactor") < -0.15
                and _number(row, "FilingFundamentalFactor") < -0.10
            )
            state["BreakWeeks"] = (
                int(state["BreakWeeks"]) + 1 if thesis_break else 0
            )
            reason = None
            if pd.notna(position_return) and position_return <= config.core_hard_stop_return:
                reason = "CORE_HARD_STOP"
            elif critical:
                reason = "CORE_FILING_RED_FLAG"
            elif (
                int(state["WeeksHeld"]) >= config.core_minimum_hold_weeks
                and int(state["BreakWeeks"]) >= config.thesis_break_confirm_weeks
            ):
                reason = "CORE_PERSISTENT_THESIS_BREAK"
            if reason:
                core.pop(ticker)
                actions[ticker] = "SELL"
                reasons[ticker] = reason

        for ticker in list(tactical):
            state = tactical[ticker]
            state["WeeksHeld"] = int(state["WeeksHeld"]) + 1
            row = _row(indexed, ticker)
            close = _number(row, "Close")
            entry = float(state["EntryClose"])
            if close > 0:
                state["PeakClose"] = max(float(state["PeakClose"]), close)
            peak = float(state["PeakClose"])
            position_return = close / entry - 1 if close > 0 and entry > 0 else np.nan
            peak_return = peak / entry - 1 if peak > 0 and entry > 0 else np.nan
            trailing = close / peak - 1 if close > 0 and peak > 0 else np.nan
            reason = None
            if not market_risk_on:
                reason = "MARKET_RISK_OFF"
            elif pd.notna(position_return) and position_return <= config.tactical_hard_stop_return:
                reason = "TACTICAL_HARD_STOP"
            elif _critical_filing_flag(row):
                reason = "TACTICAL_FILING_RED_FLAG"
            elif (
                pd.notna(peak_return)
                and peak_return >= config.tactical_trailing_activation_return
                and trailing <= config.tactical_trailing_drawdown
            ):
                reason = "TACTICAL_TRAILING_STOP"
            elif (
                int(state["WeeksHeld"]) >= config.tactical_time_stop_weeks
                and peak_return < config.tactical_required_peak_return
            ):
                reason = "TACTICAL_TIME_STOP"
            elif (
                _number(row, "MomentumFactor") < -0.15
                and _number(row, "TrendFactor") < -0.15
            ):
                reason = "TACTICAL_TREND_BREAK"
            if reason:
                tactical.pop(ticker)
                actions[ticker] = "SELL"
                reasons[ticker] = reason

        core_candidates = group.loc[group["FilingCoreQualified"]].sort_values(
            ["FilingCoreRank", "Ticker"]
        )
        for candidate in core_candidates.itertuples(index=False):
            ticker = str(candidate.Ticker)
            if len(core) >= config.core_slots:
                break
            if ticker in core or ticker in tactical or actions.get(ticker) == "SELL":
                continue
            close = float(candidate.Close)
            if not np.isfinite(close) or close <= 0:
                continue
            core[ticker] = {"EntryClose": close, "WeeksHeld": 0, "BreakWeeks": 0}
            actions[ticker] = "BUY"
            reasons[ticker] = "CORE_TRUE_VALUE_AND_FILING_QUALITY"

        replacement_candidates = core_candidates.loc[
            ~core_candidates["Ticker"].isin(set(core) | set(tactical))
        ]
        held_rows = group.loc[group["Ticker"].isin(core)]
        if (
            market_risk_on
            and len(core) == config.core_slots
            and not replacement_candidates.empty
            and not held_rows.empty
        ):
            best = replacement_candidates.iloc[0]
            weakest = held_rows.sort_values(
                ["FilingCoreScore", "Ticker"],
                ascending=[True, True],
            ).iloc[0]
            best_ticker = str(best["Ticker"])
            weakest_ticker = str(weakest["Ticker"])
            advantage = float(
                best["FilingCoreScore"] - weakest["FilingCoreScore"]
            )
            enough_history = (
                int(core[weakest_ticker]["WeeksHeld"])
                >= config.core_minimum_hold_weeks
            )
            pair = (best_ticker, weakest_ticker)
            if (
                advantage >= config.core_replacement_score_advantage
                and enough_history
            ):
                if pair == challenger_pair:
                    challenger_weeks += 1
                else:
                    challenger_pair = pair
                    challenger_weeks = 1
            else:
                challenger_pair = None
                challenger_weeks = 0
            if challenger_weeks >= config.core_replacement_confirm_weeks:
                close = float(best["Close"])
                if np.isfinite(close) and close > 0:
                    core.pop(weakest_ticker)
                    actions[weakest_ticker] = "SELL"
                    reasons[weakest_ticker] = "CORE_STRONGER_CHALLENGER"
                    core[best_ticker] = {
                        "EntryClose": close,
                        "WeeksHeld": 0,
                        "BreakWeeks": 0,
                    }
                    actions[best_ticker] = "BUY"
                    reasons[best_ticker] = "CORE_CONFIRMED_STRONGER_VALUE"
                    challenger_pair = None
                    challenger_weeks = 0
        else:
            challenger_pair = None
            challenger_weeks = 0

        tactical_candidates = group.loc[
            group["FilingTacticalQualified"] & market_risk_on
        ].sort_values(["FilingTacticalRank", "Ticker"])
        for candidate in tactical_candidates.itertuples(index=False):
            ticker = str(candidate.Ticker)
            if len(tactical) >= config.tactical_slots:
                break
            if ticker in tactical or ticker in core or actions.get(ticker) == "SELL":
                continue
            close = float(candidate.Close)
            if not np.isfinite(close) or close <= 0:
                continue
            tactical[ticker] = {
                "EntryClose": close,
                "PeakClose": close,
                "WeeksHeld": 0,
            }
            actions[ticker] = "BUY"
            reasons[ticker] = "TACTICAL_PRICE_VOLUME_WITH_FILING_GUARD"

        core_total_weight = (
            config.core_total_weight
            if market_risk_on
            else config.risk_off_core_total_weight
        )
        core_weight = core_total_weight / config.core_slots
        weights = {ticker: core_weight for ticker in core}
        weights.update(
            {
                ticker: config.tactical_position_weight
                for ticker in tactical
            }
        )
        changed = set(weights) != set(last_weights) or any(
            abs(weight - last_weights.get(ticker, 0.0)) > 1e-12
            for ticker, weight in weights.items()
        )
        if not changed:
            continue
        output_rows: list[dict[str, object]] = []
        for ticker in sorted(set(weights) | set(last_weights)):
            row = _row(indexed, ticker)
            base = row.to_dict() if row is not None else {"Date": date, "Ticker": ticker}
            sleeve = (
                "CORE"
                if ticker in core
                else "TACTICAL"
                if ticker in tactical
                else prior_sleeves.get(ticker, "UNKNOWN")
            )
            base.update(
                {
                    "Date": pd.Timestamp(date),
                    "SignalDate": pd.Timestamp(date),
                    "Ticker": ticker,
                    "Sleeve": sleeve,
                    "TargetWeight": weights.get(ticker, 0.0),
                    "TradeAction": actions.get(ticker, "HOLD"),
                    "Reason": reasons.get(ticker, "UNCHANGED_THESIS"),
                    "MarketRiskOn": market_risk_on,
                }
            )
            output_rows.append(base)
        records.append(pd.DataFrame(output_rows))
        last_weights = weights
    if not records:
        return pd.DataFrame(
            columns=["Date", "Ticker", "TargetWeight", "Sleeve", "TradeAction", "Reason"]
        )
    return pd.concat(records, ignore_index=True)


def _add_empty_filing_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for target in FILING_FEATURE_COLUMNS.values():
        if target not in result:
            result[target] = np.nan
    for column in ("AvailableDate", "AccessionNumber", "Form"):
        if column not in result:
            result[column] = pd.NaT if column == "AvailableDate" else ""
    return result


def _rank(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
    minimum_count: int,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").where(eligible)
    counts = values.notna().groupby(frame["Date"]).transform("sum")
    ranks = values.groupby(frame["Date"]).rank(pct=True, method="average") - 0.5
    return ranks.where(counts.ge(minimum_count))


def _row(indexed: pd.DataFrame, ticker: str) -> pd.Series | None:
    if ticker not in indexed.index:
        return None
    row = indexed.loc[ticker]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def _number(row: pd.Series | None, column: str) -> float:
    if row is None:
        return np.nan
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def _critical_filing_flag(row: pd.Series | None) -> bool:
    if row is None:
        return False
    return any(
        bool(row.get(column, False))
        for column in (
            "FiledGoingConcernFlag",
            "FiledMaterialWeaknessFlag",
            "FiledRestatementFlag",
        )
        if pd.notna(row.get(column))
    )
