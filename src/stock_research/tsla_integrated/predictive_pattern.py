from __future__ import annotations

"""Strict prequential probabilities for TSLA two-month pattern states."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TECHNICAL_FEATURES = (
    "Return1D",
    "Return5D",
    "Return20D",
    "Return60D",
    "RealizedVol21",
    "RealizedVol63",
    "Gap",
    "VolumeSurprise21",
    "Turnover21",
    "MACDSpread",
    "RSI14",
    "DistanceFromSMA50",
    "DistanceFromSMA200",
    "Beta126",
    "IdiosyncraticVol126",
)
SEC_FEATURES = (
    "MarginLevelZScore",
    "MarginTrend",
    "GrowthComposite",
    "ShareGrowthYoYFiled",
    "StockCompensationToRevenue",
    "ResearchAndDevelopmentToRevenue",
    "FundamentalsAgeDays",
)
MACRO_FEATURES = (
    "Treasury10Y",
    "FedFundsRate",
    "YieldCurve10Y2Y",
    "CPI",
    "UnemploymentRate",
    "VIX_Level",
    "VIX9D_Level",
    "VIX3M_Level",
    "VVIX_Level",
)
NEWS_FEATURES = (
    "NewsDirection",
    "NewsSignalActiveNumeric",
    "MarketAdjustedReaction",
    "VolumeSurprise20D",
    "EventClusters",
)


def add_two_month_excursion_targets(
    prices: pd.DataFrame,
    *,
    months: int = 2,
) -> pd.DataFrame:
    """Add terminal return and maximum close excursion through +2 months."""

    if months < 1:
        raise ValueError("months must be positive.")
    required = {"Date", "AdjClose"}
    if missing := required.difference(prices.columns):
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    frame = prices.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    close = pd.to_numeric(frame["AdjClose"], errors="coerce").to_numpy(dtype=float)
    dates = frame["Date"].to_numpy(dtype="datetime64[ns]")
    target_dates = (frame["Date"] + pd.DateOffset(months=months)).to_numpy(
        dtype="datetime64[ns]"
    )
    end_indices = np.searchsorted(dates, target_dates, side="left")
    maximum_gain = np.full(len(frame), np.nan)
    maximum_drawdown = np.full(len(frame), np.nan)
    terminal_return = np.full(len(frame), np.nan)
    end_dates = np.full(
        len(frame), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    for start, end in enumerate(end_indices):
        if end >= len(frame) or not np.isfinite(close[start]):
            continue
        future = close[start + 1 : end + 1]
        if not len(future) or not np.isfinite(future).any():
            continue
        maximum_gain[start] = np.nanmax(future) / close[start] - 1.0
        maximum_drawdown[start] = np.nanmin(future) / close[start] - 1.0
        terminal_return[start] = close[end] / close[start] - 1.0
        end_dates[start] = dates[end]
    frame["TargetEndDate"] = pd.to_datetime(end_dates)
    frame["ForwardMaxGain2M"] = maximum_gain
    frame["ForwardMaxDrawdown2M"] = maximum_drawdown
    frame["ForwardTerminalReturn2M"] = terminal_return
    return frame


def prepare_news_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the sparse stored-event proxy to numeric causal features."""

    result = frame.copy()
    score = pd.to_numeric(
        result.get("NewsScore", pd.Series(0.5, index=result.index)),
        errors="coerce",
    ).fillna(0.5)
    result["NewsDirection"] = ((score - 0.5) * 2.0).clip(-1.0, 1.0)
    active = result.get("NewsSignalActive", pd.Series(False, index=result.index))
    result["NewsSignalActiveNumeric"] = active.fillna(False).astype(float)
    for column in ("MarketAdjustedReaction", "VolumeSurprise20D", "EventClusters"):
        result[column] = pd.to_numeric(
            result.get(column, pd.Series(0.0, index=result.index)), errors="coerce"
        ).fillna(0.0)
    return result


def _make_model(c_value: float) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=3_000, random_state=20260809),
    )


def _causal_rolling_percentile(
    values: np.ndarray,
    *,
    window: int,
    minimum_history: int,
) -> np.ndarray:
    percentiles = np.full(len(values), np.nan)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        history = values[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if len(history) < minimum_history:
            continue
        percentiles[index] = (np.count_nonzero(history <= value) + 1.0) / (
            len(history) + 1.0
        )
    return percentiles


def prequential_binary_probability(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    positive_threshold: float,
    positive_when_above: bool,
    c_value: float,
    first_prediction_date: str = "2020-01-01",
    first_training_date: str = "2012-01-01",
    minimum_training_rows: int = 500,
    retrain_frequency: str = "monthly",
) -> pd.DataFrame:
    """Predict in blocks while training only on fully realized labels.

    The label for row ``i`` is usable only when its ``TargetEndDate`` is
    strictly earlier than the first date of the prediction block.
    """

    if c_value <= 0:
        raise ValueError("c_value must be positive.")
    if retrain_frequency not in {"monthly", "quarterly"}:
        raise ValueError("retrain_frequency must be monthly or quarterly.")
    required = {"Date", "TargetEndDate", target_column, *feature_columns}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing predictive columns: {sorted(missing)}")
    data = frame.copy().sort_values("Date").reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["TargetEndDate"] = pd.to_datetime(data["TargetEndDate"], errors="coerce")
    target_value = pd.to_numeric(data[target_column], errors="coerce")
    target = (
        target_value.ge(positive_threshold)
        if positive_when_above
        else target_value.le(positive_threshold)
    ).astype(float)
    target[target_value.isna()] = np.nan
    probabilities = np.full(len(data), np.nan)
    training_rows = np.full(len(data), np.nan)
    prediction_indices = data.index[data["Date"].ge(first_prediction_date)]
    period_frequency = "M" if retrain_frequency == "monthly" else "Q"
    periods = data.loc[prediction_indices, "Date"].dt.to_period(period_frequency)
    grouped = pd.Series(prediction_indices, index=prediction_indices).groupby(
        periods.to_numpy()
    )
    matrix = data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    for _, block_index in grouped:
        block = block_index.to_numpy(dtype=int)
        prediction_start = data.loc[block[0], "Date"]
        train = (
            data["Date"].ge(first_training_date)
            & data["TargetEndDate"].lt(prediction_start)
            & target.notna()
        )
        if train.sum() < minimum_training_rows or target.loc[train].nunique() < 2:
            continue
        model = _make_model(c_value)
        model.fit(matrix.loc[train], target.loc[train].astype(int))
        probabilities[block] = model.predict_proba(matrix.loc[block])[:, 1]
        training_rows[block] = int(train.sum())
    result = pd.DataFrame(
        {
            "Date": data["Date"],
            "ActualLabel": target,
            "Probability": probabilities,
            "TrainingRows": training_rows,
        }
    )
    result["CausalProbabilityPercentile"] = _causal_rolling_percentile(
        probabilities, window=252, minimum_history=126
    )
    return result


def classify_predictive_pattern(
    frame: pd.DataFrame,
    *,
    opportunity_percentile_threshold: float = 0.75,
    risk_percentile_threshold: float = 0.80,
    include_catalyst: bool = False,
) -> pd.DataFrame:
    """Combine predictive tiers with the interpretable structural families."""

    required = {
        "OpportunityPercentile",
        "RiskPercentile",
        "MOMENTUM_CONTINUATION",
        "WASHOUT_REVERSAL",
        "CATALYST_RERATING",
        "BLOWOFF_RISK",
        "BEAR_CONTINUATION_RISK",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing classification columns: {sorted(missing)}")
    result = frame.copy()
    opportunity_high = result["OpportunityPercentile"].ge(
        opportunity_percentile_threshold
    )
    risk_high = result["RiskPercentile"].ge(risk_percentile_threshold)
    upside_columns = ["MOMENTUM_CONTINUATION", "WASHOUT_REVERSAL"]
    if include_catalyst:
        upside_columns.append("CATALYST_RERATING")
    upside_top = result[upside_columns].idxmax(axis=1)
    risk_top = result[["BLOWOFF_RISK", "BEAR_CONTINUATION_RISK"]].idxmax(axis=1)
    result["PredictivePattern"] = "NEUTRAL_NO_EDGE"
    result.loc[opportunity_high & ~risk_high, "PredictivePattern"] = upside_top
    result.loc[risk_high & ~opportunity_high, "PredictivePattern"] = risk_top
    result.loc[opportunity_high & risk_high, "PredictivePattern"] = "CONFLICT_HIGH_BOTH"
    result["OpportunityHigh"] = opportunity_high
    result["RiskHigh"] = risk_high
    return result


def add_pattern_phase_flags(
    frame: pd.DataFrame,
    *,
    opportunity_percentile_threshold: float = 0.75,
    risk_percentile_threshold: float = 0.75,
    include_catalyst: bool = False,
) -> pd.DataFrame:
    """Separate observable setup, confirmation, and forecast-confidence phases.

    Multiple setups may be true at once.  This is intentional: TSLA can be in
    a strong momentum regime and a blow-off-risk regime simultaneously.
    """

    required = {
        "Return5D",
        "Return60D",
        "RSI14",
        "DistanceFromSMA200",
        "WASHOUT_REVERSAL",
        "CATALYST_RERATING",
        "BLOWOFF_RISK",
        "OpportunityPercentile",
        "RiskPercentile",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing phase columns: {sorted(missing)}")
    result = frame.copy()
    r5 = pd.to_numeric(result["Return5D"], errors="coerce")
    r60 = pd.to_numeric(result["Return60D"], errors="coerce")
    rsi = pd.to_numeric(result["RSI14"], errors="coerce")
    d200 = pd.to_numeric(result["DistanceFromSMA200"], errors="coerce")
    result["MomentumSetup"] = r60.gt(0.05) & d200.gt(0.0)
    result["WashoutSetup"] = r60.le(-0.10) & rsi.le(42.0) & d200.lt(0.0)
    result["WashoutReversalConfirmed"] = (
        result["WashoutSetup"]
        & pd.to_numeric(result["WASHOUT_REVERSAL"], errors="coerce").ge(0.62)
        & r5.gt(0.0)
    )
    result["CatalystConfirmed"] = include_catalyst & pd.to_numeric(
        result["CATALYST_RERATING"], errors="coerce"
    ).ge(0.58)
    result["BlowoffSetup"] = pd.to_numeric(
        result["BLOWOFF_RISK"], errors="coerce"
    ).ge(0.55)
    result["OpportunityHigh"] = result["OpportunityPercentile"].ge(
        opportunity_percentile_threshold
    )
    result["RiskHigh"] = result["RiskPercentile"].ge(risk_percentile_threshold)

    phases: list[str] = []
    setup_lists: list[str] = []
    for row in result.itertuples(index=False):
        setups: list[str] = []
        if row.MomentumSetup:
            setups.append("MOMENTUM")
        if row.WashoutSetup:
            setups.append("WASHOUT")
        if row.CatalystConfirmed:
            setups.append("CATALYST")
        if row.BlowoffSetup:
            setups.append("BLOWOFF")
        if row.RiskHigh:
            setups.append("BEAR_RISK")
        setup_lists.append("|".join(setups) if setups else "NONE")

        if row.CatalystConfirmed:
            phase = "CATALYST_RERATING_CONFIRMED"
        elif row.WashoutSetup:
            if row.RiskHigh:
                phase = "WASHOUT_FALLING_KNIFE_RISK"
            elif row.WashoutReversalConfirmed and row.OpportunityHigh:
                phase = "WASHOUT_REVERSAL_HIGH_EDGE"
            elif row.WashoutReversalConfirmed:
                phase = "WASHOUT_REVERSAL_EARLY_LOW_EDGE"
            else:
                phase = "WASHOUT_SETUP_UNCONFIRMED"
        elif row.MomentumSetup and row.BlowoffSetup:
            phase = "MOMENTUM_BLOWOFF_CONFLICT"
        elif row.RiskHigh:
            phase = "BEAR_CONTINUATION_RISK"
        elif row.MomentumSetup and row.OpportunityHigh:
            phase = "MOMENTUM_CONTINUATION_HIGH_EDGE"
        elif row.MomentumSetup:
            phase = "MOMENTUM_SETUP_LOW_EDGE"
        elif row.BlowoffSetup:
            phase = "BLOWOFF_RISK"
        else:
            phase = "NEUTRAL_NO_EDGE"
        phases.append(phase)
    result["ActiveSetups"] = setup_lists
    result["DetailedPatternPhase"] = phases
    return result


def build_pattern_execution_signals(
    classified: pd.DataFrame,
    *,
    scale_in_policy: str,
) -> pd.DataFrame:
    """Translate non-catalyst pattern phases into shared accumulation inputs."""

    if scale_in_policy not in {"never", "all_drawdowns", "washout_setup", "confirmed"}:
        raise ValueError("Unknown scale_in_policy.")
    required = {"Date", "AdjOpen", "AdjClose", "FedFundsRate"}
    if missing := required.difference(classified.columns):
        raise ValueError(f"Missing execution columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(classified["Date"], errors="coerce"),
            "Open": pd.to_numeric(classified["AdjOpen"], errors="coerce"),
            "Close": pd.to_numeric(classified["AdjClose"], errors="coerce"),
            "CashRate": pd.to_numeric(
                classified["FedFundsRate"], errors="coerce"
            ).fillna(0.0),
        }
    )
    if scale_in_policy == "never":
        scale_in = pd.Series(False, index=classified.index)
    elif scale_in_policy == "all_drawdowns":
        scale_in = pd.Series(True, index=classified.index)
    elif scale_in_policy == "washout_setup":
        if "WashoutSetup" not in classified:
            raise ValueError("WashoutSetup is required for washout_setup policy.")
        scale_in = classified["WashoutSetup"].fillna(False).astype(bool)
    else:
        if "WashoutReversalConfirmed" not in classified:
            raise ValueError("WashoutReversalConfirmed is required for confirmed policy.")
        scale_in = classified["WashoutReversalConfirmed"].fillna(False).astype(bool)
    result["CompositeScore"] = np.where(scale_in, 0.0, 1.0)
    result["DownsideProbability21"] = 0.0
    result["BearRegime"] = scale_in.to_numpy(dtype=bool)
    result["FastWashout"] = False
    result["BearEndBuySignal"] = False
    result["PriceDrawdown63"] = (
        result["Close"] / result["Close"].rolling(63, min_periods=1).max() - 1.0
    )
    return result
