from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DOWNSIDE_FEATURES = (
    "Return21",
    "Return63",
    "Trend50",
    "Trend200",
    "RSI14",
    "MACD",
    "MACDSignal",
    "RevenueGrowthYoY",
    "OperatingMargin",
    "FreeCashFlowMargin",
    "VixPercentile",
    "MacroConfirmationScore",
    "ModelRisk",
)


def _forward_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for index in range(len(values) - horizon):
        future_low = np.nanmin(values[index + 1 : index + horizon + 1])
        result[index] = future_low / values[index] - 1
    return pd.Series(result, index=close.index)


def add_strict_oos_downside_probability(
    features: pd.DataFrame,
    *,
    horizon: int = 21,
    drawdown_threshold: float = -0.10,
    minimum_training_rows: int = 504,
    retrain_every: int = 21,
) -> pd.DataFrame:
    """Add expanding-window probabilities without training on future labels."""

    frame = features.copy().sort_values("Date").reset_index(drop=True)
    missing = set(DOWNSIDE_FEATURES) - set(frame)
    if missing:
        raise ValueError(f"Equity downside features missing: {sorted(missing)}")
    matrix = frame.loc[:, DOWNSIDE_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    target = (
        _forward_drawdown(frame["Close"], horizon) <= drawdown_threshold
    ).astype(float)
    target.iloc[-horizon:] = np.nan
    probabilities = np.full(len(frame), np.nan)

    for prediction_start in range(
        minimum_training_rows + horizon,
        len(frame),
        retrain_every,
    ):
        last_known_label = prediction_start - horizon
        train_x = matrix.iloc[: last_known_label + 1]
        train_y = target.iloc[: last_known_label + 1]
        valid = train_y.notna() & train_x.notna().any(axis=1)
        if valid.sum() < minimum_training_rows:
            continue
        classes = train_y.loc[valid].unique()
        if len(classes) < 2:
            continue
        usable_columns = train_x.loc[valid].columns[
            train_x.loc[valid].notna().any(axis=0)
        ]
        if usable_columns.empty:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=0,
            ),
        )
        model.fit(train_x.loc[valid, usable_columns], train_y.loc[valid])
        prediction_end = min(prediction_start + retrain_every, len(frame))
        probabilities[prediction_start:prediction_end] = model.predict_proba(
            matrix.loc[
                prediction_start : prediction_end - 1,
                usable_columns,
            ]
        )[:, 1]

    frame["DownsideProbability21"] = probabilities
    # Keep the original column while the TSLA entry point remains supported.
    frame["TslaDownsideProbability21"] = probabilities
    return frame
