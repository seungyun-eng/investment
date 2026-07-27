from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from .config import ResearchConfig
from .features import feature_columns
from .models import (
    CandidateSpec,
    build_estimator,
    candidate_specs,
    predict_estimator,
    usable_feature_columns,
)
from .targets import primary_target_names


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    selections: pd.DataFrame
    candidate_scores: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_columns: list[str]


def _safe_auc(actual: pd.Series, predicted: pd.Series) -> float:
    valid = actual.notna() & predicted.notna()
    if valid.sum() == 0 or actual[valid].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(actual[valid], predicted[valid]))


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    valid = actual.notna() & predicted.notna()
    if valid.sum() == 0:
        return {"AUC": np.nan, "Brier": np.nan, "AveragePrecision": np.nan, "Count": 0}
    y = actual[valid].astype(int)
    p = predicted[valid].clip(0, 1)
    return {
        "AUC": _safe_auc(y, p),
        "Brier": float(brier_score_loss(y, p)),
        "AveragePrecision": (
            float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan
        ),
        "BaseRate": float(y.mean()),
        "Count": len(y),
    }


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    valid = actual.notna() & predicted.notna()
    if valid.sum() == 0:
        return {
            "Spearman": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "QuintileSpread": np.nan,
            "Count": 0,
        }
    y = actual[valid]
    p = predicted[valid]
    ranks = p.rank(method="first")
    buckets = pd.qcut(ranks, q=min(5, len(p)), labels=False, duplicates="drop")
    spread = (
        float(y[buckets == buckets.max()].mean() - y[buckets == buckets.min()].mean())
        if buckets.nunique() >= 2
        else np.nan
    )
    return {
        "Spearman": float(y.corr(p, method="spearman")),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "QuintileSpread": spread,
        "Count": len(y),
    }


def _inner_years(train: pd.DataFrame, config: ResearchConfig) -> list[int]:
    years = sorted(train["Date"].dt.year.unique())
    return years[-config.inner_validation_years :]


def _sample_rows(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride == 1 or frame.empty:
        return frame
    return frame.sort_values("Date").iloc[::stride].copy()


def _score_candidate(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    target_end: str,
    candidate: CandidateSpec,
    config: ResearchConfig,
) -> tuple[float, list[dict[str, object]]]:
    fold_rows: list[dict[str, object]] = []
    selected_columns = usable_feature_columns(train, features, candidate, config)
    if not selected_columns:
        return -np.inf, fold_rows
    for validation_year in _inner_years(train, config):
        validation_start = pd.Timestamp(validation_year, 1, 1)
        validation_end = pd.Timestamp(validation_year + 1, 1, 1)
        inner_train = train[
            (train["Date"] < validation_start)
            & (train[target_end] < validation_start)
            & train[target].notna()
        ]
        validation = train[
            (train["Date"] >= validation_start)
            & (train["Date"] < validation_end)
            & train[target].notna()
        ]
        inner_train = _sample_rows(inner_train, config.training_stride)
        validation = _sample_rows(validation, config.training_stride)
        if len(inner_train) < max(300, config.minimum_train_rows // 2) or len(validation) < 50:
            continue
        if candidate.task == "classification" and inner_train[target].nunique() < 2:
            continue
        estimator = build_estimator(candidate, config.random_seed)
        estimator.fit(inner_train[selected_columns], inner_train[target])
        predicted = pd.Series(
            predict_estimator(estimator, validation[selected_columns], candidate.task),
            index=validation.index,
        )
        if candidate.task == "classification":
            metric = classification_metrics(validation[target], predicted)
            auc = metric["AUC"]
            brier = metric["Brier"]
            score = (auc if np.isfinite(auc) else 0.5) - brier
        else:
            metric = regression_metrics(validation[target], predicted)
            scale = max(float(validation[target].std(ddof=1)), 1e-8)
            spearman = metric["Spearman"]
            score = (spearman if np.isfinite(spearman) else 0.0) - 0.25 * metric["MAE"] / scale
        fold_rows.append(
            {
                "ValidationYear": validation_year,
                "Score": score,
                "FeatureCount": len(selected_columns),
                **metric,
            }
        )
    if not fold_rows:
        return -np.inf, fold_rows
    return float(np.nanmean([row["Score"] for row in fold_rows])), fold_rows


def _select_candidate(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    target_end: str,
    task: str,
    outer_year: int,
    config: ResearchConfig,
) -> tuple[CandidateSpec, list[dict[str, object]]]:
    diagnostics: list[dict[str, object]] = []
    best_spec: CandidateSpec | None = None
    best_score = -np.inf
    for candidate in candidate_specs(task, config):
        score, folds = _score_candidate(
            train,
            features,
            target,
            target_end,
            candidate,
            config,
        )
        diagnostics.append(
            {
                "OuterYear": outer_year,
                "Task": task,
                "Target": target,
                "Candidate": candidate.name,
                "Family": candidate.family,
                "FeatureGroup": candidate.feature_group,
                "MeanScore": score,
                "InnerFoldCount": len(folds),
            }
        )
        if score > best_score:
            best_score = score
            best_spec = candidate
    if best_spec is None:
        raise RuntimeError(f"No valid {task} candidate for outer year {outer_year}.")
    return best_spec, diagnostics


def _fit_predict_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_features: list[str],
    target: str,
    target_end: str,
    candidate: CandidateSpec,
    test_start: pd.Timestamp,
    config: ResearchConfig,
    *,
    collect_importance: bool = False,
) -> tuple[np.ndarray, list[str], int, pd.DataFrame]:
    eligible = train[
        train[target].notna()
        & (train[target_end] < test_start)
    ]
    eligible = _sample_rows(eligible, config.training_stride)
    columns = usable_feature_columns(eligible, all_features, candidate, config)
    if len(eligible) < config.minimum_train_rows:
        raise RuntimeError(f"Only {len(eligible)} eligible rows for {target}.")
    if candidate.task == "classification" and eligible[target].nunique() < 2:
        raise RuntimeError(f"Classification target {target} has one class.")
    estimator = build_estimator(candidate, config.random_seed)
    estimator.fit(eligible[columns], eligible[target])
    importance = pd.DataFrame()
    if collect_importance:
        model = estimator.named_steps["model"]
        if candidate.family in {"logistic", "ridge"}:
            coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
            signed = coefficients[: len(columns)]
            magnitude = np.abs(signed)
        else:
            valid = test[target].notna()
            importance_test = test.loc[valid]
            if len(importance_test) > 256:
                importance_test = importance_test.sample(
                    256,
                    random_state=config.random_seed,
                )
            if not importance_test.empty:
                scoring = (
                    "neg_brier_score"
                    if candidate.task == "classification"
                    else "neg_mean_absolute_error"
                )
                measured = permutation_importance(
                    estimator,
                    importance_test[columns],
                    importance_test[target],
                    scoring=scoring,
                    n_repeats=2,
                    random_state=config.random_seed,
                )
                signed = measured.importances_mean
                magnitude = np.maximum(signed, 0)
            else:
                signed = np.zeros(len(columns))
                magnitude = np.zeros(len(columns))
        total = float(magnitude.sum())
        importance = pd.DataFrame(
            {
                "Feature": columns,
                "ImportanceMagnitude": magnitude,
                "SignedEffect": signed,
                "NormalizedImportance": magnitude / total if total > 0 else 0.0,
            }
        )
    return (
        predict_estimator(estimator, test[columns], candidate.task),
        columns,
        len(eligible),
        importance,
    )


def _metric_tables(predictions: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    risk_label = round(abs(config.primary_drawdown_threshold) * 100)
    periods: list[tuple[str, pd.DataFrame]] = [("All", predictions)]
    for year, part in predictions.groupby(predictions["Date"].dt.year):
        periods.append((str(year), part))
    for period, part in periods:
        for horizon in config.risk_horizons:
            target = f"DrawdownHit_{horizon}_{risk_label}"
            prediction = f"RiskProbability_{horizon}"
            rows.append(
                {
                    "Period": period,
                    "Task": "classification",
                    "Horizon": horizon,
                    **classification_metrics(part[target], part[prediction]),
                }
            )
        for horizon in config.return_horizons:
            target = f"ExcessReturn_{horizon}"
            prediction = f"PredictedExcessReturn_{horizon}"
            rows.append(
                {
                    "Period": period,
                    "Task": "regression",
                    "Horizon": horizon,
                    **regression_metrics(part[target], part[prediction]),
                }
            )
    return pd.DataFrame(rows)


def _calibration_table(predictions: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    label = round(abs(config.primary_drawdown_threshold) * 100)
    target = f"DrawdownHit_{config.primary_risk_horizon}_{label}"
    probability = f"RiskProbability_{config.primary_risk_horizon}"
    valid = predictions.dropna(subset=[target, probability]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["CalibrationBin"] = pd.qcut(
        valid[probability].rank(method="first"),
        q=min(10, len(valid)),
        labels=False,
        duplicates="drop",
    )
    return (
        valid.groupby("CalibrationBin", as_index=False)
        .agg(
            MeanPredicted=(probability, "mean"),
            ActualFrequency=(target, "mean"),
            Count=(target, "size"),
        )
    )


def nested_walk_forward(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    config: ResearchConfig,
) -> WalkForwardResult:
    data = features.merge(targets, on="Date", how="inner", validate="one_to_one")
    data = data.sort_values("Date").reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"])
    all_features = feature_columns(data)
    primary_risk, primary_return = primary_target_names(config)
    primary_risk_end = f"TargetEndDate_{config.primary_risk_horizon}"
    primary_return_end = f"TargetEndDate_{config.primary_return_horizon}"
    first_possible = int(data["Date"].dt.year.min()) + config.training_years
    first_year = max(config.first_test_year, first_possible)
    last_year = int(data["Date"].dt.year.max())

    prediction_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    importance_parts: list[pd.DataFrame] = []
    for outer_year in range(first_year, last_year + 1):
        test_start = pd.Timestamp(outer_year, 1, 1)
        test_end = pd.Timestamp(outer_year + 1, 1, 1)
        train_start = (
            pd.Timestamp(data["Date"].min())
            if config.expanding_training
            else pd.Timestamp(outer_year - config.training_years, 1, 1)
        )
        train = data[(data["Date"] >= train_start) & (data["Date"] < test_start)].copy()
        test = data[(data["Date"] >= test_start) & (data["Date"] < test_end)].copy()
        if len(train) < config.minimum_train_rows or test.empty:
            continue

        classifier_selection_train = train[
            train[primary_risk_end].notna()
            & (train[primary_risk_end] < test_start)
        ].copy()
        regressor_selection_train = train[
            train[primary_return_end].notna()
            & (train[primary_return_end] < test_start)
        ].copy()
        classifier, classifier_diagnostics = _select_candidate(
            classifier_selection_train,
            all_features,
            primary_risk,
            primary_risk_end,
            "classification",
            outer_year,
            config,
        )
        regressor, regressor_diagnostics = _select_candidate(
            regressor_selection_train,
            all_features,
            primary_return,
            primary_return_end,
            "regression",
            outer_year,
            config,
        )
        candidate_rows.extend(classifier_diagnostics + regressor_diagnostics)

        diagnostic_columns = [
            "Date",
            "Open",
            "Close",
            "CashRate",
            "VIX",
            "Drawdown252",
            "MacroConfirmationScore",
            "MacroStressLevel",
            "MacroStressTrend",
            "MacroStressBreadth",
            "MacroLevel_Volatility",
            "MacroLevel_Credit",
            "MacroLevel_FinancialConditions",
            "MacroLevel_Labor",
            "MacroLevel_YieldCurve",
        ]
        output = test[
            [name for name in diagnostic_columns if name in test]
        ].copy()
        for horizon in config.risk_horizons:
            target = (
                f"DrawdownHit_{horizon}_"
                f"{round(abs(config.primary_drawdown_threshold) * 100)}"
            )
            end = f"TargetEndDate_{horizon}"
            predicted, columns, count, importance = _fit_predict_target(
                train,
                test,
                all_features,
                target,
                end,
                classifier,
                test_start,
                config,
                collect_importance=horizon == config.primary_risk_horizon,
            )
            if not importance.empty:
                importance.insert(0, "Horizon", horizon)
                importance.insert(0, "Task", "classification")
                importance.insert(0, "OuterYear", outer_year)
                importance_parts.append(importance)
            output[f"RiskProbability_{horizon}"] = predicted
            output[target] = test[target].to_numpy()
            selection_rows.append(
                {
                    "OuterYear": outer_year,
                    "Task": "classification",
                    "Horizon": horizon,
                    "Target": target,
                    "Candidate": classifier.name,
                    "Family": classifier.family,
                    "FeatureGroup": classifier.feature_group,
                    "FeatureCount": len(columns),
                    "TrainRows": count,
                    "TrainStart": train_start,
                    "TrainEnd": train["Date"].max(),
                    "TestRows": len(test),
                }
            )
        for horizon in config.return_horizons:
            target = f"ExcessReturn_{horizon}"
            end = f"TargetEndDate_{horizon}"
            predicted, columns, count, importance = _fit_predict_target(
                train,
                test,
                all_features,
                target,
                end,
                regressor,
                test_start,
                config,
                collect_importance=horizon == config.primary_return_horizon,
            )
            if not importance.empty:
                importance.insert(0, "Horizon", horizon)
                importance.insert(0, "Task", "regression")
                importance.insert(0, "OuterYear", outer_year)
                importance_parts.append(importance)
            output[f"PredictedExcessReturn_{horizon}"] = predicted
            output[target] = test[target].to_numpy()
            output[f"ForwardReturn_{horizon}"] = test[
                f"ForwardReturn_{horizon}"
            ].to_numpy()
            selection_rows.append(
                {
                    "OuterYear": outer_year,
                    "Task": "regression",
                    "Horizon": horizon,
                    "Target": target,
                    "Candidate": regressor.name,
                    "Family": regressor.family,
                    "FeatureGroup": regressor.feature_group,
                    "FeatureCount": len(columns),
                    "TrainRows": count,
                    "TrainStart": train_start,
                    "TrainEnd": train["Date"].max(),
                    "TestRows": len(test),
                }
            )
        prediction_parts.append(output)

    if not prediction_parts:
        raise RuntimeError("Walk-forward evaluation produced no out-of-sample predictions.")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return WalkForwardResult(
        predictions=predictions,
        selections=pd.DataFrame(selection_rows),
        candidate_scores=pd.DataFrame(candidate_rows),
        metrics=_metric_tables(predictions, config),
        calibration=_calibration_table(predictions, config),
        feature_importance=(
            pd.concat(importance_parts, ignore_index=True)
            if importance_parts
            else pd.DataFrame()
        ),
        feature_columns=all_features,
    )
