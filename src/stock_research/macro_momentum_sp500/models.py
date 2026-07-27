from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ResearchConfig
from .features import select_feature_group


@dataclass(frozen=True)
class CandidateSpec:
    task: str
    family: str
    feature_group: str
    params: tuple[tuple[str, object], ...]

    @property
    def name(self) -> str:
        payload = "_".join(f"{key}={value}" for key, value in self.params)
        return f"{self.task}:{self.family}:{self.feature_group}:{payload}"

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["params"] = dict(self.params)
        result["name"] = self.name
        return result


def _spec(
    task: str,
    family: str,
    feature_group: str,
    **params: object,
) -> CandidateSpec:
    return CandidateSpec(
        task=task,
        family=family,
        feature_group=feature_group,
        params=tuple(sorted(params.items())),
    )


def candidate_specs(task: str, config: ResearchConfig) -> list[CandidateSpec]:
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unsupported model task: {task}")
    specs: list[CandidateSpec] = []
    groups = tuple(config.feature_groups)
    if task == "classification":
        for group in groups:
            for c_value in (0.01, 0.05, 0.2, 1.0, 5.0):
                for balanced in (False, True):
                    specs.append(
                        _spec(
                            task,
                            "logistic",
                            group,
                            C=c_value,
                            balanced=balanced,
                        )
                    )
            for learning_rate in (0.03, 0.07):
                for leaves in (7, 15, 31):
                    specs.append(
                        _spec(
                            task,
                            "hist_gradient_boosting",
                            group,
                            learning_rate=learning_rate,
                            max_leaf_nodes=leaves,
                            l2_regularization=1.0,
                        )
                    )
    else:
        for group in groups:
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                specs.append(_spec(task, "ridge", group, alpha=alpha))
            for learning_rate in (0.03, 0.07):
                for leaves in (7, 15, 31):
                    specs.append(
                        _spec(
                            task,
                            "hist_gradient_boosting",
                            group,
                            learning_rate=learning_rate,
                            max_leaf_nodes=leaves,
                            l2_regularization=1.0,
                        )
                    )
    return _balanced_sample(specs, config.search_budget_per_task, config.random_seed)


def _balanced_sample(
    specs: list[CandidateSpec],
    budget: int,
    seed: int,
) -> list[CandidateSpec]:
    if budget >= len(specs):
        return specs
    rng = np.random.default_rng(seed)
    selected: list[CandidateSpec] = []
    buckets: dict[tuple[str, str], list[CandidateSpec]] = {}
    for spec in specs:
        buckets.setdefault((spec.family, spec.feature_group), []).append(spec)
    while len(selected) < budget and buckets:
        for key in list(buckets):
            choices = buckets[key]
            if not choices:
                del buckets[key]
                continue
            index = int(rng.integers(0, len(choices)))
            selected.append(choices.pop(index))
            if len(selected) >= budget:
                break
    return selected


def usable_feature_columns(
    frame: pd.DataFrame,
    all_feature_columns: list[str],
    candidate: CandidateSpec,
    config: ResearchConfig,
) -> list[str]:
    selected = select_feature_group(all_feature_columns, candidate.feature_group, config)
    return [
        name
        for name in selected
        if frame[name].notna().mean() >= 0.20 and frame[name].nunique(dropna=True) > 1
    ]


def build_estimator(candidate: CandidateSpec, random_seed: int) -> Pipeline:
    params = dict(candidate.params)
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
        keep_empty_features=True,
    )
    if candidate.family == "logistic":
        estimator = LogisticRegression(
            C=float(params["C"]),
            class_weight="balanced" if params["balanced"] else None,
            max_iter=2000,
            random_state=random_seed,
        )
        return Pipeline(
            [("imputer", imputer), ("scale", StandardScaler()), ("model", estimator)]
        )
    if candidate.family == "ridge":
        estimator = Ridge(alpha=float(params["alpha"]))
        return Pipeline(
            [("imputer", imputer), ("scale", StandardScaler()), ("model", estimator)]
        )
    if candidate.family == "hist_gradient_boosting":
        common = {
            "learning_rate": float(params["learning_rate"]),
            "max_leaf_nodes": int(params["max_leaf_nodes"]),
            "l2_regularization": float(params["l2_regularization"]),
            "max_iter": 150,
            "min_samples_leaf": 30,
            "random_state": random_seed,
        }
        estimator = (
            HistGradientBoostingClassifier(**common)
            if candidate.task == "classification"
            else HistGradientBoostingRegressor(**common)
        )
        return Pipeline([("imputer", imputer), ("model", estimator)])
    raise ValueError(f"Unsupported model family: {candidate.family}")


def predict_estimator(estimator: Pipeline, values: pd.DataFrame, task: str) -> np.ndarray:
    if task == "classification":
        return estimator.predict_proba(values)[:, 1]
    return np.asarray(estimator.predict(values), dtype=float)
