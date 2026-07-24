from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import optuna
from optuna.importance import get_param_importances
import pandas as pd

from .backtest import run_long_only
from .strategies.technical import (
    TechnicalParams,
    technical_buy_signal,
    technical_sell_signal,
)
from .strategies.vix import VixParams, vix_buy_signal, vix_sell_signal


def _importance(study: optuna.Study) -> dict[str, float]:
    try:
        return get_param_importances(study)
    except (RuntimeError, ValueError):
        return {name: 0.0 for name in study.best_params}


def optimize_vix(
    data: pd.DataFrame,
    *,
    tpe_trials: int = 1500,
    cma_trials: int = 500,
    seed: int = 42,
) -> tuple[VixParams, float, dict[str, float]]:
    def objective_tpe(trial: optuna.Trial) -> float:
        buy = trial.suggest_float("vix_buy_th", 0, 100)
        # Fix: enforce the constraint in both TPE and CMA, not only in one stage.
        sell = trial.suggest_float("vix_sell_th", 0, buy)
        params = VixParams(
            vix_buy_th=buy,
            vix_sell_th=sell,
            rsi_buy_th=trial.suggest_float("rsi_buy_th", 0, 100),
            rsi_sell_th=trial.suggest_float("rsi_sell_th", 0, 100),
            boll_buffer=trial.suggest_float("boll_buffer", 0, 0.1),
        )
        return run_long_only(
            data, params, vix_buy_signal, vix_sell_signal
        ).summary.roi_percent

    tpe = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    tpe.optimize(objective_tpe, n_trials=tpe_trials)
    importance = _importance(tpe)
    best = dict(tpe.best_params)

    important = {name for name, value in importance.items() if value >= 0.05}
    bounds = {
        "vix_buy_th": (0.0, 100.0),
        "vix_sell_th": (0.0, 100.0),
        "rsi_buy_th": (0.0, 100.0),
        "rsi_sell_th": (0.0, 100.0),
        "boll_buffer": (0.0, 0.1),
    }

    if cma_trials > 0 and important:
        def objective_cma(trial: optuna.Trial) -> float:
            candidate = dict(best)
            for name in important:
                low, high = bounds[name]
                radius = 0.2 * (high - low)
                narrow_low = max(low, best[name] - radius)
                narrow_high = min(high, best[name] + radius)
                if name == "vix_sell_th":
                    narrow_high = min(narrow_high, candidate["vix_buy_th"])
                candidate[name] = trial.suggest_float(name, narrow_low, narrow_high)
            if candidate["vix_sell_th"] > candidate["vix_buy_th"]:
                return -1e12
            params = VixParams(**candidate)
            return run_long_only(
                data, params, vix_buy_signal, vix_sell_signal
            ).summary.roi_percent

        cma = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.CmaEsSampler(seed=seed),
        )
        cma.optimize(objective_cma, n_trials=cma_trials)
        best.update(cma.best_params)

    params = VixParams(**best)
    roi = run_long_only(
        data, params, vix_buy_signal, vix_sell_signal
    ).summary.roi_percent
    return params, roi, importance


def optimize_technical(
    data: pd.DataFrame,
    *,
    tpe_trials: int = 500,
    cma_trials: int = 200,
    seed: int = 42,
) -> tuple[TechnicalParams, float, dict[str, float]]:
    def objective_tpe(trial: optuna.Trial) -> float:
        params = TechnicalParams(
            rsi_sell_th=trial.suggest_float("rsi_sell_th", 30, 100),
            boll_buffer=trial.suggest_float("boll_buffer", 0, 0.1),
        )
        return run_long_only(
            data, params, technical_buy_signal, technical_sell_signal
        ).summary.roi_percent

    tpe = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    tpe.optimize(objective_tpe, n_trials=tpe_trials)
    importance = _importance(tpe)
    best = dict(tpe.best_params)
    important = {name for name, value in importance.items() if value >= 0.05}

    if cma_trials > 0 and important:
        bounds = {"rsi_sell_th": (30.0, 100.0), "boll_buffer": (0.0, 0.1)}

        def objective_cma(trial: optuna.Trial) -> float:
            candidate = dict(best)
            for name in important:
                low, high = bounds[name]
                radius = 0.2 * (high - low)
                candidate[name] = trial.suggest_float(
                    name,
                    max(low, best[name] - radius),
                    min(high, best[name] + radius),
                )
            params = TechnicalParams(**candidate)
            return run_long_only(
                data, params, technical_buy_signal, technical_sell_signal
            ).summary.roi_percent

        cma = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.CmaEsSampler(seed=seed),
        )
        cma.optimize(objective_cma, n_trials=cma_trials)
        best.update(cma.best_params)

    params = TechnicalParams(**best)
    roi = run_long_only(
        data, params, technical_buy_signal, technical_sell_signal
    ).summary.roi_percent
    return params, roi, importance
