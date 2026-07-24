from __future__ import annotations

import optuna
import pandas as pd
from optuna.importance import get_param_importances

from .backtest import BacktestResult, run_long_only
from .data_loading import validate_actual_vix
from .strategies.technical import (
    TechnicalParams,
    technical_buy_signal,
    technical_sell_signal,
)
from .strategies.vix import (
    VixParams,
    VixRuleConfig,
    load_vix_rule_config,
    vix_buy_signal,
    vix_sell_signal,
    vix_trade_log_details,
)


def _importance(study: optuna.Study) -> dict[str, float]:
    try:
        return get_param_importances(study)
    except (ImportError, RuntimeError, ValueError):
        return {name: 0.0 for name in study.best_params}


def _vix_params(candidate: dict[str, float], rules: VixRuleConfig) -> VixParams:
    return VixParams(
        rsi_buy_th=float(candidate["rsi_buy_th"]),
        rsi_sell_th=float(candidate["rsi_sell_th"]),
        boll_buffer=float(candidate["boll_buffer"]),
        vix_buy_level=rules.vix_buy_level,
        vix_sell_level=rules.vix_sell_level,
    )


def run_vix_backtest(data: pd.DataFrame, params: VixParams) -> BacktestResult:
    return run_long_only(
        data, params, vix_buy_signal, vix_sell_signal,
        log_details=vix_trade_log_details,
    )


def _score_vix_trial(
    trial: optuna.Trial,
    data: pd.DataFrame,
    candidate: dict[str, float],
    rules: VixRuleConfig,
) -> float:
    if candidate["rsi_sell_th"] < candidate["rsi_buy_th"] + 15.0:
        raise optuna.TrialPruned("RSI sell threshold must be at least buy + 15.")
    result = run_vix_backtest(data, _vix_params(candidate, rules))
    trial.set_user_attr("BuyCount", result.summary.buys)
    trial.set_user_attr("SignalSellCount", result.summary.sells)
    trial.set_user_attr("LiquidationCount", result.summary.liquidations)
    trial.set_user_attr("CompletedTrades", result.summary.completed_trades)
    if result.summary.completed_trades < 2:
        raise optuna.TrialPruned("Fewer than two completed BUY-to-SELL trades.")
    return result.summary.roi_percent


def optimize_vix(
    data: pd.DataFrame,
    *,
    rules: VixRuleConfig | None = None,
    tpe_trials: int = 1500,
    cma_trials: int = 500,
    seed: int = 42,
) -> tuple[VixParams, float, dict[str, float]]:
    rules = rules or load_vix_rule_config()
    data = validate_actual_vix(data.copy())

    def objective_tpe(trial: optuna.Trial) -> float:
        buy = trial.suggest_float("rsi_buy_th", 20.0, 45.0)
        candidate = {
            "rsi_buy_th": buy,
            "rsi_sell_th": trial.suggest_float(
                "rsi_sell_th", max(55.0, buy + 15.0), 80.0
            ),
            "boll_buffer": trial.suggest_float("boll_buffer", 0.0, 0.03),
        }
        return _score_vix_trial(trial, data, candidate, rules)

    tpe = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    tpe.optimize(objective_tpe, n_trials=tpe_trials)
    try:
        best = dict(tpe.best_params)
    except ValueError as exc:
        raise RuntimeError(
            "No valid VIX trial completed at least two BUY-to-SELL trades."
        ) from exc
    importance = _importance(tpe)
    bounds = {
        "rsi_buy_th": (20.0, 45.0),
        "rsi_sell_th": (55.0, 80.0),
        "boll_buffer": (0.0, 0.03),
    }
    important = {name for name, value in importance.items() if value >= 0.05}

    if cma_trials > 0 and important:
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
            return _score_vix_trial(trial, data, candidate, rules)

        cma = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.CmaEsSampler(seed=seed)
        )
        cma.optimize(objective_cma, n_trials=cma_trials)
        completed = [
            trial for trial in cma.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            best.update(cma.best_params)

    params = _vix_params(best, rules)
    result = run_vix_backtest(data, params)
    return params, result.summary.roi_percent, importance


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
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
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
                    name, max(low, best[name] - radius), min(high, best[name] + radius)
                )
            params = TechnicalParams(**candidate)
            return run_long_only(
                data, params, technical_buy_signal, technical_sell_signal
            ).summary.roi_percent

        cma = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.CmaEsSampler(seed=seed)
        )
        cma.optimize(objective_cma, n_trials=cma_trials)
        best.update(cma.best_params)
    params = TechnicalParams(**best)
    roi = run_long_only(
        data, params, technical_buy_signal, technical_sell_signal
    ).summary.roi_percent
    return params, roi, importance
