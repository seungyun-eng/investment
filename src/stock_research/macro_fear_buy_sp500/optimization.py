from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .config import FearBuyParams, FearBuySettings
from .features import build_fear_features
from .portfolio import run_signal_backtest
from .strategy import generate_fear_buy_signals


@dataclass
class OptimizationResult:
    selected_params: FearBuyParams
    candidates: pd.DataFrame


def optimize_on_development_period(
    predictions: pd.DataFrame,
    base_params: FearBuyParams,
    settings: FearBuySettings,
    *,
    quick: bool = False,
) -> OptimizationResult:
    """Select parameters on the development period only."""

    features = build_fear_features(predictions, base_params)
    development = features[
        features["Date"] <= pd.Timestamp(settings.development_end)
    ].copy()
    if development.empty:
        raise ValueError("No observations are available in the development period.")

    core_weights = settings.optimization_core_weights
    mild_scores = settings.optimization_mild_scores
    fear_scores = settings.optimization_fear_scores
    euphoria_scores = settings.optimization_euphoria_scores
    hold_sessions = settings.optimization_hold_sessions
    profit_buffers = settings.optimization_profit_buffers
    if quick:
        core_weights = core_weights[:: max(1, len(core_weights) - 1)]
        mild_scores = (base_params.mild_fear_score,)
        fear_scores = (base_params.fear_fear_score,)
        euphoria_scores = (base_params.trim_euphoria_score,)
        hold_sessions = (base_params.minimum_hold_sessions,)
        profit_buffers = (base_params.trim_profit_buffer,)

    rows: list[dict[str, object]] = []
    for core_weight in core_weights:
        for mild_score in mild_scores:
            for fear_score in fear_scores:
                if mild_score >= fear_score:
                    continue
                for euphoria_score in euphoria_scores:
                    for minimum_hold in hold_sessions:
                        for profit_buffer in profit_buffers:
                            params = replace(
                                base_params,
                                core_weight=core_weight,
                                mild_fear_score=mild_score,
                                fear_fear_score=fear_score,
                                trim_euphoria_score=euphoria_score,
                                minimum_hold_sessions=minimum_hold,
                                trim_profit_buffer=profit_buffer,
                            )
                            signals = generate_fear_buy_signals(
                                development,
                                params,
                            )
                            result = run_signal_backtest(
                                signals,
                                params,
                                settings,
                                name="DevelopmentCandidate",
                            )
                            summary = result.summary
                            rows.append(
                                {
                                    "CoreWeight": core_weight,
                                    "MildFearScore": mild_score,
                                    "FearFearScore": fear_score,
                                    "TrimEuphoriaScore": euphoria_score,
                                    "MinimumHoldSessions": minimum_hold,
                                    "TrimProfitBuffer": profit_buffer,
                                    "DevelopmentFinalValue": summary.final_value,
                                    "DevelopmentROI(%)": summary.roi_percent,
                                    "DevelopmentCAGR(%)": summary.cagr_percent,
                                    "DevelopmentMDD(%)": summary.max_drawdown_percent,
                                    "DevelopmentSharpe": summary.sharpe_ratio,
                                    "DevelopmentAverageExposure(%)": (
                                        summary.average_exposure_percent
                                    ),
                                    "DevelopmentTurnover": summary.turnover_multiple,
                                    "DevelopmentTrades": summary.rebalance_count,
                                }
                            )
    candidates = pd.DataFrame(rows).sort_values(
        ["DevelopmentCAGR(%)", "DevelopmentSharpe"],
        ascending=[False, False],
    )
    candidates = candidates.reset_index(drop=True)
    candidates.insert(0, "DevelopmentRank", candidates.index + 1)
    winner = candidates.iloc[0]
    selected = replace(
        base_params,
        core_weight=float(winner["CoreWeight"]),
        mild_fear_score=float(winner["MildFearScore"]),
        fear_fear_score=float(winner["FearFearScore"]),
        trim_euphoria_score=float(winner["TrimEuphoriaScore"]),
        minimum_hold_sessions=int(winner["MinimumHoldSessions"]),
        trim_profit_buffer=float(winner["TrimProfitBuffer"]),
    )
    return OptimizationResult(selected_params=selected, candidates=candidates)
