from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .config import FearBuyParams, FearBuySettings
from .features import build_fear_features
from .portfolio import (
    PortfolioResult,
    run_signal_backtest,
    segment_result,
)
from .strategy import generate_fear_buy_signals


def tactical_buy_forward_returns(
    result: PortfolioResult,
    *,
    horizons: tuple[int, ...] = (21, 63, 126, 252),
) -> pd.DataFrame:
    """Measure post-execution SPY returns for actual tactical buys."""

    if result.trades.empty:
        return pd.DataFrame()
    buys = result.trades[
        (result.trades["Action"] == "BUY")
        & (result.trades["Sleeve"] == "TACTICAL")
    ].copy()
    if buys.empty:
        return pd.DataFrame()
    daily = result.daily.sort_values("Date").reset_index(drop=True)
    date_to_index = {
        pd.Timestamp(date): index for index, date in enumerate(daily["Date"])
    }
    rows: list[dict[str, object]] = []
    for trade in buys.itertuples(index=False):
        trade_date = pd.Timestamp(trade.Date)
        index = date_to_index[trade_date]
        row = {
            "Date": trade_date,
            "ExecutionPrice": float(trade.ExecutionPrice),
            "TargetWeight": float(trade.TargetWeight),
            "FearScore": float(trade.FearScore),
            "Reason": str(trade.Reason),
        }
        for horizon in horizons:
            future_index = index + horizon
            row[f"SPYForwardReturn_{horizon}d(%)"] = (
                (
                    float(daily.loc[future_index, "Close"])
                    / float(daily.loc[index, "Close"])
                    - 1.0
                )
                * 100.0
                if future_index < len(daily)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def tactical_trade_summary(result: PortfolioResult) -> pd.DataFrame:
    tactical = result.trades[result.trades["Sleeve"] == "TACTICAL"]
    buys = tactical[tactical["Action"] == "BUY"]
    sells = tactical[tactical["Action"] == "SELL"]
    realized = pd.to_numeric(
        sells.get("RealizedTacticalPnL", pd.Series(dtype=float)),
        errors="coerce",
    )
    return pd.DataFrame(
        [
            {
                "TacticalBuys": len(buys),
                "TacticalSells": len(sells),
                "ProfitableSells": int((realized > 0).sum()),
                "LossSells": int((realized < 0).sum()),
                "TotalRealizedTacticalPnL": float(realized.sum()),
                "OpenTacticalShares": float(
                    result.daily["TacticalShares"].iloc[-1]
                ),
            }
        ]
    )


def evaluate_top_development_candidates_on_holdout(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    base_params: FearBuyParams,
    settings: FearBuySettings,
    *,
    benchmark_holdout_cagr: float,
    limit: int = 30,
) -> pd.DataFrame:
    """Post-selection stability check; never use these columns to select."""

    features = build_fear_features(predictions, base_params)
    rows: list[dict[str, object]] = []
    for candidate in candidates.head(limit).itertuples(index=False):
        params = replace(
            base_params,
            core_weight=float(candidate.CoreWeight),
            mild_fear_score=float(candidate.MildFearScore),
            fear_fear_score=float(candidate.FearFearScore),
            trim_euphoria_score=float(candidate.TrimEuphoriaScore),
            minimum_hold_sessions=int(candidate.MinimumHoldSessions),
            trim_profit_buffer=float(candidate.TrimProfitBuffer),
        )
        signals = generate_fear_buy_signals(features, params)
        full = run_signal_backtest(
            signals,
            params,
            settings,
            name="HoldoutStabilityDiagnostic",
        )
        holdout = segment_result(
            full,
            settings,
            start=settings.holdout_start,
        )
        summary = holdout.summary
        rows.append(
            {
                "DevelopmentRank": int(candidate.DevelopmentRank),
                "CoreWeight": params.core_weight,
                "MildFearScore": params.mild_fear_score,
                "FearFearScore": params.fear_fear_score,
                "TrimEuphoriaScore": params.trim_euphoria_score,
                "MinimumHoldSessions": params.minimum_hold_sessions,
                "TrimProfitBuffer": params.trim_profit_buffer,
                "HoldoutCAGR(%)": summary.cagr_percent,
                "HoldoutMDD(%)": summary.max_drawdown_percent,
                "HoldoutSharpe": summary.sharpe_ratio,
                "HoldoutExposure(%)": summary.average_exposure_percent,
                "HoldoutTrades": summary.rebalance_count,
                "CAGRMinusBuyHold(pp)": (
                    summary.cagr_percent - benchmark_holdout_cagr
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("DevelopmentRank").reset_index(drop=True)
