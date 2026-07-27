from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_research.macro_sp500.portfolio import (
    PerformanceSummary,
    _performance_summary,
)

from .config import FearBuyParams, FearBuySettings
from .features import build_fear_features
from .strategy import generate_fear_buy_signals


@dataclass
class PortfolioResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: PerformanceSummary
    source: str = ""


def run_signal_backtest(
    signals: pd.DataFrame,
    params: FearBuyParams,
    settings: FearBuySettings,
    *,
    name: str,
    core_weight_override: float | None = None,
) -> PortfolioResult:
    """Execute close-generated target changes at the following session's open."""

    required = {"Date", "Open", "Close", "CashRate", "TargetWeight"}
    missing = required - set(signals)
    if missing:
        raise ValueError(f"Portfolio signals are missing: {sorted(missing)}")
    frame = signals.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    core_weight = (
        params.core_weight if core_weight_override is None else core_weight_override
    )
    if not 0 <= core_weight <= 1:
        raise ValueError("core_weight_override must be in [0, 1].")

    cash = float(settings.initial_capital)
    core_shares = 0.0
    tactical_shares = 0.0
    tactical_cost_basis = 0.0
    previous_date: pd.Timestamp | None = None
    executed_target = core_weight
    fee_rate = settings.transaction_cost_bps / 10_000.0
    slippage_rate = settings.slippage_bps / 10_000.0
    trade_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        date = pd.Timestamp(row["Date"])
        if previous_date is not None and cash > 0:
            elapsed_days = max(0, (date - previous_date).days)
            prior_rate = max(-0.99, float(frame.loc[index - 1, "CashRate"]) / 100)
            cash *= (1.0 + prior_rate) ** (elapsed_days / 365.25)

        if index == 0:
            signal_row = row
            signal_date = date
            requested_target = core_weight
            target_delta = core_weight
            reason = "INITIAL_CORE"
            state = "CORE_ONLY" if core_weight < 1 else "BUY_HOLD"
        else:
            signal_row = frame.loc[index - 1]
            signal_date = pd.Timestamp(signal_row["Date"])
            requested_target = float(signal_row["TargetWeight"])
            target_delta = requested_target - executed_target
            reason = str(signal_row.get("TransitionReason", ""))
            state = str(signal_row.get("SignalState", name))

        open_price = float(row["Open"])
        close_price = float(row["Close"])
        value_at_open = cash + (core_shares + tactical_shares) * open_price
        action = ""
        sleeve = ""
        execution_price = open_price
        notional = 0.0
        fee = 0.0
        quantity = 0.0
        realized_pnl = 0.0

        should_trade = index == 0 or abs(target_delta) >= params.rebalance_band
        if should_trade and target_delta > 0:
            action = "BUY"
            sleeve = "CORE" if index == 0 else "TACTICAL"
            execution_price = open_price * (1.0 + slippage_rate)
            requested_notional = target_delta * value_at_open
            notional = min(requested_notional, cash / (1.0 + fee_rate))
            quantity = notional / execution_price if execution_price else 0.0
            fee = notional * fee_rate
            cash -= notional + fee
            if sleeve == "CORE":
                core_shares += quantity
            else:
                tactical_shares += quantity
                tactical_cost_basis += notional + fee
        elif should_trade and target_delta < 0 and tactical_shares > 0:
            action = "SELL"
            sleeve = "TACTICAL"
            execution_price = open_price * (1.0 - slippage_rate)
            requested_sale = -target_delta * value_at_open
            quantity = min(requested_sale / open_price, tactical_shares)
            notional = quantity * execution_price
            fee = notional * fee_rate
            average_cost = (
                tactical_cost_basis / tactical_shares if tactical_shares else 0.0
            )
            removed_cost = quantity * average_cost
            tactical_shares -= quantity
            tactical_cost_basis = max(0.0, tactical_cost_basis - removed_cost)
            realized_pnl = notional - fee - removed_cost
            cash += notional - fee

        if should_trade:
            executed_target = requested_target
        total_shares = core_shares + tactical_shares
        value_at_close = cash + total_shares * close_price
        actual_weight = (
            total_shares * close_price / value_at_close if value_at_close else 0.0
        )
        tactical_average_cost = (
            tactical_cost_basis / tactical_shares if tactical_shares else float("nan")
        )
        tactical_unrealized_roi = (
            (close_price / tactical_average_cost - 1.0) * 100.0
            if tactical_shares and tactical_average_cost
            else float("nan")
        )
        roi = (value_at_close / settings.initial_capital - 1.0) * 100.0

        if action:
            trade_rows.append(
                {
                    "Date": date,
                    "SignalDate": signal_date,
                    "Action": action,
                    "Sleeve": sleeve,
                    "ExecutionPrice": execution_price,
                    "Quantity": quantity,
                    "Notional": notional,
                    "Fee": fee,
                    "RealizedTacticalPnL": realized_pnl,
                    "TargetWeight": requested_target,
                    "ActualWeightAfterClose": actual_weight,
                    "State": state,
                    "Reason": reason,
                    "VIX": row.get("VIX", float("nan")),
                    "VixPercentile": signal_row.get(
                        "VixPercentile",
                        float("nan"),
                    ),
                    "FearScore": signal_row.get("FearScore", float("nan")),
                    "EuphoriaScore": signal_row.get(
                        "EuphoriaScore",
                        float("nan"),
                    ),
                    "Drawdown252": signal_row.get(
                        "Drawdown252",
                        float("nan"),
                    ),
                    "ROI": roi,
                }
            )
        daily_rows.append(
            {
                "Date": date,
                "Open": open_price,
                "Close": close_price,
                "CashRate": float(row["CashRate"]),
                "VIX": row.get("VIX", float("nan")),
                "VixPercentile": row.get("VixPercentile", float("nan")),
                "FearScore": row.get("FearScore", float("nan")),
                "EuphoriaScore": row.get("EuphoriaScore", float("nan")),
                "Drawdown252": row.get("Drawdown252", float("nan")),
                "SignalTargetWeight": float(row["TargetWeight"]),
                "ExecutedTargetWeight": executed_target,
                "ActualWeight": actual_weight,
                "State": state,
                "Reason": reason,
                "Cash": cash,
                "CoreShares": core_shares,
                "TacticalShares": tactical_shares,
                "TacticalAverageCost": tactical_average_cost,
                "TacticalUnrealizedROI": tactical_unrealized_roi,
                "TotalValue": value_at_close,
                "TotalInjected": settings.initial_capital,
                "ROI": roi,
            }
        )
        previous_date = date

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    return PortfolioResult(
        daily=daily,
        trades=trades,
        summary=_performance_summary(
            daily,
            trades,
            initial_capital=settings.initial_capital,
        ),
        source=f"{name}: current production fear-buy signal function",
    )


def run_fear_buy_backtest(
    predictions: pd.DataFrame,
    params: FearBuyParams,
    settings: FearBuySettings,
) -> PortfolioResult:
    features = build_fear_features(predictions, params)
    signals = generate_fear_buy_signals(features, params)
    return run_signal_backtest(
        signals,
        params,
        settings,
        name="MacroFearBuy",
    )


def run_constant_weight_benchmark(
    predictions: pd.DataFrame,
    params: FearBuyParams,
    settings: FearBuySettings,
    *,
    weight: float,
    name: str,
) -> PortfolioResult:
    signals = predictions[["Date", "Open", "Close", "CashRate"]].copy()
    for column in ("VIX", "Drawdown252"):
        if column in predictions:
            signals[column] = predictions[column]
    signals["TargetWeight"] = weight
    signals["SignalState"] = name
    signals["TransitionReason"] = ""
    return run_signal_backtest(
        signals,
        params,
        settings,
        name=name,
        core_weight_override=weight,
    )


def segment_result(
    result: PortfolioResult,
    settings: FearBuySettings,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    rebase: bool = True,
) -> PortfolioResult:
    daily = result.daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    if start is not None:
        daily = daily[daily["Date"] >= pd.Timestamp(start)]
    if end is not None:
        daily = daily[daily["Date"] <= pd.Timestamp(end)]
    daily = daily.reset_index(drop=True)
    if daily.empty:
        raise ValueError("Requested evaluation segment is empty.")
    scale = (
        settings.initial_capital / float(daily["TotalValue"].iloc[0])
        if rebase
        else 1.0
    )
    for column in (
        "Cash",
        "CoreShares",
        "TacticalShares",
        "TotalValue",
    ):
        if column in daily:
            daily[column] = daily[column] * scale
    daily["TotalInjected"] = settings.initial_capital
    daily["ROI"] = (
        daily["TotalValue"] / settings.initial_capital - 1.0
    ) * 100.0

    trades = result.trades.copy()
    if not trades.empty:
        trades["Date"] = pd.to_datetime(trades["Date"])
        if start is not None:
            trades = trades[trades["Date"] >= pd.Timestamp(start)]
        if end is not None:
            trades = trades[trades["Date"] <= pd.Timestamp(end)]
        trades = trades.reset_index(drop=True)
        for column in ("Notional", "Fee", "RealizedTacticalPnL"):
            if column in trades:
                trades[column] = trades[column] * scale
    return PortfolioResult(
        daily=daily,
        trades=trades,
        summary=_performance_summary(
            daily,
            trades,
            initial_capital=settings.initial_capital,
        ),
        source=result.source,
    )


def comparison_table(
    portfolios: dict[str, PortfolioResult],
    settings: FearBuySettings,
) -> pd.DataFrame:
    periods = {
        "Full OOS": (None, None, False),
        "Development": (None, settings.development_end, False),
        "Untouched holdout": (settings.holdout_start, None, True),
    }
    rows: list[dict[str, object]] = []
    for period, (start, end, rebase) in periods.items():
        for name, result in portfolios.items():
            segment = segment_result(
                result,
                settings,
                start=start,
                end=end,
                rebase=rebase,
            )
            summary = segment.summary
            rows.append(
                {
                    "Period": period,
                    "Portfolio": name,
                    "FinalValue": summary.final_value,
                    "ROI(%)": summary.roi_percent,
                    "CAGR(%)": summary.cagr_percent,
                    "MDD(%)": summary.max_drawdown_percent,
                    "Sharpe": summary.sharpe_ratio,
                    "Sortino": summary.sortino_ratio,
                    "Calmar": summary.calmar_ratio,
                    "AverageExposure(%)": summary.average_exposure_percent,
                    "Turnover": summary.turnover_multiple,
                    "Trades": summary.rebalance_count,
                }
            )
    return pd.DataFrame(rows)
