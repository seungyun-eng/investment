from __future__ import annotations

import pandas as pd

from .config_v2 import MacroSp500V2Params, MacroSp500V2Settings
from .portfolio import PortfolioResult, _performance_summary
from .strategy_v2 import CrisisMemory, generate_v2_target_weights


def run_v2_portfolio_from_signals(
    features: pd.DataFrame,
    signals: pd.DataFrame,
    settings: MacroSp500V2Settings,
) -> PortfolioResult:
    frame = features.sort_values("Date").reset_index(drop=True)
    signals = signals.sort_values("Date").reset_index(drop=True)
    if frame.empty or len(frame) != len(signals):
        raise ValueError("V2 features and signals must have equal nonzero length.")
    if not pd.to_datetime(frame["Date"]).equals(pd.to_datetime(signals["Date"])):
        raise ValueError("V2 feature and signal dates do not align.")

    dates = pd.to_datetime(frame["Date"]).to_numpy()
    opens = frame["Open"].to_numpy(dtype=float)
    closes = frame["Close"].to_numpy(dtype=float)
    vix = frame["VIX"].to_numpy(dtype=float)
    percentiles = frame["VixPercentile"].to_numpy(dtype=float)
    drawdowns = frame["Drawdown"].to_numpy(dtype=float)
    cash_rates = frame.get(
        "CashRate",
        pd.Series(settings.fallback_cash_annual_rate, index=frame.index),
    ).fillna(settings.fallback_cash_annual_rate).to_numpy(dtype=float)
    targets = signals["TargetWeight"].to_numpy(dtype=float)
    cores = signals["CoreWeight"].to_numpy(dtype=float)
    bands = signals["RebalanceBand"].to_numpy(dtype=float)
    states = signals["State"].to_numpy(dtype=str)
    reasons = signals["Reason"].to_numpy(dtype=str)

    cash = float(settings.initial_capital)
    shares = 0.0
    last_executed_target: float | None = None
    previous_date: pd.Timestamp | None = None
    fee_rate = settings.transaction_cost_bps / 10_000.0
    slippage_rate = settings.slippage_bps / 10_000.0
    trades: list[dict[str, object]] = []
    cash_history: list[float] = []
    share_history: list[float] = []
    value_history: list[float] = []
    weight_history: list[float] = []
    roi_history: list[float] = []

    for index in range(len(frame)):
        date = pd.Timestamp(dates[index])
        if previous_date is not None and cash > 0:
            elapsed_days = max(0, (date - previous_date).days)
            annual_rate = max(-0.99, cash_rates[index - 1] / 100.0)
            cash *= (1.0 + annual_rate) ** (elapsed_days / 365.25)

        if index == 0:
            target = cores[index]
            band = bands[index]
            signal_date = date
            reason = "INITIAL_CORE"
            state = "NORMAL"
        else:
            target = targets[index - 1]
            band = bands[index - 1]
            signal_date = pd.Timestamp(dates[index - 1])
            reason = reasons[index - 1]
            state = states[index - 1]

        open_price = opens[index]
        value_open = cash + shares * open_price
        current_weight = shares * open_price / value_open if value_open else 0.0
        requested = target * value_open - shares * open_price
        requested_fraction = abs(requested) / value_open if value_open else 0.0
        target_changed = (
            last_executed_target is None
            or abs(target - last_executed_target) > 1e-10
        )
        should_rebalance = target_changed or abs(current_weight - target) >= band
        should_trade = (
            should_rebalance
            and (
                requested_fraction >= settings.minimum_trade_fraction
                or last_executed_target is None
            )
        )

        action = ""
        execution_price = open_price
        notional = 0.0
        fee = 0.0
        if should_trade and requested > 0:
            action = "BUY"
            execution_price = open_price * (1.0 + slippage_rate)
            notional = min(requested, cash / (1.0 + fee_rate))
            quantity = notional / execution_price
            fee = notional * fee_rate
            shares += quantity
            cash -= notional + fee
        elif should_trade and requested < 0:
            action = "SELL"
            execution_price = open_price * (1.0 - slippage_rate)
            quantity = min(-requested / open_price, shares)
            notional = quantity * execution_price
            fee = notional * fee_rate
            shares -= quantity
            cash += notional - fee

        if action:
            last_executed_target = target
            trades.append(
                {
                    "Date": date,
                    "SignalDate": signal_date,
                    "Action": action,
                    "ExecutionPrice": execution_price,
                    "Notional": notional,
                    "Fee": fee,
                    "TargetWeight": target,
                    "State": state,
                    "Reason": reason,
                    "VIX": vix[index],
                    "VixPercentile": percentiles[index],
                    "Drawdown": drawdowns[index],
                }
            )

        value_close = cash + shares * closes[index]
        actual_weight = shares * closes[index] / value_close if value_close else 0.0
        cash_history.append(cash)
        share_history.append(shares)
        value_history.append(value_close)
        weight_history.append(actual_weight)
        roi_history.append(
            (value_close / settings.initial_capital - 1.0) * 100.0
        )
        previous_date = date

    daily = pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "Close": closes,
            "VIX": vix,
            "VixPercentile": percentiles,
            "Drawdown": drawdowns,
            "CashRate": cash_rates,
            "State": states,
            "NextTargetWeight": targets,
            "ActualWeight": weight_history,
            "Cash": cash_history,
            "Shares": share_history,
            "TotalValue": value_history,
            "TotalInjected": settings.initial_capital,
            "ROI": roi_history,
        }
    )
    trade_frame = pd.DataFrame(trades)
    return PortfolioResult(
        daily=daily,
        trades=trade_frame,
        summary=_performance_summary(
            daily,
            trade_frame,
            initial_capital=settings.initial_capital,
        ),
    )


def run_v2_backtest(
    features: pd.DataFrame,
    params: MacroSp500V2Params,
    settings: MacroSp500V2Settings,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> PortfolioResult:
    frame = features.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    if start is not None:
        frame = frame[frame["Date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["Date"] <= pd.Timestamp(end)]
    frame = frame.reset_index(drop=True)
    signals, _ = generate_v2_target_weights(frame, params, settings)
    return run_v2_portfolio_from_signals(frame, signals, settings)


def run_static_weight_v2(
    features: pd.DataFrame,
    settings: MacroSp500V2Settings,
    *,
    weight: float,
) -> PortfolioResult:
    signals = pd.DataFrame(
        {
            "Date": features["Date"],
            "State": "STATIC",
            "TargetWeight": weight,
            "CoreWeight": weight,
            "RebalanceBand": 0.05,
            "Reason": "STATIC_WEIGHT",
        }
    )
    return run_v2_portfolio_from_signals(features, signals, settings)


def run_v2_with_memory(
    features: pd.DataFrame,
    params: MacroSp500V2Params,
    settings: MacroSp500V2Settings,
    memory: CrisisMemory | None,
) -> tuple[pd.DataFrame, CrisisMemory]:
    return generate_v2_target_weights(
        features,
        params,
        settings,
        initial_memory=memory,
    )
