from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from stock_research.macro_sp500.portfolio import PerformanceSummary, _performance_summary

from .config import ResearchConfig


@dataclass
class AllocationResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: PerformanceSummary
    source: str = ""


def prediction_target_weights(
    predictions: pd.DataFrame,
    config: ResearchConfig,
    *,
    risk_threshold: float | None = None,
    caution_threshold: float | None = None,
    defensive_weight: float | None = None,
    caution_weight: float | None = None,
) -> pd.DataFrame:
    frame = predictions.copy()
    risk_columns = [
        f"RiskProbability_{horizon}"
        for horizon in config.risk_horizons
        if f"RiskProbability_{horizon}" in frame
    ]
    return_columns = [
        f"PredictedExcessReturn_{horizon}"
        for horizon in config.return_horizons
        if horizon in {63, config.primary_return_horizon}
        and f"PredictedExcessReturn_{horizon}" in frame
    ]
    if not risk_columns or not return_columns:
        raise ValueError("Predictions require risk and expected-return columns.")
    frame["RiskScore"] = frame[risk_columns].mean(axis=1)
    frame["ExpectedExcessReturn"] = frame[return_columns].mean(axis=1)
    high_risk = frame["RiskScore"] >= (
        config.risk_probability_threshold if risk_threshold is None else risk_threshold
    )
    caution_risk = frame["RiskScore"] >= (
        config.caution_probability_threshold
        if caution_threshold is None
        else caution_threshold
    )
    negative_return = frame["ExpectedExcessReturn"] < config.negative_return_threshold
    defensive = config.defensive_weight if defensive_weight is None else defensive_weight
    caution = config.caution_weight if caution_weight is None else caution_weight
    frame["TargetWeight"] = 1.0
    frame.loc[caution_risk & negative_return, "TargetWeight"] = caution
    frame.loc[high_risk & negative_return, "TargetWeight"] = defensive
    frame["SignalState"] = "NORMAL"
    frame.loc[caution_risk & negative_return, "SignalState"] = "CAUTION"
    frame.loc[high_risk & negative_return, "SignalState"] = "DEFENSIVE"
    return frame


def _entry_reason(
    joint: bool,
    macro_only: bool,
    early_warning: bool,
    *,
    joint_label: str,
    macro_only_label: str,
    early_warning_label: str,
) -> str:
    """Which path actually decided the transition, most-specific first, so a
    trade log can show whether early-warning ever cast the deciding vote."""

    if early_warning and not joint and not macro_only:
        return early_warning_label
    if macro_only and not joint:
        return macro_only_label
    return joint_label


def stateful_macro_target_weights(
    predictions: pd.DataFrame,
    config: ResearchConfig,
    *,
    early_warning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create weekly, hysteretic macro allocation states from OOS predictions.

    `early_warning`, if given, is a `{Date, EarlyWarningCaution,
    EarlyWarningDefensive}` frame of booleans (e.g. from a fast breadth
    +persistence rule) that can ALSO trigger CAUTION/DEFENSIVE -- a third,
    independent path alongside the existing joint (risk AND macro) and
    macro-only paths, not a replacement for either. Omitting it (the
    default) reproduces the original behaviour exactly, so existing callers
    are unaffected.
    """

    required = {
        "Date",
        "Open",
        "Close",
        "CashRate",
        "MacroConfirmationScore",
    }
    risk_columns = {
        horizon: f"RiskProbability_{horizon}"
        for horizon in config.state_risk_horizons
    }
    required.update(risk_columns.values())
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Stateful macro predictions are missing: {sorted(missing)}")

    frame = predictions.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    if early_warning is not None:
        gate = early_warning.copy()
        gate["Date"] = pd.to_datetime(gate["Date"])
        frame = frame.merge(
            gate[["Date", "EarlyWarningCaution", "EarlyWarningDefensive"]],
            on="Date",
            how="left",
        )
        frame["EarlyWarningCaution"] = frame["EarlyWarningCaution"].fillna(False).astype(bool)
        frame["EarlyWarningDefensive"] = frame["EarlyWarningDefensive"].fillna(False).astype(bool)
    else:
        frame["EarlyWarningCaution"] = False
        frame["EarlyWarningDefensive"] = False
    total_weight = float(sum(config.state_risk_weights))
    raw_risk = pd.Series(0.0, index=frame.index)
    for horizon, weight in zip(
        config.state_risk_horizons,
        config.state_risk_weights,
        strict=True,
    ):
        raw_risk += pd.to_numeric(frame[risk_columns[horizon]], errors="coerce") * weight
    frame["RawRiskScore"] = raw_risk / total_weight
    frame["RiskScore"] = frame["RawRiskScore"].ewm(
        span=config.state_risk_smoothing_days,
        adjust=False,
        min_periods=1,
    ).mean()
    frame["RawMacroScore"] = pd.to_numeric(
        frame["MacroConfirmationScore"],
        errors="coerce",
    ).fillna(0.50)
    frame["MacroScore"] = frame["RawMacroScore"].ewm(
        span=config.state_macro_smoothing_days,
        adjust=False,
        min_periods=1,
    ).mean()
    frame["MacroRising"] = frame["RawMacroScore"] >= frame["MacroScore"]
    frame["PriceMomentum"] = frame["Close"].pct_change(config.state_momentum_days)
    frame["TrendSMA"] = frame["Close"].rolling(
        config.state_trend_sma_days,
        min_periods=config.state_trend_sma_days,
    ).mean()
    frame["MomentumRecovery"] = (
        (frame["RiskScore"] <= config.state_recovery_risk)
        & (frame["MacroScore"] <= config.state_momentum_recovery_macro)
        & ~frame["MacroRising"]
        & (frame["PriceMomentum"] > 0)
        & (frame["Close"] > frame["TrendSMA"])
    )
    return_columns = [
        name
        for name in ("PredictedExcessReturn_63", "PredictedExcessReturn_126")
        if name in frame
    ]
    frame["ExpectedExcessReturn"] = (
        frame[return_columns].mean(axis=1) if return_columns else float("nan")
    )
    week = frame["Date"].dt.to_period("W-FRI")
    frame["DecisionDay"] = week.ne(week.shift(-1))

    target_by_state = {
        "NORMAL": 1.0,
        "CAUTION": config.caution_weight,
        "DEFENSIVE": config.defensive_weight,
        "RECOVERY": config.caution_weight,
    }
    state = "NORMAL"
    state_age = 0
    normal_reference_close = float(frame.loc[0, "Close"])
    caution_count = 0
    defensive_count = 0
    exit_count = 0
    recovery_count = 0
    states: list[str] = []
    targets: list[float] = []
    ages: list[int] = []
    reasons: list[str] = []
    loss_blocks: list[bool] = []
    reference_prices: list[float] = []

    for _, row in frame.iterrows():
        risk = float(row["RiskScore"])
        macro = float(row["MacroScore"])
        raw_macro = float(row["RawMacroScore"])
        macro_rising = bool(row["MacroRising"])
        momentum_recovery = bool(row["MomentumRecovery"])
        close = float(row["Close"])
        decision_day = bool(row["DecisionDay"])
        early_warning_caution = bool(row["EarlyWarningCaution"])
        early_warning_defensive = bool(row["EarlyWarningDefensive"])
        transition_reason = ""
        loss_gate_blocked = False
        joint_emergency = (
            risk >= config.state_emergency_risk
            and macro >= config.state_emergency_macro
        )
        macro_only_emergency = raw_macro >= config.state_macro_only_emergency
        emergency = joint_emergency or macro_only_emergency
        new_state = state

        if emergency and state != "DEFENSIVE":
            new_state = "DEFENSIVE"
            transition_reason = (
                "EMERGENCY_MACRO_SHOCK"
                if macro_only_emergency and not joint_emergency
                else "EMERGENCY_RISK_AND_MACRO"
            )
        elif state == "NORMAL" and decision_day:
            joint_caution = (
                risk >= config.state_caution_entry_risk
                and macro >= config.state_caution_macro
            )
            macro_only_caution = (
                macro >= config.state_macro_only_caution and macro_rising
            )
            caution_condition = joint_caution or macro_only_caution or early_warning_caution
            joint_defensive = (
                risk >= config.state_defensive_entry_risk
                and macro >= config.state_defensive_macro
            )
            macro_only_defensive = (
                macro >= config.state_macro_only_defensive and macro_rising
            )
            defensive_condition = joint_defensive or macro_only_defensive or early_warning_defensive
            selling_below_reference = close < (
                normal_reference_close * (1 - config.state_loss_tolerance)
            )
            strong_loss_exit = (
                risk >= config.state_loss_exit_risk
                or macro >= config.state_loss_exit_macro
            )
            cooldown_complete = state_age >= config.state_normal_cooldown_days
            if (
                caution_condition
                and cooldown_complete
                and selling_below_reference
                and not strong_loss_exit
            ):
                loss_gate_blocked = True
                caution_count = 0
                defensive_count = 0
            else:
                caution_count = caution_count + 1 if (
                    caution_condition and cooldown_complete
                ) else 0
                defensive_count = defensive_count + 1 if (
                    defensive_condition and cooldown_complete
                ) else 0
            if defensive_count >= config.state_entry_confirmations:
                new_state = "DEFENSIVE"
                transition_reason = _entry_reason(
                    joint_defensive, macro_only_defensive, early_warning_defensive,
                    joint_label="CONFIRMED_DEFENSIVE_MACRO_RISK",
                    macro_only_label="CONFIRMED_MACRO_ONLY_DEFENSIVE",
                    early_warning_label="CONFIRMED_EARLY_WARNING_DEFENSIVE",
                )
            elif caution_count >= config.state_entry_confirmations:
                new_state = "CAUTION"
                transition_reason = _entry_reason(
                    joint_caution, macro_only_caution, early_warning_caution,
                    joint_label="CONFIRMED_CAUTION_MACRO_RISK",
                    macro_only_label="CONFIRMED_MACRO_ONLY_CAUTION",
                    early_warning_label="CONFIRMED_EARLY_WARNING_CAUTION",
                )

        elif state in {"CAUTION", "DEFENSIVE"} and decision_day:
            joint_defensive = (
                risk >= config.state_defensive_entry_risk
                and macro >= config.state_defensive_macro
            )
            macro_only_defensive = (
                macro >= config.state_macro_only_defensive and macro_rising
            )
            defensive_condition = joint_defensive or macro_only_defensive or early_warning_defensive
            safe_condition = (
                (
                    risk <= config.state_exit_risk
                    and macro <= config.state_exit_macro
                )
                or momentum_recovery
            ) and state_age >= config.state_min_hold_days
            defensive_count = defensive_count + 1 if (
                state == "CAUTION" and defensive_condition
            ) else 0
            exit_count = exit_count + 1 if safe_condition else 0
            if (
                state == "CAUTION"
                and defensive_count >= config.state_entry_confirmations
            ):
                new_state = "DEFENSIVE"
                transition_reason = _entry_reason(
                    joint_defensive, macro_only_defensive, early_warning_defensive,
                    joint_label="CAUTION_ESCALATED_TO_DEFENSIVE",
                    macro_only_label="CAUTION_ESCALATED_BY_MACRO_ONLY",
                    early_warning_label="CAUTION_ESCALATED_BY_EARLY_WARNING",
                )
            elif exit_count >= config.state_exit_confirmations:
                new_state = "RECOVERY"
                transition_reason = "CONFIRMED_MACRO_RISK_RELIEF"

        elif state == "RECOVERY" and decision_day:
            joint_relapse_caution = (
                risk >= config.state_caution_entry_risk
                and macro >= config.state_caution_macro
            )
            macro_only_caution = (
                macro >= config.state_macro_only_caution and macro_rising
            )
            relapse_caution = joint_relapse_caution or macro_only_caution or early_warning_caution
            joint_relapse_defensive = (
                risk >= config.state_defensive_entry_risk
                and macro >= config.state_defensive_macro
            )
            macro_only_defensive = (
                macro >= config.state_macro_only_defensive and macro_rising
            )
            relapse_defensive = joint_relapse_defensive or macro_only_defensive or early_warning_defensive
            recovery_condition = (
                (
                    risk <= config.state_recovery_risk
                    and macro <= config.state_recovery_macro
                )
                or momentum_recovery
            ) and state_age >= config.state_recovery_days
            caution_count = caution_count + 1 if relapse_caution else 0
            defensive_count = defensive_count + 1 if relapse_defensive else 0
            recovery_count = recovery_count + 1 if recovery_condition else 0
            if defensive_count >= config.state_entry_confirmations:
                new_state = "DEFENSIVE"
                transition_reason = _entry_reason(
                    joint_relapse_defensive, macro_only_defensive, early_warning_defensive,
                    joint_label="RECOVERY_FAILED_DEFENSIVE",
                    macro_only_label="RECOVERY_FAILED_MACRO_ONLY_DEFENSIVE",
                    early_warning_label="RECOVERY_FAILED_EARLY_WARNING_DEFENSIVE",
                )
            elif caution_count >= config.state_entry_confirmations:
                new_state = "CAUTION"
                transition_reason = _entry_reason(
                    joint_relapse_caution, macro_only_caution, early_warning_caution,
                    joint_label="RECOVERY_FAILED_CAUTION",
                    macro_only_label="RECOVERY_FAILED_MACRO_ONLY_CAUTION",
                    early_warning_label="RECOVERY_FAILED_EARLY_WARNING_CAUTION",
                )
            elif recovery_count >= config.state_exit_confirmations:
                new_state = "NORMAL"
                transition_reason = "CONFIRMED_RETURN_TO_NORMAL"

        if new_state != state:
            state = new_state
            state_age = 0
            caution_count = 0
            defensive_count = 0
            exit_count = 0
            recovery_count = 0
            if state == "NORMAL":
                normal_reference_close = close

        states.append(state)
        targets.append(target_by_state[state])
        ages.append(state_age)
        reasons.append(transition_reason)
        loss_blocks.append(loss_gate_blocked)
        reference_prices.append(normal_reference_close)
        state_age += 1

    frame["SignalState"] = states
    frame["TargetWeight"] = targets
    frame["StateAge"] = ages
    frame["TransitionReason"] = reasons
    frame["LossGateBlocked"] = loss_blocks
    frame["NormalReferenceClose"] = reference_prices
    return frame


def run_weight_backtest(
    signals: pd.DataFrame,
    config: ResearchConfig,
    *,
    name: str,
    transaction_cost_bps: float | None = None,
    slippage_bps: float | None = None,
    rebalance_band: float | None = None,
) -> AllocationResult:
    required = {"Date", "Open", "Close", "CashRate", "TargetWeight"}
    missing = required - set(signals)
    if missing:
        raise ValueError(f"Portfolio signals are missing columns: {sorted(missing)}")
    frame = signals.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    cost = config.transaction_cost_bps if transaction_cost_bps is None else transaction_cost_bps
    slippage = config.slippage_bps if slippage_bps is None else slippage_bps
    fee_rate = cost / 10_000
    slippage_rate = slippage / 10_000
    band = config.rebalance_band if rebalance_band is None else rebalance_band
    cash = float(config.initial_capital)
    shares = 0.0
    previous_date: pd.Timestamp | None = None
    trade_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        date = pd.Timestamp(row["Date"])
        if previous_date is not None and cash > 0:
            elapsed_days = max(0, (date - previous_date).days)
            annual_rate = max(-0.99, float(frame.loc[index - 1, "CashRate"]) / 100)
            cash *= (1 + annual_rate) ** (elapsed_days / 365.25)

        signal_index = max(0, index - 1)
        signal_row = frame.loc[signal_index]
        target = 1.0 if index == 0 else float(signal_row["TargetWeight"])
        signal_date = pd.Timestamp(signal_row["Date"])
        state = str(signal_row["SignalState"]) if "SignalState" in frame else name
        risk_score = signal_row.get("RiskScore", float("nan"))
        macro_score = signal_row.get(
            "MacroScore",
            signal_row.get("MacroConfirmationScore", float("nan")),
        )
        expected_return = signal_row.get("ExpectedExcessReturn", float("nan"))
        transition_reason = str(signal_row.get("TransitionReason", ""))
        state_age = signal_row.get("StateAge", float("nan"))
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        value_open = cash + shares * open_price
        current_weight = shares * open_price / value_open if value_open else 0.0
        gap = target - current_weight
        should_trade = index == 0 or abs(gap) >= band
        action = ""
        notional = 0.0
        fee = 0.0
        execution_price = open_price

        if should_trade and gap > 0:
            action = "BUY"
            execution_price = open_price * (1 + slippage_rate)
            desired = gap * value_open
            notional = min(desired, cash / (1 + fee_rate))
            quantity = notional / execution_price
            fee = notional * fee_rate
            shares += quantity
            cash -= notional + fee
        elif should_trade and gap < 0:
            action = "SELL"
            execution_price = open_price * (1 - slippage_rate)
            desired = -gap * value_open
            quantity = min(desired / open_price, shares)
            notional = quantity * execution_price
            fee = notional * fee_rate
            shares -= quantity
            cash += notional - fee

        value_close = cash + shares * close_price
        actual_weight = shares * close_price / value_close if value_close else 0.0
        if action:
            trade_rows.append(
                {
                    "Date": date,
                    "SignalDate": signal_date,
                    "Action": action,
                    "ExecutionPrice": execution_price,
                    "Notional": notional,
                    "Fee": fee,
                    "TargetWeight": target,
                    "State": state,
                    "RiskScore": risk_score,
                    "MacroScore": macro_score,
                    "ExpectedExcessReturn": expected_return,
                    "TransitionReason": transition_reason,
                    "StateAge": state_age,
                }
            )
        daily_rows.append(
            {
                "Date": date,
                "Open": open_price,
                "Close": close_price,
                "CashRate": float(row["CashRate"]),
                "TargetWeight": target,
                "ActualWeight": actual_weight,
                "Cash": cash,
                "Shares": shares,
                "TotalValue": value_close,
                "TotalInjected": config.initial_capital,
                "ROI": (value_close / config.initial_capital - 1) * 100,
                "State": state,
                "RiskScore": risk_score,
                "MacroScore": macro_score,
                "ExpectedExcessReturn": expected_return,
                "TransitionReason": transition_reason,
                "StateAge": state_age,
            }
        )
        previous_date = date

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    return AllocationResult(
        daily=daily,
        trades=trades,
        summary=_performance_summary(
            daily,
            trades,
            initial_capital=config.initial_capital,
        ),
        source="generated from current prediction signals",
    )


def constant_weight_signals(predictions: pd.DataFrame, weight: float) -> pd.DataFrame:
    output = predictions[["Date", "Open", "Close", "CashRate"]].copy()
    output["TargetWeight"] = weight
    output["SignalState"] = f"STATIC_{weight:.2f}"
    return output


def run_portfolio_comparison(
    predictions: pd.DataFrame,
    config: ResearchConfig,
    *,
    v2_folder: str | Path | None = None,
) -> dict[str, AllocationResult]:
    model_signals = prediction_target_weights(predictions, config)
    stateful_signals = stateful_macro_target_weights(predictions, config)
    results = {
        "StatefulMacro": run_weight_backtest(
            stateful_signals,
            config,
            name="StatefulMacro",
            rebalance_band=config.state_rebalance_band,
        ),
        "MacroMomentum": run_weight_backtest(
            model_signals,
            config,
            name="MacroMomentum",
        ),
        "BuyHold": run_weight_backtest(
            constant_weight_signals(predictions, 1.0),
            config,
            name="BuyHold",
        ),
        "Static70": run_weight_backtest(
            constant_weight_signals(predictions, 0.70),
            config,
            name="Static70",
        ),
        "Static76": run_weight_backtest(
            constant_weight_signals(predictions, 0.76),
            config,
            name="Static76",
        ),
    }
    if v2_folder is not None:
        v2 = load_prior_v2_oos_benchmark(
            v2_folder,
            start=predictions["Date"].min(),
            end=predictions["Date"].max(),
            config=config,
        )
        if v2 is not None:
            results["PriorV2"] = v2
    return results


def load_prior_v2_oos_benchmark(
    folder: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: ResearchConfig,
) -> AllocationResult | None:
    """Load and rebase the prior V2 strict-OOS result over the comparison period."""

    folder = Path(folder)
    daily_hits = sorted(
        folder.glob("*_oos_daily_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not daily_hits:
        return None
    daily_path = daily_hits[0]
    daily = pd.read_csv(daily_path, parse_dates=["Date"])
    required = {"Date", "TotalValue", "ActualWeight"}
    if not required <= set(daily):
        return None
    daily = daily[
        (daily["Date"] >= pd.Timestamp(start)) & (daily["Date"] <= pd.Timestamp(end))
    ].copy()
    if daily.empty:
        return None
    scale = config.initial_capital / float(daily["TotalValue"].iloc[0])
    for name in ("TotalValue", "Cash"):
        if name in daily:
            daily[name] = daily[name] * scale
    daily["TotalInjected"] = config.initial_capital
    daily["ROI"] = (daily["TotalValue"] / config.initial_capital - 1) * 100

    trade_hits = sorted(
        folder.glob("*_oos_rebalances_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    trades = pd.DataFrame()
    if trade_hits:
        trades = pd.read_csv(trade_hits[0], parse_dates=["Date"])
        trades = trades[
            (trades["Date"] >= pd.Timestamp(start))
            & (trades["Date"] <= pd.Timestamp(end))
        ].copy()
        for name in ("Notional", "Fee"):
            if name in trades:
                trades[name] = trades[name] * scale
    return AllocationResult(
        daily=daily,
        trades=trades,
        summary=_performance_summary(
            daily,
            trades,
            initial_capital=config.initial_capital,
        ),
        source=f"rebased prior V2 strict-OOS file: {daily_path}",
    )


def performance_table(results: dict[str, AllocationResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        summary = result.summary
        rows.append(
            {
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


def trade_cycle_diagnostics(result: AllocationResult) -> pd.DataFrame:
    """Pair exposure reductions and restorations using execution-price diagnostics."""

    if result.trades.empty:
        return pd.DataFrame()
    trades = result.trades.sort_values("Date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    normal_entry: pd.Series | None = None
    defensive_entry: pd.Series | None = None
    for _, trade in trades.iterrows():
        target = float(trade["TargetWeight"])
        if trade["Action"] == "BUY" and target >= 0.999:
            if defensive_entry is not None:
                sell_price = float(defensive_entry["ExecutionPrice"])
                buy_price = float(trade["ExecutionPrice"])
                rows.append(
                    {
                        "CycleType": "DEFENSIVE_TO_NORMAL",
                        "StartDate": defensive_entry["Date"],
                        "EndDate": trade["Date"],
                        "CalendarDays": (
                            pd.Timestamp(trade["Date"])
                            - pd.Timestamp(defensive_entry["Date"])
                        ).days,
                        "StartPrice": sell_price,
                        "EndPrice": buy_price,
                        "PriceChange(%)": (buy_price / sell_price - 1) * 100,
                        "Adverse": buy_price > sell_price,
                    }
                )
                defensive_entry = None
            normal_entry = trade
        elif trade["Action"] == "SELL" and target < 0.999:
            if normal_entry is not None:
                buy_price = float(normal_entry["ExecutionPrice"])
                sell_price = float(trade["ExecutionPrice"])
                rows.append(
                    {
                        "CycleType": "NORMAL_TO_DEFENSIVE",
                        "StartDate": normal_entry["Date"],
                        "EndDate": trade["Date"],
                        "CalendarDays": (
                            pd.Timestamp(trade["Date"])
                            - pd.Timestamp(normal_entry["Date"])
                        ).days,
                        "StartPrice": buy_price,
                        "EndPrice": sell_price,
                        "PriceChange(%)": (sell_price / buy_price - 1) * 100,
                        "Adverse": sell_price < buy_price,
                    }
                )
                normal_entry = None
            if defensive_entry is None:
                defensive_entry = trade
    return pd.DataFrame(rows)


def allocation_sensitivity(
    predictions: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for risk_threshold in (0.50, 0.60, 0.70):
        for defensive_weight in (0.0, 0.25, 0.50):
            for cost_bps in (0.0, config.transaction_cost_bps, 20.0):
                signals = prediction_target_weights(
                    predictions,
                    config,
                    risk_threshold=risk_threshold,
                    caution_threshold=min(risk_threshold, config.caution_probability_threshold),
                    defensive_weight=defensive_weight,
                )
                result = run_weight_backtest(
                    signals,
                    config,
                    name="Sensitivity",
                    transaction_cost_bps=cost_bps,
                    slippage_bps=cost_bps,
                )
                rows.append(
                    {
                        "Strategy": "StatelessMacro",
                        "RiskThreshold": risk_threshold,
                        "DefensiveWeight": defensive_weight,
                        "CostAndSlippageEachBps": cost_bps,
                        "ROI(%)": result.summary.roi_percent,
                        "CAGR(%)": result.summary.cagr_percent,
                        "MDD(%)": result.summary.max_drawdown_percent,
                        "Sharpe": result.summary.sharpe_ratio,
                        "AverageExposure(%)": result.summary.average_exposure_percent,
                        "Trades": result.summary.rebalance_count,
                    }
                )
    return pd.DataFrame(rows)


def stateful_allocation_sensitivity(
    predictions: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Evaluate nearby state rules by calling the production signal function."""

    rows: list[dict[str, object]] = []
    for caution_risk in (0.50, config.state_caution_entry_risk, 0.60):
        for confirmations in (1, config.state_entry_confirmations, 3):
            for minimum_hold in (10, config.state_min_hold_days, 40):
                candidate = replace(
                    config,
                    state_caution_entry_risk=caution_risk,
                    state_entry_confirmations=confirmations,
                    state_min_hold_days=minimum_hold,
                )
                signals = stateful_macro_target_weights(predictions, candidate)
                result = run_weight_backtest(
                    signals,
                    candidate,
                    name="StatefulSensitivity",
                    rebalance_band=candidate.state_rebalance_band,
                )
                rows.append(
                    {
                        "Strategy": "StatefulMacro",
                        "CautionRisk": caution_risk,
                        "EntryConfirmations": confirmations,
                        "MinimumHoldDays": minimum_hold,
                        "CostAndSlippageEachBps": candidate.transaction_cost_bps,
                        "ROI(%)": result.summary.roi_percent,
                        "CAGR(%)": result.summary.cagr_percent,
                        "MDD(%)": result.summary.max_drawdown_percent,
                        "Sharpe": result.summary.sharpe_ratio,
                        "AverageExposure(%)": result.summary.average_exposure_percent,
                        "Trades": result.summary.rebalance_count,
                    }
                )
    return pd.DataFrame(rows)
