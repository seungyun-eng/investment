from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FearBuyParams


@dataclass(frozen=True)
class ContributionConfig:
    initial_lump_sum: float = 40_000.0
    monthly_contribution: float = 4_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.initial_lump_sum <= 0:
            raise ValueError("initial_lump_sum must be positive.")
        if self.monthly_contribution < 0:
            raise ValueError("monthly_contribution must be non-negative.")
        if min(self.transaction_cost_bps, self.slippage_bps) < 0:
            raise ValueError("Trading costs must be non-negative.")


@dataclass(frozen=True)
class ContributionDeploymentPolicy:
    """How much accumulated contribution cash each fear level may deploy."""

    mild_fraction: float = 1.0
    fear_fraction: float = 1.0
    panic_fraction: float = 1.0
    cooldown_sessions: int = 0

    def __post_init__(self) -> None:
        fractions = (
            self.mild_fraction,
            self.fear_fraction,
            self.panic_fraction,
        )
        if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
            raise ValueError("Deployment fractions must be in [0, 1].")
        if not fractions[0] <= fractions[1] <= fractions[2]:
            raise ValueError(
                "Deployment fractions must not decrease as fear deepens."
            )
        if self.cooldown_sessions < 0:
            raise ValueError("cooldown_sessions must be non-negative.")

    def fraction_for(self, trigger_level: str) -> float:
        return {
            "MILD_FEAR": self.mild_fraction,
            "FEAR": self.fear_fraction,
            "PANIC": self.panic_fraction,
        }.get(trigger_level, 0.0)


@dataclass(frozen=True)
class ContributionSummary:
    initial_lump_sum: float
    monthly_contribution: float
    contribution_count: int
    total_injected: float
    final_value: float
    net_profit: float
    roi_percent: float
    time_weighted_cagr_percent: float
    money_weighted_return_percent: float
    max_drawdown_percent: float
    sharpe_ratio: float
    average_exposure_percent: float
    trade_count: int


@dataclass
class ContributionResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: ContributionSummary
    source: str = ""


def _xirr(
    dates: list[pd.Timestamp],
    cash_flows: list[float],
) -> float:
    start = dates[0]
    years = np.asarray([(date - start).days / 365.25 for date in dates])
    flows = np.asarray(cash_flows, dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(flows / np.power(1.0 + rate, years)))

    lower = -0.9999
    upper = 1.0
    lower_value = npv(lower)
    upper_value = npv(upper)
    while lower_value * upper_value > 0 and upper < 1_000:
        upper *= 2.0
        upper_value = npv(upper)
    if lower_value * upper_value > 0:
        return float("nan")
    for _ in range(200):
        middle = (lower + upper) / 2.0
        middle_value = npv(middle)
        if abs(middle_value) < 1e-8:
            return middle
        if lower_value * middle_value <= 0:
            upper = middle
        else:
            lower = middle
            lower_value = middle_value
    return (lower + upper) / 2.0


def _contribution_summary(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    config: ContributionConfig,
) -> ContributionSummary:
    total_injected = float(daily["TotalInjected"].iloc[-1])
    final_value = float(daily["TotalValue"].iloc[-1])
    returns = daily["FlowAdjustedReturn"].dropna()
    wealth = daily["FlowAdjustedIndex"]
    elapsed_days = max(
        1,
        int((daily["Date"].iloc[-1] - daily["Date"].iloc[0]).days),
    )
    years = elapsed_days / 365.25
    twr_cagr = (
        (float(wealth.iloc[-1]) / 100.0) ** (1.0 / years) - 1.0
    ) * 100.0
    drawdown = wealth / wealth.cummax() - 1.0
    volatility = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / volatility * np.sqrt(252))
        if volatility > 0
        else float("nan")
    )
    contribution_rows = daily[daily["Contribution"] > 0]
    flow_dates = [pd.Timestamp(daily["Date"].iloc[0])]
    flow_values = [-config.initial_lump_sum]
    for row in contribution_rows.itertuples(index=False):
        flow_dates.append(pd.Timestamp(row.Date))
        flow_values.append(-float(row.Contribution))
    flow_dates.append(pd.Timestamp(daily["Date"].iloc[-1]))
    flow_values.append(final_value)
    return ContributionSummary(
        initial_lump_sum=config.initial_lump_sum,
        monthly_contribution=config.monthly_contribution,
        contribution_count=len(contribution_rows),
        total_injected=total_injected,
        final_value=final_value,
        net_profit=final_value - total_injected,
        roi_percent=(final_value / total_injected - 1.0) * 100.0,
        time_weighted_cagr_percent=twr_cagr,
        money_weighted_return_percent=_xirr(flow_dates, flow_values) * 100.0,
        max_drawdown_percent=float(drawdown.min() * 100.0),
        sharpe_ratio=sharpe,
        average_exposure_percent=float(daily["ActualWeight"].mean() * 100.0),
        trade_count=len(trades),
    )


def run_contribution_backtest(
    signals: pd.DataFrame,
    params: FearBuyParams,
    config: ContributionConfig,
    *,
    name: str,
    core_weight_override: float | None = None,
    invest_contributions_without_signal: bool = False,
    deployment_policy: ContributionDeploymentPolicy | None = None,
) -> ContributionResult:
    """Backtest an initial allocation plus first-session monthly cash deposits.

    By default, monthly deposits remain in cash until an active fear signal
    permits deployment. Benchmarks can opt into immediate contribution
    investment with ``invest_contributions_without_signal``.
    """

    required = {"Date", "Open", "Close", "CashRate", "TargetWeight"}
    missing = required - set(signals)
    if missing:
        raise ValueError(f"Contribution signals are missing: {sorted(missing)}")
    frame = signals.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame.empty:
        raise ValueError("Contribution backtest requires at least one row.")
    core_weight = (
        params.core_weight if core_weight_override is None else core_weight_override
    )
    policy = deployment_policy or ContributionDeploymentPolicy()
    fee_rate = config.transaction_cost_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    cash = float(config.initial_lump_sum)
    pending_contribution_cash = 0.0
    total_injected = float(config.initial_lump_sum)
    core_shares = 0.0
    tactical_shares = 0.0
    tactical_cost_basis = 0.0
    executed_target = core_weight
    previous_date: pd.Timestamp | None = None
    previous_value: float | None = None
    previous_month: pd.Period | None = None
    last_contribution_deployment_index = -1_000_000
    episode_peak_severity = 0
    flow_index = 100.0
    trade_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        date = pd.Timestamp(row["Date"])
        if previous_date is not None and cash > 0:
            days = max(0, (date - previous_date).days)
            prior_rate = max(-0.99, float(frame.loc[index - 1, "CashRate"]) / 100)
            cash_growth = (1.0 + prior_rate) ** (days / 365.25)
            cash *= cash_growth
            pending_contribution_cash *= cash_growth
        current_month = date.to_period("M")
        contribution = 0.0
        if previous_month is not None and current_month != previous_month:
            contribution = float(config.monthly_contribution)
            cash += contribution
            if not invest_contributions_without_signal:
                pending_contribution_cash += contribution
            total_injected += contribution

        if index == 0:
            signal_row = row
            signal_date = date
            requested_target = core_weight
            target_changed = True
            signal_reason = "INITIAL_LUMP_SUM"
            signal_state = "CORE_ONLY" if core_weight < 1 else "BUY_HOLD"
        else:
            signal_row = frame.loc[index - 1]
            signal_date = pd.Timestamp(signal_row["Date"])
            requested_target = float(signal_row["TargetWeight"])
            target_changed = (
                abs(requested_target - executed_target) >= params.rebalance_band
            )
            signal_reason = str(signal_row.get("TransitionReason", ""))
            signal_state = str(signal_row.get("SignalState", name))

        open_price = float(row["Open"])
        close_price = float(row["Close"])
        total_shares = core_shares + tactical_shares
        value_at_open = cash + total_shares * open_price
        current_stock_value = total_shares * open_price
        target_increased = requested_target > (
            executed_target + params.rebalance_band
        )
        desired_deployment_target = float(
            signal_row.get("DesiredTargetWeight", requested_target)
        )
        trigger_level = str(signal_row.get("TriggerLevel", "NO_FEAR"))
        trigger_severity = {
            "MILD_FEAR": 1,
            "FEAR": 2,
            "PANIC": 3,
        }.get(trigger_level, 0)
        active_fear_signal = (
            bool(signal_row.get("DecisionDay", False))
            and trigger_severity > 0
        )
        escalation = False
        if bool(signal_row.get("DecisionDay", False)):
            if trigger_severity == 0:
                episode_peak_severity = 0
            else:
                escalation = trigger_severity > episode_peak_severity
                episode_peak_severity = max(
                    episode_peak_severity,
                    trigger_severity,
                )
        cooldown_ready = (
            index - last_contribution_deployment_index
            >= policy.cooldown_sessions
        )
        deployment_fraction = policy.fraction_for(trigger_level)
        contribution_deployment_signal = (
            active_fear_signal
            and pending_contribution_cash > 1e-8
            and deployment_fraction > 0.0
            and (escalation or cooldown_ready)
        )
        buy_target = requested_target
        if active_fear_signal and not target_increased:
            buy_target = min(requested_target, desired_deployment_target)
        buy_gap = buy_target * value_at_open - current_stock_value
        sell_gap = requested_target * value_at_open - current_stock_value
        order_rows: list[tuple[str, str, float, str]] = []

        should_target_buy = buy_gap > 1e-8 and (
            index == 0
            or target_increased
            or (
                invest_contributions_without_signal
                and contribution > 0
            )
        )
        if should_target_buy:
            strategic_cash = max(0.0, cash - pending_contribution_cash)
            affordable_notional = strategic_cash / (1.0 + fee_rate)
            target_order = min(buy_gap, affordable_notional)
            desired_core_value = core_weight * value_at_open
            core_gap = max(0.0, desired_core_value - core_shares * open_price)
            core_order = min(target_order, core_gap)
            tactical_order = max(0.0, target_order - core_order)
            if core_order > 1e-8:
                order_rows.append(("BUY", "CORE", core_order, "STRATEGIC"))
            if tactical_order > 1e-8:
                order_rows.append(
                    ("BUY", "TACTICAL", tactical_order, "STRATEGIC")
                )
        if contribution_deployment_signal:
            contribution_budget = (
                pending_contribution_cash * deployment_fraction
            )
            contribution_notional = contribution_budget / (
                1.0 + fee_rate
            )
            order_rows.append(
                (
                    "BUY",
                    "TACTICAL",
                    contribution_notional,
                    "ACCUMULATED_CONTRIBUTION",
                )
            )
        if (
            not should_target_buy
            and sell_gap < -1e-8
            and target_changed
            and tactical_shares > 0
        ):
            order_rows.append(
                (
                    "SELL",
                    "TACTICAL",
                    min(-sell_gap, tactical_shares * open_price),
                    "STRATEGIC",
                )
            )

        for action, sleeve, requested_notional, cash_source in order_rows:
            execution_price = open_price
            quantity = 0.0
            notional = 0.0
            fee = 0.0
            realized_pnl = 0.0
            if action == "BUY":
                execution_price = open_price * (1.0 + slippage_rate)
                notional = min(requested_notional, cash / (1.0 + fee_rate))
                quantity = notional / execution_price if execution_price else 0.0
                fee = notional * fee_rate
                cash = max(0.0, cash - notional - fee)
                if cash_source == "ACCUMULATED_CONTRIBUTION":
                    pending_contribution_cash = max(
                        0.0,
                        pending_contribution_cash - notional - fee,
                    )
                if sleeve == "CORE":
                    core_shares += quantity
                else:
                    tactical_shares += quantity
                    tactical_cost_basis += notional + fee
            else:
                execution_price = open_price * (1.0 - slippage_rate)
                quantity = min(requested_notional / open_price, tactical_shares)
                notional = quantity * execution_price
                fee = notional * fee_rate
                average_cost = (
                    tactical_cost_basis / tactical_shares
                    if tactical_shares
                    else 0.0
                )
                removed_cost = quantity * average_cost
                tactical_shares -= quantity
                tactical_cost_basis = max(
                    0.0,
                    tactical_cost_basis - removed_cost,
                )
                realized_pnl = notional - fee - removed_cost
                cash += notional - fee
            reason = signal_reason
            if cash_source == "ACCUMULATED_CONTRIBUTION":
                reason = f"{trigger_level}_ACCUMULATED_CASH_BUY"
            elif (
                invest_contributions_without_signal
                and contribution > 0
                and not target_changed
            ):
                reason = f"MONTHLY_CONTRIBUTION_{sleeve}"
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
                    "Contribution": contribution,
                    "TotalInjected": total_injected,
                    "TargetWeight": requested_target,
                    "DeploymentTargetWeight": buy_target,
                    "CashSource": cash_source,
                    "DeploymentFraction": deployment_fraction,
                    "State": signal_state,
                    "Reason": reason,
                    "FearScore": signal_row.get("FearScore", float("nan")),
                    "EuphoriaScore": signal_row.get(
                        "EuphoriaScore",
                        float("nan"),
                    ),
                }
            )

        if target_changed:
            executed_target = requested_target
        total_shares = core_shares + tactical_shares
        value_at_close = cash + total_shares * close_price
        actual_weight = (
            total_shares * close_price / value_at_close if value_at_close else 0.0
        )
        if previous_value is None:
            flow_adjusted_return = (
                value_at_close / config.initial_lump_sum - 1.0
            )
        else:
            flow_adjusted_return = (
                value_at_close / (previous_value + contribution) - 1.0
            )
        flow_index *= 1.0 + flow_adjusted_return
        daily_rows.append(
            {
                "Date": date,
                "Open": open_price,
                "Close": close_price,
                "CashRate": row.get("CashRate", float("nan")),
                "VIX": row.get("VIX", float("nan")),
                "VixPercentile": row.get("VixPercentile", float("nan")),
                "FearScore": row.get("FearScore", float("nan")),
                "EuphoriaScore": row.get("EuphoriaScore", float("nan")),
                "ModelRiskPercentile": row.get(
                    "ModelRiskPercentile",
                    float("nan"),
                ),
                "MacroConfirmationScore": row.get(
                    "MacroConfirmationScore",
                    float("nan"),
                ),
                "Drawdown252": row.get("Drawdown252", float("nan")),
                "Contribution": contribution,
                "TotalInjected": total_injected,
                "TargetWeight": requested_target,
                "DeploymentTargetWeight": buy_target,
                "ActiveFearSignal": active_fear_signal,
                "ContributionDeploymentSignal": (
                    contribution_deployment_signal
                ),
                "DeploymentFraction": deployment_fraction,
                "ActualWeight": actual_weight,
                "Cash": cash,
                "PendingContributionCash": pending_contribution_cash,
                "CoreShares": core_shares,
                "TacticalShares": tactical_shares,
                "TotalValue": value_at_close,
                "NetProfit": value_at_close - total_injected,
                "ROI": (value_at_close / total_injected - 1.0) * 100.0,
                "FlowAdjustedReturn": flow_adjusted_return,
                "FlowAdjustedIndex": flow_index,
                "State": signal_state,
                "Reason": signal_reason,
            }
        )
        previous_date = date
        previous_value = value_at_close
        previous_month = current_month
        if contribution_deployment_signal:
            last_contribution_deployment_index = index

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    return ContributionResult(
        daily=daily,
        trades=trades,
        summary=_contribution_summary(daily, trades, config),
        source=f"{name}: monthly contribution scenario",
    )


def contribution_comparison_table(
    results: dict[str, ContributionResult],
    *,
    period: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        summary = result.summary
        rows.append(
            {
                "Period": period,
                "Portfolio": name,
                "InitialLumpSum": summary.initial_lump_sum,
                "MonthlyContribution": summary.monthly_contribution,
                "ContributionCount": summary.contribution_count,
                "TotalInjected": summary.total_injected,
                "FinalValue": summary.final_value,
                "NetProfit": summary.net_profit,
                "ROI(%)": summary.roi_percent,
                "TWR_CAGR(%)": summary.time_weighted_cagr_percent,
                "XIRR(%)": summary.money_weighted_return_percent,
                "MDD(%)": summary.max_drawdown_percent,
                "Sharpe": summary.sharpe_ratio,
                "AverageExposure(%)": summary.average_exposure_percent,
                "Trades": summary.trade_count,
            }
        )
    return pd.DataFrame(rows)
