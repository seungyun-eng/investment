from __future__ import annotations

"""Cycle-anchored profit taking with causal Technical re-entry execution."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import IntegratedResult, IntegratedSummary


@dataclass(frozen=True)
class CycleProfitParams:
    """Execution rules whose profit target is anchored to each new buy cycle.

    ``cycle_sell_fractions`` are fractions of the shares held at the start of
    the cycle, not fractions of the remaining position.  An actual Technical
    re-entry purchase starts a new cycle and re-arms every profit target.
    Portfolio average cost remains separate and is used only for accounting.
    """

    initial_allocation: float = 1.0
    bearish_score_max: float = 0.5
    downside_probability_min: float = 0.0
    drawdown_levels: tuple[float, ...] = (-0.10, -0.20, -0.30)
    buy_equity_fractions: tuple[float, ...] = (0.15, 0.20, 0.30)
    profit_targets: tuple[float, ...] = (0.50,)
    cycle_sell_fractions: tuple[float, ...] = (0.50,)
    drawdown_reset_level: float = -0.05
    minimum_sessions_between_buys: int = 10
    require_bear_regime_for_scale_in: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.initial_allocation <= 1:
            raise ValueError("initial_allocation must be in [0, 1].")
        if not 0 <= self.bearish_score_max <= 1:
            raise ValueError("bearish_score_max must be in [0, 1].")
        if not 0 <= self.downside_probability_min <= 1:
            raise ValueError("downside_probability_min must be in [0, 1].")
        if len(self.drawdown_levels) != len(self.buy_equity_fractions):
            raise ValueError("drawdown levels and buy fractions must align.")
        if any(level >= 0 for level in self.drawdown_levels):
            raise ValueError("drawdown levels must be negative.")
        if tuple(sorted(self.drawdown_levels, reverse=True)) != self.drawdown_levels:
            raise ValueError("drawdown levels must run from shallow to deep.")
        if any(not 0 < fraction <= 1 for fraction in self.buy_equity_fractions):
            raise ValueError("buy fractions must be in (0, 1].")
        if len(self.profit_targets) != len(self.cycle_sell_fractions):
            raise ValueError("profit targets and cycle sell fractions must align.")
        if tuple(sorted(self.profit_targets)) != self.profit_targets:
            raise ValueError("profit targets must be ascending.")
        if any(target <= 0 for target in self.profit_targets):
            raise ValueError("profit targets must be positive.")
        if any(not 0 < fraction <= 1 for fraction in self.cycle_sell_fractions):
            raise ValueError("cycle sell fractions must be in (0, 1].")
        if sum(self.cycle_sell_fractions) > 1 + 1e-12:
            raise ValueError("cycle sell fractions cannot exceed 100% of cycle shares.")
        if not min(self.drawdown_levels) < self.drawdown_reset_level <= 0:
            raise ValueError("drawdown reset must be above the deepest buy level.")
        if self.minimum_sessions_between_buys < 0:
            raise ValueError("minimum_sessions_between_buys must be non-negative.")


def run_cycle_profit_backtest(
    signals: pd.DataFrame,
    params: CycleProfitParams,
    *,
    initial_capital: float = 40_000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> IntegratedResult:
    """Execute cycle profit targets and prior-close Technical buys at next open."""

    frame = signals.sort_values("Date").reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("signals must contain at least one row.")
    required = {"Date", "Open", "Close", "CompositeScore"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"signals missing required columns: {sorted(missing)}")

    dates = pd.to_datetime(frame["Date"]).to_numpy()
    opens = pd.to_numeric(frame["Open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(frame["CompositeScore"], errors="coerce").to_numpy(dtype=float)
    downside_column = (
        "DownsideProbability21"
        if "DownsideProbability21" in frame
        else "TslaDownsideProbability21"
    )
    downside = pd.to_numeric(
        frame.get(downside_column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    bear_regime = (
        frame.get("BearRegime", pd.Series(False, index=frame.index)).fillna(False)
        | frame.get("FastWashout", pd.Series(False, index=frame.index)).fillna(False)
    ).to_numpy(dtype=bool)
    cash_rates = pd.to_numeric(
        frame.get("CashRate", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    if "PriceDrawdown63" in frame:
        price_drawdown = pd.to_numeric(
            frame["PriceDrawdown63"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        close_series = pd.Series(closes)
        price_drawdown = (
            close_series / close_series.rolling(63, min_periods=1).max() - 1
        ).to_numpy(dtype=float)

    buy_multiplier = 1 + (transaction_cost_bps + slippage_bps) / 10_000
    sell_multiplier = 1 - (transaction_cost_bps + slippage_bps) / 10_000
    cash = float(initial_capital)
    shares = 0.0
    average_cost: float | None = None
    cycle_anchor: float | None = None
    cycle_start_shares = 0.0
    cycle_id = 0
    cycle_buy_notional = 0.0
    cycle_buy_shares = 0.0
    collecting_cycle_buys = False
    rearm_pending = False
    fired_drawdowns: set[float] = set()
    fired_profit_targets: set[float] = set()
    last_buy_index = -10**9
    trades: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    completed_sales = 0

    for index in range(len(frame)):
        open_price = opens[index]
        close_price = closes[index]
        if not np.isfinite(open_price) or open_price <= 0:
            raise ValueError(f"Invalid open price at row {index}.")
        if not np.isfinite(close_price) or close_price <= 0:
            raise ValueError(f"Invalid close price at row {index}.")

        cash *= 1 + max(cash_rates[index], 0.0) / 100 / 252
        actions: list[str] = []
        executed_notional = 0.0
        bought_shares = 0.0
        sold_shares = 0.0

        if index == 0 and params.initial_allocation > 0:
            spend = cash * params.initial_allocation
            execution = open_price * buy_multiplier
            bought = spend / execution
            cash -= spend
            shares += bought
            average_cost = execution
            cycle_id = 1
            cycle_anchor = execution
            cycle_start_shares = shares
            cycle_buy_notional = bought * execution
            cycle_buy_shares = bought
            collecting_cycle_buys = True
            last_buy_index = index
            bought_shares += bought
            executed_notional += spend
            actions.append("BUY_INITIAL")

        if index > 0:
            if shares > 0 and cycle_anchor is not None:
                cycle_return = open_price / cycle_anchor - 1
                for target, fraction in zip(
                    params.profit_targets,
                    params.cycle_sell_fractions,
                    strict=True,
                ):
                    if target in fired_profit_targets or cycle_return < target:
                        continue
                    sold = min(shares, cycle_start_shares * fraction)
                    if sold <= 0:
                        continue
                    execution = open_price * sell_multiplier
                    proceeds = sold * execution
                    shares -= sold
                    cash += proceeds
                    sold_shares += sold
                    executed_notional += proceeds
                    fired_profit_targets.add(target)
                    completed_sales += 1
                    rearm_pending = True
                    collecting_cycle_buys = False
                    actions.append(f"SELL_CYCLE_TP_{int(round(target * 100))}")

            previous_drawdown = price_drawdown[index - 1]
            previous_score = scores[index - 1]
            previous_downside = downside[index - 1]
            score_recovered = np.isfinite(previous_score) and previous_score > params.bearish_score_max
            regime_recovered = params.require_bear_regime_for_scale_in and not bear_regime[index - 1]
            if (
                np.isfinite(previous_drawdown)
                and previous_drawdown >= params.drawdown_reset_level
                and (score_recovered or regime_recovered)
            ):
                fired_drawdowns.clear()

            downside_confirmed = (
                params.downside_probability_min == 0
                or (
                    np.isfinite(previous_downside)
                    and previous_downside >= params.downside_probability_min
                )
            )
            bearish = (
                np.isfinite(previous_score)
                and previous_score <= params.bearish_score_max
                and downside_confirmed
                and (
                    not params.require_bear_regime_for_scale_in
                    or bear_regime[index - 1]
                )
            )
            buy_spacing_ok = index - last_buy_index >= params.minimum_sessions_between_buys
            if bearish and buy_spacing_ok and cash > 0 and shares > 0:
                eligible_levels = [
                    (level, fraction)
                    for level, fraction in zip(
                        params.drawdown_levels,
                        params.buy_equity_fractions,
                        strict=True,
                    )
                    if level not in fired_drawdowns and previous_drawdown <= level
                ]
                if eligible_levels:
                    equity_at_open = cash + shares * open_price
                    desired = equity_at_open * sum(fraction for _, fraction in eligible_levels)
                    spend = min(cash, desired)
                    if spend > 0:
                        execution = open_price * buy_multiplier
                        bought = spend / execution
                        existing_cost = shares * (average_cost or 0.0)
                        shares += bought
                        cash -= spend
                        average_cost = (existing_cost + bought * execution) / shares
                        fired_drawdowns.update(level for level, _ in eligible_levels)
                        last_buy_index = index
                        bought_shares += bought
                        executed_notional += spend

                        if rearm_pending or not collecting_cycle_buys:
                            cycle_id += 1
                            cycle_buy_notional = 0.0
                            cycle_buy_shares = 0.0
                            fired_profit_targets.clear()
                            collecting_cycle_buys = True
                            rearm_pending = False
                        cycle_buy_notional += bought * execution
                        cycle_buy_shares += bought
                        cycle_anchor = cycle_buy_notional / cycle_buy_shares
                        cycle_start_shares = shares
                        actions.append(
                            "BUY_CYCLE_DRAWDOWN_"
                            + "_".join(
                                str(int(round(abs(level) * 100)))
                                for level, _ in eligible_levels
                            )
                        )

        action = "+".join(actions) if actions else "HOLD"
        equity = cash + shares * close_price
        row = {
            "Date": dates[index],
            "Open": open_price,
            "Close": close_price,
            "Action": action,
            "State": "LONG" if shares > 0 else "CASH",
            "Cash": cash,
            "Shares": shares,
            "AverageCost": average_cost,
            "CycleId": cycle_id,
            "CycleAnchor": cycle_anchor,
            "CycleStartShares": cycle_start_shares,
            "FiredProfitTargetCount": len(fired_profit_targets),
            "Equity": equity,
            "CompositeScore": scores[index],
            "DownsideProbability21": downside[index],
            "PriceDrawdown63": price_drawdown[index],
            "BoughtShares": bought_shares,
            "SoldShares": sold_shares,
            "ExecutedNotional": executed_notional,
        }
        daily.append(row)
        if action != "HOLD":
            trades.append(row.copy())

    daily_frame = pd.DataFrame(daily)
    final_value = float(daily_frame["Equity"].iloc[-1])
    drawdown = daily_frame["Equity"] / np.maximum.accumulate(
        daily_frame["Equity"].to_numpy(dtype=float)
    ) - 1
    return IntegratedResult(
        daily=daily_frame,
        trades=pd.DataFrame(trades),
        summary=IntegratedSummary(
            initial_capital=initial_capital,
            total_injected=initial_capital,
            final_value=final_value,
            roi_percent=(final_value / initial_capital - 1) * 100,
            max_drawdown_percent=float(drawdown.min() * 100),
            completed_trades=completed_sales,
        ),
    )
