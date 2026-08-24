from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import IntegratedResult, IntegratedSummary


@dataclass(frozen=True)
class ScoreAccumulationParams:
    """Execution rules for buying a scored washout and selling into strength.

    The score itself is deliberately not recalculated here.  Callers must pass
    the output of ``generate_integrated_signals`` so development selection and
    final simulation use the exact same point-in-time signal values.
    """

    initial_allocation: float = 0.50
    bearish_score_max: float = 0.45
    downside_probability_min: float = 0.55
    drawdown_levels: tuple[float, ...] = (-0.10, -0.20, -0.30)
    buy_equity_fractions: tuple[float, ...] = (0.15, 0.20, 0.30)
    profit_targets: tuple[float, ...] = (0.50,)
    profit_sell_fractions: tuple[float, ...] = (0.50,)
    drawdown_reset_level: float = -0.05
    minimum_sessions_between_buys: int = 10
    require_bear_regime_for_scale_in: bool = False
    bear_end_entry_fraction: float = 0.0

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
        if len(self.profit_targets) != len(self.profit_sell_fractions):
            raise ValueError("profit targets and sell fractions must align.")
        if tuple(sorted(self.profit_targets)) != self.profit_targets:
            raise ValueError("profit targets must be ascending.")
        if any(target <= 0 for target in self.profit_targets):
            raise ValueError("profit targets must be positive.")
        if any(not 0 < fraction <= 1 for fraction in self.profit_sell_fractions):
            raise ValueError("profit sell fractions must be in (0, 1].")
        if not min(self.drawdown_levels) < self.drawdown_reset_level <= 0:
            raise ValueError("drawdown reset must be above the deepest buy level.")
        if self.minimum_sessions_between_buys < 0:
            raise ValueError("minimum_sessions_between_buys must be non-negative.")
        if not 0 <= self.bear_end_entry_fraction <= 1:
            raise ValueError("bear_end_entry_fraction must be in [0, 1].")


def run_score_accumulation_backtest(
    signals: pd.DataFrame,
    params: ScoreAccumulationParams,
    *,
    initial_capital: float = 40_000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> IntegratedResult:
    """Buy scored drawdowns and scale out at average-cost profit targets.

    Composite/downside values and closing-price drawdowns from session *t* are
    acted on at session *t+1* open.  Profit targets are evaluated using the
    executable open, so no intraday high/close look-ahead is introduced.
    """

    frame = signals.sort_values("Date").reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("signals must contain at least one row.")
    required = {"Date", "Open", "Close", "CompositeScore"}
    missing = required.difference(frame.columns)
    if missing:
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
    bear_end_buy = frame.get(
        "BearEndBuySignal", pd.Series(False, index=frame.index)
    ).fillna(False).to_numpy(dtype=bool)
    cash_rates = pd.to_numeric(
        frame.get("CashRate", pd.Series(0.0, index=frame.index)),
        errors="coerce",
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

        if index == 0 and params.initial_allocation > 0:
            spend = cash * params.initial_allocation
            execution = open_price * buy_multiplier
            bought = spend / execution
            cash -= spend
            shares += bought
            average_cost = execution
            last_buy_index = index
            actions.append("BUY_INITIAL")
            executed_notional += spend

        if index > 0:
            # Sell first when an executable open reaches a profit target.
            # Multiple targets crossed by a gap execute in ascending order.
            if shares > 0 and average_cost is not None:
                open_return = open_price / average_cost - 1
                for target, fraction in zip(
                    params.profit_targets,
                    params.profit_sell_fractions,
                    strict=True,
                ):
                    if target in fired_profit_targets or open_return < target:
                        continue
                    sold = shares * fraction
                    execution = open_price * sell_multiplier
                    proceeds = sold * execution
                    shares -= sold
                    cash += proceeds
                    fired_profit_targets.add(target)
                    completed_sales += 1
                    actions.append(f"SELL_TP_{int(round(target * 100))}")
                    executed_notional += proceeds

            previous_drawdown = price_drawdown[index - 1]
            previous_score = scores[index - 1]
            previous_downside = downside[index - 1]
            score_recovered = (
                np.isfinite(previous_score)
                and previous_score > params.bearish_score_max
            )
            regime_recovered = (
                params.require_bear_regime_for_scale_in
                and not bear_regime[index - 1]
            )
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
            buy_spacing_ok = (
                index - last_buy_index >= params.minimum_sessions_between_buys
            )
            # A drawdown purchase is averaging down an existing holding.
            # A cash-only account must use the separate bear-end entry path.
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
                    desired = equity_at_open * sum(
                        fraction for _, fraction in eligible_levels
                    )
                    spend = min(cash, desired)
                    if spend > 0:
                        execution = open_price * buy_multiplier
                        bought = spend / execution
                        existing_cost = shares * (average_cost or 0.0)
                        shares += bought
                        cash -= spend
                        average_cost = (existing_cost + bought * execution) / shares
                        fired_drawdowns.update(level for level, _ in eligible_levels)
                        # Restart the user's +50% ladder only if the blended
                        # position is genuinely back below its first target.
                        # Otherwise a high-priced dip addition could trigger
                        # an unintended sell at the very next open.
                        if (
                            not params.profit_targets
                            or open_price / average_cost - 1
                            < min(params.profit_targets)
                        ):
                            fired_profit_targets.clear()
                        last_buy_index = index
                        actions.append(
                            "BUY_DRAWDOWN_"
                            + "_".join(str(int(round(abs(level) * 100))) for level, _ in eligible_levels)
                        )
                        executed_notional += spend

            if (
                shares == 0
                and cash > 0
                and params.bear_end_entry_fraction > 0
                and bear_end_buy[index - 1]
            ):
                spend = cash * params.bear_end_entry_fraction
                execution = open_price * buy_multiplier
                bought = spend / execution
                cash -= spend
                shares = bought
                average_cost = execution
                last_buy_index = index
                fired_profit_targets.clear()
                if np.isfinite(previous_drawdown):
                    fired_drawdowns.update(
                        level
                        for level in params.drawdown_levels
                        if previous_drawdown <= level
                    )
                actions.append("BUY_BEAR_END")
                executed_notional += spend

        action = "+".join(actions) if actions else "HOLD"
        equity = cash + shares * close_price
        unrealized_return = (
            close_price / average_cost - 1
            if shares > 0 and average_cost is not None
            else np.nan
        )
        row = {
            "Date": dates[index],
            "Open": open_price,
            "Close": close_price,
            "Action": action,
            "State": "LONG" if shares > 0 else "CASH",
            "Cash": cash,
            "Shares": shares,
            "AverageCost": average_cost,
            "Equity": equity,
            "CompositeScore": scores[index],
            "DownsideProbability21": downside[index],
            "PriceDrawdown63": price_drawdown[index],
            "UnrealizedReturn": unrealized_return,
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
