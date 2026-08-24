from __future__ import annotations

"""Live, state-aware recommendations for the TSLA cycle-profit strategy.

The module does not place broker orders.  It converts the same Technical
re-entry mask used by optimization and simulation into a next-session order
plan, while keeping the cycle anchor and already-used drawdown levels explicit.
"""

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .pattern_optimization import ReentryFilterParams, generate_reentry_signal


@dataclass(frozen=True)
class LiveCyclePolicy:
    strategy: str = "SINGLE_70_SELL_50_TECHNICAL"
    initial_allocation: float = 0.25
    profit_target: float = 0.70
    sell_fraction: float = 0.50
    drawdown_levels: tuple[float, ...] = (-0.10, -0.20, -0.30)
    buy_equity_fractions: tuple[float, ...] = (0.15, 0.20, 0.30)
    drawdown_reset_level: float = -0.05
    minimum_sessions_between_buys: int = 10
    reentry: ReentryFilterParams = field(default_factory=ReentryFilterParams)

    def __post_init__(self) -> None:
        if not 0 < self.initial_allocation <= 1:
            raise ValueError("initial_allocation must be in (0, 1].")
        if self.profit_target <= 0:
            raise ValueError("profit_target must be positive.")
        if not 0 < self.sell_fraction <= 1:
            raise ValueError("sell_fraction must be in (0, 1].")
        if len(self.drawdown_levels) != len(self.buy_equity_fractions):
            raise ValueError("drawdown levels and buy fractions must align.")
        if self.minimum_sessions_between_buys < 0:
            raise ValueError("minimum_sessions_between_buys must be non-negative.")


@dataclass
class LiveCycleAccount:
    """Broker balances plus the minimum strategy state that a broker lacks."""

    cash: float
    shares: float = 0.0
    average_cost: float | None = None
    cycle_id: int = 0
    cycle_anchor: float | None = None
    cycle_start_shares: float = 0.0
    cycle_buy_notional: float = 0.0
    cycle_buy_shares: float = 0.0
    collecting_cycle_buys: bool = False
    rearm_pending: bool = False
    fired_drawdowns: list[float] = field(default_factory=list)
    fired_profit_target: bool = False
    last_buy_date: str | None = None
    state_as_of: str | None = None

    def validate(self) -> None:
        if self.cash < -1e-9 or self.shares < -1e-12:
            raise ValueError("cash and shares cannot be negative.")
        if self.cycle_start_shares < -1e-12:
            raise ValueError("cycle_start_shares cannot be negative.")
        if self.shares > 0 and self.cycle_anchor is None:
            raise ValueError("cycle_anchor is required when shares are held.")
        if self.shares > 0 and self.cycle_start_shares <= 0:
            raise ValueError("cycle_start_shares is required when shares are held.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveCycleAccount":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        account = cls(**{key: value for key, value in payload.items() if key in known})
        account.validate()
        return account

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_live_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate the exact three Technical inputs and the 63-day drawdown."""

    required = {"Date", "AdjClose"}
    if missing := required.difference(prices.columns):
        raise ValueError(f"Missing live price columns: {sorted(missing)}")
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["AdjClose"] = pd.to_numeric(frame["AdjClose"], errors="coerce")
    frame = (
        frame.dropna(subset=["Date", "AdjClose"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 200:
        raise ValueError("At least 200 valid daily closes are required.")
    close = frame["AdjClose"]
    frame["Return60D"] = close.pct_change(60, fill_method=None)
    change = close.diff()
    gain = change.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-change.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative = gain / loss.replace(0, np.nan)
    frame["RSI14"] = (100 - 100 / (1 + relative)).where(loss.ne(0), 100.0)
    sma200 = close.rolling(200, min_periods=200).mean()
    frame["DistanceFromSMA200"] = close / sma200 - 1.0
    frame["PriceDrawdown63"] = close / close.rolling(63, min_periods=1).max() - 1.0
    frame["TechnicalReentrySignal"] = generate_reentry_signal(
        frame, ReentryFilterParams()
    )
    return frame


def trading_sessions_since(
    trading_dates: pd.Series,
    last_buy_date: str | None,
    as_of_date: str | pd.Timestamp,
) -> int | None:
    if last_buy_date is None:
        return None
    dates = pd.to_datetime(trading_dates, errors="coerce").dropna()
    last_buy = pd.Timestamp(last_buy_date)
    as_of = pd.Timestamp(as_of_date)
    return int(dates.gt(last_buy).mul(dates.le(as_of)).sum())


def business_days_after(as_of_date: str | pd.Timestamp, today: date) -> int:
    """Approximate data freshness; one missing weekday is tolerated by caller."""

    start = pd.Timestamp(as_of_date).normalize() + pd.offsets.BDay(1)
    end = pd.Timestamp(today).normalize()
    if start > end:
        return 0
    return int(len(pd.bdate_range(start, end)))


def recommend_next_session(
    latest: pd.Series | dict[str, Any],
    account: LiveCycleAccount,
    policy: LiveCyclePolicy,
    *,
    sessions_since_last_buy: int | None,
) -> dict[str, Any]:
    """Return a next-open plan and position weights without placing an order."""

    account.validate()
    row = latest if isinstance(latest, pd.Series) else pd.Series(latest)
    close = float(row["AdjClose"])
    signal = bool(row["TechnicalReentrySignal"])
    drawdown = float(row["PriceDrawdown63"])
    equity = account.cash + account.shares * close
    if equity <= 0:
        raise ValueError("Account equity must be positive.")
    position_before = account.shares * close / equity
    cash_before = account.cash / equity

    effective_fired = set(float(value) for value in account.fired_drawdowns)
    reset_drawdowns = drawdown >= policy.drawdown_reset_level and not signal
    if reset_drawdowns:
        effective_fired.clear()

    next_target_price = (
        account.cycle_anchor * (1 + policy.profit_target)
        if account.cycle_anchor is not None and not account.fired_profit_target
        else None
    )
    conditional_sell_shares = (
        min(account.shares, account.cycle_start_shares * policy.sell_fraction)
        if next_target_price is not None
        else 0.0
    )
    conditional_sell_fraction = conditional_sell_shares * close / equity

    action = "HOLD"
    side: str | None = None
    equity_fraction = 0.0
    notional = 0.0
    estimated_shares = 0.0
    eligible_levels: list[float] = []
    reason = "No new entry or scale-in condition is actionable at the next open."

    if account.shares <= 1e-12:
        action = "BUY_INITIAL_NEXT_OPEN"
        side = "BUY"
        equity_fraction = min(policy.initial_allocation, cash_before)
        notional = equity * equity_fraction
        estimated_shares = notional / close
        reason = "Fresh/flat account: establish the validated 25% initial allocation."
    else:
        spacing_ok = (
            sessions_since_last_buy is not None
            and sessions_since_last_buy >= policy.minimum_sessions_between_buys
        )
        eligible_levels = [
            level
            for level in policy.drawdown_levels
            if level not in effective_fired and drawdown <= level
        ]
        if signal and sessions_since_last_buy is None:
            action = "HOLD_MISSING_LAST_BUY_DATE"
            reason = (
                "Technical re-entry is true, but last_buy_date is missing; "
                "the 10-session spacing rule cannot be verified."
            )
        elif signal and spacing_ok and eligible_levels and account.cash > 0:
            desired_fraction = sum(
                fraction
                for level, fraction in zip(
                    policy.drawdown_levels,
                    policy.buy_equity_fractions,
                    strict=True,
                )
                if level in eligible_levels
            )
            equity_fraction = min(desired_fraction, cash_before)
            notional = equity * equity_fraction
            estimated_shares = notional / close
            side = "BUY"
            action = "BUY_SCALE_NEXT_OPEN"
            reason = (
                "Technical re-entry is true, new drawdown levels are eligible, "
                "and the 10-session spacing rule is satisfied."
            )
        elif signal and not spacing_ok:
            action = "HOLD_BUY_SPACING"
            reason = (
                f"Technical re-entry is true, but only {sessions_since_last_buy} "
                f"sessions have elapsed; {policy.minimum_sessions_between_buys} are required."
            )
        elif signal and not eligible_levels:
            reason = "Technical re-entry is true, but all current drawdown levels were already used."
        elif signal and account.cash <= 0:
            reason = "Technical re-entry is true, but no cash is available for a scale-in."

    position_after = min(1.0, position_before + equity_fraction)
    return {
        "Strategy": policy.strategy,
        "SignalAsOfClose": pd.Timestamp(row["Date"]).date().isoformat(),
        "ReferenceClose": close,
        "Action": action,
        "OrderSide": side,
        "OrderEquityFraction": equity_fraction,
        "OrderNotionalAtReferenceClose": notional,
        "EstimatedSharesAtReferenceClose": estimated_shares,
        "PositionBeforeFraction": position_before,
        "CashBeforeFraction": cash_before,
        "EstimatedPositionAfterFraction": position_after,
        "EstimatedCashAfterFraction": max(0.0, 1.0 - position_after),
        "Reason": reason,
        "Technical": {
            "Return60D": float(row["Return60D"]),
            "Return60DPass": bool(float(row["Return60D"]) <= policy.reentry.return60_max),
            "RSI14": float(row["RSI14"]),
            "RSI14Pass": bool(float(row["RSI14"]) <= policy.reentry.rsi_max),
            "DistanceFromSMA200": float(row["DistanceFromSMA200"]),
            "BelowSMA200Pass": bool(
                float(row["DistanceFromSMA200"]) <= policy.reentry.distance_sma200_max
            ),
            "TechnicalReentrySignal": signal,
            "PriceDrawdown63": drawdown,
        },
        "ScaleIn": {
            "EligibleNewDrawdownLevels": eligible_levels,
            "EffectiveFiredDrawdownLevels": sorted(effective_fired, reverse=True),
            "DrawdownLevelsReset": reset_drawdowns,
            "SessionsSinceLastBuy": sessions_since_last_buy,
            "MinimumSessionsBetweenBuys": policy.minimum_sessions_between_buys,
        },
        "ProfitTaking": {
            "CycleAnchor": account.cycle_anchor,
            "TargetReturn": policy.profit_target,
            "NextTargetPrice": next_target_price,
            "TargetReachedAtReferenceClose": bool(
                next_target_price is not None and close >= next_target_price
            ),
            "Instruction": (
                f"SELL {conditional_sell_shares:.8f} shares if next open is at or above "
                f"{next_target_price:.4f}"
                if next_target_price is not None
                else "No armed profit target."
            ),
            "ConditionalSellShares": conditional_sell_shares,
            "ConditionalSellEquityFractionAtReferenceClose": conditional_sell_fraction,
            "EstimatedPositionAfterConditionalSellFraction": max(
                0.0, position_before - conditional_sell_fraction
            ),
            "SellFractionOfCycleStartShares": policy.sell_fraction,
        },
        "ExecutionConvention": (
            "Use the completed close signal at the next tradable open. Profit target is "
            "checked against that open. Percentages are fractions of equity immediately "
            "before execution; actual shares use the broker fill price."
        ),
    }


def apply_filled_order(
    account: LiveCycleAccount,
    *,
    side: str,
    fill_price: float,
    fill_shares: float,
    fill_date: str,
    fees: float = 0.0,
    fired_drawdown_levels: list[float] | None = None,
) -> LiveCycleAccount:
    """Apply a user-confirmed broker fill to the persistent cycle state."""

    account.validate()
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL.")
    if fill_price <= 0 or fill_shares <= 0 or fees < 0:
        raise ValueError("fill price/shares must be positive and fees non-negative.")
    pd.Timestamp(fill_date)  # validate before mutating the account
    gross = fill_price * fill_shares

    if side == "BUY":
        total_cost = gross + fees
        if total_cost > account.cash + 1e-8:
            raise ValueError("Buy fill exceeds available cash.")
        previous_shares = account.shares
        previous_book_cost = previous_shares * (account.average_cost or 0.0)
        account.cash -= total_cost
        account.shares += fill_shares
        account.average_cost = (
            previous_book_cost + total_cost
        ) / account.shares

        starts_cycle = previous_shares <= 1e-12 or account.rearm_pending or not account.collecting_cycle_buys
        if starts_cycle:
            account.cycle_id += 1
            account.cycle_buy_notional = 0.0
            account.cycle_buy_shares = 0.0
            account.fired_profit_target = False
            account.collecting_cycle_buys = True
            account.rearm_pending = False
        account.cycle_buy_notional += gross
        account.cycle_buy_shares += fill_shares
        account.cycle_anchor = account.cycle_buy_notional / account.cycle_buy_shares
        account.cycle_start_shares = account.shares
        account.fired_drawdowns = sorted(
            set(account.fired_drawdowns).union(fired_drawdown_levels or []),
            reverse=True,
        )
        account.last_buy_date = pd.Timestamp(fill_date).date().isoformat()
    else:
        if fill_shares > account.shares + 1e-10:
            raise ValueError("Sell fill exceeds shares held.")
        proceeds = gross - fees
        if proceeds < 0:
            raise ValueError("Fees cannot exceed sale proceeds.")
        account.cash += proceeds
        account.shares = max(0.0, account.shares - fill_shares)
        account.fired_profit_target = True
        account.rearm_pending = True
        account.collecting_cycle_buys = False

    account.state_as_of = pd.Timestamp(fill_date).date().isoformat()
    account.validate()
    return account
