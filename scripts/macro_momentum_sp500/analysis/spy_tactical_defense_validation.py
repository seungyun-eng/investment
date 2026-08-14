"""Validation-only analysis for the SPY tactical-defense scratch rule.

This file deliberately does not import or modify production strategy code.  It
reconciles the indicator warm-up discrepancy, applies the repository-standard
5 bp fee + 5 bp slippage at next-session adjusted open, and evaluates the fixed
quick-recovery parameters on periods outside the 2020-2026 tuning window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_research.paths import load_paths

SELL_CONFIRM_DAYS = 3
FALLBACK_CONFIRM_DAYS = 3
RSI_TIER_35 = 0.30
RSI_TIER_30 = 0.80
QUICK_RECOVERY_DEPTH_THRESH = -0.03
QUICK_RECOVERY_WINDOW_DAYS = 3
ONE_WAY_FEE_BPS = 5.0
ONE_WAY_SLIPPAGE_BPS = 5.0

CONFIGS = {
    "V1": {},
    "V1_QUICK": {"use_quick_recovery": True},
    "V1_QUICK_FB5_REJECTED": {
        "use_quick_recovery": True,
        "fb_pct": 0.05,
    },
    "V1_QUICK_FB5_CAP_REJECTED": {
        "use_quick_recovery": True,
        "fb_pct": 0.05,
        "use_capitulation": True,
    },
}


@dataclass
class SimulationResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, float | int]


def _indicators(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    close = pd.to_numeric(output["Adj Close"], errors="coerce")
    output["MA150"] = close.rolling(150, min_periods=150).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    output["RSI14"] = 100 - 100 / (1 + average_gain / average_loss)
    return output


def load_prices() -> pd.DataFrame:
    path = load_paths().macro / "SPY Adjusted Historical Data.csv"
    frame = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    required = {"Date", "Adj Open", "Adj Close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SPY file is missing columns: {sorted(missing)}")
    return _indicators(frame)


def _stats(
    nav: pd.Series,
    dates: pd.Series,
    *,
    initial_value: float | None = None,
) -> dict[str, float]:
    starting_value = float(nav.iloc[0]) if initial_value is None else initial_value
    total = float(nav.iloc[-1] / starting_value - 1)
    years = float((dates.iloc[-1] - dates.iloc[0]).days / 365.25)
    cagr = float((nav.iloc[-1] / starting_value) ** (1 / years) - 1)
    nav_for_stats = (
        pd.concat([pd.Series([starting_value]), nav.reset_index(drop=True)], ignore_index=True)
        if initial_value is not None
        else nav
    )
    drawdown = nav_for_stats / nav_for_stats.cummax() - 1
    returns = nav_for_stats.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    return {
        "total_return_percent": total * 100,
        "cagr_percent": cagr * 100,
        "mdd_percent": float(drawdown.min() * 100),
        "sharpe": sharpe,
    }


def run_reported_close_to_close(
    full: pd.DataFrame,
    *,
    start: str,
    end: str,
    warmup_before_start: bool,
    use_quick_recovery: bool = False,
    fb_pct: float = 0.0,
    use_capitulation: bool = False,
    cap_days: int = 4,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[dict[str, float | int], dict[str, int]]:
    """Reproduce the scratch script's close-to-close weight convention.

    This exists only to reconcile the published figures.  It is not considered
    execution-safe because some current-close decisions receive that same
    session's entire close-to-close return.
    """

    frame = full.loc[full["Date"].between(start, end)].copy().reset_index(drop=True)
    if not warmup_before_start:
        frame = _indicators(frame)
    close = frame["Adj Close"].astype(float).to_numpy()
    ma150 = frame["MA150"].astype(float).to_numpy()
    rsi = frame["RSI14"].astype(float).to_numpy()
    weights = np.zeros(len(frame))
    weight = 1.0
    state = "invested"
    sell_price = None
    sell_index = None
    sell_depth = None
    below_streak = 0
    fallback_streak = 0
    rsi_bought_since_sell = False
    capitulation_index = None
    capitulation_price = None
    counts: dict[str, int] = {}

    def record(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for index in range(1, len(frame)):
        price = close[index]
        ma = ma150[index]
        if np.isnan(ma):
            weights[index] = weight
            continue
        prior_below = close[index - 1] < ma150[index - 1]
        below_streak = below_streak + 1 if prior_below else 0
        if state == "invested":
            if below_streak >= SELL_CONFIRM_DAYS and weight > 0:
                weight = 0.0
                state = "partial"
                sell_price = price
                sell_index = index
                sell_depth = close[index - 1] / ma150[index - 1] - 1
                fallback_streak = 0
                rsi_bought_since_sell = False
                capitulation_index = None
                capitulation_price = None
                record("SELL")
        else:
            bought = False
            if (
                use_quick_recovery
                and not rsi_bought_since_sell
                and sell_index is not None
                and index - sell_index <= QUICK_RECOVERY_WINDOW_DAYS
                and weight < 1.0
                and sell_depth is not None
                and sell_depth > QUICK_RECOVERY_DEPTH_THRESH
                and sell_price is not None
                and price > sell_price
            ):
                weight = 1.0
                state = "invested"
                bought = True
                record("QUICKBUY")
            if not bought:
                prior_rsi = rsi[index - 1]
                rsi_target = (
                    RSI_TIER_30
                    if prior_rsi < 30
                    else RSI_TIER_35
                    if prior_rsi < 35
                    else 0.0
                )
                if rsi_target > weight:
                    if weight < RSI_TIER_30 <= rsi_target:
                        capitulation_index = index
                        capitulation_price = price
                    weight = rsi_target
                    rsi_bought_since_sell = True
                    record("RSIBUY")
                if (
                    use_capitulation
                    and capitulation_index is not None
                    and capitulation_price is not None
                    and weight < 1.0
                    and index - capitulation_index <= cap_days
                    and price < capitulation_price
                ):
                    weight = 1.0
                    state = "invested"
                    record("CAPBUY")
                if state != "invested" and weight < 1.0:
                    threshold = ma150[index - 1] * (1 - fb_pct)
                    prior_ok = close[index - 1] > threshold
                    fallback_streak = fallback_streak + 1 if prior_ok else 0
                    if fallback_streak >= FALLBACK_CONFIRM_DAYS:
                        weight = 1.0
                        state = "invested"
                        record("FALLBACK")
        weights[index] = weight

    close_returns = pd.Series(close).pct_change().fillna(0.0)
    turnover = pd.Series(weights).diff().abs().fillna(abs(weights[0]))
    one_way_cost = (fee_bps + slippage_bps) / 10_000
    net_returns = weights * close_returns - turnover * one_way_cost
    nav = (1 + net_returns).cumprod()
    summary: dict[str, float | int] = _stats(nav, frame["Date"])
    summary.update(
        {
            "orders": int(sum(counts.values())),
            "sell_orders": int(counts.get("SELL", 0)),
            "turnover": float(turnover.sum()),
        }
    )
    return summary, counts


def run_next_open(
    full: pd.DataFrame,
    *,
    start: str,
    end: str,
    use_quick_recovery: bool = False,
    fb_pct: float = 0.0,
    use_capitulation: bool = False,
    cap_days: int = 4,
    fee_bps: float = ONE_WAY_FEE_BPS,
    slippage_bps: float = ONE_WAY_SLIPPAGE_BPS,
) -> SimulationResult:
    """Causal close-signal -> next adjusted-open execution simulation."""

    frame = full.loc[full["Date"].between(start, end)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No prices in {start}..{end}")
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    cash = 1.0
    shares = 0.0
    target = 1.0
    pending_target: float | None = 1.0
    state = "invested"
    below_streak = 0
    fallback_streak = 0
    pending_reason = "INITIAL"
    sell_execution_price: float | None = None
    sell_execution_index: int | None = None
    sell_depth: float | None = None
    pending_sell_depth: float | None = None
    rsi_bought_since_sell = False
    capitulation_index: int | None = None
    capitulation_price: float | None = None
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        open_price = float(row["Adj Open"])
        close_price = float(row["Adj Close"])
        value_at_open = cash + shares * open_price
        current_stock_value = shares * open_price
        requested = (
            pending_target * value_at_open - current_stock_value
            if pending_target is not None
            else 0.0
        )
        traded = False
        if requested > max(1e-12, value_at_open * 1e-10):
            execution_price = open_price * (1 + slippage_rate)
            affordable = cash / (1 + fee_rate)
            executed_notional = min(requested, affordable)
            quantity = executed_notional / execution_price
            fee = executed_notional * fee_rate
            shares += quantity
            cash -= executed_notional + fee
            traded = True
            action = "BUY"
        elif requested < -max(1e-12, value_at_open * 1e-10):
            execution_price = open_price * (1 - slippage_rate)
            quantity = min(-requested / open_price, shares)
            executed_notional = quantity * execution_price
            fee = executed_notional * fee_rate
            shares -= quantity
            cash += executed_notional - fee
            traded = True
            action = "SELL"
        if traded:
            trade_rows.append(
                {
                    "Date": row["Date"],
                    "Action": action,
                    "Reason": pending_reason,
                    "TargetWeight": pending_target,
                    "ExecutionPrice": execution_price,
                    "Fee": fee,
                }
            )
            if action == "SELL" and pending_target == 0.0:
                sell_execution_price = execution_price
                sell_execution_index = index
                sell_depth = pending_sell_depth
        pending_target = None

        value_at_close = cash + shares * close_price
        actual_weight = shares * close_price / value_at_close if value_at_close else 0.0
        daily_rows.append(
            {
                "Date": row["Date"],
                "TotalValue": value_at_close,
                "ActualWeight": actual_weight,
                "Close": close_price,
            }
        )

        ma = row["MA150"]
        rsi = row["RSI14"]
        if pd.isna(ma):
            continue
        below_streak = below_streak + 1 if close_price < float(ma) else 0
        next_target = target
        next_reason = "HOLD"
        if state == "invested":
            if below_streak >= SELL_CONFIRM_DAYS and target > 0:
                next_target = 0.0
                next_reason = "SELL_3D_BELOW_MA150"
                state = "partial"
                pending_sell_depth = close_price / float(ma) - 1
                fallback_streak = 0
                rsi_bought_since_sell = False
                capitulation_index = None
                capitulation_price = None
        else:
            bought = False
            days_after_sell = None if sell_execution_index is None else index - sell_execution_index
            if (
                use_quick_recovery
                and not rsi_bought_since_sell
                and sell_execution_price is not None
                and days_after_sell is not None
                and 1 <= days_after_sell <= QUICK_RECOVERY_WINDOW_DAYS
                and sell_depth is not None
                and sell_depth > QUICK_RECOVERY_DEPTH_THRESH
                and close_price > sell_execution_price
                and target < 1.0
            ):
                next_target = 1.0
                next_reason = "QUICK_RECOVERY"
                state = "invested"
                bought = True
            if not bought:
                rsi_target = RSI_TIER_30 if rsi < 30 else RSI_TIER_35 if rsi < 35 else 0.0
                if rsi_target > target:
                    next_target = rsi_target
                    next_reason = "RSI_TIER"
                    rsi_bought_since_sell = True
                    if target < RSI_TIER_30 <= rsi_target:
                        capitulation_index = index
                        capitulation_price = close_price
                if (
                    use_capitulation
                    and capitulation_index is not None
                    and capitulation_price is not None
                    and target < 1.0
                    and index - capitulation_index <= cap_days
                    and close_price < capitulation_price
                ):
                    next_target = 1.0
                    next_reason = "CAPITULATION"
                    state = "invested"
                if state != "invested" and next_target < 1.0:
                    threshold = float(ma) * (1 - fb_pct)
                    fallback_streak = fallback_streak + 1 if close_price > threshold else 0
                    if fallback_streak >= FALLBACK_CONFIRM_DAYS:
                        next_target = 1.0
                        next_reason = "FALLBACK"
                        state = "invested"
        if next_target != target:
            target = next_target
            pending_target = next_target
            pending_reason = next_reason

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    summary = _stats(daily["TotalValue"], daily["Date"], initial_value=1.0)
    summary.update(
        {
            "orders": len(trades),
            "sell_orders": int((trades["Action"] == "SELL").sum()) if not trades.empty else 0,
            "total_fees_percent_of_initial": (
                float(trades["Fee"].sum() * 100) if not trades.empty else 0.0
            ),
        }
    )
    return SimulationResult(daily=daily, trades=trades, summary=summary)


def run_buy_hold(
    full: pd.DataFrame,
    *,
    start: str,
    end: str,
    fee_bps: float = ONE_WAY_FEE_BPS,
    slippage_bps: float = ONE_WAY_SLIPPAGE_BPS,
) -> dict[str, float | int]:
    frame = full.loc[full["Date"].between(start, end)].copy().reset_index(drop=True)
    open_price = float(frame.loc[0, "Adj Open"]) * (1 + slippage_bps / 10_000)
    capital_after_fee = 1.0 / (1 + fee_bps / 10_000)
    shares = capital_after_fee / open_price
    nav = shares * frame["Adj Close"].astype(float)
    result = _stats(nav, frame["Date"], initial_value=1.0)
    result.update({"orders": 1, "sell_orders": 0})
    return result


def main() -> None:
    prices = load_prices()
    print("[REPORTED SCRATCH SEMANTICS: baseline reconciliation]")
    for warmup in (False, True):
        label = "WARM_INDICATORS" if warmup else "COLD_2020_INDICATORS"
        print(f"\n{label}")
        for name, parameters in CONFIGS.items():
            summary, counts = run_reported_close_to_close(
                prices,
                start="2020-01-02",
                end="2026-07-31",
                warmup_before_start=warmup,
                **parameters,
            )
            print(name, summary, counts)

    print("\n[REPORTED SCRATCH SEMANTICS: one-way cost sensitivity]")
    for label, fee_bps, slippage_bps in (
        ("NO_COST", 0.0, 0.0),
        ("REPO_STANDARD_5BP_FEE_PLUS_5BP_SLIPPAGE", 5.0, 5.0),
        ("HIGH_5BP_FEE_PLUS_10BP_SLIPPAGE", 5.0, 10.0),
    ):
        print(f"\n{label}")
        for name, parameters in CONFIGS.items():
            summary, counts = run_reported_close_to_close(
                prices,
                start="2020-01-02",
                end="2026-07-31",
                warmup_before_start=False,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                **parameters,
            )
            print(name, summary, counts)

    print("\n[NEXT-OPEN CAUSAL AUDIT: repo-standard 5 bp fee + 5 bp slippage]")
    periods = {
        "DISCOVERY_STYLE_1994_2017": ("1994-01-03", "2017-12-29"),
        "HISTORICAL_HOLDOUT_2018_2019": ("2018-01-02", "2019-12-31"),
        "PRE_TUNING_COMBINED_1994_2019": ("1994-01-03", "2019-12-31"),
        "REPORTED_TUNING_WINDOW_2020_2026": ("2020-01-02", "2026-07-31"),
        "EARLY_SUBPERIOD_2020_2023": ("2020-01-02", "2023-12-29"),
        "CONTAMINATED_LATE_SUBPERIOD_2024_2026": ("2024-01-02", "2026-07-31"),
    }
    for period, (start, end) in periods.items():
        print(f"\n[{period}] {start}..{end}")
        for name, parameters in CONFIGS.items():
            result = run_next_open(prices, start=start, end=end, **parameters)
            print(name, result.summary)
        print("BUY_HOLD", run_buy_hold(prices, start=start, end=end))


if __name__ == "__main__":
    main()
