from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyParams

FACTOR_COLUMNS = (
    "MomentumFactor",
    "TrendFactor",
    "GrowthFactor",
    "QualityFactor",
    "RiskControlFactor",
)


def weekly_signal_dates(
    dates: pd.Series | pd.DatetimeIndex,
    rebalance_weekday: int = 4,
) -> pd.DatetimeIndex:
    values = pd.Series(pd.to_datetime(dates).unique()).dropna().sort_values()
    if values.empty:
        return pd.DatetimeIndex([])
    weekday_names = ("MON", "TUE", "WED", "THU", "FRI")
    weekly = values.groupby(
        values.dt.to_period(f"W-{weekday_names[rebalance_weekday]}")
    ).max()
    return pd.DatetimeIndex(weekly.to_numpy())


def score_panel(
    panel: pd.DataFrame,
    params: StrategyParams,
    *,
    compact: bool = False,
    presorted: bool = False,
) -> pd.DataFrame:
    """Apply one frozen parameter set to a feature panel."""

    if compact:
        columns = [
            "Date",
            "Ticker",
            "Eligible",
            "Close",
            "Trend200",
            "Return126",
            *FACTOR_COLUMNS,
        ]
        frame = panel.loc[
            :,
            [column for column in columns if column in panel.columns],
        ].copy()
        if "Close" not in frame:
            frame["Close"] = np.nan
    else:
        frame = panel.copy()
    if presorted:
        frame = frame.reset_index(drop=True)
    else:
        frame = frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    score = pd.Series(0.0, index=frame.index)
    for column, weight in params.factor_weights.items():
        factor = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        score = score + factor * weight
    frame["AlphaScore"] = score.where(frame["Eligible"])
    frame["Qualified"] = (
        frame["Eligible"]
        & frame["Trend200"].ge(params.trend_floor)
        & frame["Return126"].ge(params.momentum_floor)
    )
    frame["Rank"] = (
        frame["AlphaScore"]
        .where(frame["Qualified"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    if compact:
        return frame.loc[
            :,
            [
                "Date",
                "Ticker",
                "Close",
                "Trend200",
                "Return126",
                "Qualified",
                "Rank",
            ],
        ]
    return frame


def generate_rebalance_targets(
    scored_signal_days: pd.DataFrame,
    params: StrategyParams,
    *,
    compact: bool = False,
) -> pd.DataFrame:
    """Create stateful top-K targets with an exit-rank no-trade band."""

    frame = scored_signal_days.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)
    held: set[str] = set()
    entry_reference_prices: dict[str, float] = {}
    holding_rebalances: dict[str, int] = {}
    records: list[pd.DataFrame] = []
    for date, group in frame.groupby("Date", sort=True):
        ranked = group.loc[group["Qualified"]].sort_values(
            ["Rank", "Ticker"]
        )
        rank_by_ticker = ranked.set_index("Ticker")["Rank"].to_dict()
        rows_by_ticker = group.set_index("Ticker", drop=False)
        exit_reasons: dict[str, str] = {}
        reference_returns: dict[str, float] = {}
        retained: set[str] = set()
        for ticker in held:
            row = (
                rows_by_ticker.loc[ticker]
                if ticker in rows_by_ticker.index
                else None
            )
            entry_price = entry_reference_prices.get(ticker)
            current_close = _row_number(row, "Close")
            reference_return = _reference_return(
                current_close,
                entry_price,
            )
            if reference_return is not None:
                reference_returns[ticker] = reference_return
            exit_reason = _exit_reason(
                row,
                rank_by_ticker.get(ticker, np.inf),
                reference_return,
                holding_rebalances.get(ticker, 1),
                params,
            )
            if exit_reason is None:
                retained.add(ticker)
            else:
                exit_reasons[ticker] = exit_reason
        selected = set(retained)
        for ticker in ranked["Ticker"]:
            if len(selected) >= params.top_k:
                break
            selected.add(str(ticker))
        prior = set(held)
        held = selected
        added = held - prior
        removed = prior - held
        for ticker in retained:
            holding_rebalances[ticker] = (
                holding_rebalances.get(ticker, 1) + 1
            )
        for ticker in added:
            row = (
                rows_by_ticker.loc[ticker]
                if ticker in rows_by_ticker.index
                else None
            )
            entry_reference_prices[ticker] = (
                _row_number(row, "Close") or np.nan
            )
            holding_rebalances[ticker] = 1
            reference_returns[ticker] = 0.0
        target_weight = 1.0 / len(held) if held else 0.0

        if compact:
            selected_tickers = sorted(held)
            if selected_tickers:
                current = pd.DataFrame(
                    {
                        "Date": pd.Timestamp(date),
                        "Ticker": selected_tickers,
                        "TargetWeight": target_weight,
                        "SignalDate": pd.Timestamp(date),
                    }
                )
            else:
                current = pd.DataFrame(
                    {
                        "Date": [pd.Timestamp(date)],
                        "Ticker": [str(group["Ticker"].iloc[0])],
                        "TargetWeight": [0.0],
                        "SignalDate": [pd.Timestamp(date)],
                    }
                )
            records.append(current)
            for ticker in removed:
                entry_reference_prices.pop(ticker, None)
                holding_rebalances.pop(ticker, None)
            continue

        current = group.copy()
        current["TargetWeight"] = np.where(
            current["Ticker"].isin(held),
            target_weight,
            0.0,
        )
        current["ModelSelected"] = current["Ticker"].isin(held)
        current["EntryReferencePrice"] = current["Ticker"].map(
            entry_reference_prices
        )
        current["SignalReferenceReturn"] = current["Ticker"].map(
            reference_returns
        )
        current["HoldingRebalances"] = current["Ticker"].map(
            holding_rebalances
        )
        current["ExitReason"] = current["Ticker"].map(exit_reasons)
        current["TradeAction"] = [
            _trade_action(
                str(ticker),
                prior,
                held,
                bool(qualified),
                rank,
                params.top_k,
            )
            for ticker, qualified, rank in zip(
                current["Ticker"],
                current["Qualified"],
                current["Rank"],
                strict=True,
            )
        ]
        current["SignalDate"] = pd.Timestamp(date)
        records.append(current)
        for ticker in removed:
            entry_reference_prices.pop(ticker, None)
            holding_rebalances.pop(ticker, None)
    if not records:
        if compact:
            return pd.DataFrame(
                columns=["Date", "Ticker", "TargetWeight", "SignalDate"]
            )
        columns = list(frame.columns) + [
            "TargetWeight",
            "ModelSelected",
            "TradeAction",
            "SignalDate",
        ]
        return pd.DataFrame(columns=columns)
    return pd.concat(records, ignore_index=True)


def _exit_reason(
    row: pd.Series | pd.DataFrame | None,
    rank: float,
    reference_return: float | None,
    held_rebalances: int,
    params: StrategyParams,
) -> str | None:
    qualified = bool(row is not None and _row_bool(row, "Qualified"))
    rank_value = float(rank) if pd.notna(rank) else np.inf
    ordinary_exit = not qualified or rank_value > params.exit_rank

    if not params.loss_aware_exit_enabled:
        return "RANK_OR_FILTER_EXIT" if ordinary_exit else None
    if reference_return is None:
        return "MISSING_REFERENCE_EXIT" if ordinary_exit else None
    if reference_return <= params.hard_stop_return:
        return "HARD_STOP"
    if not ordinary_exit:
        return None
    if reference_return >= params.minimum_exit_gain:
        return "PROFITABLE_ROTATION"
    if held_rebalances < params.minimum_hold_rebalances:
        return None

    severe_breakdown = bool(
        rank_value > params.conviction_exit_rank
        and _row_number(row, "Trend200")
        <= params.conviction_trend_floor
        and _row_number(row, "Return126")
        <= params.conviction_momentum_floor
    )
    return "CONVICTION_BREAKDOWN" if severe_breakdown else None


def _reference_return(
    current_close: float | None,
    entry_price: float | None,
) -> float | None:
    if (
        current_close is None
        or entry_price is None
        or pd.isna(current_close)
        or pd.isna(entry_price)
        or entry_price <= 0
    ):
        return None
    return current_close / entry_price - 1


def _row_number(
    row: pd.Series | pd.DataFrame | None,
    column: str,
) -> float:
    if row is None:
        return np.nan
    value = row[column] if column in row else np.nan
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _row_bool(
    row: pd.Series | pd.DataFrame,
    column: str,
) -> bool:
    value = row[column] if column in row else False
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return bool(value)


def generate_equal_weight_targets(
    signal_days: pd.DataFrame,
) -> pd.DataFrame:
    """Build a same-universe equal-weight benchmark on the same schedule."""

    records: list[pd.DataFrame] = []
    for date, group in signal_days.groupby("Date", sort=True):
        current = group[["Date", "Ticker", "Eligible"]].copy()
        count = int(current["Eligible"].sum())
        current["TargetWeight"] = np.where(
            current["Eligible"] & (count > 0),
            1.0 / count if count else 0.0,
            0.0,
        )
        current["SignalDate"] = pd.Timestamp(date)
        records.append(current)
    if not records:
        return pd.DataFrame(
            columns=["Date", "Ticker", "Eligible", "TargetWeight", "SignalDate"]
        )
    return pd.concat(records, ignore_index=True)


def build_daily_recommendations(
    scored_daily: pd.DataFrame,
    targets: pd.DataFrame,
    params: StrategyParams,
) -> pd.DataFrame:
    """Expand weekly executable targets into a human-readable daily signal table."""

    frame = scored_daily.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True).copy()
    target_by_date = {
        pd.Timestamp(date): group.set_index("Ticker")
        for date, group in targets.groupby("Date", sort=True)
    }
    selected: set[str] = set()
    weights: dict[str, float] = {}
    entry_reference_prices: dict[str, float] = {}
    holding_counts: dict[str, int] = {}
    outputs: list[pd.DataFrame] = []
    for date, group in frame.groupby("Date", sort=True):
        date = pd.Timestamp(date)
        current = group.copy()
        scheduled = target_by_date.get(date)
        is_rebalance = scheduled is not None
        if scheduled is not None:
            weights = {
                str(ticker): float(weight)
                for ticker, weight in scheduled["TargetWeight"].items()
                if float(weight) > 0
            }
            selected = set(weights)
            action_map = scheduled["TradeAction"].to_dict()
            entry_reference_prices = {
                str(ticker): float(price)
                for ticker, price in scheduled[
                    "EntryReferencePrice"
                ].items()
                if ticker in selected and pd.notna(price)
            }
            holding_counts = {
                str(ticker): int(count)
                for ticker, count in scheduled["HoldingRebalances"].items()
                if ticker in selected and pd.notna(count)
            }
            scheduled_entry_prices = (
                scheduled["EntryReferencePrice"].dropna().to_dict()
            )
            scheduled_holding_counts = (
                scheduled["HoldingRebalances"].dropna().to_dict()
            )
            exit_reason_map = scheduled["ExitReason"].dropna().to_dict()
        else:
            action_map = {}
            scheduled_entry_prices = {}
            scheduled_holding_counts = {}
            exit_reason_map = {}
        current["IsRebalanceSignal"] = is_rebalance
        current["ModelSelected"] = current["Ticker"].isin(selected)
        current["TargetWeight"] = current["Ticker"].map(weights).fillna(0.0)
        current["TradeAction"] = current["Ticker"].map(action_map).fillna("NONE")
        current["EntryReferencePrice"] = current["Ticker"].map(
            entry_reference_prices
        )
        if is_rebalance:
            current["EntryReferencePrice"] = current[
                "EntryReferencePrice"
            ].fillna(current["Ticker"].map(scheduled_entry_prices))
        current["SignalReferenceReturn"] = (
            pd.to_numeric(current["Close"], errors="coerce")
            / current["EntryReferencePrice"]
            - 1
        ).where(
            current["ModelSelected"]
            | current["TradeAction"].eq("SELL")
        )
        current["HoldingRebalances"] = current["Ticker"].map(
            holding_counts
        )
        if is_rebalance:
            current["HoldingRebalances"] = current[
                "HoldingRebalances"
            ].fillna(current["Ticker"].map(scheduled_holding_counts))
        current["ExitReason"] = current["Ticker"].map(exit_reason_map)
        current["DailySignal"] = [
            _daily_signal(
                str(ticker),
                ticker in selected,
                bool(qualified),
                rank,
                params.top_k,
                params.exit_rank,
                action_map.get(str(ticker)),
            )
            for ticker, qualified, rank in zip(
                current["Ticker"],
                current["Qualified"],
                current["Rank"],
                strict=True,
            )
        ]
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True) if outputs else frame


def signal_day_panel(
    panel: pd.DataFrame,
    start: str,
    end: str,
    rebalance_weekday: int = 4,
) -> pd.DataFrame:
    period = panel.loc[panel["Date"].between(start, end)].copy()
    dates = weekly_signal_dates(period["Date"], rebalance_weekday)
    return period.loc[period["Date"].isin(dates)].copy()


def _trade_action(
    ticker: str,
    prior: set[str],
    selected: set[str],
    qualified: bool,
    rank: float,
    top_k: int,
) -> str:
    if ticker not in prior and ticker in selected:
        return "BUY"
    if ticker in prior and ticker not in selected:
        return "SELL"
    if ticker in selected:
        return "HOLD"
    if qualified and pd.notna(rank) and rank <= top_k:
        return "WATCH"
    return "AVOID"


def _daily_signal(
    ticker: str,
    selected: bool,
    qualified: bool,
    rank: float,
    top_k: int,
    exit_rank: int,
    trade_action: str | None,
) -> str:
    if trade_action in {"BUY", "SELL"}:
        return trade_action
    if selected:
        if not qualified or pd.isna(rank) or rank > exit_rank:
            return "REDUCE_WATCH"
        return "HOLD"
    if qualified and pd.notna(rank) and rank <= top_k:
        return "BUY_WATCH"
    return "AVOID"
