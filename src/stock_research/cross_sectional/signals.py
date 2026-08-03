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
            "UniverseMember",
            "Close",
            "Trend200",
            "Return126",
            "Drawdown126",
            "Volatility63",
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
    frame["EntryBlocked"] = (
        frame["Qualified"]
        & params.overheated_entry_enabled
        & pd.to_numeric(
            frame.get("Return126", np.nan),
            errors="coerce",
        ).ge(params.overheated_return126)
        & pd.to_numeric(
            frame.get("Trend200", np.nan),
            errors="coerce",
        ).ge(params.overheated_trend200)
        & pd.to_numeric(
            frame.get(
                "Drawdown126",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        ).ge(params.overheated_drawdown126_floor)
    )
    frame["EntryBlockReason"] = np.where(
        frame["EntryBlocked"],
        "OVERHEATED_ENTRY",
        None,
    )
    frame["Rank"] = (
        frame["AlphaScore"]
        .where(frame["Qualified"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    if compact:
        compact_columns = [
                "Date",
                "Ticker",
                "Close",
                "Trend200",
                "Return126",
                "Drawdown126",
                "Volatility63",
                "AlphaScore",
                "Qualified",
                "Rank",
                "EntryBlocked",
                "EntryBlockReason",
            ]
        return frame.loc[
            :,
            [column for column in compact_columns if column in frame],
        ]
    return frame


def generate_rebalance_targets(
    scored_signal_days: pd.DataFrame,
    params: StrategyParams,
    *,
    compact: bool = False,
    force_universe_exit: bool = False,
) -> pd.DataFrame:
    """Create stateful top-K targets with an exit-rank no-trade band."""

    frame = scored_signal_days.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)
    held: set[str] = set()
    entry_reference_prices: dict[str, float] = {}
    peak_reference_prices: dict[str, float] = {}
    holding_rebalances: dict[str, int] = {}
    profit_exit_streaks: dict[str, int] = {}
    records: list[pd.DataFrame] = []
    for date, group in frame.groupby("Date", sort=True):
        ranked = group.loc[group["Qualified"]].sort_values(
            ["Rank", "Ticker"]
        )
        rank_by_ticker = ranked.set_index("Ticker")["Rank"].to_dict()
        rows_by_ticker = group.set_index("Ticker", drop=False)
        exit_reasons: dict[str, str] = {}
        reference_returns: dict[str, float] = {}
        peak_reference_returns: dict[str, float] = {}
        trailing_drawdowns: dict[str, float] = {}
        replacement_advantages: dict[str, float] = {}
        profit_exit_streak_values: dict[str, int] = {}
        available_challengers = ranked.loc[
            ~ranked["Ticker"].isin(held)
            & ~ranked.get(
                "EntryBlocked",
                pd.Series(False, index=ranked.index),
            ).fillna(False)
        ]
        best_replacement_score = (
            float(available_challengers["AlphaScore"].max())
            if (
                not available_challengers.empty
                and "AlphaScore" in available_challengers
            )
            else None
        )
        retained: set[str] = set()
        for ticker in held:
            row = (
                rows_by_ticker.loc[ticker]
                if ticker in rows_by_ticker.index
                else None
            )
            entry_price = entry_reference_prices.get(ticker)
            current_close = _row_number(row, "Close")
            prior_peak = peak_reference_prices.get(ticker)
            current_peak = _updated_peak(prior_peak, current_close)
            if current_peak is not None:
                peak_reference_prices[ticker] = current_peak
            reference_return = _reference_return(
                current_close,
                entry_price,
            )
            if reference_return is not None:
                reference_returns[ticker] = reference_return
            peak_reference_return = _reference_return(
                current_peak,
                entry_price,
            )
            if peak_reference_return is not None:
                peak_reference_returns[ticker] = peak_reference_return
            trailing_drawdown = _reference_return(
                current_close,
                current_peak,
            )
            if trailing_drawdown is not None:
                trailing_drawdowns[ticker] = trailing_drawdown
            current_score = _row_number(row, "AlphaScore")
            replacement_advantage = _replacement_advantage(
                best_replacement_score,
                current_score,
            )
            if replacement_advantage is not None:
                replacement_advantages[ticker] = replacement_advantage
            is_profit_exit_candidate = _profit_rotation_candidate(
                row,
                rank_by_ticker.get(ticker, np.inf),
                reference_return,
                replacement_advantage,
                params,
            )
            profit_exit_streak = (
                profit_exit_streaks.get(ticker, 0) + 1
                if is_profit_exit_candidate
                else 0
            )
            profit_exit_streaks[ticker] = profit_exit_streak
            profit_exit_streak_values[ticker] = profit_exit_streak
            exit_reason = _exit_reason(
                row,
                rank_by_ticker.get(ticker, np.inf),
                reference_return,
                holding_rebalances.get(ticker, 1),
                params,
                force_universe_exit=force_universe_exit,
                peak_reference_price=current_peak,
                peak_reference_return=peak_reference_return,
                profit_exit_streak=profit_exit_streak,
                replacement_advantage=replacement_advantage,
            )
            if exit_reason is None:
                retained.add(ticker)
            else:
                exit_reasons[ticker] = exit_reason
        selected = set(retained)
        for ticker in ranked["Ticker"]:
            if len(selected) >= params.top_k:
                break
            if ticker in exit_reasons:
                continue
            row = rows_by_ticker.loc[ticker]
            if ticker not in held and _entry_block_reason(row, params):
                continue
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
            peak_reference_prices[ticker] = (
                _row_number(row, "Close") or np.nan
            )
            holding_rebalances[ticker] = 1
            profit_exit_streaks[ticker] = 0
            profit_exit_streak_values[ticker] = 0
            reference_returns[ticker] = 0.0
            peak_reference_returns[ticker] = 0.0
        position_weights = _position_weights(
            held, rows_by_ticker, params.weighting_scheme
        )

        if compact:
            selected_tickers = sorted(held)
            if selected_tickers:
                current = pd.DataFrame(
                    {
                        "Date": pd.Timestamp(date),
                        "Ticker": selected_tickers,
                        "TargetWeight": [
                            position_weights[ticker]
                            for ticker in selected_tickers
                        ],
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
                peak_reference_prices.pop(ticker, None)
                holding_rebalances.pop(ticker, None)
                profit_exit_streaks.pop(ticker, None)
            continue

        current = group.copy()
        current["TargetWeight"] = (
            current["Ticker"].map(position_weights).fillna(0.0)
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
        current["PeakReferencePrice"] = current["Ticker"].map(
            peak_reference_prices
        )
        current["PeakReferenceReturn"] = current["Ticker"].map(
            peak_reference_returns
        )
        current["TrailingDrawdown"] = current["Ticker"].map(
            trailing_drawdowns
        )
        current["BestReplacementAlphaScore"] = best_replacement_score
        current["ReplacementScoreAdvantage"] = current["Ticker"].map(
            replacement_advantages
        )
        current["ProfitExitStreak"] = current["Ticker"].map(
            profit_exit_streak_values
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
            peak_reference_prices.pop(ticker, None)
            holding_rebalances.pop(ticker, None)
            profit_exit_streaks.pop(ticker, None)
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


def _position_weights(
    held: set[str],
    rows_by_ticker: pd.DataFrame,
    scheme: str,
) -> dict[str, float]:
    """Split 100% target exposure across held tickers by the chosen scheme.

    Falls back to equal weight whenever a required input is missing for any
    held ticker, so a partial data gap never produces an unweighted or
    negative allocation.
    """

    if not held:
        return {}
    if scheme == "equal":
        weight = 1.0 / len(held)
        return {ticker: weight for ticker in held}

    def _lookup(ticker: str, column: str) -> float:
        row = (
            rows_by_ticker.loc[ticker]
            if ticker in rows_by_ticker.index
            else None
        )
        return _row_number(row, column)

    if scheme == "score_proportional":
        scores = {ticker: _lookup(ticker, "AlphaScore") for ticker in held}
        if any(pd.isna(value) for value in scores.values()):
            weight = 1.0 / len(held)
            return {ticker: weight for ticker in held}
        floor = min(scores.values())
        shifted = {
            ticker: value - floor + 1e-6 for ticker, value in scores.items()
        }
        total = sum(shifted.values())
        if total <= 0:
            weight = 1.0 / len(held)
            return {ticker: weight for ticker in held}
        return {ticker: value / total for ticker, value in shifted.items()}

    if scheme == "inverse_volatility":
        volatility = {
            ticker: _lookup(ticker, "Volatility63") for ticker in held
        }
        if any(
            pd.isna(value) or value <= 0 for value in volatility.values()
        ):
            weight = 1.0 / len(held)
            return {ticker: weight for ticker in held}
        inverse = {ticker: 1.0 / value for ticker, value in volatility.items()}
        total = sum(inverse.values())
        return {ticker: value / total for ticker, value in inverse.items()}

    raise ValueError(f"Unknown weighting_scheme: {scheme}")


def _exit_reason(
    row: pd.Series | pd.DataFrame | None,
    rank: float,
    reference_return: float | None,
    held_rebalances: int,
    params: StrategyParams,
    *,
    force_universe_exit: bool = False,
    peak_reference_price: float | None = None,
    peak_reference_return: float | None = None,
    profit_exit_streak: int = 0,
    replacement_advantage: float | None = None,
) -> str | None:
    if (
        force_universe_exit
        and row is not None
        and "UniverseMember" in row
        and not _row_bool(row, "UniverseMember")
    ):
        return "UNIVERSE_EXIT"
    qualified = bool(row is not None and _row_bool(row, "Qualified"))
    rank_value = float(rank) if pd.notna(rank) else np.inf
    ordinary_exit = not qualified or rank_value > params.exit_rank

    if not params.loss_aware_exit_enabled:
        return "RANK_OR_FILTER_EXIT" if ordinary_exit else None
    if reference_return is None:
        return "MISSING_REFERENCE_EXIT" if ordinary_exit else None
    if reference_return <= params.hard_stop_return:
        return "HARD_STOP"
    current_close = _row_number(row, "Close")
    trailing_drawdown = _reference_return(
        current_close,
        peak_reference_price,
    )
    if (
        params.trailing_stop_enabled
        and peak_reference_price is not None
        and peak_reference_return is not None
        and peak_reference_return >= params.trailing_stop_activation_gain
        and trailing_drawdown is not None
        and trailing_drawdown <= params.trailing_stop_drawdown
    ):
        return "TRAILING_STOP"
    if reference_return >= params.minimum_exit_gain:
        if _profit_rotation_candidate(
            row,
            rank_value,
            reference_return,
            replacement_advantage,
            params,
        ) and (
            profit_exit_streak
            >= params.profit_rotation_confirmation_rebalances
        ):
            return "PROFITABLE_ROTATION"
        return None
    if not ordinary_exit:
        return None
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


def _profit_rotation_candidate(
    row: pd.Series | pd.DataFrame | None,
    rank: float,
    reference_return: float | None,
    replacement_advantage: float | None,
    params: StrategyParams,
) -> bool:
    if reference_return is None or reference_return < params.minimum_exit_gain:
        return False
    qualified = bool(row is not None and _row_bool(row, "Qualified"))
    rank_value = float(rank) if pd.notna(rank) else np.inf
    exit_rank = params.profit_rotation_exit_rank or params.exit_rank
    if qualified and rank_value <= exit_rank:
        return False
    if params.replacement_score_advantage <= 0:
        return True
    return bool(
        replacement_advantage is not None
        and replacement_advantage >= params.replacement_score_advantage
    )


def _replacement_advantage(
    replacement_score: float | None,
    current_score: float | None,
) -> float | None:
    if (
        replacement_score is None
        or current_score is None
        or pd.isna(replacement_score)
        or pd.isna(current_score)
    ):
        return None
    return float(replacement_score - current_score)


def _updated_peak(
    prior_peak: float | None,
    current_close: float | None,
) -> float | None:
    values = [
        float(value)
        for value in (prior_peak, current_close)
        if value is not None and pd.notna(value) and value > 0
    ]
    return max(values) if values else None


def _entry_block_reason(
    row: pd.Series | pd.DataFrame | None,
    params: StrategyParams,
) -> str | None:
    if row is None or not params.overheated_entry_enabled:
        return None
    if _row_bool(row, "EntryBlocked"):
        return "OVERHEATED_ENTRY"
    return None


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
    value = row.get(column, np.nan)
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _row_bool(
    row: pd.Series | pd.DataFrame,
    column: str,
) -> bool:
    value = row.get(column, False)
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
            diagnostic_maps = {
                column: scheduled[column].dropna().to_dict()
                for column in (
                    "PeakReferencePrice",
                    "PeakReferenceReturn",
                    "TrailingDrawdown",
                    "BestReplacementAlphaScore",
                    "ReplacementScoreAdvantage",
                    "ProfitExitStreak",
                )
                if column in scheduled
            }
        else:
            action_map = {}
            scheduled_entry_prices = {}
            scheduled_holding_counts = {}
            exit_reason_map = {}
            diagnostic_maps = {}
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
        for column, values in diagnostic_maps.items():
            current[column] = current["Ticker"].map(values)
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
                bool(entry_blocked),
            )
            for ticker, qualified, rank, entry_blocked in zip(
                current["Ticker"],
                current["Qualified"],
                current["Rank"],
                current.get(
                    "EntryBlocked",
                    pd.Series(False, index=current.index),
                ),
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
    if period.empty:
        return period
    weekday_names = ("MON", "TUE", "WED", "THU", "FRI")
    coverage = (
        period.groupby("Date", as_index=False)["Ticker"]
        .nunique()
        .rename(columns={"Ticker": "CrossSectionCoverage"})
    )
    coverage["Week"] = coverage["Date"].dt.to_period(
        f"W-{weekday_names[rebalance_weekday]}"
    )
    dates = pd.DatetimeIndex(
        coverage.sort_values(
            ["Week", "CrossSectionCoverage", "Date"],
            ascending=[True, False, False],
        )
        .drop_duplicates("Week", keep="first")["Date"]
        .sort_values()
    )
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
    entry_blocked: bool = False,
) -> str:
    if trade_action in {"BUY", "SELL"}:
        return trade_action
    if selected:
        if not qualified or pd.isna(rank) or rank > exit_rank:
            return "REDUCE_WATCH"
        return "HOLD"
    if entry_blocked:
        return "ENTRY_BLOCKED"
    if qualified and pd.notna(rank) and rank <= top_k:
        return "BUY_WATCH"
    return "AVOID"
