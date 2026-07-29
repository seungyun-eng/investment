from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioSummary:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    total_injected: float
    final_value: float
    roi_percent: float
    cagr_percent: float
    max_drawdown_percent: float
    sharpe_ratio: float
    turnover_multiple: float
    annualized_turnover: float
    ticker_trades: int
    rebalance_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "StartDate": self.start_date,
            "EndDate": self.end_date,
            "InitialCapital": self.initial_capital,
            "TotalInjected": self.total_injected,
            "FinalValue": self.final_value,
            "ROI": self.roi_percent,
            "CAGR": self.cagr_percent,
            "MaxDrawdown": self.max_drawdown_percent,
            "Sharpe": self.sharpe_ratio,
            "TurnoverMultiple": self.turnover_multiple,
            "AnnualizedTurnover": self.annualized_turnover,
            "TickerTrades": self.ticker_trades,
            "Rebalances": self.rebalance_count,
        }


@dataclass(frozen=True)
class PortfolioResult:
    daily: pd.DataFrame
    executions: pd.DataFrame
    summary: PortfolioSummary
    attribution: pd.DataFrame | None = None
    delistings: pd.DataFrame | None = None


@dataclass(frozen=True)
class PreparedMarket:
    start: str
    end: str
    open_prices: pd.DataFrame
    valuation_prices: pd.DataFrame
    fallback_trade_prices: pd.DataFrame
    dates: pd.DatetimeIndex
    tickers: pd.Index


def prepare_market(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> PreparedMarket:
    period = panel.loc[panel["Date"].between(start, end)].copy()
    if period.empty:
        raise ValueError(f"No price data between {start} and {end}")
    open_prices = period.pivot(index="Date", columns="Ticker", values="Open")
    close_prices = period.pivot(index="Date", columns="Ticker", values="Close")
    dates = pd.DatetimeIndex(close_prices.index).sort_values()
    tickers = close_prices.columns.sort_values()
    open_prices = open_prices.reindex(index=dates, columns=tickers)
    valuation_prices = close_prices.reindex(
        index=dates,
        columns=tickers,
    ).ffill()
    return PreparedMarket(
        start=start,
        end=end,
        open_prices=open_prices,
        valuation_prices=valuation_prices,
        fallback_trade_prices=valuation_prices.shift(1),
        dates=dates,
        tickers=tickers,
    )


def run_portfolio_backtest(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    start: str,
    end: str,
    initial_capital: float,
    transaction_cost_bps: float,
    prepared_market: PreparedMarket | None = None,
    record_attribution: bool = False,
    delisting_events: pd.DataFrame | None = None,
) -> PortfolioResult:
    """Execute close-generated targets at the next available session's open."""

    market = prepared_market or prepare_market(panel, start=start, end=end)
    if market.start != start or market.end != end:
        raise ValueError("Prepared market period does not match requested period")
    open_prices = market.open_prices
    valuation_prices = market.valuation_prices
    fallback_trade_prices = market.fallback_trade_prices
    dates = market.dates
    tickers = market.tickers
    open_values = open_prices.to_numpy(dtype=float, copy=False)
    valuation_values = valuation_prices.to_numpy(dtype=float, copy=False)
    fallback_values = fallback_trade_prices.to_numpy(dtype=float, copy=False)

    period_targets = targets.loc[targets["Date"].between(start, end)].copy()
    execution_schedule = _execution_schedule(period_targets, dates, tickers)
    delisting_schedule = _delisting_schedule(
        delisting_events,
        dates,
        tickers,
    )

    shares = np.zeros(len(tickers), dtype=float)
    permanently_untradable = np.zeros(len(tickers), dtype=bool)
    cash = float(initial_capital)
    total_turnover = 0.0
    ticker_trades = 0
    rebalance_count = 0
    daily_records: list[dict[str, object]] = []
    execution_records: list[dict[str, object]] = []
    attribution_records: list[dict[str, object]] = []
    delisting_records: list[dict[str, object]] = []
    cost_rate = transaction_cost_bps / 10_000
    previous_equity = float(initial_capital)

    for position, date in enumerate(dates):
        open_row = np.where(
            np.isnan(open_values[position]),
            fallback_values[position],
            open_values[position],
        )
        close_row = valuation_values[position]
        previous_close = (
            np.where(
                np.isnan(valuation_values[position - 1]),
                0.0,
                valuation_values[position - 1],
            )
            if position > 0
            else np.zeros(len(tickers), dtype=float)
        )
        delisting_pnl = np.zeros(len(tickers), dtype=float)
        for ticker_index, delisting_return, source_date in (
            delisting_schedule.get(date, [])
        ):
            held_shares = float(shares[ticker_index])
            prior_price = float(previous_close[ticker_index])
            if held_shares != 0 and prior_price <= 0:
                raise RuntimeError(
                    "Cannot settle a held delisted position without a valid "
                    f"prior close: {tickers[ticker_index]} on {date.date()}"
                )
            settlement_price = (
                prior_price * (1.0 + delisting_return)
                if prior_price > 0
                else 0.0
            )
            proceeds = held_shares * settlement_price
            delisting_pnl[ticker_index] = (
                held_shares * (settlement_price - prior_price)
            )
            cash += proceeds
            shares[ticker_index] = 0.0
            permanently_untradable[ticker_index] = True
            delisting_records.append(
                {
                    "EffectiveDate": date,
                    "SourceEffectiveDate": source_date,
                    "Ticker": str(tickers[ticker_index]),
                    "DelistingReturn": delisting_return,
                    "PriorClose": prior_price,
                    "SettlementPrice": settlement_price,
                    "SharesSettled": held_shares,
                    "Proceeds": proceeds,
                    "DelistingPnL": delisting_pnl[ticker_index],
                }
            )
        open_row[permanently_untradable] = np.nan
        open_for_valuation = np.where(np.isnan(open_row), 0.0, open_row)
        close_for_valuation = np.where(np.isnan(close_row), 0.0, close_row)
        if record_attribution and position > 0:
            overnight_pnl = shares * (
                open_for_valuation - previous_close
            )
            overnight_pnl += delisting_pnl
        else:
            overnight_pnl = delisting_pnl.copy()
        transaction_cost_by_ticker = np.zeros(
            len(tickers),
            dtype=float,
        )
        if date in execution_schedule:
            scheduled_target = execution_schedule[date]
            target = scheduled_target.to_numpy(dtype=float, copy=True)
            tradable = (open_row > 0) & ~np.isnan(open_row)
            if (~tradable).any():
                target = np.where(tradable, target, 0.0)
                target_sum = float(target.sum())
                if target_sum > 0:
                    target = target / target_sum
            pre_trade_equity = cash + float(
                np.sum(shares * open_for_valuation)
            )
            current_notional = shares * open_for_valuation
            desired_notional = target * pre_trade_equity
            changes = desired_notional - current_notional
            turnover = float(np.abs(changes).sum())
            transaction_cost = turnover * cost_rate
            if record_attribution and turnover > 0:
                transaction_cost_by_ticker = (
                    np.abs(changes) / turnover * transaction_cost
                )
            net_equity = max(pre_trade_equity - transaction_cost, 0.0)
            desired_after_cost = target * net_equity
            changed = np.abs(changes) > max(pre_trade_equity, 1.0) * 1e-8
            ticker_trades += int(changed.sum())
            rebalance_count += 1
            total_turnover += turnover
            shares = np.divide(
                desired_after_cost,
                open_row,
                out=np.zeros_like(desired_after_cost),
                where=tradable,
            )
            cash = net_equity - float(desired_after_cost.sum())
            execution_records.append(
                {
                    "ExecutionDate": date,
                    "SignalDate": scheduled_target.attrs["SignalDate"],
                    "PreTradeEquity": pre_trade_equity,
                    "Turnover": turnover,
                    "TransactionCost": transaction_cost,
                    "SelectedCount": int((target > 0).sum()),
                }
            )
        equity = cash + float(np.sum(shares * close_for_valuation))
        if record_attribution:
            intraday_pnl = shares * (
                close_for_valuation - open_for_valuation
            )
            gross_price_pnl = overnight_pnl + intraday_pnl
            net_pnl = gross_price_pnl - transaction_cost_by_ticker
            expected_pnl = equity - previous_equity
            reconciliation_error = expected_pnl - float(net_pnl.sum())
            tolerance = 1e-8 * max(abs(equity), 1.0)
            if abs(reconciliation_error) > tolerance:
                raise RuntimeError(
                    "Ticker attribution does not reconcile to portfolio "
                    f"equity on {date.date()}: {reconciliation_error}"
                )
            ending_notional = shares * close_for_valuation
            active = (
                (np.abs(gross_price_pnl) > 1e-12)
                | (transaction_cost_by_ticker > 1e-12)
                | (np.abs(ending_notional) > 1e-12)
            )
            for ticker_index in np.flatnonzero(active):
                attribution_records.append(
                    {
                        "Date": date,
                        "Ticker": str(tickers[ticker_index]),
                        "OvernightPnL": overnight_pnl[ticker_index],
                        "IntradayPnL": intraday_pnl[ticker_index],
                        "GrossPricePnL": gross_price_pnl[ticker_index],
                        "DelistingPnL": delisting_pnl[ticker_index],
                        "TransactionCost": (
                            transaction_cost_by_ticker[ticker_index]
                        ),
                        "NetPnL": net_pnl[ticker_index],
                        "EndingNotional": ending_notional[ticker_index],
                        "EndingWeight": (
                            ending_notional[ticker_index] / equity
                            if equity > 0
                            else 0.0
                        ),
                    }
                )
        daily_records.append(
            {
                "Date": date,
                "Equity": equity,
                "Cash": cash,
                "GrossExposure": (
                    float(np.sum(shares * close_for_valuation)) / equity
                    if equity > 0
                    else 0.0
                ),
                "SelectedCount": int((shares > 0).sum()),
            }
        )
        previous_equity = equity

    daily = pd.DataFrame(daily_records)
    summary = _summarize(
        daily,
        initial_capital=initial_capital,
        total_turnover=total_turnover,
        ticker_trades=ticker_trades,
        rebalance_count=rebalance_count,
    )
    return PortfolioResult(
        daily=daily,
        executions=pd.DataFrame(execution_records),
        summary=summary,
        attribution=(
            pd.DataFrame(
                attribution_records,
                columns=[
                    "Date",
                    "Ticker",
                    "OvernightPnL",
                    "IntradayPnL",
                    "GrossPricePnL",
                    "DelistingPnL",
                    "TransactionCost",
                    "NetPnL",
                    "EndingNotional",
                    "EndingWeight",
                ],
            )
            if record_attribution
            else None
        ),
        delistings=pd.DataFrame(
            delisting_records,
            columns=[
                "EffectiveDate",
                "SourceEffectiveDate",
                "Ticker",
                "DelistingReturn",
                "PriorClose",
                "SettlementPrice",
                "SharesSettled",
                "Proceeds",
                "DelistingPnL",
            ],
        ),
    )


def _execution_schedule(
    targets: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> dict[pd.Timestamp, pd.Series]:
    schedule: dict[pd.Timestamp, pd.Series] = {}
    date_positions = {date: index for index, date in enumerate(dates)}
    for signal_date, group in targets.groupby("Date", sort=True):
        signal_date = pd.Timestamp(signal_date)
        position = date_positions.get(signal_date)
        if position is None or position + 1 >= len(dates):
            continue
        execution_date = dates[position + 1]
        weights = (
            group.set_index("Ticker")["TargetWeight"]
            .astype(float)
            .reindex(tickers, fill_value=0.0)
        )
        weights.attrs["SignalDate"] = signal_date
        schedule[execution_date] = weights
    return schedule


def _delisting_schedule(
    events: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> dict[pd.Timestamp, list[tuple[int, float, pd.Timestamp]]]:
    if events is None or events.empty:
        return {}
    required = {"Ticker", "EffectiveDate", "DelistingReturn"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(
            f"Delisting events are missing required columns: {missing}"
        )
    frame = events.loc[:, sorted(required)].copy()
    frame["Ticker"] = frame["Ticker"].astype(str)
    frame["EffectiveDate"] = pd.to_datetime(
        frame["EffectiveDate"],
        errors="raise",
    )
    frame["DelistingReturn"] = pd.to_numeric(
        frame["DelistingReturn"],
        errors="raise",
    )
    if frame.duplicated(["Ticker", "EffectiveDate"]).any():
        raise ValueError("Duplicate ticker/effective-date delisting event")
    invalid_return = (
        ~np.isfinite(frame["DelistingReturn"])
        | frame["DelistingReturn"].lt(-1.0)
    )
    if invalid_return.any():
        raise ValueError(
            "DelistingReturn must be finite and greater than or equal to -1"
        )
    ticker_positions = {
        str(ticker): index for index, ticker in enumerate(tickers)
    }
    unknown = sorted(set(frame["Ticker"]) - set(ticker_positions))
    if unknown:
        raise ValueError(
            f"Delisting events contain tickers absent from market: {unknown}"
        )
    schedule: dict[
        pd.Timestamp,
        list[tuple[int, float, pd.Timestamp]],
    ] = {}
    for row in frame.itertuples(index=False):
        source_date = pd.Timestamp(row.EffectiveDate)
        position = int(dates.searchsorted(source_date, side="left"))
        if position >= len(dates):
            continue
        execution_date = pd.Timestamp(dates[position])
        schedule.setdefault(execution_date, []).append(
            (
                ticker_positions[str(row.Ticker)],
                float(row.DelistingReturn),
                source_date,
            )
        )
    return schedule


def _summarize(
    daily: pd.DataFrame,
    *,
    initial_capital: float,
    total_turnover: float,
    ticker_trades: int,
    rebalance_count: int,
) -> PortfolioSummary:
    start_date = pd.Timestamp(daily["Date"].iloc[0])
    end_date = pd.Timestamp(daily["Date"].iloc[-1])
    final_value = float(daily["Equity"].iloc[-1])
    roi = (final_value / initial_capital - 1) * 100
    years = max((end_date - start_date).days / 365.25, 1 / 252)
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    running_peak = daily["Equity"].cummax()
    max_drawdown = float((daily["Equity"] / running_peak - 1).min() * 100)
    returns = daily["Equity"].pct_change(fill_method=None).dropna()
    volatility = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / volatility * np.sqrt(252))
        if volatility > 0
        else 0.0
    )
    turnover_multiple = total_turnover / initial_capital
    return PortfolioSummary(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        total_injected=initial_capital,
        final_value=final_value,
        roi_percent=roi,
        cagr_percent=cagr,
        max_drawdown_percent=max_drawdown,
        sharpe_ratio=sharpe,
        turnover_multiple=turnover_multiple,
        annualized_turnover=turnover_multiple / years,
        ticker_trades=ticker_trades,
        rebalance_count=rebalance_count,
    )
