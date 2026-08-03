from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from .filing_signals import FilingHybridConfig, add_market_regime


@dataclass(frozen=True)
class WatchlistEntry:
    ticker: str
    eligible_from: pd.Timestamp
    note: str = ""


def watchlist_entries(values: Iterable[Mapping[str, object]]) -> list[WatchlistEntry]:
    """Validate the user-maintained watchlist and its causal start dates."""

    entries: list[WatchlistEntry] = []
    seen: set[str] = set()
    for value in values:
        ticker = str(value.get("ticker", "")).strip().upper()
        if not ticker:
            raise ValueError("Every watchlist entry requires a ticker")
        if ticker in seen:
            raise ValueError(f"Duplicate watchlist ticker: {ticker}")
        eligible_from = pd.Timestamp(value.get("eligible_from"))
        if pd.isna(eligible_from):
            raise ValueError(f"Watchlist ticker {ticker} requires eligible_from")
        entries.append(
            WatchlistEntry(
                ticker=ticker,
                eligible_from=eligible_from.normalize(),
                note=str(value.get("note", "")).strip(),
            )
        )
        seen.add(ticker)
    return entries


def build_top_n_plus_watchlist_membership(
    top_n_membership: pd.DataFrame,
    watchlist: Iterable[WatchlistEntry],
    *,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Union dated Top-N snapshots with names known by each snapshot date.

    Extra snapshots are inserted on watchlist eligibility dates so an IPO can
    enter the research universe without waiting for the next annual Top-N
    snapshot. The most recent prior Top-N set is carried forward unchanged.
    """

    required = {"AsOfDate", "DataSymbol", "Rank"}
    missing = sorted(required - set(top_n_membership))
    if missing:
        raise ValueError(f"Top-N membership is missing columns: {missing}")
    base = top_n_membership.copy()
    base["AsOfDate"] = pd.to_datetime(base["AsOfDate"], errors="raise").dt.normalize()
    base["DataSymbol"] = base["DataSymbol"].astype(str).str.upper().str.strip()
    entries = list(watchlist)
    dates = set(base["AsOfDate"])
    first_snapshot = base["AsOfDate"].min()
    last_allowed = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else base["AsOfDate"].max()
    )
    dates.update(
        entry.eligible_from
        for entry in entries
        if first_snapshot <= entry.eligible_from <= last_allowed
    )
    if end_date is not None and last_allowed >= first_snapshot:
        dates.add(last_allowed)

    frames: list[pd.DataFrame] = []
    snapshot_dates = sorted(base["AsOfDate"].unique())
    for as_of in sorted(dates):
        prior_dates = [value for value in snapshot_dates if value <= as_of]
        if not prior_dates:
            continue
        source_date = max(prior_dates)
        top_rows = base.loc[base["AsOfDate"].eq(source_date)].copy()
        top_rows["AsOfDate"] = as_of
        top_rows["MembershipBucket"] = "TOP_N"
        top_rows["MembershipSourceDate"] = source_date
        rows = [top_rows]
        existing = set(top_rows["DataSymbol"])
        next_rank = int(pd.to_numeric(top_rows["Rank"]).max()) + 1
        extras: list[dict[str, object]] = []
        for entry in entries:
            if entry.eligible_from > as_of or entry.ticker in existing:
                continue
            extras.append(
                {
                    "AsOfDate": as_of,
                    "HistoricalTicker": entry.ticker,
                    "DataSymbol": entry.ticker,
                    "Company": entry.ticker,
                    "Rank": next_rank,
                    "PublishedRank": pd.NA,
                    "MarketCap": pd.NA,
                    "Selected": True,
                    "MembershipSource": "USER_WATCHLIST",
                    "MembershipBucket": "WATCHLIST",
                    "MembershipSourceDate": entry.eligible_from,
                    "WatchlistNote": entry.note,
                }
            )
            next_rank += 1
        if extras:
            rows.append(pd.DataFrame(extras))
        frames.append(
            pd.concat(
                [row.dropna(axis=1, how="all") for row in rows],
                ignore_index=True,
                sort=False,
            )
        )

    result = pd.concat(frames, ignore_index=True, sort=False)
    result["Selected"] = True
    if result.duplicated(["AsOfDate", "DataSymbol"]).any():
        raise RuntimeError("Duplicate ticker in Top-N plus watchlist snapshot")
    return result.sort_values(["AsOfDate", "Rank", "DataSymbol"]).reset_index(
        drop=True
    )


def apply_full_cash_market_gate(
    panel: pd.DataFrame,
    spy_prices: pd.DataFrame,
    *,
    slow_sessions: int = 200,
    fast_sessions: int = 50,
    band: float = 0.01,
) -> pd.DataFrame:
    """Make every stock ineligible while the causal SPY regime is risk-off.

    The shared target generator then emits exits and a 100% cash allocation.
    Using eligibility rather than scaling weights keeps backtest and live trade
    state identical when the regime changes.
    """

    config = FilingHybridConfig(
        market_regime_enabled=True,
        market_regime_sma_sessions=slow_sessions,
        market_regime_fast_sessions=fast_sessions,
        market_regime_band=band,
        risk_off_core_total_weight=0.0,
    )
    frame = add_market_regime(panel, spy_prices, config)
    frame["EligibleBeforeMarketGate"] = frame["Eligible"].fillna(False)
    frame["UniverseMemberBeforeMarketGate"] = frame.get(
        "UniverseMember", pd.Series(True, index=frame.index)
    ).fillna(False)
    frame["Eligible"] = (
        frame["EligibleBeforeMarketGate"] & frame["MarketRiskOn"]
    )
    frame["UniverseMember"] = (
        frame["UniverseMemberBeforeMarketGate"] & frame["MarketRiskOn"]
    )
    frame["MarketGateReason"] = ""
    frame.loc[~frame["MarketRiskOn"], "MarketGateReason"] = "SPY_TREND_RISK_OFF"
    return frame


def allocation_with_cash(targets: pd.DataFrame, date: object | None = None) -> pd.DataFrame:
    """Return the latest target allocation with an explicit CASH row."""

    if targets.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "TargetWeight"])
    target_date = (
        pd.Timestamp(date)
        if date is not None
        else pd.Timestamp(targets["Date"].max())
    )
    latest = targets.loc[pd.to_datetime(targets["Date"]).eq(target_date)].copy()
    latest["TargetWeight"] = pd.to_numeric(
        latest["TargetWeight"], errors="coerce"
    ).fillna(0.0)
    invested = float(latest["TargetWeight"].clip(lower=0).sum())
    cash = max(0.0, 1.0 - invested)
    selected = latest.loc[latest["TargetWeight"].gt(0)].copy()
    cash_row = pd.DataFrame(
        [{"Date": target_date, "Ticker": "CASH", "TargetWeight": cash}]
    )
    return pd.concat([selected, cash_row], ignore_index=True, sort=False)


def new_account_allocation(
    scored: pd.DataFrame,
    *,
    top_k: int,
    date: object | None = None,
) -> pd.DataFrame:
    """Build a fresh-account allocation without inheriting old hold state."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if scored.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "TargetWeight"])
    target_date = (
        pd.Timestamp(date)
        if date is not None
        else pd.Timestamp(scored["Date"].max())
    )
    latest = scored.loc[pd.to_datetime(scored["Date"]).eq(target_date)].copy()
    risk_on = bool(latest.get("MarketRiskOn", pd.Series([True])).iloc[0])
    ranks = pd.to_numeric(latest.get("Rank"), errors="coerce")
    qualified = latest.get(
        "Qualified", pd.Series(False, index=latest.index)
    ).fillna(False).astype(bool)
    selected = latest.loc[risk_on & qualified & ranks.le(top_k)].copy()
    selected = selected.sort_values("Rank").head(top_k)
    selected["TargetWeight"] = 1.0 / top_k
    invested = float(selected["TargetWeight"].sum())
    cash_row = pd.DataFrame(
        [
            {
                "Date": target_date,
                "Ticker": "CASH",
                "TargetWeight": max(0.0, 1.0 - invested),
            }
        ]
    )
    return pd.concat([selected, cash_row], ignore_index=True, sort=False)
