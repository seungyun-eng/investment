from __future__ import annotations

"""Data collection for dashboard-added tickers: price history (Yahoo Finance
chart API via plain `requests`) and SEC filing point-in-time features
(reusing the existing sync_sec_filings pipeline).

Note: yfinance's curl_cffi HTTP backend cannot open a CA bundle at a path
containing non-ASCII characters (this repo lives under a Korean-named
OneDrive folder) -- libcurl on Windows resolves that path with the system
ANSI code page, which cannot represent Korean characters, so every yfinance
call fails with `curl: (77) error setting certificate verify locations`
regardless of Python-level encoding fixes. Plain `requests` (already a
project dependency, already used for the SEC EDGAR client) uses Python's own
ssl module instead and is unaffected, so this hits Yahoo's public chart API
directly rather than going through yfinance.
"""

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from stock_research.cross_sectional.sec_filings import (
    SecFilingSettings,
    sync_sec_filings,
)
from stock_research.dashboard import state as dashboard_state
from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths

_EPOCH_START = 0  # Unix epoch; Yahoo clamps to each symbol's real listing date.

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
DASHBOARD_PRICE_DIR = "Dashboard Data/Prices"
DASHBOARD_FILING_LABEL = "dashboard"
DASHBOARD_CANDIDATE_FILING_LABEL = "dashboard_candidates"
# sync_sec_filings() caches raw SEC downloads globally (by ticker, not by
# output_label), but its per-document TEXT ANALYSIS cache
# (filing_text_features.csv, keyed by document SHA256) lives under each
# output_label's own directory. The first-ever dashboard sync would
# otherwise re-run regex text analysis over every cached document from
# scratch (measured: 28 tickers, ~100s+) even though an earlier session
# already computed the identical analysis under a different output_label.
# Seeding from that prior run's cache turns the first dashboard sync from
# "reanalyze ~600 documents" into "read ~600 cached rows".
_SEED_TEXT_FEATURE_LABEL = "live_top10_plus_watchlist"


def _session_is_settled(meta: dict) -> bool:
    """True when `regularMarketPrice` is a closing print, not a live quote.

    Yahoo advertises the *next* session in `currentTradingPeriod` once the
    current one ends, so a last trade time before that window's start means
    the session already closed.  Immediately after the bell the window still
    describes the session that just ended, which the `>= end` arm covers.
    """

    period = ((meta.get("currentTradingPeriod") or {}).get("regular")) or {}
    start, end, traded = period.get("start"), period.get("end"), meta.get("regularMarketTime")
    if not isinstance(traded, (int, float)):
        return False
    if isinstance(end, (int, float)) and traded >= end:
        return True
    return isinstance(start, (int, float)) and traded < start


def _fill_settled_close(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Restore the just-closed session's close when Yahoo leaves it null.

    Yahoo publishes the final bar's open/high/low/volume before it fills in
    `close`, so the freshest completed session was being dropped by the
    dropna below and the dashboard stayed a day behind the market.  `meta`
    already carries the settled price for that same date, and it was verified
    against the monitor's independent Nasdaq feed (NVDA 225.01, MCO 477.17,
    PLTR 172.55 on 2026-08-17).  Only the final row is repaired, and only
    while the session is genuinely over, so a live intraday quote is never
    written as a daily close.
    """

    price, traded = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if frame.empty or not isinstance(price, (int, float)):
        return frame
    last = frame.index[-1]
    if pd.notna(frame.at[last, "Close"]) or not _session_is_settled(meta):
        return frame
    settled_date = pd.to_datetime(traded, unit="s", utc=True).tz_convert(None).normalize()
    if frame.at[last, "Date"] != settled_date:
        return frame
    frame.at[last, "Close"] = float(price)
    return frame


def download_price_history(ticker: str, dest_path: Path) -> dict[str, object]:
    """Fetch full daily price history for `ticker` and write it in the
    positional Date,Close,Open,High,Low,Volume order that
    tsla_integrated.data.load_equity_prices expects (that loader keys off
    column position, not header names, so this is directly compatible with
    every existing panel-building function)."""

    # `range=max` silently downgrades very long spans to ~monthly bars on
    # Yahoo's chart API; explicit period1/period2 bounds preserve true daily
    # granularity for the whole history instead (verified: 234 vs 4868 rows
    # for the same 19-year SMCI request).
    response = requests.get(
        YAHOO_CHART_URL.format(ticker=ticker),
        params={
            "period1": _EPOCH_START,
            "period2": int(time.time()),
            "interval": "1d",
            "events": "div,splits",
        },
        headers={"User-Agent": "Mozilla/5.0 (dashboard price downloader)"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance error for {ticker}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"No price data returned for {ticker}")
    result = results[0]
    timestamps = result.get("timestamp")
    if not timestamps:
        raise ValueError(f"No price history available for {ticker}")
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
            "Close": quote["close"],
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Volume": quote["volume"],
        }
    )
    frame = _fill_settled_close(frame, result.get("meta") or {})
    frame = frame.dropna(subset=["Date", "Close", "Open"]).sort_values("Date")
    frame = frame.drop_duplicates("Date", keep="last").reset_index(drop=True)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(frame, dest_path, index=False)
    return {
        "ticker": ticker,
        "rows": len(frame),
        "start_date": str(frame["Date"].min().date()) if not frame.empty else None,
        "end_date": str(frame["Date"].max().date()) if not frame.empty else None,
        "path": str(dest_path),
    }


def _seed_text_feature_cache(paths: ProjectPaths) -> None:
    dest_dir = paths.processed / "SEC Filings" / DASHBOARD_FILING_LABEL
    dest = dest_dir / "filing_text_features.csv"
    if dest.exists():
        return
    seed = paths.processed / "SEC Filings" / _SEED_TEXT_FEATURE_LABEL / "filing_text_features.csv"
    if not seed.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed, dest)


def sync_filings_for_dashboard(
    paths: ProjectPaths,
    state: "dashboard_state.DashboardState",
    *,
    refresh_metadata: bool = False,
):
    _seed_text_feature_cache(paths)
    tickers = [t.ticker for t in state.tickers]
    settings = SecFilingSettings()
    return sync_sec_filings(
        paths,
        tickers=tickers,
        settings=settings,
        user_agent=state.sec_user_agent,
        output_label=DASHBOARD_FILING_LABEL,
        refresh_metadata=refresh_metadata,
    )


def sync_filings_for_candidates(
    paths: ProjectPaths,
    state: "dashboard_state.DashboardState",
    tickers: list[str],
    *,
    refresh_metadata: bool = True,
):
    """Refresh only requested candidates, not the entire live universe.

    Candidate SEC artifacts stay separate until a ticker passes review. This
    avoids re-downloading metadata for every live ticker and prevents one
    late request in a large batch from making a valid candidate look empty.
    """

    normalized = list(dict.fromkeys(ticker.upper().strip() for ticker in tickers))
    if not normalized:
        raise ValueError("No candidate tickers were supplied")
    return sync_sec_filings(
        paths,
        tickers=normalized,
        settings=SecFilingSettings(),
        user_agent=state.sec_user_agent,
        output_label=DASHBOARD_CANDIDATE_FILING_LABEL,
        refresh_metadata=refresh_metadata,
    )


def collect_ticker(paths: ProjectPaths, ticker: str, company: str) -> dict[str, object]:
    """Full collection flow for a newly-added ticker: download price history
    via Yahoo, add it to the dashboard universe, then re-sync SEC filings for
    the whole (now-expanded) universe -- cheap for already-synced tickers
    since sync_sec_filings caches every raw SEC download on disk and only
    hits the network for what's actually new."""

    ticker = ticker.upper().strip()
    dest_path = paths.stock_root / DASHBOARD_PRICE_DIR / f"{ticker}.csv"
    price_summary = download_price_history(ticker, dest_path)

    relative_price_path = f"{DASHBOARD_PRICE_DIR}/{ticker}.csv"
    state = dashboard_state.add_ticker(
        paths, ticker, company or ticker, relative_price_path, source="yfinance"
    )

    filing_summary: dict[str, object] = {"status": "skipped", "reason": "no SEC_USER_AGENT configured"}
    try:
        artifacts = sync_filings_for_dashboard(paths, state)
        filing_summary = {
            "status": "ok",
            "point_in_time_features_csv": str(artifacts.point_in_time_features_csv),
        }
    except ValueError as exc:
        filing_summary = {"status": "failed", "reason": str(exc)}

    return {"ticker": ticker, "price": price_summary, "filings": filing_summary}


def refresh_all_prices(
    paths: ProjectPaths,
    state: "dashboard_state.DashboardState",
    *,
    max_workers: int = 4,
) -> dict[str, object]:
    """Refresh every dashboard ticker into the sibling Dashboard Data folder.

    Successful tickers are switched to the refreshed file only after its
    atomic write succeeds. A failed ticker keeps its prior configured path so
    one Yahoo error cannot invalidate the whole daily dashboard.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    entries = {entry.ticker: entry for entry in state.tickers}

    def refresh(ticker: str) -> dict[str, object]:
        destination = paths.stock_root / DASHBOARD_PRICE_DIR / f"{ticker}.csv"
        return download_price_history(ticker, destination)

    successes: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(entries)))) as executor:
        futures = {executor.submit(refresh, ticker): ticker for ticker in entries}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                summary = future.result()
                successes.append(summary)
                entries[ticker].price_path = f"{DASHBOARD_PRICE_DIR}/{ticker}.csv"
                entries[ticker].source = "yahoo_refresh"
            except Exception as exc:  # noqa: BLE001
                failures.append({"ticker": ticker, "error": str(exc)})

    dashboard_state.save_state(paths, state)
    latest_dates = [row.get("end_date") for row in successes if row.get("end_date")]
    return {
        "requested": len(entries),
        "succeeded": len(successes),
        "failed": len(failures),
        "latest_date": max(latest_dates) if latest_dates else None,
        "prices": sorted(successes, key=lambda row: str(row["ticker"])),
        "failures": sorted(failures, key=lambda row: row["ticker"]),
    }
