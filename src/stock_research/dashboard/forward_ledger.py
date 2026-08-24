from __future__ import annotations

"""Point-in-time forward ledger: a NAV series that starts at
registry.json's forward_ledger_start and only ever appends going forward.

Each entry in registry["universe_changes"] defines one *segment* -- a
stretch of time with one fixed universe (a snapshot file under
config/dashboard_model_registry/universe_snapshots/). Each segment is
replayed independently through the existing engine.run_backtest (which
always starts a fresh simulation at its own INITIAL_CAPITAL), then the
segments' growth factors are chained into one continuous NAV series --
segment N's ending NAV becomes segment N+1's starting capital, minus a
one-time re-entry cost. Past segments are never recomputed: only the last
(currently open) segment's end date moves forward on each refresh.
"""

import json
from pathlib import Path

import pandas as pd

from stock_research.dashboard import data_collection, engine
from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard.signal_health import DEFAULT_ROLLING_WINDOWS, FACTOR_COLUMNS, build_signal_health, rolling_average
from stock_research.paths import ProjectPaths

REGISTRY_DIR = "config/dashboard_model_registry"
LEDGER_RESULTS_FOLDER = "mobile_investment_app"
LEDGER_FILENAME = "forward_ledger.json"
BENCHMARK_TICKERS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100"}
REENTRY_COST_BPS = 10.0  # same transaction-cost assumption used everywhere else in this model


def registry_dir(paths: ProjectPaths) -> Path:
    return paths.repo_root / REGISTRY_DIR


def load_registry(paths: ProjectPaths) -> dict[str, object]:
    return json.loads((registry_dir(paths) / "registry.json").read_text(encoding="utf-8"))


def build_segment(paths: ProjectPaths, snapshot_path: Path, *, start: str, end: str, top_k: int, universe_version: str) -> dict[str, object]:
    state = dashboard_state.state_from_snapshot(snapshot_path)
    # Build the scored panel once; both the NAV simulation and the signal-health
    # diagnostics read from it, so the panel-building step never runs twice.
    panel = engine.build_scored_panel(paths, state, start=start, end=end, top_k=top_k)
    result = engine.simulate_portfolio(panel, start=start, end=end)
    health = build_signal_health(panel.scored, panel.factored, k=panel.top_k)
    return {
        "start": start,
        "end": end,
        "universe_snapshot": snapshot_path.name,
        "universe_version": universe_version,
        "summary": result["summary"],
        "equity_curve": [{"date": row["date"], "equity": row["equity"]} for row in result["equity_curve"]],
        "signal_health": health,
    }


def build_segment_if_signaled(
    paths: ProjectPaths,
    snapshot_path: Path,
    *,
    start: str,
    end: str,
    top_k: int,
    universe_version: str,
) -> dict[str, object] | None:
    """Skip a universe segment that ended before its first weekly signal.

    A midweek universe change can leave the prior snapshot with no completed
    Friday signal.  Such a segment never held a portfolio and therefore must
    not abort the entire forward-ledger publication or incur a re-entry cost.
    """

    try:
        return build_segment(
            paths,
            snapshot_path,
            start=start,
            end=end,
            top_k=top_k,
            universe_version=universe_version,
        )
    except ValueError as error:
        if str(error).startswith("No trading data in range "):
            return None
        raise


def chain_segments(
    segments: list[dict[str, object]],
    *,
    initial_nav: float = 100_000.0,
    reentry_cost_bps: float = REENTRY_COST_BPS,
) -> list[dict[str, object]]:
    """Turn independently-simulated segments (each its own fresh
    INITIAL_CAPITAL run) into one continuous NAV series. Within a segment,
    NAV tracks that segment's own equity-curve growth factor scaled by the
    NAV carried in from the prior segment; a re-entry cost is applied once
    at each segment boundary, matching the project's standard 10bps
    transaction-cost assumption."""

    series: list[dict[str, object]] = []
    segment_start_nav = initial_nav
    for segment in segments:
        curve = segment["equity_curve"]
        if not curve:
            continue
        if series:  # a prior segment already contributed rows -- this is a real boundary
            segment_start_nav *= 1 - reentry_cost_bps / 10_000.0
        base_equity = curve[0]["equity"]
        for row in curve:
            growth = (row["equity"] / base_equity) if base_equity else 1.0
            series.append({"date": row["date"], "nav": segment_start_nav * growth})
        segment_start_nav = series[-1]["nav"]
    return series


def benchmark_series(prices: dict[str, float], dates: list[str], *, initial_nav: float = 100_000.0) -> list[dict[str, object]]:
    """Buy-and-hold benchmark: initial_nav at the first date `prices` covers,
    growing with that date's close-to-close return thereafter. Dates missing
    a price (holiday mismatch) are skipped, not treated as flat."""

    covered = [d for d in dates if d in prices]
    if not covered:
        return []
    base = prices[covered[0]]
    return [{"date": d, "nav": initial_nav * prices[d] / base} for d in covered]


def load_benchmark_prices(paths: ProjectPaths, ticker: str) -> dict[str, float]:
    dest = paths.stock_root / "Dashboard Data" / "Benchmarks" / f"{ticker}.csv"
    data_collection.download_price_history(ticker, dest)
    frame = pd.read_csv(dest)
    return {str(pd.Timestamp(d).date()): c for d, c in zip(frame["Date"], frame["Close"])}


def chain_signal_health(segments: list[dict[str, object]]) -> dict[str, object]:
    """Concatenate each segment's own signal-health series in chronological
    order (FINAL observations from an earlier segment never change; only
    the newest segment's trailing PENDING date can still mature), and
    NAV-chain the Universe EW curves with `chain_segments` -- same
    boundary/re-entry-cost treatment as the main NAV series, so it stays a
    real point-in-time benchmark instead of retroactively applying whatever
    universe is current. Known, accepted imprecision: a segment's very last
    signal date is always PENDING from that segment's own point of view,
    even after the next segment's prices exist to finalize it -- narrow
    (one date per universe change) and conservative (under- not
    over-reports confidence), not a look-ahead risk."""

    portfolio_ic: list[dict] = []
    factor_ic: dict[str, list[dict]] = {factor: [] for factor in FACTOR_COLUMNS}
    top_k_spread: list[dict] = []
    ew_segments = []
    last_health: dict[str, object] | None = None

    for segment in segments:
        health = segment["signal_health"]
        portfolio_ic.extend(health["portfolio_ic"])
        for factor, payload in health["factor_ic"].items():
            factor_ic.setdefault(factor, []).extend(payload["series"])
        top_k_spread.extend(health["top_k_spread"])
        ew_segments.append({"equity_curve": [{"date": row["date"], "equity": row["nav"]} for row in health["universe_equal_weight_curve_gross"]]})
        last_health = health

    return {
        "universe_size": last_health["universe_size"] if last_health else 0,
        "top_k": last_health["top_k"] if last_health else 0,
        "selection_rate": last_health["selection_rate"] if last_health else 0.0,
        "completed_weeks": sum(1 for row in portfolio_ic if row["status"] == "FINAL"),
        "portfolio_ic": portfolio_ic,
        "portfolio_ic_rolling": rolling_average(portfolio_ic, "ic", DEFAULT_ROLLING_WINDOWS),
        "factor_ic": {
            factor: {"series": series, "rolling": rolling_average(series, "ic", DEFAULT_ROLLING_WINDOWS)}
            for factor, series in factor_ic.items()
            if series
        },
        "top_k_spread": top_k_spread,
        "top_k_spread_rolling": rolling_average(top_k_spread, "spread", DEFAULT_ROLLING_WINDOWS),
        "universe_equal_weight_curve_gross": [
            {"date": row["date"], "nav": row["nav"]} for row in chain_segments(ew_segments)
        ],
    }


def build_forward_ledger(paths: ProjectPaths, *, end: str, top_k: int = 5) -> dict[str, object]:
    registry = load_registry(paths)
    snapshots_dir = registry_dir(paths) / "universe_snapshots"
    changes = registry["universe_changes"]

    segments = []
    for index, change in enumerate(changes):
        is_last = index == len(changes) - 1
        segment_end = end if is_last else changes[index + 1]["date"]
        segment = build_segment_if_signaled(
            paths,
            snapshots_dir / change["snapshot"],
            start=change["date"],
            end=segment_end,
            top_k=top_k,
            universe_version=change.get("universe_version", f"U{index+1:03d}"),
        )
        if segment is not None:
            segments.append(segment)

    nav_series = chain_segments(segments)
    all_dates = [row["date"] for row in nav_series]
    benchmarks = {
        ticker: benchmark_series(load_benchmark_prices(paths, ticker), all_dates)
        for ticker in BENCHMARK_TICKERS
    }

    return {
        "start_date": registry["forward_ledger_start"],
        "initial_nav": 100_000.0,
        "model_version": registry["active_model_version"],
        "current_universe_snapshot": changes[-1]["snapshot"],
        "current_universe_version": changes[-1].get("universe_version", f"U{len(changes):03d}"),
        "series": nav_series,
        "benchmarks": benchmarks,
        "signal_health": chain_signal_health(segments),
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "universe_snapshot": segment["universe_snapshot"],
                "universe_version": segment["universe_version"],
                "summary": segment["summary"],
            }
            for segment in segments
        ],
    }
