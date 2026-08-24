from __future__ import annotations

"""Persisted dashboard state: user-editable ticker universe + settings.

Kept as a small JSON file rather than a database -- this is a single-user
local research tool, not a multi-tenant service. Seeded on first run from
the 28-ticker SEC-synced universe already validated in this session
(config/cross_sectional/sec_synced_28_universe_swap_test.json), so the
dashboard is immediately useful without requiring the user to add every
ticker by hand.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from stock_research.paths import ProjectPaths

STATE_PATH_RELATIVE = "config/cross_sectional/dashboard_universe.json"
SEED_UNIVERSE_CONFIG = "config/cross_sectional/sec_synced_28_universe_swap_test.json"
DEFAULT_TOP_K = 5


@dataclass
class TickerEntry:
    ticker: str
    company: str
    price_path: str  # relative to paths.stock_root
    source: str  # "seed" | "yfinance"
    added_date: str


@dataclass
class DashboardState:
    top_k: int
    sec_user_agent: str
    tickers: list[TickerEntry] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "sec_user_agent": self.sec_user_agent,
            "tickers": [asdict(t) for t in self.tickers],
        }


def _state_path(paths: ProjectPaths) -> Path:
    return paths.repo_root / STATE_PATH_RELATIVE


def _seed_state(paths: ProjectPaths) -> DashboardState:
    seed_config_path = paths.repo_root / SEED_UNIVERSE_CONFIG
    seed = json.loads(seed_config_path.read_text(encoding="utf-8"))
    today = date.today().isoformat()
    tickers = [
        TickerEntry(
            ticker=str(row["ticker"]).upper(),
            company=str(row["company"]),
            price_path=str(row["price_path"]),
            source="seed",
            added_date=today,
        )
        for row in seed["universe"]
    ]
    default_user_agent = "Personal Research Dashboard seungyun@berkeley.edu"
    return DashboardState(top_k=DEFAULT_TOP_K, sec_user_agent=default_user_agent, tickers=tickers)


def state_from_dict(raw: dict[str, object]) -> DashboardState:
    return DashboardState(
        top_k=int(raw.get("top_k", DEFAULT_TOP_K)),
        sec_user_agent=str(raw.get("sec_user_agent", "")),
        tickers=[TickerEntry(**row) for row in raw.get("tickers", [])],  # type: ignore[arg-type]
    )


def load_state(paths: ProjectPaths) -> DashboardState:
    path = _state_path(paths)
    if not path.exists():
        state = _seed_state(paths)
        save_state(paths, state)
        return state
    return state_from_dict(json.loads(path.read_text(encoding="utf-8")))


def state_from_snapshot(path: Path) -> DashboardState:
    """Load a frozen universe_snapshots/<date>.json file (same shape as the
    live dashboard_universe.json) without touching the live state -- used to
    replay a past universe segment for the forward ledger."""

    return state_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_state(paths: ProjectPaths, state: DashboardState) -> None:
    path = _state_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=path.stem + "_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(state.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def add_ticker(paths: ProjectPaths, ticker: str, company: str, price_path: str, source: str) -> DashboardState:
    state = load_state(paths)
    ticker = ticker.upper().strip()
    if any(t.ticker == ticker for t in state.tickers):
        raise ValueError(f"{ticker} is already in the universe")
    state.tickers.append(
        TickerEntry(
            ticker=ticker,
            company=company,
            price_path=price_path,
            source=source,
            added_date=date.today().isoformat(),
        )
    )
    save_state(paths, state)
    return state


def remove_ticker(paths: ProjectPaths, ticker: str) -> DashboardState:
    state = load_state(paths)
    ticker = ticker.upper().strip()
    remaining = [t for t in state.tickers if t.ticker != ticker]
    if len(remaining) == len(state.tickers):
        raise ValueError(f"{ticker} is not in the universe")
    state.tickers = remaining
    save_state(paths, state)
    return state


def set_top_k(paths: ProjectPaths, top_k: int) -> DashboardState:
    state = load_state(paths)
    state.top_k = int(top_k)
    save_state(paths, state)
    return state
