"""Shared, research-only inputs for the A (vol-target) vs B (dip-buy) race.

Never writes the frozen artifact, never touches config/, never deploys.
Everything runs on the same U001 universe snapshot, same 10bp cost, same
next-session-open execution and same ROI definition as frozen V7.3.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

REGISTRY_DIR = ROOT / "config/dashboard_model_registry"
FROZEN_PATH = REGISTRY_DIR / "frozen_backtests/v7_3.json"
SNAPSHOT_PATH = REGISTRY_DIR / "universe_snapshots/2026-08-12.json"
PANEL_CACHE = OUT / "panel_cache.pkl"

SELECTION_END = "2025-12-31"
OOS_START = "2026-01-02"
COST_BPS = 10.0


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, payload) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    handle, tmp = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel(force: bool = False):
    """Build (or reuse) the exact frozen-V7.3 scored panel."""
    if PANEL_CACHE.exists() and not force:
        with PANEL_CACHE.open("rb") as fh:
            return pickle.load(fh)
    from stock_research.dashboard import engine
    from stock_research.dashboard.state import state_from_snapshot
    from stock_research.paths import load_paths

    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    state = state_from_snapshot(SNAPSHOT_PATH)
    start, end = frozen["period"]["start"], frozen["period"]["end"]
    panel = engine.build_scored_panel(load_paths(), state, start=start, end=end, top_k=5)
    bundle = {"panel": panel, "start": start, "end": end, "frozen": frozen}
    with PANEL_CACHE.open("wb") as fh:
        pickle.dump(bundle, fh)
    return bundle


def frozen_signal_days(factored: pd.DataFrame, start: str, end: str, rebalance_weekday: int) -> pd.DataFrame:
    """Signal-date selection exactly as the frozen V7.3 run did.

    Production changed after the freeze to drop an incomplete current week, so
    a fresh replay lands on a slightly different weekly grid and no longer
    reproduces the frozen 213.31% baseline.  Same research-only compatibility
    shim as research/momentum_chase_audit/common.py; shared code is untouched.
    """
    period = factored.loc[factored["Date"].between(start, end)].copy()
    if period.empty:
        return period
    weekday_names = ("MON", "TUE", "WED", "THU", "FRI")
    coverage = (
        period.groupby("Date", as_index=False)["Ticker"].nunique()
        .rename(columns={"Ticker": "CrossSectionCoverage"})
    )
    coverage["Week"] = coverage["Date"].dt.to_period("W-%s" % weekday_names[rebalance_weekday])
    dates = pd.DatetimeIndex(
        coverage.sort_values(["Week", "CrossSectionCoverage", "Date"], ascending=[True, False, False])
        .drop_duplicates("Week", keep="first")["Date"].sort_values()
    )
    return period.loc[period["Date"].isin(dates)].copy()
