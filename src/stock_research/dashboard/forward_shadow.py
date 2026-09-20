"""Append-only forward-shadow accounting for the frozen Alpha Desk finalists.

The module deliberately does not own any strategy parameters.  It reads the
sealed recommendations selected by ``ALPHA_DESK_FINAL_CHAMPION_AUDIT_V1`` and
starts new, independent $100,000 paper accounts after the 2026-09-11 freeze.
Historical research NAV is never copied into, or joined to, these ledgers.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from stock_research.paths import ProjectPaths, load_paths

FREEZE_DATE = pd.Timestamp("2026-09-11")
INITIAL_NAV = 100_000.0
COST_BPS = 25.0
MODEL_SYSTEMIC = "ADS_SHADOW_SYSTEMIC_JOINT_V1_20260919"
MODEL_MEGA = "ADS_SHADOW_MEGABLEND_B_U1_V1_20260919"
MODEL_VOL15 = "ADS_SHADOW_SYSTEMIC_JOINT_VOL15_V1_20260919"
MODEL_V73 = "ADS_SHADOW_FROZEN_V73_U001_V1_20260919"
MODEL_W100 = "ADS_REFERENCE_MEGABLEND_B_W100_V1_20260919"
PRIMARY_MODELS = (MODEL_SYSTEMIC, MODEL_MEGA, MODEL_VOL15, MODEL_V73)
ALL_MODELS = (*PRIMARY_MODELS, MODEL_W100)
MEGA = ("AAPL", "MSFT", "GOOGL")

OUTPUT_FILES = {
    "recommendations": "SHADOW_RECOMMENDATIONS.parquet",
    "holdings": "SHADOW_HOLDINGS.parquet",
    "trades": "SHADOW_TRADES.parquet",
    "nav": "SHADOW_NAV.parquet",
    "risk": "SHADOW_RISK_STATE.parquet",
    "comparison": "SHADOW_MODEL_COMPARISON.parquet",
    "universe": "SHADOW_UNIVERSE_COMPARISON.parquet",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    short_name: str
    role: str
    universe: str
    primary: bool
    synthetic: bool = False


SPECS = {
    MODEL_SYSTEMIC: ModelSpec(
        MODEL_SYSTEMIC, "SYSTEMIC_JOINT", "Aggressive", "U1_CURRENT_GROWTH_368", True
    ),
    MODEL_MEGA: ModelSpec(
        MODEL_MEGA, "MEGABLEND_B_U1", "Balanced", "U1_CURRENT_GROWTH_368", True
    ),
    MODEL_VOL15: ModelSpec(
        MODEL_VOL15,
        "SYSTEMIC_JOINT_VOL15",
        "Risk-Controlled",
        "U1_CURRENT_GROWTH_368",
        True,
        True,
    ),
    MODEL_V73: ModelSpec(
        MODEL_V73, "FROZEN_V73_U001", "Challenger", "U001_FIXED_31", True
    ),
    MODEL_W100: ModelSpec(
        MODEL_W100, "MEGABLEND_B_W100", "Universe Reference", "W100_MONTHLY", False
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _normalize_for_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if "date" in column.lower() or column.endswith("_at"):
            try:
                result[column] = pd.to_datetime(result[column], errors="raise")
            except (TypeError, ValueError):
                pass
    return result


def append_immutable(
    path: Path, incoming: pd.DataFrame, keys: list[str], *, sort: list[str] | None = None
) -> pd.DataFrame:
    """Append unseen keys and reject any attempt to revise an existing snapshot."""
    incoming = _normalize_for_comparison(incoming)
    if incoming.duplicated(keys).any():
        raise ValueError(f"duplicate incoming keys for {path.name}: {keys}")
    if path.exists():
        existing = _normalize_for_comparison(pd.read_parquet(path))
        common = existing.merge(incoming, on=keys, suffixes=("_old", "_new"))
        for column in existing.columns:
            if column in keys:
                continue
            old, new = f"{column}_old", f"{column}_new"
            if old not in common or new not in common:
                raise RuntimeError(f"append schema changed for {path.name}: {column}")
            a, b = common[old], common[new]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                same = np.isclose(
                    pd.to_numeric(a, errors="coerce"),
                    pd.to_numeric(b, errors="coerce"),
                    rtol=0,
                    atol=1e-10,
                    equal_nan=True,
                )
            else:
                same = a.astype("string").fillna("<NA>").eq(
                    b.astype("string").fillna("<NA>")
                )
            if not bool(np.all(same)):
                raise RuntimeError(
                    f"immutable shadow row changed in {path.name}: {column}"
                )
        unseen = incoming.merge(existing[keys], on=keys, how="left", indicator=True)
        unseen = unseen.loc[unseen["_merge"].eq("left_only"), incoming.columns]
        combined = pd.concat([existing, unseen], ignore_index=True)
    else:
        combined = incoming.copy()
    if sort:
        combined = combined.sort_values(sort, kind="stable").reset_index(drop=True)
    _atomic_parquet(path, combined)
    return combined


def _source_paths(paths: ProjectPaths) -> dict[str, Path]:
    results = paths.results
    return {
        "audit": results
        / "bravo_desk/alpha_desk_final_champion_audit_v1/20260919",
        "comparison": results
        / "bravo_desk/comparison_2015_2026_v1/20260917_fixed_common_period",
        "systemic": results
        / "bravo_desk/systemic_risk_engine_v1/20260917_fixed_systemic",
        "mega": results
        / "bravo_desk/abe_rates_2015_2026_v1/20260917_fixed_six",
        "architecture": results
        / "bravo_desk/alpha_master_portfolio_architecture_v1/20260919",
        "v73": results / "v73_frozen_macro_annual/20260916T060055_824125Z",
        "w100": results / "bravo_desk/dynamic_universe_ab_v1b/20260919",
        "clean": results / "bravo_desk/clean_historical_v2/20260918",
    }


def verify_frozen_sources(paths: ProjectPaths) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _source_paths(paths)
    manifest_path = source["audit"] / "FROZEN_SHADOW_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if [row["frozen_version_id"] for row in manifest["models"]] != list(PRIMARY_MODELS):
        raise RuntimeError("final-audit shadow set differs from registered four-model set")
    checks: list[dict[str, Any]] = []
    for model in manifest["models"]:
        for raw_path, expected in model["source_files"].items():
            path = Path(raw_path)
            if not path.exists():
                # Frozen manifests were created on Windows.  The cloud runner
                # restores the same sealed tree below its configured Results
                # root, so remap only the suffix after that explicit boundary.
                normalized = str(raw_path).replace("\\", "/")
                marker = "/Results/"
                if marker not in normalized:
                    raise FileNotFoundError(path)
                path = paths.results / normalized.split(marker, 1)[1]
            actual = _sha256(path)
            checks.append(
                {
                    "model_id": model["frozen_version_id"],
                    "path": str(path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "unchanged": actual == expected,
                }
            )
    if not all(row["unchanged"] for row in checks):
        raise RuntimeError("one or more frozen strategy source files changed")
    return manifest, checks


def _company_map(source: dict[str, Path]) -> dict[str, str]:
    panel = pd.read_parquet(
        source["comparison"] / "panel.parquet", columns=["Ticker", "company_name"]
    )
    values = (
        panel.dropna(subset=["company_name"])
        .drop_duplicates("Ticker", keep="last")
        .set_index("Ticker")["company_name"]
        .astype(str)
        .to_dict()
    )
    v73 = pd.read_parquet(
        source["v73"] / "base_model_targets.parquet", columns=["Ticker", "Company"]
    )
    values.update(
        v73.drop_duplicates("Ticker", keep="last").set_index("Ticker")["Company"].to_dict()
    )
    return values


def _next_session(signal_date: pd.Timestamp) -> pd.Timestamp:
    date = signal_date + pd.Timedelta(days=1)
    while date.weekday() >= 5:
        date += pd.Timedelta(days=1)
    return date.normalize()


def _momentum_rows(
    plan: pd.DataFrame,
    model_id: str,
    company: dict[str, str],
    *,
    sleeve: str,
    exposure: float = 1.0,
) -> list[dict[str, Any]]:
    date = pd.to_datetime(plan["Date"]).max()
    rows = plan.loc[pd.to_datetime(plan["Date"]).eq(date)].copy()
    rows = rows.sort_values(["rank", "Ticker"], kind="stable")
    target = {str(row.Ticker): float(row.weight) for row in rows.itertuples()}
    score = {str(row.Ticker): float(row.score) for row in rows.itertuples()}
    rank = {str(row.Ticker): int(row.rank) for row in rows.itertuples()}
    if sleeve == "MEGA30":
        target = {ticker: value * 0.7 for ticker, value in target.items()}
        for ticker in MEGA:
            target[ticker] = target.get(ticker, 0.0) + 0.1
            score.setdefault(ticker, math.nan)
            rank.setdefault(ticker, 10_000 + MEGA.index(ticker))
    target = {ticker: value * exposure for ticker, value in target.items()}
    ordered = sorted(target, key=lambda ticker: (rank.get(ticker, 99_999), ticker))
    return [
        {
            "model_id": model_id,
            "signal_date": date,
            "effective_trade_date": _next_session(date),
            "Ticker": ticker,
            "company": company.get(ticker, ticker),
            "rank": rank.get(ticker),
            "score": score.get(ticker),
            "target_weight": target[ticker],
            "recommendation_basis": (
                "UNDERLYING_SYSTEMIC_RISK_SCALED" if model_id == MODEL_VOL15 else sleeve
            ),
        }
        for ticker in ordered
    ]


def seed_recommendations(paths: ProjectPaths, generated_at: datetime) -> pd.DataFrame:
    source = _source_paths(paths)
    company = _company_map(source)
    core = pd.read_parquet(source["comparison"] / "core_plan.parquet")
    core = core[pd.to_datetime(core.Date).eq(FREEZE_DATE)]
    if len(core) != 10:
        raise RuntimeError("sealed U1 core plan does not have ten freeze-date names")
    w100 = pd.read_parquet(source["w100"] / "plans/monthly__W100.parquet")
    w100 = w100[pd.to_datetime(w100.Date).eq(FREEZE_DATE)]
    if len(w100) != 10:
        raise RuntimeError("sealed W100 plan does not have ten freeze-date names")
    rows: list[dict[str, Any]] = []
    rows += _momentum_rows(core, MODEL_SYSTEMIC, company, sleeve="CORE", exposure=1.0)
    rows += _momentum_rows(core, MODEL_MEGA, company, sleeve="MEGA30", exposure=1.0)
    rows += _momentum_rows(w100, MODEL_W100, company, sleeve="MEGA30", exposure=1.0)

    v73 = pd.read_parquet(source["v73"] / "base_model_targets.parquet")
    v73 = v73[pd.to_datetime(v73.Date).eq(FREEZE_DATE)].copy()
    v73 = v73.loc[v73.Rank.notna()].sort_values(["Rank", "Ticker"], kind="stable").head(10)
    for row in v73.itertuples():
        rows.append(
            {
                "model_id": MODEL_V73,
                "signal_date": FREEZE_DATE,
                "effective_trade_date": _next_session(FREEZE_DATE),
                "Ticker": str(row.Ticker),
                "company": str(row.Company),
                "rank": int(row.Rank),
                "score": float(row.AlphaScore),
                "target_weight": float(row.TargetWeight),
                "recommendation_basis": "FROZEN_V73_ALPHA_SCORE_AND_HOLD_RULE",
            }
        )
    result = pd.DataFrame(rows)
    # Recommendation snapshots are content-addressable by signal date.  A
    # rerun timestamp belongs in SHADOW_STATUS, not in an immutable signal row.
    result["generated_at"] = pd.Timestamp("2026-09-19T00:00:00Z")
    result["data_timestamp"] = FREEZE_DATE
    result["cost_bps"] = COST_BPS
    result["strategy_version"] = result.model_id
    result["universe_version"] = result.model_id.map(lambda value: SPECS[value].universe)
    result["is_primary"] = result.model_id.map(lambda value: SPECS[value].primary)
    result["risk_state"] = "NORMAL"
    result["action"] = np.where(result.target_weight.gt(0), "BUY", "HOLD")
    return result


def _universe_tickers(paths: ProjectPaths) -> set[str]:
    source = _source_paths(paths)
    panel = pd.read_parquet(source["comparison"] / "panel.parquet", columns=["Ticker"])
    w100 = pd.read_parquet(
        source["w100"] / "plans/monthly__W100.parquet", columns=["Date", "Ticker"]
    )
    w100_latest = w100.loc[pd.to_datetime(w100.Date).eq(FREEZE_DATE), "Ticker"]
    membership = pd.read_parquet(
        source["w100"] / "TRACK1_WIDTH_MEMBERSHIP.parquet"
    )
    membership["refresh"] = pd.to_datetime(membership.refresh)
    latest_refresh = membership.loc[
        membership.cadence.eq("monthly") & membership.universe.eq("W100"),
        "refresh",
    ].max()
    w100_candidates = membership.loc[
        membership.cadence.eq("monthly")
        & membership.universe.eq("W100")
        & membership.refresh.eq(latest_refresh),
        "ticker",
    ]
    v73 = pd.read_parquet(
        source["v73"] / "base_model_targets.parquet", columns=["Ticker"]
    )
    return (
        set(panel.Ticker.astype(str))
        | set(w100_latest.astype(str))
        | set(w100_candidates.astype(str))
        | set(v73.Ticker.astype(str))
        | set(MEGA)
        | {"SPY", "QQQ"}
    )


def _fetch_yahoo_adjusted(
    ticker: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cutoff: pd.Timestamp,
    timeout_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read provider-adjusted OHLC and numeric volume without rewriting history."""
    response = requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(ticker, safe="-"),
        params={
            "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
            "period2": int(pd.Timestamp(end, tz="UTC").timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo chart error for {ticker}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(), {}
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_rows = result.get("indicators", {}).get("quote") or []
    adjusted_rows = result.get("indicators", {}).get("adjclose") or []
    if not timestamps or not quote_rows or not adjusted_rows:
        return pd.DataFrame(), {}
    length = len(timestamps)

    def values(source: dict[str, Any], name: str) -> list[Any]:
        raw = list(source.get(name) or [])
        return (raw + [None] * length)[:length]

    quote_values = quote_rows[0]
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert(None)
            .normalize(),
            "Open": pd.to_numeric(values(quote_values, "open"), errors="coerce"),
            "High": pd.to_numeric(values(quote_values, "high"), errors="coerce"),
            "Low": pd.to_numeric(values(quote_values, "low"), errors="coerce"),
            "Close": pd.to_numeric(values(quote_values, "close"), errors="coerce"),
            "Volume_numeric": pd.to_numeric(
                values(quote_values, "volume"), errors="coerce"
            ),
            "AdjClose": pd.to_numeric(
                values(adjusted_rows[0], "adjclose"), errors="coerce"
            ),
        }
    )
    frame = frame.loc[frame.Date.le(cutoff)].dropna(
        subset=["Date", "Open", "High", "Low", "Close", "AdjClose"]
    )
    ratio = frame.AdjClose / frame.Close
    frame["AdjOpen"] = frame.Open * ratio
    frame["AdjHigh"] = frame.High * ratio
    frame["AdjLow"] = frame.Low * ratio
    payload_hash = hashlib.sha256(response.content).hexdigest()
    return frame, {"payload_sha256": payload_hash}


def refresh_prices(
    paths: ProjectPaths,
    output: Path,
    *,
    cutoff: pd.Timestamp,
    workers: int = 8,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Append provider-adjusted OHLC without revising an already stored session."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    source = _source_paths(paths)
    tickers = sorted(_universe_tickers(paths))
    cached_path = output / "data/SHADOW_PRICES.parquet"
    if cached_path.exists():
        cached = pd.read_parquet(cached_path)
        cached["Date"] = pd.to_datetime(cached.Date).dt.normalize()
    else:
        base = pd.read_parquet(source["comparison"] / "raw.parquet")
        base["Date"] = pd.to_datetime(base.Date).dt.normalize()
        missing_base = set(tickers).difference(base.Ticker.astype(str))
        if missing_base:
            clean = pd.read_parquet(source["clean"] / "SINGLE_SOURCE_PRICES.parquet")
            clean["Date"] = pd.to_datetime(clean.Date).dt.normalize()
            base = pd.concat(
                [base, clean.loc[clean.Ticker.isin(missing_base), base.columns]],
                ignore_index=True,
            )
        cached = base.loc[base.Ticker.isin(tickers)].copy()
    last = cached.groupby("Ticker").Date.max().to_dict()

    def one(ticker: str) -> tuple[str, pd.DataFrame, dict[str, Any]]:
        start = pd.Timestamp(last.get(ticker, FREEZE_DATE)) + pd.Timedelta(days=1)
        if start > cutoff:
            return ticker, pd.DataFrame(), {"Ticker": ticker, "status": "CURRENT"}
        frame, metadata = _fetch_yahoo_adjusted(
            ticker,
            start=start,
            end=cutoff + pd.Timedelta(days=1),
            cutoff=cutoff,
            timeout_seconds=30,
        )
        if frame.empty:
            return ticker, frame, {"Ticker": ticker, "status": "NO_ROWS"}
        out = pd.DataFrame(
            {
                "Date": pd.to_datetime(frame.Date).dt.normalize(),
                "Ticker": ticker,
                "AdjOpen": frame.AdjOpen,
                "AdjClose": frame.AdjClose,
                "AdjHigh": frame.AdjHigh,
                "AdjLow": frame.AdjLow,
                "Close": frame.Close,
                "Volume_numeric": frame.Volume_numeric,
            }
        )
        out = out.dropna(subset=["Date", "AdjOpen", "AdjClose"])
        return ticker, out, {
            "Ticker": ticker,
            "status": "OK",
            "rows": len(out),
            "payload_sha256": metadata["payload_sha256"],
        }

    new: list[pd.DataFrame] = []
    log: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(one, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, frame, status = future.result()
                if len(frame):
                    new.append(frame)
                log.append(status)
            except Exception as error:  # noqa: BLE001 - isolate per-symbol provider failure
                log.append({"Ticker": ticker, "status": "ERROR", "error": str(error)})
    incoming = pd.concat(new, ignore_index=True) if new else pd.DataFrame()
    combined = (
        pd.concat([cached, incoming], ignore_index=True, sort=False)
        if len(incoming)
        else cached.copy()
    )
    combined = (
        combined.sort_values(["Ticker", "Date"], kind="stable")
        .drop_duplicates(["Ticker", "Date"], keep="first")
        .reset_index(drop=True)
    )
    _atomic_parquet(cached_path, combined)
    return combined, log


def _current_book(
    closes: pd.DataFrame,
    eligible: Iterable[str],
    previous: Iterable[str],
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    c = closes.where(np.isfinite(closes) & closes.gt(0))
    score = c.shift(21) / c.shift(252) - 1
    if signal_date not in score.index:
        raise RuntimeError(f"signal date {signal_date.date()} absent from price matrix")
    values = score.loc[signal_date, sorted(set(eligible))].dropna()
    ranked = values.rename("score").reset_index().rename(columns={"index": "Ticker"})
    ranked = ranked.sort_values(["score", "Ticker"], ascending=[False, True], kind="stable")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    lookup = ranked.set_index("Ticker")
    held = [ticker for ticker in previous if ticker in lookup.index and lookup.at[ticker, "rank"] <= 20]
    for ticker in ranked.head(10).Ticker:
        if len(held) >= 10:
            break
        if ticker not in held:
            held.append(ticker)
    result = lookup.loc[held].reset_index()
    result["Date"] = signal_date
    result["weight"] = 0.1
    result["retained"] = result.Ticker.isin(set(previous))
    return result[["Date", "Ticker", "weight", "score", "rank", "retained"]]


def refresh_momentum_recommendations(
    paths: ProjectPaths,
    prices: pd.DataFrame,
    generated_at: datetime,
) -> pd.DataFrame:
    source = _source_paths(paths)
    company = _company_map(source)
    calendar = (
        prices.loc[prices.Ticker.eq("SPY"), "Date"].drop_duplicates().sort_values()
    )
    fridays = calendar[pd.to_datetime(calendar).dt.weekday.eq(4)]
    if not len(fridays):
        return pd.DataFrame()
    signal_date = pd.Timestamp(fridays.iloc[-1]).normalize()
    if signal_date <= FREEZE_DATE:
        return pd.DataFrame()
    closes = prices.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
    panel = pd.read_parquet(source["comparison"] / "panel.parquet", columns=["Date", "Ticker", "eligible"])
    final_panel = panel[pd.to_datetime(panel.Date).eq(FREEZE_DATE)]
    u1_eligible = set(final_panel.loc[final_panel.eligible.eq(True), "Ticker"].astype(str))
    old_core = pd.read_parquet(source["comparison"] / "core_plan.parquet")
    previous_u1 = old_core.loc[pd.to_datetime(old_core.Date).eq(FREEZE_DATE), "Ticker"].astype(str)
    core = _current_book(closes, u1_eligible, previous_u1, signal_date)

    membership = pd.read_parquet(source["w100"] / "TRACK1_WIDTH_MEMBERSHIP.parquet")
    membership["refresh"] = pd.to_datetime(membership.refresh)
    refresh = membership.loc[
        membership.cadence.eq("monthly") & membership.universe.eq("W100"), "refresh"
    ].max()
    w100_names = set(
        membership.loc[
            membership.cadence.eq("monthly")
            & membership.universe.eq("W100")
            & membership.refresh.eq(refresh),
            "ticker",
        ].astype(str)
    )
    old_w100 = pd.read_parquet(source["w100"] / "plans/monthly__W100.parquet")
    previous_w100 = old_w100.loc[
        pd.to_datetime(old_w100.Date).eq(FREEZE_DATE), "Ticker"
    ].astype(str)
    w100 = _current_book(closes, w100_names, previous_w100, signal_date)

    rows: list[dict[str, Any]] = []
    rows += _momentum_rows(core, MODEL_SYSTEMIC, company, sleeve="CORE")
    rows += _momentum_rows(core, MODEL_MEGA, company, sleeve="MEGA30")
    rows += _momentum_rows(w100, MODEL_W100, company, sleeve="MEGA30")
    result = pd.DataFrame(rows)
    result["generated_at"] = signal_date.tz_localize("UTC") + pd.Timedelta(hours=23)
    result["data_timestamp"] = signal_date
    result["cost_bps"] = COST_BPS
    result["strategy_version"] = result.model_id
    result["universe_version"] = result.model_id.map(lambda value: SPECS[value].universe)
    result["is_primary"] = result.model_id.map(lambda value: SPECS[value].primary)
    result["risk_state"] = "NORMAL"
    result["action"] = "HOLD"
    return result


def refresh_v73_recommendations(
    paths: ProjectPaths,
    prices: pd.DataFrame,
    output: Path,
    generated_at: datetime,
) -> pd.DataFrame:
    """Extend the exact U001/V7.3 recipe in an isolated stock root.

    The sealed 2019-start score state is regenerated through the newest Friday.
    The 2026-09-11 score/rank/target rows must reproduce before any newer signal
    is admitted to the forward ledger.
    """
    from stock_research.dashboard import engine as dashboard_engine
    from stock_research.dashboard.state import state_from_snapshot

    source = _source_paths(paths)
    calendar = prices.loc[prices.Ticker.eq("SPY"), "Date"].drop_duplicates().sort_values()
    fridays = calendar[pd.to_datetime(calendar).dt.weekday.eq(4)]
    if not len(fridays):
        return pd.DataFrame()
    signal_date = pd.Timestamp(fridays.iloc[-1]).normalize()
    if signal_date <= FREEZE_DATE:
        return pd.DataFrame()

    state = state_from_snapshot(
        paths.repo_root / "config/dashboard_model_registry/universe_snapshots/2026-08-12.json"
    )
    stock_root = output / "data/v73_stock_root"
    for member in state.tickers:
        extension = prices.loc[
            prices.Ticker.eq(member.ticker) & prices.Date.gt(FREEZE_DATE)
        ].copy()
        if extension.empty or extension.Date.max() < signal_date:
            raise RuntimeError(f"V7.3 isolated price is stale: {member.ticker}")
        extension = extension[
            ["Date", "AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "Volume_numeric"]
        ].rename(
            columns={
                "AdjClose": "Close",
                "AdjOpen": "Open",
                "AdjHigh": "High",
                "AdjLow": "Low",
                "Volume_numeric": "Volume",
            }
        )
        canonical_price = (
            source["v73"] / "research_data" / member.price_path
        )
        canonical = pd.read_csv(canonical_price)
        canonical["Date"] = pd.to_datetime(canonical.Date).dt.normalize()
        export = (
            pd.concat([canonical, extension], ignore_index=True)
            .sort_values("Date", kind="stable")
            .drop_duplicates("Date", keep="first")
        )
        destination = stock_root / member.price_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        export.sort_values("Date").to_csv(temporary, index=False)
        temporary.replace(destination)
    isolated_paths = load_paths(stock_root)
    filings = pd.read_csv(
        source["v73"]
        / "research_data/Processed Data/SEC Filings/combined/point_in_time_features.csv"
    )
    panel = dashboard_engine.build_scored_panel(
        isolated_paths,
        state,
        start="2019-01-02",
        end=str(signal_date.date()),
        top_k=5,
        filing_features_override=filings,
    )
    canonical = pd.read_parquet(source["v73"] / "base_model_targets.parquet")
    canonical = canonical.loc[pd.to_datetime(canonical.Date).eq(FREEZE_DATE)].sort_values("Ticker")
    replay = panel.targets.loc[pd.to_datetime(panel.targets.Date).eq(FREEZE_DATE)].sort_values("Ticker")
    if canonical.Ticker.tolist() != replay.Ticker.tolist():
        raise RuntimeError("V7.3 freeze-date ticker replay differs")
    for column in ("AlphaScore", "Rank", "TargetWeight", "ModelSelected"):
        left, right = canonical[column].reset_index(drop=True), replay[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left):
            exact = np.isclose(left, right, rtol=0, atol=1e-12, equal_nan=True).all()
        else:
            exact = left.astype("string").fillna("<NA>").eq(
                right.astype("string").fillna("<NA>")
            ).all()
        if not exact:
            raise RuntimeError(f"V7.3 freeze-date parity failed: {column}")
    latest = panel.targets.loc[pd.to_datetime(panel.targets.Date).eq(signal_date)].copy()
    latest = latest.loc[
        latest.Rank.notna()
        & (latest.Rank.le(10) | latest.TargetWeight.gt(0))
    ].sort_values(["Rank", "Ticker"])
    rows = []
    for row in latest.itertuples():
        rows.append(
            {
                "model_id": MODEL_V73,
                "signal_date": signal_date,
                "effective_trade_date": _next_session(signal_date),
                "Ticker": str(row.Ticker),
                "company": str(row.Company),
                "rank": int(row.Rank),
                "score": float(row.AlphaScore),
                "target_weight": float(row.TargetWeight),
                "recommendation_basis": "FROZEN_V73_ALPHA_SCORE_AND_HOLD_RULE",
                "generated_at": signal_date.tz_localize("UTC") + pd.Timedelta(hours=23),
                "data_timestamp": signal_date,
                "cost_bps": COST_BPS,
                "strategy_version": MODEL_V73,
                "universe_version": SPECS[MODEL_V73].universe,
                "is_primary": True,
                "risk_state": "NORMAL",
                "action": "HOLD",
            }
        )
    return pd.DataFrame(rows)


def _systemic_risk_timeline(
    paths: ProjectPaths,
    prices: pd.DataFrame,
    output: Path,
    unscaled_nav: pd.DataFrame,
) -> pd.DataFrame:
    """Continue the frozen JOINT latch with completed-session inputs only."""
    from stock_research.dashboard.frozen_systemic_risk import (
        build_market_features,
    )
    from stock_research.macro_momentum_sp500.tactical_defense import (
        TUNED_DEFENSE,
        evaluate_defense,
    )

    source = _source_paths(paths)
    sessions = pd.DatetimeIndex(
        prices.loc[prices.Ticker.eq("SPY"), "Date"].drop_duplicates().sort_values()
    )
    closes = prices.pivot(index="Date", columns="Ticker", values="AdjClose").reindex(
        sessions
    )
    anchor = pd.read_parquet(
        source["comparison"] / "panel.parquet",
        columns=["Date", "Ticker", "eligible"],
    )
    extended = SimpleNamespace(frame=anchor, sessions=sessions, closes=closes)
    raw = prices[["Date", "Ticker", "Close", "Volume_numeric"]].copy()

    vix_path = output / "data/SHADOW_VIX.parquet"
    stored_vix = (
        pd.read_parquet(vix_path)
        if vix_path.exists()
        else pd.read_parquet(source["systemic"] / "vix.parquet")
    )
    latest_vix = []
    for symbol in ("VIX", "VIX3M"):
        response = requests.get(
            f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv",
            timeout=30,
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        frame["Date"] = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
        frame = frame[["Date", "CLOSE"]].rename(columns={"CLOSE": symbol})
        latest_vix.append(frame)
    incoming_vix = latest_vix[0].merge(latest_vix[1], on="Date", how="outer")
    vix = (
        pd.concat([stored_vix, incoming_vix], ignore_index=True)
        .sort_values("Date", kind="stable")
        .drop_duplicates("Date", keep="first")
    )
    _atomic_parquet(vix_path, vix)
    market = build_market_features(extended, raw, vix).reset_index(drop=True)

    m0 = pd.read_parquet(
        source["systemic"] / "accounts/rank1__M0__BASE25/nav.parquet"
    )[["Date", "daily_return"]]
    forward = unscaled_nav.loc[
        unscaled_nav.model_id.eq(MODEL_SYSTEMIC) & unscaled_nav.Date.gt(FREEZE_DATE),
        ["Date", "daily_return"],
    ]
    returns = (
        pd.concat([m0.loc[m0.Date.le(FREEZE_DATE)], forward], ignore_index=True)
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
    )
    squares = returns.daily_return.pow(2)
    denominator = squares.rolling(20, min_periods=20).sum()
    returns["downside_ratio20"] = (
        returns.daily_return.clip(upper=0).pow(2).rolling(20, min_periods=20).sum()
        / denominator.where(denominator.gt(0))
    )
    signals = market.merge(
        returns[["Date", "downside_ratio20"]], on="Date", how="left"
    )
    spy = prices.loc[prices.Ticker.eq("SPY"), ["Date", "AdjClose"]].rename(
        columns={"AdjClose": "Close"}
    )
    macro_state = evaluate_defense(spy, TUNED_DEFENSE)
    macro = pd.DataFrame(
        {
            "Date": spy.Date.to_numpy(),
            "legacy_target_exposure": macro_state.exposure,
        }
    )
    signals = signals.merge(macro, on="Date", how="left", validate="one_to_one")

    prior = pd.read_parquet(
        source["systemic"]
        / "accounts/rank1__JOINT__BASE25/controller_audit.parquet"
    ).sort_values("Date")
    last = prior.iloc[-1]
    gate_factor, clear = float(last.factor), int(last.clear)
    rows = []
    for row in signals.loc[signals.Date.gt(pd.Timestamp(last.Date))].itertuples():
        inputs = (row.ar_shift, row.turbulence_pct, row.downside_ratio20)
        value = (
            float(row.ar_shift >= 1 and row.turbulence_pct >= 0.90 and row.downside_ratio20 > 0.5)
            if all(np.isfinite(value) for value in inputs)
            else math.nan
        )
        if not np.isfinite(value):
            clear = 0
        elif value >= 0.5:
            gate_factor, clear = 0.5, 0
        else:
            clear += 1
            if clear >= 5:
                gate_factor = 1.0
        joint_factor = float(gate_factor)
        macro_exposure = float(row.legacy_target_exposure)
        factor = joint_factor * macro_exposure
        rows.append(
            {
                "signal_date": pd.Timestamp(row.Date),
                "alarm": value,
                "factor": factor,
                "joint_factor": joint_factor,
                "macro_exposure": macro_exposure,
                "clear": int(clear),
                "ar_shift": float(row.ar_shift),
                "turbulence_pct": float(row.turbulence_pct),
                "downside_ratio20": float(row.downside_ratio20),
            }
        )
    result = pd.DataFrame(rows)
    _atomic_parquet(output / "data/SHADOW_SYSTEMIC_RISK_INPUTS.parquet", result)
    return result


def _apply_systemic_risk(
    recommendations: pd.DataFrame, timeline: pd.DataFrame
) -> pd.DataFrame:
    base = recommendations.loc[recommendations.model_id.eq(MODEL_SYSTEMIC)].copy()
    other = recommendations.loc[~recommendations.model_id.eq(MODEL_SYSTEMIC)].copy()
    signal_dates = set(pd.to_datetime(base.signal_date))
    rows: list[pd.DataFrame] = []
    previous_factor: float | None = None
    for risk in timeline.itertuples():
        date = pd.Timestamp(risk.signal_date)
        changed = previous_factor is None or not math.isclose(
            float(risk.factor), previous_factor, abs_tol=1e-12
        )
        previous_factor = float(risk.factor)
        if date < FREEZE_DATE or (date not in signal_dates and not changed):
            continue
        available = base.loc[pd.to_datetime(base.signal_date).le(date)]
        source_date = pd.to_datetime(available.signal_date).max()
        snapshot = available.loc[pd.to_datetime(available.signal_date).eq(source_date)].copy()
        snapshot["signal_date"] = date
        snapshot["effective_trade_date"] = _next_session(date)
        snapshot["target_weight"] = snapshot.target_weight * float(risk.factor)
        snapshot["risk_state"] = (
            f"JOINT_{float(risk.joint_factor):.1f}_M1_{float(risk.macro_exposure):.1f}"
        )
        snapshot["recommendation_basis"] = "FROZEN_SYSTEMIC_JOINT"
        snapshot["data_timestamp"] = date
        snapshot["generated_at"] = date.tz_localize("UTC") + pd.Timedelta(hours=23)
        rows.append(snapshot)
    if not rows:
        raise RuntimeError("no forward SYSTEMIC JOINT state could be produced")
    systemic = pd.concat(rows, ignore_index=True)
    return pd.concat([other, systemic], ignore_index=True, sort=False)


def _vol15_recommendations(
    paths: ProjectPaths,
    recommendations: pd.DataFrame,
    systemic_nav: pd.DataFrame,
) -> pd.DataFrame:
    """Make the frozen 15%-vol rule executable at each next session open."""
    source = _source_paths(paths)
    historical = pd.read_parquet(
        source["systemic"] / "accounts/rank1__JOINT__BASE25/nav.parquet"
    )[["Date", "daily_return"]]
    forward = systemic_nav.loc[
        systemic_nav.model_id.eq(MODEL_SYSTEMIC) & systemic_nav.Date.gt(FREEZE_DATE),
        ["Date", "daily_return"],
    ]
    base = (
        pd.concat([historical.loc[historical.Date.le(FREEZE_DATE)], forward])
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .set_index("Date")
    )
    sigma = base.daily_return.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)
    exposure = (0.15 / sigma).clip(upper=1.0)
    systemic = recommendations.loc[recommendations.model_id.eq(MODEL_SYSTEMIC)].copy()
    rows = []
    for date, value in exposure.items():
        date = pd.Timestamp(date)
        if date < FREEZE_DATE or not np.isfinite(value):
            continue
        available = systemic.loc[pd.to_datetime(systemic.signal_date).le(date)]
        if available.empty:
            continue
        source_date = pd.to_datetime(available.signal_date).max()
        snapshot = available.loc[pd.to_datetime(available.signal_date).eq(source_date)].copy()
        snapshot["model_id"] = MODEL_VOL15
        snapshot["strategy_version"] = MODEL_VOL15
        snapshot["signal_date"] = date
        snapshot["effective_trade_date"] = _next_session(date)
        snapshot["target_weight"] = snapshot.target_weight * float(value)
        snapshot["recommendation_basis"] = "FROZEN_SYSTEMIC_JOINT_TIMES_VOL15"
        snapshot["risk_state"] = f"VOL15_EXPOSURE_{float(value):.6f}"
        snapshot["universe_version"] = SPECS[MODEL_VOL15].universe
        snapshot["is_primary"] = True
        snapshot["data_timestamp"] = date
        snapshot["generated_at"] = date.tz_localize("UTC") + pd.Timedelta(hours=23)
        rows.append(snapshot)
    if not rows:
        raise RuntimeError("no VOL15 exposure rows produced")
    return pd.concat(rows, ignore_index=True)


def _target_map(recommendations: pd.DataFrame, model_id: str, date: pd.Timestamp) -> dict[str, float]:
    candidates = recommendations.loc[
        recommendations.model_id.eq(model_id)
        & pd.to_datetime(recommendations.effective_trade_date).le(date)
    ]
    if candidates.empty:
        return {}
    signal = pd.to_datetime(candidates.signal_date).max()
    latest = candidates.loc[pd.to_datetime(candidates.signal_date).eq(signal)]
    return {
        str(row.Ticker): float(row.target_weight)
        for row in latest.itertuples()
        if float(row.target_weight) > 0
    }


def run_accounts(
    recommendations: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    model_ids: Iterable[str] = ALL_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild forward rows deterministically; append guard prevents revisions."""
    opens = prices.pivot(index="Date", columns="Ticker", values="AdjOpen").sort_index()
    closes = prices.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
    sessions = pd.DatetimeIndex(opens.index)
    sessions = sessions[sessions >= FREEZE_DATE]
    holdings_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        cash = INITIAL_NAV
        quantity: dict[str, float] = {}
        current_target: dict[str, float] = {}
        cumulative_cost = 0.0
        cumulative_turnover = 0.0
        peak = INITIAL_NAV
        previous_nav = INITIAL_NAV
        returns: list[float] = []
        signals = recommendations.loc[recommendations.model_id.eq(model_id)].copy()
        effective_dates = set(pd.to_datetime(signals.effective_trade_date))
        for date in sessions:
            if date == FREEZE_DATE:
                nav = INITIAL_NAV
                market_value = 0.0
                turnover = 0.0
            else:
                turnover = 0.0
                if date in effective_dates:
                    execution_sequence = 0
                    desired = _target_map(recommendations, model_id, date)
                    before = {
                        ticker: quantity.get(ticker, 0.0) * float(opens.at[date, ticker])
                        for ticker in quantity
                        if ticker in opens and pd.notna(opens.at[date, ticker])
                    }
                    gross_before = cash + sum(before.values())
                    # Sell first.  The buy budget is based on post-sale cash and
                    # the pre-trade NAV; each side pays its own 25bp cost.
                    for ticker in sorted(set(quantity) - set(desired)):
                        execution_sequence += 1
                        price = float(opens.at[date, ticker])
                        qty = quantity.pop(ticker)
                        notional = qty * price
                        cost = notional * COST_BPS / 10_000
                        cash += notional - cost
                        cumulative_cost += cost
                        turnover += notional
                        trade_rows.append(
                            {
                                "model_id": model_id,
                                "signal_date": signals.loc[
                                    pd.to_datetime(signals.effective_trade_date).eq(date), "signal_date"
                                ].max(),
                                "trade_date": date,
                                "Ticker": ticker,
                                "side": "SELL",
                                "quantity": qty,
                                "price": price,
                                "notional": notional,
                                "transaction_cost": cost,
                                "post_quantity": 0.0,
                                "reason": "NEXT_OPEN_TARGET_EXIT",
                                "execution": "NEXT_ACTUAL_SESSION_OPEN_SELL_BEFORE_BUY",
                                "execution_sequence": execution_sequence,
                            }
                        )
                    # Compute the whole rebalance first, then reduce existing
                    # positions before funding any buys.  Alphabetical order
                    # must never accidentally put a buy ahead of a sell.
                    adjustments = []
                    for ticker in sorted(desired):
                        price = float(opens.at[date, ticker])
                        wanted_value = gross_before * desired[ticker]
                        existing_value = quantity.get(ticker, 0.0) * price
                        delta = wanted_value - existing_value
                        if abs(delta) > 1e-8:
                            adjustments.append(("BUY" if delta > 0 else "SELL", ticker, price, delta))
                    adjustments.sort(key=lambda value: (value[0] == "BUY", value[1]))
                    for side, ticker, price, delta in adjustments:
                        execution_sequence += 1
                        notional = abs(delta)
                        cost = notional * COST_BPS / 10_000
                        qty = notional / price
                        if side == "BUY":
                            # Fees reduce cash; scale the final buy if tiny
                            # floating-point differences exceed cash.
                            required = notional + cost
                            if required > cash:
                                notional = cash / (1 + COST_BPS / 10_000)
                                cost = notional * COST_BPS / 10_000
                                qty = notional / price
                            cash -= notional + cost
                            quantity[ticker] = quantity.get(ticker, 0.0) + qty
                        else:
                            qty = min(qty, quantity.get(ticker, 0.0))
                            notional = qty * price
                            cost = notional * COST_BPS / 10_000
                            cash += notional - cost
                            quantity[ticker] = quantity.get(ticker, 0.0) - qty
                            if quantity[ticker] <= 1e-12:
                                quantity.pop(ticker, None)
                        cumulative_cost += cost
                        turnover += notional
                        trade_rows.append(
                            {
                                "model_id": model_id,
                                "signal_date": signals.loc[
                                    pd.to_datetime(signals.effective_trade_date).eq(date), "signal_date"
                                ].max(),
                                "trade_date": date,
                                "Ticker": ticker,
                                "side": side,
                                "quantity": qty,
                                "price": price,
                                "notional": notional,
                                "transaction_cost": cost,
                                "post_quantity": quantity.get(ticker, 0.0),
                                "reason": "NEXT_OPEN_FROZEN_TARGET_REBALANCE",
                                "execution": "NEXT_ACTUAL_SESSION_OPEN_SELL_BEFORE_BUY",
                                "execution_sequence": execution_sequence,
                            }
                        )
                    current_target = desired
                market_value = 0.0
                for ticker, qty in sorted(quantity.items()):
                    close = float(closes.at[date, ticker])
                    value = qty * close
                    market_value += value
                    holdings_rows.append(
                        {
                            "model_id": model_id,
                            "Date": date,
                            "Ticker": ticker,
                            "quantity": qty,
                            "close": close,
                            "market_value": value,
                            "target_weight": current_target.get(ticker, 0.0),
                        }
                    )
                nav = cash + market_value
            daily_return = nav / previous_nav - 1 if date != FREEZE_DATE else 0.0
            returns.append(daily_return)
            peak = max(peak, nav)
            current_drawdown = nav / peak - 1
            series = np.asarray(returns, dtype=float)
            sharpe = (
                float(np.sqrt(252) * series.mean() / series.std(ddof=1))
                if len(series) >= 2 and series.std(ddof=1) > 0
                else math.nan
            )
            cumulative_turnover += turnover / previous_nav if previous_nav > 0 else 0.0
            nav_rows.append(
                {
                    "model_id": model_id,
                    "Date": date,
                    "nav": nav,
                    "cash": cash,
                    "market_value": market_value,
                    "gross_exposure": market_value / nav if nav > 0 else math.nan,
                    "daily_pnl": nav - previous_nav,
                    "daily_return": daily_return,
                    "current_drawdown": current_drawdown,
                    "shadow_mdd": min(
                        [0.0, current_drawdown]
                        + [
                            row["current_drawdown"]
                            for row in nav_rows
                            if row["model_id"] == model_id
                        ]
                    ),
                    "sharpe_rf0": sharpe,
                    "turnover": cumulative_turnover,
                    "transaction_cost": cumulative_cost,
                }
            )
            previous_nav = nav
    nav = pd.DataFrame(nav_rows)
    if len(nav):
        nav["weekly_pnl"] = nav.groupby("model_id")["nav"].transform(
            lambda s: s - s.shift(5).fillna(INITIAL_NAV)
        )
        nav["monthly_pnl"] = nav.groupby("model_id")["nav"].transform(
            lambda s: s - s.shift(21).fillna(INITIAL_NAV)
        )
    holdings = pd.DataFrame(
        holdings_rows,
        columns=[
            "model_id",
            "Date",
            "Ticker",
            "quantity",
            "close",
            "market_value",
            "target_weight",
        ],
    )
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "model_id",
            "signal_date",
            "trade_date",
            "Ticker",
            "side",
            "quantity",
            "price",
            "notional",
            "transaction_cost",
            "post_quantity",
            "reason",
            "execution",
            "execution_sequence",
        ],
    )
    return holdings, trades, nav


def apply_actions(recommendations: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    result = recommendations.copy()
    actions: list[str] = []
    for row in result.itertuples():
        before = holdings.loc[
            holdings.model_id.eq(row.model_id)
            & pd.to_datetime(holdings.Date).le(pd.Timestamp(row.signal_date)),
            "Ticker",
        ]
        held = str(row.Ticker) in set(before.astype(str))
        if float(row.target_weight) > 0:
            actions.append("HOLD" if held else "BUY")
        else:
            actions.append("SELL" if held else "HOLD")
    result["action"] = actions
    return result


def risk_rows(recommendations: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id in ALL_MODELS:
        rec = recommendations.loc[recommendations.model_id.eq(model_id)]
        signal_date = pd.to_datetime(rec.signal_date).max()
        latest = rec.loc[pd.to_datetime(rec.signal_date).eq(signal_date)]
        target = float(latest.target_weight.sum())
        account = nav.loc[nav.model_id.eq(model_id)].sort_values("Date").iloc[-1]
        state = str(latest.risk_state.iloc[0])
        rows.append(
            {
                "model_id": model_id,
                "Date": account.Date,
                "signal_date": signal_date,
                "risk_state": state,
                "target_exposure": target,
                "actual_exposure": float(account.gross_exposure),
                "cash": float(account.cash),
                "source": (
                    "FROZEN_DAILY_VOL15_ON_SYSTEMIC_RETURNS"
                    if model_id == MODEL_VOL15
                    else "FROZEN_SYSTEMIC_JOINT_LATCH"
                    if model_id == MODEL_SYSTEMIC
                    else "FROZEN_MODEL_POLICY"
                ),
            }
        )
    return pd.DataFrame(rows)


def comparison_rows(recommendations: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    focus = {
        "SYSTEMIC": MODEL_SYSTEMIC,
        "B_U1": MODEL_MEGA,
        "V7_3": MODEL_V73,
    }
    latest: dict[str, set[str]] = {}
    held: dict[str, set[str]] = {}
    dates: dict[str, pd.Timestamp] = {}
    for label, model_id in focus.items():
        frame = recommendations.loc[recommendations.model_id.eq(model_id)]
        date = pd.to_datetime(frame.signal_date).max()
        dates[label] = date
        latest[label] = set(
            frame.loc[pd.to_datetime(frame.signal_date).eq(date) & frame.target_weight.gt(0), "Ticker"]
        )
        h = holdings.loc[holdings.model_id.eq(model_id)]
        held[label] = set(h.loc[pd.to_datetime(h.Date).eq(pd.to_datetime(h.Date).max()), "Ticker"])
    union = sorted(set().union(*latest.values()))
    as_of_date = (
        pd.to_datetime(holdings.Date).max()
        if len(holdings)
        else FREEZE_DATE
    )
    rows = []
    for ticker in union:
        flags = {label: ticker in names for label, names in latest.items()}
        count = sum(flags.values())
        rows.append(
            {
                "as_of_date": as_of_date,
                "signal_date": max(dates.values()),
                "Ticker": ticker,
                **flags,
                "consensus_count": count,
                "model_overlap": f"{count}/3",
                "unique_recommendation": count == 1,
                "recommendation_divergence": count == 1,
                "holdings_overlap_count": sum(ticker in names for names in held.values()),
            }
        )
    return pd.DataFrame(rows)


def universe_comparison_rows(recommendations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, model_id in (("B_U1", MODEL_MEGA), ("B_W100", MODEL_W100)):
        frame = recommendations.loc[recommendations.model_id.eq(model_id)]
        date = pd.to_datetime(frame.signal_date).max()
        frame = frame.loc[pd.to_datetime(frame.signal_date).eq(date) & frame.target_weight.gt(0)]
        for row in frame.itertuples():
            rows.append(
                {
                    "signal_date": date,
                    "variant": label,
                    "Ticker": row.Ticker,
                    "rank": row.rank,
                    "score": row.score,
                    "target_weight": row.target_weight,
                    "universe_version": SPECS[model_id].universe,
                }
            )
    out = pd.DataFrame(rows)
    u1 = set(out.loc[out.variant.eq("B_U1"), "Ticker"])
    w100 = set(out.loc[out.variant.eq("B_W100"), "Ticker"])
    out["in_both"] = out.Ticker.isin(u1 & w100)
    out["only_in_variant"] = ~out.in_both
    out["recommendation_jaccard"] = len(u1 & w100) / len(u1 | w100) if u1 | w100 else 1.0
    return out


def verification_payload(
    output: Path,
    recommendations: pd.DataFrame,
    holdings: pd.DataFrame,
    trades: pd.DataFrame,
    nav: pd.DataFrame,
    frozen_checks: list[dict[str, Any]],
    status: dict[str, Any],
) -> dict[str, Any]:
    hashes = {
        name: _sha256(output / filename)
        for name, filename in OUTPUT_FILES.items()
        if (output / filename).exists()
    }
    input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "frozen": [row["actual_sha256"] for row in frozen_checks],
                "prices": _sha256(output / "data/SHADOW_PRICES.parquet"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    prior_path = output / "SHADOW_VERIFICATION.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    same_input = (
        prior.get("input_fingerprint") == input_fingerprint
        and prior.get("dataset_sha256") == hashes
    )
    no_future = bool(
        (
            pd.to_datetime(recommendations.data_timestamp)
            <= pd.to_datetime(recommendations.signal_date)
        ).all()
    )
    next_open = True
    if len(trades):
        effective = recommendations[
            ["model_id", "signal_date", "Ticker", "effective_trade_date"]
        ].drop_duplicates()
        joined = trades.merge(
            effective,
            on=["model_id", "signal_date", "Ticker"],
            how="left",
            validate="many_to_one",
        )
        next_open = bool(
            joined.effective_trade_date.notna().all()
            and pd.to_datetime(joined.trade_date)
            .eq(pd.to_datetime(joined.effective_trade_date))
            .all()
        )
    costs = bool(
        np.isclose(
            trades.transaction_cost,
            trades.notional * COST_BPS / 10_000,
            rtol=0,
            atol=1e-9,
        ).all()
    )
    sell_first = True
    for _, group in trades.groupby(["model_id", "trade_date"]):
        sells = group.loc[group.side.eq("SELL"), "execution_sequence"]
        buys = group.loc[group.side.eq("BUY"), "execution_sequence"]
        if len(sells) and len(buys) and sells.max() >= buys.min():
            sell_first = False
            break
    marked = holdings.groupby(["model_id", "Date"]).market_value.sum()
    check_nav = nav.set_index(["model_id", "Date"])
    values = marked.reindex(check_nav.index, fill_value=0.0) + check_nav.cash
    daily_mark = bool(np.isclose(values, check_nav.nav, rtol=0, atol=1e-7).all())
    allowed = set(ALL_MODELS)
    model_isolation = all(
        set(frame.model_id.astype(str)).issubset(allowed)
        for frame in (recommendations, holdings, trades, nav)
    )
    separation = bool(pd.to_datetime(nav.Date).min() == FREEZE_DATE)
    checks = {
        "same_input_same_recommendation": same_input,
        "no_future_data": no_future,
        "next_open_execution": next_open,
        "transaction_costs_25bp": costs,
        "sell_before_buy": sell_first,
        "daily_mark_to_market": daily_mark,
        "frozen_hash_unchanged": all(row["unchanged"] for row in frozen_checks),
        "model_version_isolation": model_isolation,
        "historical_forward_separation": separation,
        "stale_data_warning_active": status["status"] == "STALE_WARNING"
        if status["stale_models"] or status["price_failures"]
        else status["status"] == "OK",
    }
    return {
        "status": "PASS" if all(checks.values()) else "PENDING_REPLAY" if not same_input and all(v for k, v in checks.items() if k != "same_input_same_recommendation") else "FAIL",
        "checks": checks,
        "input_fingerprint": input_fingerprint,
        "dataset_sha256": hashes,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].dt.strftime("%Y-%m-%d")
    cleaned = cleaned.replace({np.nan: None, pd.NaT: None})
    return cleaned.to_dict("records")


def export_dashboard_json(
    output: Path,
    destination: Path,
    recommendations: pd.DataFrame,
    holdings: pd.DataFrame,
    trades: pd.DataFrame,
    nav: pd.DataFrame,
    risk: pd.DataFrame,
    comparison: pd.DataFrame,
    universe: pd.DataFrame,
    status: dict[str, Any],
) -> None:
    models = []
    for model_id in ALL_MODELS:
        spec = SPECS[model_id]
        rec = recommendations.loc[recommendations.model_id.eq(model_id)]
        signal_date = pd.to_datetime(rec.signal_date).max()
        current_rec = rec.loc[pd.to_datetime(rec.signal_date).eq(signal_date)].sort_values(
            ["rank", "Ticker"], na_position="last"
        )
        account = nav.loc[nav.model_id.eq(model_id)].sort_values("Date")
        latest_date = pd.to_datetime(account.Date).max()
        current_holdings = holdings.loc[
            holdings.model_id.eq(model_id) & pd.to_datetime(holdings.Date).eq(latest_date)
        ]
        latest_nav = account.iloc[-1]
        models.append(
            {
                "model_id": model_id,
                "short_name": spec.short_name,
                "role": spec.role,
                "universe": spec.universe,
                "primary": spec.primary,
                "synthetic_reference": spec.synthetic,
                "signal_date": str(signal_date.date()),
                "recommendations": _records(current_rec.head(10)),
                "holdings": _records(current_holdings),
                "risk": _records(risk.loc[risk.model_id.eq(model_id)]),
                "metrics": {
                    "nav": float(latest_nav.nav),
                    "cash": float(latest_nav.cash),
                    "exposure": float(latest_nav.gross_exposure),
                    "daily_pnl": float(latest_nav.daily_pnl),
                    "weekly_pnl": float(latest_nav.weekly_pnl),
                    "monthly_pnl": float(latest_nav.monthly_pnl),
                    "current_drawdown": float(latest_nav.current_drawdown),
                    "shadow_mdd": float(latest_nav.shadow_mdd),
                    "sharpe": None if pd.isna(latest_nav.sharpe_rf0) else float(latest_nav.sharpe_rf0),
                    "turnover": float(latest_nav.turnover),
                    "transaction_cost": float(latest_nav.transaction_cost),
                },
                "nav_series": _records(account[["Date", "nav", "current_drawdown"]]),
            }
        )
    payload = {
        "stage": "ALPHA_DESK_FORWARD_SHADOW_V1",
        "historical_label": "HISTORICAL DEVELOPMENT",
        "forward_label": "FORWARD SHADOW — AFTER FREEZE",
        "freeze_date": str(FREEZE_DATE.date()),
        "shadow_start_date": str(_next_session(FREEZE_DATE).date()),
        "status": status,
        "models": models,
        "model_comparison": _records(comparison),
        "universe_comparison": _records(universe),
        "recent_trades": _records(trades.sort_values("trade_date").tail(100)),
    }
    _atomic_json(destination, payload)


def run_forward_shadow(
    paths: ProjectPaths,
    *,
    output: Path | None = None,
    public_json: Path | None = None,
    refresh_market: bool = True,
    cutoff: pd.Timestamp | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    output = output or paths.results / "alpha_desk_forward_shadow_v1"
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC)
    cutoff = pd.Timestamp(cutoff or pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None)).normalize()
    manifest, frozen_checks = verify_frozen_sources(paths)
    git_commit = _git_head(paths.repo_root)

    seed = seed_recommendations(paths, generated_at)
    base_path = output / "data/SHADOW_BASE_RECOMMENDATIONS.parquet"
    base_recommendations = append_immutable(
        base_path,
        seed,
        ["model_id", "signal_date", "Ticker"],
        sort=["signal_date", "model_id", "rank", "Ticker"],
    )
    price_log: list[dict[str, Any]] = []
    if refresh_market:
        prices, price_log = refresh_prices(paths, output, cutoff=cutoff, workers=workers)
    else:
        cached = output / "data/SHADOW_PRICES.parquet"
        prices = pd.read_parquet(cached) if cached.exists() else pd.read_parquet(
            _source_paths(paths)["comparison"] / "raw.parquet"
        )
    prices["Date"] = pd.to_datetime(prices.Date).dt.normalize()
    try:
        newer = refresh_momentum_recommendations(paths, prices, generated_at)
    except Exception as error:  # noqa: BLE001 - fail closed to a stale-status payload
        newer = pd.DataFrame()
        price_log.append({"Ticker": "MOMENTUM_SIGNAL", "status": "ERROR", "error": str(error)})
    try:
        v73_newer = refresh_v73_recommendations(
            paths, prices, output, generated_at
        )
    except Exception as error:  # noqa: BLE001 - parity failures must become stale warnings
        v73_newer = pd.DataFrame()
        price_log.append({"Ticker": "V73_SIGNAL", "status": "ERROR", "error": str(error)})
    if len(v73_newer):
        newer = pd.concat([newer, v73_newer], ignore_index=True, sort=False)
    if len(newer):
        base_recommendations = append_immutable(
            base_path,
            newer,
            ["model_id", "signal_date", "Ticker"],
            sort=["signal_date", "model_id", "rank", "Ticker"],
        )

    # The JOINT alarm is defined on an unoverlaid reference account.  Build
    # that reference first, continue the frozen latch, then make VOL15 depend
    # on the resulting SYSTEMIC daily returns.  No parameter is refit here.
    _, _, reference_nav = run_accounts(
        base_recommendations, prices, model_ids=[MODEL_SYSTEMIC]
    )
    systemic_timeline = _systemic_risk_timeline(
        paths, prices, output, reference_nav
    )
    recommendations = _apply_systemic_risk(base_recommendations, systemic_timeline)
    _, _, systemic_nav = run_accounts(
        recommendations, prices, model_ids=[MODEL_SYSTEMIC]
    )
    vol15 = _vol15_recommendations(paths, recommendations, systemic_nav)
    recommendations = pd.concat([recommendations, vol15], ignore_index=True, sort=False)
    holdings, trades, nav = run_accounts(recommendations, prices)
    recommendations = apply_actions(recommendations, holdings)
    rec_path = output / OUTPUT_FILES["recommendations"]
    recommendations = append_immutable(
        rec_path,
        recommendations,
        ["model_id", "signal_date", "Ticker"],
        sort=["signal_date", "model_id", "rank", "Ticker"],
    )
    risk = risk_rows(recommendations, nav)
    comparison = comparison_rows(recommendations, holdings)
    universe = universe_comparison_rows(recommendations)
    datasets = {
        "holdings": (holdings, ["model_id", "Date", "Ticker"]),
        "trades": (trades, ["model_id", "trade_date", "Ticker", "side"]),
        "nav": (nav, ["model_id", "Date"]),
        "risk": (risk, ["model_id", "Date"]),
        "comparison": (comparison, ["as_of_date", "signal_date", "Ticker"]),
        "universe": (universe, ["signal_date", "variant", "Ticker"]),
    }
    stored: dict[str, pd.DataFrame] = {}
    for name, (frame, keys) in datasets.items():
        stored[name] = append_immutable(
            output / OUTPUT_FILES[name], frame, keys, sort=keys
        )

    completed = prices.loc[prices.Ticker.eq("SPY"), "Date"].max()
    latest_signal_by_model = {
        model_id: str(
            pd.to_datetime(recommendations.loc[recommendations.model_id.eq(model_id), "signal_date"])
            .max()
            .date()
        )
        for model_id in ALL_MODELS
    }
    errors = [row for row in price_log if row.get("status") == "ERROR"]
    expected_signal = pd.to_datetime(
        prices.loc[
            prices.Ticker.eq("SPY") & pd.to_datetime(prices.Date).dt.weekday.eq(4),
            "Date",
        ]
    ).max()
    stale_models = [
        model_id
        for model_id, value in latest_signal_by_model.items()
        if pd.Timestamp(value) < expected_signal
    ]
    status = {
        "status": "STALE_WARNING" if errors or stale_models else "OK",
        "generated_at": generated_at.isoformat(),
        "freeze_date": str(FREEZE_DATE.date()),
        "shadow_start_date": str(_next_session(FREEZE_DATE).date()),
        "market_as_of": str(pd.Timestamp(completed).date()),
        "latest_signal_by_model": latest_signal_by_model,
        "stale_models": stale_models,
        "price_failures": errors,
        "historical_forward_separated": True,
        "broker_orders_enabled": False,
        "scheduler": "EXISTING_ALPHA_DESK_GITHUB_ACTIONS",
    }
    _atomic_json(output / "SHADOW_STATUS.json", status)
    verification = verification_payload(
        output,
        recommendations,
        stored["holdings"],
        stored["trades"],
        stored["nav"],
        frozen_checks,
        status,
    )
    _atomic_json(output / "SHADOW_VERIFICATION.json", verification)
    version = {
        "stage": "ALPHA_DESK_FORWARD_SHADOW_V1",
        "created_at": generated_at.isoformat(),
        "git_commit": git_commit,
        "freeze_date": str(FREEZE_DATE.date()),
        "models": [
            {
                **SPECS[model_id].__dict__,
                "strategy_version": model_id,
                "cost_bps_per_side": COST_BPS,
                "execution": "COMPLETED_SIGNAL_CLOSE_TO_NEXT_ACTUAL_SESSION_OPEN",
            }
            for model_id in ALL_MODELS
        ],
        "final_audit_manifest": manifest,
        "frozen_source_checks": frozen_checks,
        "required_files": [
            *OUTPUT_FILES.values(),
            "SHADOW_STATUS.json",
            "SHADOW_VERSION_MANIFEST.json",
            "SHADOW_VERIFICATION.json",
        ],
    }
    _atomic_json(output / "SHADOW_VERSION_MANIFEST.json", version)
    if public_json is not None:
        export_dashboard_json(
            output,
            public_json,
            recommendations,
            stored["holdings"],
            stored["trades"],
            stored["nav"],
            stored["risk"],
            stored["comparison"],
            stored["universe"],
            status,
        )
    return {
        "output": str(output),
        "status": status,
        "verification": verification,
        "rows": {name: len(frame) for name, frame in stored.items()},
        "recommendation_rows": len(recommendations),
    }
