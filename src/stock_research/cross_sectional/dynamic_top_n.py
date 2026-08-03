from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths
from stock_research.tickers import load_tickers

from .big_tech_10 import (
    build_single_asset_equity,
    calendar_return_rows,
    summarize_equity_curve,
)
from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .optimization import optimize_strategy
from .pit_universe_builder import local_lookup_ticker
from .pit_validation import apply_membership_to_panel
from .portfolio import PortfolioResult, run_portfolio_backtest
from .signals import (
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)
from .v7_pit_evaluation import (
    build_v7_source_panel,
    load_raw_company_prices,
    load_ready_tickers,
    normalize_change_membership,
)
from .v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)


@dataclass(frozen=True)
class DynamicTopNArtifacts:
    output_dir: Path
    candidate_summary_csv: Path
    period_summary_csv: Path
    calendar_returns_csv: Path
    equity_csv: Path
    membership_csv: Path
    membership_coverage_csv: Path
    signal_coverage_csv: Path
    data_audit_csv: Path
    price_only_audit_csv: Path
    selected_signals_csv: Path
    executions_csv: Path
    manifest_json: Path


def build_sp500_top_n_membership(
    direct_rankings: pd.DataFrame,
    sp500_membership: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Filter a dated published US market-cap ranking to S&P members."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    direct_required = {"AsOfDate", "Ticker", "Rank", "MarketCap"}
    membership_required = {"AsOfDate", "DataSymbol"}
    direct_missing = sorted(direct_required - set(direct_rankings.columns))
    membership_missing = sorted(
        membership_required - set(sp500_membership.columns)
    )
    if direct_missing:
        raise ValueError(
            f"Direct rankings are missing columns: {direct_missing}"
        )
    if membership_missing:
        raise ValueError(
            f"S&P membership is missing columns: {membership_missing}"
        )

    direct = direct_rankings.copy()
    direct["AsOfDate"] = pd.to_datetime(direct["AsOfDate"], errors="raise")
    direct["PublishedRank"] = pd.to_numeric(
        direct["Rank"], errors="raise"
    )
    direct["MarketCap"] = pd.to_numeric(
        direct["MarketCap"], errors="raise"
    )
    direct["HistoricalTicker"] = (
        direct["Ticker"].astype(str).str.upper().str.strip()
    )
    direct["DataSymbol"] = direct["HistoricalTicker"].map(
        local_lookup_ticker
    )

    members = sp500_membership.copy()
    members["AsOfDate"] = pd.to_datetime(
        members["AsOfDate"], errors="raise"
    )
    members["DataSymbol"] = (
        members["DataSymbol"].astype(str).str.upper().str.strip()
    )
    if "Selected" in members:
        selected = (
            members["Selected"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        members = members.loc[selected].copy()

    frames: list[pd.DataFrame] = []
    for as_of, ranking in direct.groupby("AsOfDate", sort=True):
        member_symbols = set(
            members.loc[
                members["AsOfDate"].eq(as_of), "DataSymbol"
            ]
        )
        if not member_symbols:
            raise ValueError(f"No S&P membership snapshot for {as_of.date()}")
        selected = (
            ranking.loc[ranking["DataSymbol"].isin(member_symbols)]
            .sort_values(["PublishedRank", "MarketCap"], ascending=[True, False])
            .drop_duplicates("DataSymbol", keep="first")
            .head(top_n)
            .copy()
        )
        if len(selected) != top_n:
            raise ValueError(
                f"Only {len(selected)} S&P names found in the published "
                f"ranking for {as_of.date()}; expected {top_n}"
            )
        selected["Rank"] = range(1, top_n + 1)
        selected["Selected"] = True
        selected["MembershipSource"] = (
            "PUBLISHED_US_MARKET_CAP_RANK_FILTERED_TO_SP500"
        )
        frames.append(selected)

    result = pd.concat(frames, ignore_index=True)
    columns = [
        "AsOfDate",
        "HistoricalTicker",
        "DataSymbol",
        "Company",
        "Rank",
        "PublishedRank",
        "MarketCap",
        "Selected",
        "MembershipSource",
    ]
    available = [column for column in columns if column in result]
    return result.loc[:, available].sort_values(
        ["AsOfDate", "Rank"]
    ).reset_index(drop=True)


def run_dynamic_sp500_top_n(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    config: dict[str, Any],
    direct_rankings_path: str | Path,
    sp500_membership_path: str | Path,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    spy_path: str | Path,
    reference_manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> DynamicTopNArtifacts:
    top_n = int(config.get("top_n", 15))
    direct_rankings = pd.read_csv(direct_rankings_path)
    sp500_membership = pd.read_csv(sp500_membership_path)
    membership = build_sp500_top_n_membership(
        direct_rankings,
        sp500_membership,
        top_n=top_n,
    )

    ready_tickers = load_ready_tickers(backfill_status_path)
    members, discovery_audit = discover_universe(paths, ticker_config_path)
    union_tickers = set(membership["DataSymbol"])
    discoverable = {member.ticker for member in members}
    model_tickers = union_tickers & ready_tickers & discoverable
    selected_members = [
        member for member in members if member.ticker in model_tickers
    ]
    if len(model_tickers) < settings.minimum_cross_section_size:
        raise ValueError(
            "Too few model-ready Top-N union members: "
            f"{len(model_tickers)}"
        )

    warmup_start = str(
        (pd.Timestamp(settings.train_start) - pd.DateOffset(years=2)).date()
    )
    base_panel, build_audit = build_v7_source_panel(
        paths,
        selected_members,
        settings,
        ready_tickers=model_tickers,
        warmup_start=warmup_start,
        progress_every=10,
    )
    observed = add_v7_technical_observations(base_panel)
    normalized_membership = normalize_change_membership(membership)
    pit_panel = apply_membership_to_panel(
        observed,
        normalized_membership,
        settings,
    )
    technical = add_v7_technical_factors(pit_panel, settings)
    variant_name = str(
        config.get("technical_variant", "V7_3_MA_MACD_OBV_SLOT5")
    )
    variant = next(
        item for item in TECHNICAL_VARIANTS if item.name == variant_name
    )
    scoring_panel = scoring_panel_for_variant(technical, variant)
    optimization = optimize_strategy(scoring_panel, settings)

    full_start = settings.train_start
    full_end = str(pd.Timestamp(scoring_panel["Date"].max()).date())
    selected_result, selected_targets = _run_strategy(
        scoring_panel,
        optimization.params,
        settings,
        start=full_start,
        end=full_end,
    )
    signal_days = signal_day_panel(
        scoring_panel,
        full_start,
        full_end,
        settings.rebalance_weekday,
    )
    model_benchmark = run_portfolio_backtest(
        scoring_panel,
        generate_equal_weight_targets(signal_days),
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )

    curves: dict[str, pd.DataFrame] = {
        "DYNAMIC_SP500_TOP15_V7_REOPTIMIZED": selected_result.daily[
            ["Date", "Equity"]
        ],
        "DYNAMIC_TOP15_MODEL_READY_EQUAL_WEIGHT_WEEKLY": (
            model_benchmark.daily[["Date", "Equity"]]
        ),
    }
    reference_params: StrategyParams | None = None
    if reference_manifest_path is not None:
        reference_payload = json.loads(
            Path(reference_manifest_path).read_text(encoding="utf-8")
        )
        reference_params = StrategyParams.from_dict(
            dict(reference_payload["selected_params"])
        )
        transferred, _ = _run_strategy(
            scoring_panel,
            reference_params,
            settings,
            start=full_start,
            end=full_end,
        )
        curves["DYNAMIC_TOP15_V7_BIGTECH_PARAMS_TRANSFER"] = (
            transferred.daily[["Date", "Equity"]]
        )

    price_panel, price_only_audit = _build_price_only_panel(
        paths,
        ticker_config_path,
        membership,
        start=full_start,
    )
    price_signal_days = signal_day_panel(
        price_panel,
        full_start,
        full_end,
        settings.rebalance_weekday,
    )
    price_benchmark = run_portfolio_backtest(
        price_panel,
        generate_equal_weight_targets(price_signal_days),
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    curves["DYNAMIC_TOP15_PRICE_ONLY_EQUAL_WEIGHT_WEEKLY"] = (
        price_benchmark.daily[["Date", "Equity"]]
    )
    spy = build_single_asset_equity(
        load_sp500_proxy(spy_path),
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
    )
    curves["SPY_BUY_HOLD"] = spy

    periods = {
        "TRAIN_2020_2024": (settings.train_start, settings.train_end),
        **{
            f"{label}_REPORT_ONLY": (start, min(end, full_end))
            for label, (start, end) in settings.validation_periods.items()
            if start <= full_end
        },
        "FULL_2020_2026": (full_start, full_end),
    }
    period_rows: list[dict[str, object]] = []
    calendar_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    for series, curve in curves.items():
        clean = curve.sort_values("Date").drop_duplicates("Date", keep="last")
        output = clean.copy()
        output.insert(0, "Series", series)
        equity_frames.append(output)
        for period, (start, end) in periods.items():
            period_rows.append(
                {
                    "Series": series,
                    "Period": period,
                    **summarize_equity_curve(clean, start=start, end=end),
                }
            )
        calendar_rows.extend(calendar_return_rows(clean, series=series))

    coverage = _membership_coverage(
        membership,
        model_tickers=model_tickers,
        price_tickers=set(price_panel["Ticker"]),
    )
    signal_coverage = _signal_coverage(signal_days)
    data_audit = discovery_audit.loc[
        discovery_audit["Ticker"].isin(union_tickers)
    ].merge(build_audit, on=["Ticker", "Company"], how="left")

    generated_at = datetime.now(UTC)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "dynamic_sp500_top15"
            / f"{timestamp}_v7_financial_technical"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_summary_csv": destination / "candidate_summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "calendar_returns_csv": destination / "calendar_returns.csv",
        "equity_csv": destination / "equity.csv",
        "membership_csv": destination / "top15_membership.csv",
        "membership_coverage_csv": destination / "membership_coverage.csv",
        "signal_coverage_csv": destination / "signal_coverage.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "price_only_audit_csv": destination / "price_only_audit.csv",
        "selected_signals_csv": destination / "selected_signals.csv",
        "executions_csv": destination / "executions.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(
        optimization.candidates, outputs["candidate_summary_csv"], index=False
    )
    atomic_to_csv(
        pd.DataFrame(period_rows), outputs["period_summary_csv"], index=False
    )
    atomic_to_csv(
        pd.DataFrame(calendar_rows),
        outputs["calendar_returns_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(equity_frames, ignore_index=True),
        outputs["equity_csv"],
        index=False,
    )
    atomic_to_csv(membership, outputs["membership_csv"], index=False)
    atomic_to_csv(coverage, outputs["membership_coverage_csv"], index=False)
    atomic_to_csv(
        signal_coverage, outputs["signal_coverage_csv"], index=False
    )
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    atomic_to_csv(
        price_only_audit, outputs["price_only_audit_csv"], index=False
    )
    selected_rows = selected_targets.loc[
        selected_targets["ModelSelected"].fillna(False)
        | selected_targets["TradeAction"].isin(["BUY", "SELL"])
    ]
    atomic_to_csv(
        selected_rows, outputs["selected_signals_csv"], index=False
    )
    atomic_to_csv(
        selected_result.executions, outputs["executions_csv"], index=False
    )
    _atomic_json(
        {
            "generated_at": generated_at.isoformat(),
            "task": "Dynamic S&P 500 Top 15 V7 financial-technical validation",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "top_n": top_n,
            "snapshot_frequency": "ANNUAL_JAN_1",
            "technical_variant": variant_name,
            "membership_rule": (
                "published Jan-1 US market-cap ranking filtered to the "
                "same-date S&P 500 membership snapshot"
            ),
            "strict_membership_exit": settings.force_universe_exit,
            "union_ticker_count": len(union_tickers),
            "model_ready_union_ticker_count": len(model_tickers),
            "model_missing_union_tickers": sorted(
                union_tickers - model_tickers
            ),
            "selection_mode": optimization.selection_mode,
            "selected_params": optimization.params.as_dict(),
            "reference_params": (
                reference_params.as_dict()
                if reference_params is not None
                else None
            ),
            "training_period": [settings.train_start, settings.train_end],
            "report_only_period": ["2025-01-01", full_end],
            "candidate_count": len(optimization.candidates),
            "signal_timing": "weekly close signal; next session open execution",
            "transaction_cost_bps": settings.transaction_cost_bps,
            "roi_formula": "(final_value / total_injected - 1) * 100",
            "financial_point_in_time": False,
            "source_hashes": {
                "direct_rankings": _sha256(Path(direct_rankings_path)),
                "sp500_membership": _sha256(Path(sp500_membership_path)),
                "backfill_status": _sha256(Path(backfill_status_path)),
            },
            "caveats": [
                "BRK-B has price data but no valid local quarterly financial file.",
                "Financial history is restated rather than true point-in-time.",
                "The published ranking is a free historical source, not CRSP.",
                "Local close prices are split-normalized without dividend reinvestment.",
                "2025 and 2026 were already observed and are report-only diagnostics.",
            ],
            "config": config,
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return DynamicTopNArtifacts(output_dir=destination, **outputs)


def _run_strategy(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    *,
    start: str,
    end: str,
) -> tuple[PortfolioResult, pd.DataFrame]:
    signal_days = signal_day_panel(
        panel, start, end, settings.rebalance_weekday
    )
    targets = generate_rebalance_targets(
        score_panel(signal_days, params),
        params,
        force_universe_exit=settings.force_universe_exit,
    )
    result = run_portfolio_backtest(
        panel,
        targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        record_attribution=True,
    )
    return result, targets


def _build_price_only_panel(
    paths: ProjectPaths,
    ticker_config_path: str | Path,
    membership: pd.DataFrame,
    *,
    start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = load_tickers(ticker_config_path)
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for ticker in sorted(set(membership["DataSymbol"])):
        config = configs.get(ticker)
        audit: dict[str, object] = {
            "Ticker": ticker,
            "Status": "FAILED",
            "Error": "",
        }
        if config is None:
            audit["Error"] = "MISSING_TICKER_CONFIG"
            audits.append(audit)
            continue
        processed = sorted(
            paths.processed.glob(f"{config.display_name}_*.csv"),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        fallback = processed[0] if processed else None
        try:
            prices, source_kind, source_path = load_raw_company_prices(
                paths.raw_prices / config.display_name,
                fallback_path=fallback,
                earliest_date=start,
            )
            price = prices[["Date", "Open", "Close"]].copy()
            price["Ticker"] = ticker
            frames.append(price)
            audit.update(
                {
                    "Status": "INCLUDED",
                    "Company": config.display_name,
                    "PriceSourceKind": source_kind,
                    "PriceSource": source_path,
                    "PriceStart": price["Date"].min(),
                    "PriceEnd": price["Date"].max(),
                    "PriceRows": len(price),
                }
            )
        except Exception as exc:  # noqa: BLE001
            audit["Error"] = f"{type(exc).__name__}: {exc}"
        audits.append(audit)
    if not frames:
        raise ValueError("No price-only Top-N members could be loaded")
    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["Date", "Ticker"]
    )
    snapshots = pd.DataFrame(
        {"AsOfDate": membership["AsOfDate"].drop_duplicates().sort_values()}
    )
    dates = pd.DataFrame(
        {"Date": panel["Date"].drop_duplicates().sort_values()}
    )
    dated = pd.merge_asof(
        dates,
        snapshots,
        left_on="Date",
        right_on="AsOfDate",
        direction="backward",
    )
    panel = panel.merge(dated, on="Date", how="left", validate="many_to_one")
    selected = membership[["AsOfDate", "DataSymbol"]].rename(
        columns={"DataSymbol": "Ticker"}
    )
    selected["UniverseMember"] = True
    panel = panel.merge(
        selected,
        on=["AsOfDate", "Ticker"],
        how="left",
        validate="many_to_one",
    )
    panel["UniverseMember"] = panel["UniverseMember"].eq(True)
    panel["Eligible"] = (
        panel["UniverseMember"]
        & panel["Open"].gt(0)
        & panel["Close"].gt(0)
    )
    return panel.reset_index(drop=True), pd.DataFrame(audits)


def _membership_coverage(
    membership: pd.DataFrame,
    *,
    model_tickers: set[str],
    price_tickers: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for as_of, group in membership.groupby("AsOfDate", sort=True):
        requested = set(group["DataSymbol"])
        model_ready = requested & model_tickers
        price_ready = requested & price_tickers
        rows.append(
            {
                "AsOfDate": pd.Timestamp(as_of),
                "RequestedCount": len(requested),
                "ModelReadyCount": len(model_ready),
                "PriceReadyCount": len(price_ready),
                "ModelMissingTickers": ",".join(
                    sorted(requested - model_ready)
                ),
                "PriceMissingTickers": ",".join(
                    sorted(requested - price_ready)
                ),
            }
        )
    return pd.DataFrame(rows)


def _signal_coverage(signal_days: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in signal_days.groupby("Date", sort=True):
        member = group["UniverseMember"].fillna(False)
        eligible = group["Eligible"].fillna(False)
        rows.append(
            {
                "SignalDate": pd.Timestamp(date),
                "UniverseMemberRows": int(member.sum()),
                "EligibleMembers": int(eligible.sum()),
                "MissingEligibleTickers": ",".join(
                    sorted(group.loc[member & ~eligible, "Ticker"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=f"{path.stem}_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
