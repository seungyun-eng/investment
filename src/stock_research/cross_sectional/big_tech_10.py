from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths

from .config import ResearchSettings
from .data import UniverseMember
from .features import build_panel
from .optimization import optimize_strategy
from .portfolio import run_portfolio_backtest
from .signals import (
    generate_equal_weight_targets,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)
from .v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)


@dataclass(frozen=True)
class BigTech10Artifacts:
    output_dir: Path
    candidate_summary_csv: Path
    period_summary_csv: Path
    calendar_returns_csv: Path
    equity_csv: Path
    individual_buy_hold_csv: Path
    data_audit_csv: Path
    manifest_json: Path


def run_big_tech_10_optimization(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    config: dict[str, Any],
    spy_path: str | Path,
    output_dir: str | Path | None = None,
) -> BigTech10Artifacts:
    members = _members_from_config(paths, config["universe"])
    panel, data_audit = build_panel(members, settings)
    observed = add_v7_technical_observations(panel)
    technical = add_v7_technical_factors(observed, settings)
    variant_name = str(config.get("technical_variant", "V7_3_MA_MACD_OBV_SLOT5"))
    variant = next(
        item for item in TECHNICAL_VARIANTS if item.name == variant_name
    )
    scoring_panel = scoring_panel_for_variant(technical, variant)
    optimization = optimize_strategy(scoring_panel, settings)
    selected_params = optimization.params

    full_start = settings.train_start
    full_end = str(pd.Timestamp(scoring_panel["Date"].max()).date())
    signal_days = signal_day_panel(
        scoring_panel,
        full_start,
        full_end,
        settings.rebalance_weekday,
    )
    selected_targets = generate_rebalance_targets(
        score_panel(signal_days, selected_params),
        selected_params,
    )
    equal_weight_targets = generate_equal_weight_targets(signal_days)
    selected = run_portfolio_backtest(
        scoring_panel,
        selected_targets,
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        record_attribution=True,
    )
    equal_weight = run_portfolio_backtest(
        scoring_panel,
        equal_weight_targets,
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    buy_hold = build_equal_weight_buy_hold_equity(
        scoring_panel,
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
    )
    spy = load_sp500_proxy(spy_path)
    spy = build_single_asset_equity(
        spy,
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
    )

    curves = {
        "BIG_TECH_10_V7_OPTIMIZED": selected.daily[["Date", "Equity"]],
        "BIG_TECH_10_EQUAL_WEIGHT_WEEKLY": equal_weight.daily[
            ["Date", "Equity"]
        ],
        "BIG_TECH_10_BUY_HOLD": buy_hold,
        "SPY_BUY_HOLD": spy,
    }
    period_rows: list[dict[str, object]] = []
    calendar_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    periods = {
        "TRAIN_2020_2024": (settings.train_start, settings.train_end),
        "FULL_2020_2026": (full_start, full_end),
    }
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

    individual = individual_buy_hold_summary(
        scoring_panel,
        start=full_start,
        end=full_end,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "big_tech_10"
            / f"{timestamp}_v7_financial_technical"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_summary_csv": destination / "candidate_summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "calendar_returns_csv": destination / "calendar_returns.csv",
        "equity_csv": destination / "equity.csv",
        "individual_buy_hold_csv": destination / "individual_buy_hold.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(
        optimization.candidates,
        outputs["candidate_summary_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.DataFrame(period_rows),
        outputs["period_summary_csv"],
        index=False,
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
    atomic_to_csv(individual, outputs["individual_buy_hold_csv"], index=False)
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "Big Tech 10 V7 financial-technical optimization",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "universe_fixed_with_hindsight": True,
            "universe": [item["ticker"] for item in config["universe"]],
            "technical_variant": variant_name,
            "training_period": [settings.train_start, settings.train_end],
            "report_only_period": ["2025-01-01", full_end],
            "candidate_count": len(optimization.candidates),
            "selection_mode": optimization.selection_mode,
            "selected_params": selected_params.as_dict(),
            "signal_timing": "weekly close signal; next session open execution",
            "roi_formula": "(final_value / total_injected - 1) * 100",
            "financial_release_lag_days": settings.financial_release_lag_days,
            "financial_point_in_time": False,
            "caveats": [
                "The ten current winners were fixed with hindsight.",
                "Financial history is restated rather than true point-in-time.",
                "Local close prices include split normalization but not dividend reinvestment.",
                "2025 and 2026 were already observed and are report-only diagnostics.",
            ],
            "config": config,
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return BigTech10Artifacts(output_dir=destination, **outputs)


def build_equal_weight_buy_hold_equity(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    initial_capital: float,
) -> pd.DataFrame:
    prices = (
        panel.loc[panel["Date"].between(start, end)]
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
        .dropna()
    )
    if prices.empty:
        raise ValueError("No common Big Tech 10 price history")
    normalized = prices / prices.iloc[0]
    equity = normalized.mean(axis=1) * initial_capital
    return pd.DataFrame({"Date": equity.index, "Equity": equity.to_numpy()})


def build_single_asset_equity(
    prices: pd.DataFrame,
    *,
    start: str,
    end: str,
    initial_capital: float,
) -> pd.DataFrame:
    period = prices.loc[prices["Date"].between(start, end), ["Date", "Close"]]
    period = period.sort_values("Date").drop_duplicates("Date", keep="last")
    if period.empty:
        raise ValueError(f"No single-asset prices between {start} and {end}")
    equity = period["Close"] / float(period.iloc[0]["Close"]) * initial_capital
    return pd.DataFrame({"Date": period["Date"], "Equity": equity})


def summarize_equity_curve(
    curve: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> dict[str, object]:
    period = curve.loc[curve["Date"].between(start, end)].copy()
    if period.empty:
        raise ValueError(f"No equity observations between {start} and {end}")
    first = float(period.iloc[0]["Equity"])
    last = float(period.iloc[-1]["Equity"])
    years = max(
        (pd.Timestamp(period.iloc[-1]["Date"]) - pd.Timestamp(period.iloc[0]["Date"])).days
        / 365.25,
        1.0 / 252.0,
    )
    returns = period["Equity"].pct_change(fill_method=None).dropna()
    volatility = float(returns.std(ddof=1)) if len(returns) >= 2 else 0.0
    sharpe = (
        float(returns.mean() / volatility * np.sqrt(252.0))
        if volatility > 0
        else 0.0
    )
    drawdown = period["Equity"] / period["Equity"].cummax() - 1.0
    return {
        "StartDate": pd.Timestamp(period.iloc[0]["Date"]),
        "EndDate": pd.Timestamp(period.iloc[-1]["Date"]),
        "StartValue": first,
        "FinalValue": last,
        "ROI": (last / first - 1.0) * 100.0,
        "CAGR": ((last / first) ** (1.0 / years) - 1.0) * 100.0,
        "MaxDrawdown": float(drawdown.min() * 100.0),
        "Sharpe": sharpe,
    }


def calendar_return_rows(
    curve: pd.DataFrame,
    *,
    series: str,
) -> list[dict[str, object]]:
    clean = curve.sort_values("Date").reset_index(drop=True)
    returns = clean["Equity"].pct_change(fill_method=None).fillna(0.0)
    annual = (1.0 + returns).groupby(clean["Date"].dt.year).prod() - 1.0
    return [
        {"Series": series, "Year": int(year), "Return": float(value * 100.0)}
        for year, value in annual.items()
    ]


def individual_buy_hold_summary(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, group in panel.loc[panel["Date"].between(start, end)].groupby(
        "Ticker"
    ):
        curve = build_single_asset_equity(
            group[["Date", "Close"]],
            start=start,
            end=end,
            initial_capital=100_000.0,
        )
        rows.append({"Ticker": ticker, **summarize_equity_curve(curve, start=start, end=end)})
    return pd.DataFrame(rows).sort_values("CAGR", ascending=False).reset_index(drop=True)


def _members_from_config(
    paths: ProjectPaths,
    universe: list[dict[str, Any]],
) -> list[UniverseMember]:
    members: list[UniverseMember] = []
    for item in universe:
        ticker = str(item["ticker"]).upper()
        price_path = (paths.stock_root / str(item["price_path"])).resolve()
        financial_path = paths.financial_raw / f"{ticker}_financials_Q.xlsx"
        if not price_path.exists():
            raise FileNotFoundError(price_path)
        if not financial_path.exists():
            raise FileNotFoundError(financial_path)
        members.append(
            UniverseMember(
                ticker=ticker,
                company=str(item["company"]),
                price_path=price_path,
                financial_path=financial_path,
            )
        )
    if len(members) != 10 or len({item.ticker for item in members}) != 10:
        raise ValueError("Big Tech universe must contain ten unique tickers")
    return members


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
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
