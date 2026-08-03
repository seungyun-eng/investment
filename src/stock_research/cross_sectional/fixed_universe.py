from __future__ import annotations

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

from .big_tech_10 import (
    build_single_asset_equity,
    calendar_return_rows,
    summarize_equity_curve,
)
from .config import ResearchSettings, StrategyParams
from .data import UniverseMember
from .features import build_panel
from .optimization import optimize_strategy
from .portfolio import PortfolioResult, run_portfolio_backtest
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
class FixedUniverseArtifacts:
    output_dir: Path
    candidate_summary_csv: Path
    period_summary_csv: Path
    calendar_returns_csv: Path
    equity_csv: Path
    purchase_audit_csv: Path
    data_audit_csv: Path
    selected_signals_csv: Path
    executions_csv: Path
    manifest_json: Path


def run_fixed_universe_optimization(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    config: dict[str, Any],
    spy_path: str | Path,
    reference_manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> FixedUniverseArtifacts:
    members = members_from_config(paths, config["universe"])
    panel, data_audit = build_panel(members, settings)
    observed = add_v7_technical_observations(panel)
    technical = add_v7_technical_factors(observed, settings)
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
    selected, selected_targets = _run_strategy(
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
    equal_weight = run_portfolio_backtest(
        scoring_panel,
        generate_equal_weight_targets(signal_days),
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    spy_prices = load_sp500_proxy(spy_path)
    spy = build_single_asset_equity(
        spy_prices,
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
    )
    staggered, purchase_audit = build_staggered_equal_slot_equity(
        scoring_panel,
        reference_dates=pd.DatetimeIndex(spy["Date"]),
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
    )

    curves: dict[str, pd.DataFrame] = {
        "FIXED_KNOWN_2020_16_V7_REOPTIMIZED": selected.daily[
            ["Date", "Equity"]
        ],
        "FIXED_KNOWN_2020_16_EQUAL_WEIGHT_WEEKLY": equal_weight.daily[
            ["Date", "Equity"]
        ],
        "FIXED_KNOWN_2020_16_STAGGERED_BUY_HOLD": staggered,
        "SPY_BUY_HOLD": spy,
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
        curves["FIXED_KNOWN_2020_16_BIGTECH_PARAMS_TRANSFER"] = (
            transferred.daily[["Date", "Equity"]]
        )

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

    generated_at = datetime.now(UTC)
    label = str(config.get("output_label", "fixed_universe"))
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "fixed_universe"
            / f"{timestamp}_{label}"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_summary_csv": destination / "candidate_summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "calendar_returns_csv": destination / "calendar_returns.csv",
        "equity_csv": destination / "equity.csv",
        "purchase_audit_csv": destination / "purchase_audit.csv",
        "data_audit_csv": destination / "data_audit.csv",
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
    atomic_to_csv(purchase_audit, outputs["purchase_audit_csv"], index=False)
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    selected_rows = selected_targets.loc[
        selected_targets["ModelSelected"].fillna(False)
        | selected_targets["TradeAction"].isin(["BUY", "SELL"])
    ]
    atomic_to_csv(
        selected_rows, outputs["selected_signals_csv"], index=False
    )
    atomic_to_csv(selected.executions, outputs["executions_csv"], index=False)
    _atomic_json(
        {
            "generated_at": generated_at.isoformat(),
            "task": "Fixed ex-ante thematic universe V7 optimization",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "universe_assumption": (
                "The named companies were fixed from the start of 2020; "
                "pre-IPO allocations remained in cash until first local open."
            ),
            "universe": [member.ticker for member in members],
            "technical_variant": variant_name,
            "training_period": [settings.train_start, settings.train_end],
            "report_only_period": ["2025-01-01", full_end],
            "candidate_count": len(optimization.candidates),
            "selection_mode": optimization.selection_mode,
            "selected_params": optimization.params.as_dict(),
            "reference_params": (
                reference_params.as_dict()
                if reference_params is not None
                else None
            ),
            "signal_timing": "weekly close signal; next session open execution",
            "transaction_cost_bps": settings.transaction_cost_bps,
            "roi_formula": "(final_value / total_injected - 1) * 100",
            "financial_point_in_time": False,
            "caveats": [
                "The claimed 2020 interest list is a user-supplied ex-ante assumption.",
                "PLTR and U allocations stay in cash until their 2020 listings.",
                "Financial history is restated rather than true point-in-time.",
                "Company close prices exclude dividend reinvestment.",
                "2025 and 2026 were already observed and are report-only diagnostics.",
            ],
            "config": config,
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return FixedUniverseArtifacts(output_dir=destination, **outputs)


def build_staggered_equal_slot_equity(
    panel: pd.DataFrame,
    *,
    reference_dates: pd.DatetimeIndex,
    start: str,
    end: str,
    initial_capital: float,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = reference_dates[
        (reference_dates >= pd.Timestamp(start))
        & (reference_dates <= pd.Timestamp(end))
    ]
    tickers = sorted(panel["Ticker"].unique())
    if dates.empty or not tickers:
        raise ValueError("No dates or tickers for staggered buy-and-hold")
    slot = initial_capital / len(tickers)
    equity = pd.Series(0.0, index=dates)
    audit_rows: list[dict[str, object]] = []
    cost_rate = transaction_cost_bps / 10_000.0
    for ticker in tickers:
        prices = (
            panel.loc[
                panel["Ticker"].eq(ticker) & panel["Date"].between(start, end),
                ["Date", "Open", "Close"],
            ]
            .dropna(subset=["Open", "Close"])
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .set_index("Date")
        )
        if prices.empty:
            raise ValueError(f"No prices for {ticker}")
        buy_date = pd.Timestamp(prices.index[0])
        buy_open = float(prices.iloc[0]["Open"])
        shares = slot * (1.0 - cost_rate) / buy_open
        values = pd.Series(slot, index=dates)
        closes = prices["Close"].reindex(dates).ffill()
        values.loc[buy_date:] = closes.loc[buy_date:] * shares
        equity += values
        final_value = float(values.iloc[-1])
        audit_rows.append(
            {
                "Ticker": ticker,
                "ReservedSlot": slot,
                "BuyDate": buy_date,
                "BuyOpen": buy_open,
                "TransactionCost": slot * cost_rate,
                "Shares": shares,
                "FinalSlotValue": final_value,
                "SlotROI": (final_value / slot - 1.0) * 100.0,
            }
        )
    # The first dated observation represents capital immediately before the
    # opening purchases. Subsequent values retain each actual opening fill.
    equity.iloc[0] = initial_capital
    return (
        pd.DataFrame({"Date": dates, "Equity": equity.to_numpy()}),
        pd.DataFrame(audit_rows).sort_values(
            "SlotROI", ascending=False
        ).reset_index(drop=True),
    )


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
        score_panel(signal_days, params), params
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


def members_from_config(
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
    tickers = {member.ticker for member in members}
    if len(members) < 2 or len(tickers) != len(members):
        raise ValueError("Fixed universe requires unique tickers")
    return members


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
