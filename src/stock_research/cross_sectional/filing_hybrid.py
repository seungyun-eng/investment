from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
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
from .config import ResearchSettings
from .features import build_panel
from .filing_signals import (
    FilingHybridConfig,
    add_filing_factors,
    add_filing_hybrid_scores,
    add_market_regime,
    generate_filing_hybrid_targets,
    merge_filing_features,
)
from .fixed_universe import members_from_config
from .portfolio import run_portfolio_backtest
from .signals import signal_day_panel
from .v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)


@dataclass(frozen=True)
class FilingHybridArtifacts:
    output_dir: Path
    summary_csv: Path
    period_summary_csv: Path
    calendar_returns_csv: Path
    equity_csv: Path
    targets_csv: Path
    executions_csv: Path
    latest_scores_csv: Path
    data_audit_csv: Path
    manifest_json: Path


def run_filing_hybrid(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    universe_config: dict[str, Any],
    filing_features_path: str | Path,
    spy_path: str | Path,
    hybrid_config: FilingHybridConfig | None = None,
    output_dir: str | Path | None = None,
) -> FilingHybridArtifacts:
    """Run the fixed 2020-known universe with SEC filing-aware sleeves."""

    config = hybrid_config or FilingHybridConfig()
    members = members_from_config(paths, universe_config["universe"])
    panel, data_audit = build_panel(members, settings)
    observed = add_v7_technical_observations(panel)
    technical = add_v7_technical_factors(observed, settings)
    variant_name = str(
        universe_config.get("technical_variant", "V7_3_MA_MACD_OBV_SLOT5")
    )
    variant = next(
        value for value in TECHNICAL_VARIANTS if value.name == variant_name
    )
    scoring_panel = scoring_panel_for_variant(technical, variant)
    filing_features = pd.read_csv(filing_features_path)
    spy_prices = load_sp500_proxy(spy_path)
    merged = merge_filing_features(scoring_panel, filing_features)
    factored = add_filing_factors(
        merged,
        minimum_cross_section_size=min(
            settings.minimum_cross_section_size,
            max(4, len(members) // 2),
        ),
    )
    scored = add_filing_hybrid_scores(factored)
    scored = add_market_regime(scored, spy_prices, config)

    start = settings.train_start
    end = str(pd.Timestamp(scored["Date"].max()).date())
    signal_days = signal_day_panel(
        scored, start, end, settings.rebalance_weekday
    )
    targets = generate_filing_hybrid_targets(signal_days, config)
    strategy = run_portfolio_backtest(
        scored,
        targets,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        record_attribution=True,
    )
    spy = build_single_asset_equity(
        spy_prices,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
    )
    curves = {
        "FILING_HYBRID_FIXED_RULES": strategy.daily[["Date", "Equity"]],
        "SPY_BUY_HOLD": spy,
    }
    periods = {
        "TRAIN_2020_2024": (settings.train_start, settings.train_end),
        **{
            f"{label}_REPORT_ONLY": (period_start, min(period_end, end))
            for label, (period_start, period_end) in settings.validation_periods.items()
            if period_start <= end
        },
        "FULL_2020_2026": (start, end),
    }
    summary_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    calendar_rows: list[dict[str, object]] = []
    equity_rows: list[pd.DataFrame] = []
    for name, curve in curves.items():
        clean = curve.sort_values("Date").drop_duplicates("Date", keep="last")
        summary_rows.append(
            {"Series": name, **summarize_equity_curve(clean, start=start, end=end)}
        )
        for period, (period_start, period_end) in periods.items():
            period_rows.append(
                {
                    "Series": name,
                    "Period": period,
                    **summarize_equity_curve(
                        clean, start=period_start, end=period_end
                    ),
                }
            )
        calendar_rows.extend(calendar_return_rows(clean, series=name))
        output = clean.copy()
        output.insert(0, "Series", name)
        equity_rows.append(output)

    generated_at = datetime.now(UTC)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else paths.results
        / "Cross_Sectional"
        / "filing_hybrid"
        / f"{timestamp}_known_2020_growth_16"
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_csv": destination / "summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "calendar_returns_csv": destination / "calendar_returns.csv",
        "equity_csv": destination / "equity.csv",
        "targets_csv": destination / "targets.csv",
        "executions_csv": destination / "executions.csv",
        "latest_scores_csv": destination / "latest_scores.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "manifest_json": destination / "manifest.json",
    }
    latest_date = scored["Date"].max()
    latest_columns = [
        "Date",
        "Ticker",
        "Company",
        "FilingCoreScore",
        "FilingCoreRank",
        "FilingCoreQualified",
        "FilingTacticalScore",
        "FilingTacticalRank",
        "FilingTacticalQualified",
        "TrueValueFactor",
        "FilingFundamentalFactor",
        "FilingCoverageCount",
        "AvailableDate",
        "AccessionNumber",
        "Form",
    ]
    atomic_to_csv(pd.DataFrame(summary_rows), outputs["summary_csv"], index=False)
    atomic_to_csv(
        pd.DataFrame(period_rows), outputs["period_summary_csv"], index=False
    )
    atomic_to_csv(
        pd.DataFrame(calendar_rows), outputs["calendar_returns_csv"], index=False
    )
    atomic_to_csv(
        pd.concat(equity_rows, ignore_index=True), outputs["equity_csv"], index=False
    )
    atomic_to_csv(targets, outputs["targets_csv"], index=False)
    atomic_to_csv(strategy.executions, outputs["executions_csv"], index=False)
    atomic_to_csv(
        scored.loc[scored["Date"].eq(latest_date), latest_columns],
        outputs["latest_scores_csv"],
        index=False,
    )
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    _atomic_json(
        outputs["manifest_json"],
        {
            "generated_at": generated_at.isoformat(),
            "task": "SEC filing-aware fixed-universe core and tactical prototype",
            "model_status": "POST_HOC_FIXED_RULE_PROTOTYPE",
            "validation_is_fresh": False,
            "optimization_performed": False,
            "universe": [member.ticker for member in members],
            "hybrid_config": asdict(config),
            "score_rules": {
                "true_value": (
                    "30% growth, 25% existing quality/valuation, 20% filed "
                    "durability, 15% filed balance sheet, 10% disclosure safety"
                ),
                "core": "80% true value, 10% trend, 10% risk control",
                "tactical": (
                    "45% MA/MACD/OBV, 30% momentum, 15% growth, "
                    "10% filed durability/balance"
                ),
            },
            "signal_timing": "weekly close signal; next session open execution",
            "filing_timing": "eligible the calendar day after SEC acceptance",
            "transaction_cost_bps": settings.transaction_cost_bps,
            "roi_formula": "(final_value / total_injected - 1) * 100",
            "caveats": [
                "Rules were designed after observing the historical period.",
                "This run is a diagnostic, not a fresh out-of-sample result.",
                (
                    "SEC filing features use accession-specific availability, "
                    "but inherited GrowthFactor and QualityFactor inputs still "
                    "contain restated Macrotrends history."
                ),
                "Text phrase density is an experimental risk feature.",
                "Industry growth and guidance-versus-actual history remain separate work.",
            ],
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    return FilingHybridArtifacts(output_dir=destination, **outputs)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
