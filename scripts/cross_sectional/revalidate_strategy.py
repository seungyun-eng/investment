from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.config import (
    load_settings,
    settings_from_dict,
)
from stock_research.cross_sectional.data import discover_universe
from stock_research.cross_sectional.features import build_panel
from stock_research.cross_sectional.research import (
    _atomic_json,
    _comparison_row,
    _evaluate_period,
    _financial_feature_coverage,
    _selection_statistics,
    _signal_output_columns,
    load_selected_strategy,
)
from stock_research.cross_sectional.signals import (
    build_daily_recommendations,
    generate_rebalance_targets,
    score_panel,
    signal_day_panel,
)
from stock_research.io_utils import atomic_to_csv
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate frozen cross-sectional parameters after a signal "
            "implementation correction, without re-optimizing."
        )
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config")
    parser.add_argument("--ticker-config", default="config/tickers.json")
    parser.add_argument("--stock-root")
    parser.add_argument("--label", default="frozen_revalidation")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    strategy_path = Path(args.strategy).expanduser().resolve()
    source = json.loads(strategy_path.read_text(encoding="utf-8"))
    settings = (
        load_settings(paths.repo_root / args.config)
        if args.config
        else settings_from_dict(source["settings"])
    )
    params = load_selected_strategy(strategy_path)
    members, discovery_audit = discover_universe(
        paths,
        paths.repo_root / args.ticker_config,
    )
    panel, data_audit = build_panel(members, settings)
    scored_daily = score_panel(panel, params)

    training = _evaluate_period(
        panel,
        scored_daily,
        params,
        settings,
        settings.train_start,
        settings.train_end,
    )
    current_train_roi = training["strategy"].summary.roi_percent
    original_train_roi = float(source["training_metrics"]["TrainROI"])
    if abs(current_train_roi - original_train_roi) > 1e-9:
        raise RuntimeError(
            "Training ROI changed under the corrected implementation; "
            "full re-optimization is required instead of frozen revalidation."
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.label
    ).strip("_")
    output_dir = (
        paths.results
        / "Cross_Sectional"
        / "rank_signals"
        / f"{timestamp}_{safe_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, object]] = []
    for label, (start, end) in settings.validation_periods.items():
        result = _evaluate_period(
            panel,
            scored_daily,
            params,
            settings,
            start,
            end,
        )
        strategy = result["strategy"]
        benchmark = result["benchmark"]
        validation_rows.append(
            {"Period": label, **_comparison_row(strategy, benchmark)}
        )
        atomic_to_csv(
            strategy.daily,
            output_dir / f"validation_{label}_equity.csv",
            index=False,
        )
        atomic_to_csv(
            strategy.executions,
            output_dir / f"validation_{label}_executions.csv",
            index=False,
        )
        atomic_to_csv(
            _selection_statistics(panel, result["targets"], start, end),
            output_dir / f"validation_{label}_ticker_stats.csv",
            index=False,
        )

    latest_end = str(pd.Timestamp(panel["Date"].max()).date())
    live_start = min(
        start for start, _ in settings.validation_periods.values()
    )
    signal_days = signal_day_panel(
        panel,
        live_start,
        latest_end,
        settings.rebalance_weekday,
    )
    targets = generate_rebalance_targets(
        score_panel(signal_days, params),
        params,
    )
    recommendations = build_daily_recommendations(
        scored_daily.loc[scored_daily["Date"].between(live_start, latest_end)],
        targets,
        params,
    )
    latest_date = pd.Timestamp(recommendations["Date"].max())
    validation_summary = pd.DataFrame(validation_rows)
    passed = bool(
        (
            validation_summary["PositiveROI"]
            & validation_summary["BeatEqualWeight"]
        ).all()
    )
    model_status = (
        "POST_HOC_CORRECTED_PASS"
        if passed
        else "POST_HOC_CORRECTED_FAILED"
    )
    latest = _signal_output_columns(
        recommendations.loc[recommendations["Date"].eq(latest_date)]
    )
    latest.insert(0, "ModelStatus", model_status)
    history = _signal_output_columns(recommendations)
    history.insert(0, "ModelStatus", model_status)

    audit = discovery_audit.merge(
        data_audit,
        on=["Ticker", "Company"],
        how="left",
    )
    audit["TrainingEligible"] = audit["TrainingEligibleSessions"].fillna(0).gt(0)
    audit["LatestFinancialAgeDays"] = (
        latest_date - pd.to_datetime(audit["FinancialEnd"])
    ).dt.days
    audit["LatestFinancialStale"] = (
        audit["LatestFinancialAgeDays"] > settings.max_financial_age_days
    )
    coverage = _financial_feature_coverage(panel, settings, latest_date)

    atomic_to_csv(
        validation_summary,
        output_dir / "validation_summary.csv",
        index=False,
    )
    atomic_to_csv(
        latest,
        output_dir / "latest_daily_signals.csv",
        index=False,
    )
    atomic_to_csv(
        history,
        output_dir / "daily_signal_history.csv",
        index=False,
    )
    atomic_to_csv(
        audit,
        output_dir / "universe_data_audit.csv",
        index=False,
    )
    atomic_to_csv(
        coverage,
        output_dir / "financial_feature_coverage.csv",
        index=False,
    )

    manifest = dict(source)
    manifest["generated_at"] = datetime.now(UTC).isoformat()
    manifest["model_status"] = model_status
    manifest["validation"] = validation_summary.to_dict(orient="records")
    manifest["latest_price_date"] = latest_date
    manifest["output_dir"] = str(output_dir)
    manifest["revalidation"] = {
        "source_strategy": str(strategy_path),
        "mode": "frozen_parameters_after_sparse_factor_guard",
        "training_roi_original": original_train_roi,
        "training_roi_recheck": current_train_roi,
        "training_metrics_unchanged": True,
    }
    methodology = dict(manifest["methodology"])
    methodology["minimum_factor_cross_section_size"] = (
        settings.minimum_cross_section_size
    )
    methodology["validation_is_fresh"] = False
    manifest["methodology"] = methodology
    _atomic_json(manifest, output_dir / "selected_strategy.json")

    print(f"Output: {output_dir}")
    print(f"Training ROI unchanged: {current_train_roi:.6f}%")
    print(validation_summary.to_string(index=False))
    print("\nLatest selected names:")
    print(
        latest.loc[latest["ModelSelected"], [
            "Ticker",
            "DailySignal",
            "TradeAction",
            "TargetWeight",
            "Rank",
            "AlphaScore",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
