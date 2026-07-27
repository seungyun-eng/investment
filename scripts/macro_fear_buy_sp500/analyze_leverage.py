from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_fear_buy_sp500.config import FearBuyParams
from stock_research.macro_fear_buy_sp500.contributions import (
    ContributionConfig,
    run_contribution_backtest,
)
from stock_research.macro_fear_buy_sp500.features import build_fear_features
from stock_research.macro_fear_buy_sp500.leverage import (
    conditional_two_x_risk_hedge,
    conditional_two_x_short_hedge,
    daily_reset_instrument,
    tactical_two_x_overlay,
)
from stock_research.macro_fear_buy_sp500.mass_optimization import (
    candidate_features,
    evaluate_frozen_candidate,
)
from stock_research.macro_fear_buy_sp500.strategy import (
    generate_fear_buy_signals,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply +2x long and -2x short variants to frozen mass-optimization "
            "winners without using holdout to select hedge parameters."
        )
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--mass-manifest", type=Path)
    parser.add_argument("--initial", type=float, default=40_000.0)
    parser.add_argument("--monthly", type=float, default=4_000.0)
    parser.add_argument("--development-end", default="2016-12-30")
    parser.add_argument("--holdout-start", default="2017-01-03")
    parser.add_argument("--allow-small", action="store_true")
    return parser.parse_args()


def _latest(folder: Path, pattern: str) -> Path:
    hits = list(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return max(hits, key=lambda path: path.stat().st_mtime)


def _atomic_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".json.tmp",
        prefix=path.stem + "_",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _summary_row(
    *,
    category: str,
    period: str,
    variant: str,
    summary: object,
) -> dict[str, object]:
    return {
        "SelectionCategory": category,
        "Period": period,
        "Variant": variant,
        "FinalValue": summary.final_value,
        "NetProfit": summary.net_profit,
        "ROI(%)": summary.roi_percent,
        "XIRR(%)": summary.money_weighted_return_percent,
        "TWR_CAGR(%)": summary.time_weighted_cagr_percent,
        "MDD(%)": summary.max_drawdown_percent,
        "Sharpe": summary.sharpe_ratio,
        "AverageOverlayWeight(%)": getattr(
            summary,
            "average_overlay_weight_percent",
            0.0,
        ),
    }


def _candidate_payload(
    frozen: dict[str, object],
) -> dict[str, float | int]:
    strategy = dict(frozen["strategy"])
    strategy.update(dict(frozen["deployment_policy"]))
    return strategy


def _run_selected_hedge(
    daily: pd.DataFrame,
    selected: dict[str, object],
) -> object:
    if selected["mode"] == "PreRisk":
        return conditional_two_x_risk_hedge(
            daily,
            maximum_capital_fraction=float(
                selected["maximum_capital_fraction"]
            ),
            model_risk_threshold=float(selected["model_risk_threshold"]),
            macro_threshold=float(selected["macro_threshold"]),
        )
    return conditional_two_x_short_hedge(
        daily,
        maximum_capital_fraction=float(
            selected["maximum_capital_fraction"]
        ),
        euphoria_threshold=float(selected["euphoria_threshold"]),
        max_fear_score=0.45,
        max_vix_percentile=0.60,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    output_root = (
        paths.results
        / "SP500"
        / "macro_fear_buy_sp500"
        / "monthly_contributions"
        / "massive_optimization"
    )
    manifest_path = (
        args.mass_manifest
        or _latest(output_root, "manifest_*.json")
    ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["candidate_count"]) < 100_000 and not args.allow_small:
        raise ValueError("Refusing a mass run with fewer than 100,000 candidates.")
    prediction_path = Path(manifest["prediction_source"])
    predictions = pd.read_csv(prediction_path, parse_dates=["Date"])
    baseline_params = FearBuyParams(
        **manifest["selection_categories"][0]["strategy"]
    )
    features = build_fear_features(predictions, baseline_params)
    config = ContributionConfig(
        initial_lump_sum=args.initial,
        monthly_contribution=args.monthly,
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
    )
    periods = {
        "Development": (None, args.development_end),
        "Holdout": (args.holdout_start, None),
        "Full": (None, None),
    }
    rows: list[dict[str, object]] = []
    hedge_selections: list[dict[str, object]] = []

    for frozen in manifest["selection_categories"]:
        category = str(frozen["selection_category"])
        candidate = _candidate_payload(frozen)
        period_results: dict[str, object] = {}
        period_signals: dict[str, pd.DataFrame] = {}
        for period, (start, end) in periods.items():
            period_features = features.copy()
            if start:
                period_features = period_features[
                    period_features["Date"] >= pd.Timestamp(start)
                ]
            if end:
                period_features = period_features[
                    period_features["Date"] <= pd.Timestamp(end)
                ]
            params, policy, result = evaluate_frozen_candidate(
                period_features.reset_index(drop=True),
                candidate,
                config,
                name=f"{category}{period}",
            )
            period_results[period] = result
            signal_features = period_features.reset_index(drop=True).copy()
            period_signals[period] = generate_fear_buy_signals(
                candidate_features(signal_features, params),
                params,
            )
            rows.append(
                _summary_row(
                    category=category,
                    period=period,
                    variant="Base1x",
                    summary=result.summary,
                )
            )
            tactical = tactical_two_x_overlay(result.daily)
            rows.append(
                _summary_row(
                    category=category,
                    period=period,
                    variant="Tactical2xLongApprox",
                    summary=tactical,
                )
            )
            for multiple, variant in (
                (2.0, "Full2xLong"),
                (-2.0, "Full2xShortStress"),
            ):
                leveraged_signals = daily_reset_instrument(
                    period_signals[period],
                    multiple=multiple,
                )
                leveraged_result = run_contribution_backtest(
                    leveraged_signals,
                    params,
                    config,
                    name=f"{category}{variant}",
                    deployment_policy=policy,
                )
                rows.append(
                    _summary_row(
                        category=category,
                        period=period,
                        variant=variant,
                        summary=leveraged_result.summary,
                    )
                )

        development = period_results["Development"]
        hedge_candidates: list[dict[str, object]] = []
        for fraction in (0.05, 0.10, 0.15, 0.20):
            for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
                summary = conditional_two_x_short_hedge(
                    development.daily,
                    maximum_capital_fraction=fraction,
                    euphoria_threshold=threshold,
                    max_fear_score=0.45,
                    max_vix_percentile=0.60,
                )
                profit_ratio = (
                    summary.net_profit / development.summary.net_profit
                )
                eligible = profit_ratio >= 0.85
                hedge_candidates.append(
                    {
                        "mode": "Euphoria",
                        "maximum_capital_fraction": fraction,
                        "euphoria_threshold": threshold,
                        "model_risk_threshold": float("nan"),
                        "macro_threshold": float("nan"),
                        "development_profit_ratio_vs_base": profit_ratio,
                        "development_mdd_percent": (
                            summary.max_drawdown_percent
                        ),
                        "development_sharpe": summary.sharpe_ratio,
                        "score": (
                            summary.max_drawdown_percent
                            + 0.25 * summary.sharpe_ratio
                            if eligible
                            else float("-inf")
                        ),
                    }
                )
            for risk_threshold in (0.70, 0.80, 0.90, 0.95):
                for macro_threshold in (0.40, 0.50, 0.60):
                    summary = conditional_two_x_risk_hedge(
                        development.daily,
                        maximum_capital_fraction=fraction,
                        model_risk_threshold=risk_threshold,
                        macro_threshold=macro_threshold,
                    )
                    profit_ratio = (
                        summary.net_profit / development.summary.net_profit
                    )
                    eligible = profit_ratio >= 0.85
                    hedge_candidates.append(
                        {
                            "mode": "PreRisk",
                            "maximum_capital_fraction": fraction,
                            "euphoria_threshold": float("nan"),
                            "model_risk_threshold": risk_threshold,
                            "macro_threshold": macro_threshold,
                            "development_profit_ratio_vs_base": profit_ratio,
                            "development_mdd_percent": (
                                summary.max_drawdown_percent
                            ),
                            "development_sharpe": summary.sharpe_ratio,
                            "score": (
                                summary.max_drawdown_percent
                                + 0.25 * summary.sharpe_ratio
                                if eligible
                                else float("-inf")
                            ),
                        }
                    )
        hedge_frame = pd.DataFrame(hedge_candidates).sort_values(
            ["score", "development_profit_ratio_vs_base"],
            ascending=False,
        )
        selected_hedge = hedge_frame.iloc[0].to_dict()
        hedge_selections.append(
            {
                "selection_category": category,
                **selected_hedge,
                "selected_on": f"Date <= {args.development_end}",
            }
        )
        for period, result in period_results.items():
            summary = _run_selected_hedge(result.daily, selected_hedge)
            rows.append(
                _summary_row(
                    category=category,
                    period=period,
                    variant="Conditional2xShortHedgeApprox",
                    summary=summary,
                )
            )

    comparison = pd.DataFrame(rows)
    base_profit = comparison[
        comparison["Variant"] == "Base1x"
    ][["SelectionCategory", "Period", "NetProfit"]].rename(
        columns={"NetProfit": "BaseNetProfit"}
    )
    comparison = comparison.merge(
        base_profit,
        on=["SelectionCategory", "Period"],
        how="left",
    )
    comparison["ProfitRatioVsBase"] = (
        comparison["NetProfit"] / comparison["BaseNetProfit"]
    )
    timestamp = datetime.now(UTC).astimezone().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    comparison_path = atomic_to_csv(
        comparison,
        output_root / f"leverage_comparison_{timestamp}.csv",
        index=False,
    )
    manifest_output = _atomic_json(
        {
            "source_mass_manifest": str(manifest_path),
            "selected_on": f"Date <= {args.development_end}",
            "untouched_holdout_start": args.holdout_start,
            "synthetic_instrument_assumptions": {
                "daily_reset": True,
                "annual_expense_ratio": 0.0095,
                "annual_financing_spread": 0.01,
                "conditional_hedge_transaction_cost_bps": 10.0,
                "full_2x_variants": (
                    "Exact portfolio rerun with synthetic daily-reset prices."
                ),
                "tactical_and_hedge_variants": (
                    "Flow-adjusted return overlays; approximate, not an ETF "
                    "execution backtest."
                ),
            },
            "conditional_short_hedges": hedge_selections,
            "comparison_file": str(comparison_path),
        },
        output_root / f"leverage_manifest_{timestamp}.json",
    )
    print(comparison.to_string(index=False))
    print(f"comparison={comparison_path}")
    print(f"manifest={manifest_output}")


if __name__ == "__main__":
    main()
