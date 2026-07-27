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
from stock_research.macro_fear_buy_sp500.contribution_reporting import (
    generate_contribution_report,
)
from stock_research.macro_fear_buy_sp500.contributions import (
    ContributionConfig,
    ContributionDeploymentPolicy,
    contribution_comparison_table,
    run_contribution_backtest,
)
from stock_research.macro_fear_buy_sp500.features import build_fear_features
from stock_research.macro_fear_buy_sp500.strategy import generate_fear_buy_signals
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the $40k initial + monthly contribution fear-buy scenario."
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--selected-params", type=Path)
    parser.add_argument("--initial", type=float, default=40_000.0)
    parser.add_argument("--monthly", type=float, default=4_000.0)
    parser.add_argument("--mild-fraction", type=float, default=1.0)
    parser.add_argument("--fear-fraction", type=float, default=1.0)
    parser.add_argument("--panic-fraction", type=float, default=1.0)
    parser.add_argument("--deployment-cooldown", type=int, default=0)
    parser.add_argument("--holdout-start", default="2017-01-03")
    return parser.parse_args()


def _latest(folder: Path, pattern: str) -> Path:
    hits = list(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return max(hits, key=lambda path: path.stat().st_mtime)


def _constant_signals(
    predictions: pd.DataFrame,
    *,
    target: float,
    state: str,
) -> pd.DataFrame:
    signals = predictions[["Date", "Open", "Close", "CashRate"]].copy()
    for column in ("VIX", "Drawdown252"):
        if column in predictions:
            signals[column] = predictions[column]
    signals["TargetWeight"] = target
    signals["SignalState"] = state
    signals["TransitionReason"] = ""
    return signals


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


def _run_period(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    params: FearBuyParams,
    contribution_config: ContributionConfig,
    deployment_policy: ContributionDeploymentPolicy,
) -> dict[str, object]:
    fear_signals = generate_fear_buy_signals(features, params)
    return {
        "MacroFearBuy": run_contribution_backtest(
            fear_signals,
            params,
            contribution_config,
            name="MacroFearBuy",
            deployment_policy=deployment_policy,
        ),
        "MonthlyBuyHold": run_contribution_backtest(
            _constant_signals(
                predictions,
                target=1.0,
                state="BUY_HOLD",
            ),
            params,
            contribution_config,
            name="MonthlyBuyHold",
            core_weight_override=1.0,
            invest_contributions_without_signal=True,
        ),
        "Initial80ContributionTarget": run_contribution_backtest(
            _constant_signals(
                predictions,
                target=0.80,
                state="INITIAL_80_TARGET",
            ),
            params,
            contribution_config,
            name="Initial80ContributionTarget",
            core_weight_override=0.80,
            invest_contributions_without_signal=True,
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    research_folder = paths.results / "SP500" / "macro_fear_buy_sp500"
    prediction_path = (
        args.predictions
        or _latest(
            paths.results / "SP500" / "macro_momentum_sp500",
            "oos_predictions_*.csv",
        )
    ).resolve()
    selected_path = (
        args.selected_params
        or _latest(research_folder, "selected_params_*.json")
    ).resolve()
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if bool(selected_payload.get("quick_diagnostic")):
        raise ValueError("Refusing to use parameters from a quick diagnostic run.")
    params = FearBuyParams(**selected_payload["strategy"])
    contribution_config = ContributionConfig(
        initial_lump_sum=args.initial,
        monthly_contribution=args.monthly,
        transaction_cost_bps=float(
            selected_payload["research"]["transaction_cost_bps"]
        ),
        slippage_bps=float(selected_payload["research"]["slippage_bps"]),
    )
    deployment_policy = ContributionDeploymentPolicy(
        mild_fraction=args.mild_fraction,
        fear_fraction=args.fear_fraction,
        panic_fraction=args.panic_fraction,
        cooldown_sessions=args.deployment_cooldown,
    )

    print(f"Stage=load predictions Source={prediction_path}")
    predictions = pd.read_csv(prediction_path, parse_dates=["Date"])
    features = build_fear_features(predictions, params)
    full_results = _run_period(
        predictions,
        features,
        params,
        contribution_config,
        deployment_policy,
    )
    holdout_predictions = predictions[
        predictions["Date"] >= pd.Timestamp(args.holdout_start)
    ].reset_index(drop=True)
    holdout_features = features[
        features["Date"] >= pd.Timestamp(args.holdout_start)
    ].reset_index(drop=True)
    holdout_results = _run_period(
        holdout_predictions,
        holdout_features,
        params,
        contribution_config,
        deployment_policy,
    )
    comparison = pd.concat(
        [
            contribution_comparison_table(
                full_results,
                period="Full OOS 2007–2026",
            ),
            contribution_comparison_table(
                holdout_results,
                period=f"Fresh start {args.holdout_start}+",
            ),
        ],
        ignore_index=True,
    )

    output_folder = research_folder / "monthly_contributions"
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    outputs: dict[str, Path] = {
        "comparison": atomic_to_csv(
            comparison,
            output_folder / f"contribution_comparison_{timestamp}.csv",
            index=False,
        )
    }
    for prefix, results in (
        ("full", full_results),
        ("holdout", holdout_results),
    ):
        outputs[f"{prefix}_fear_daily"] = atomic_to_csv(
            results["MacroFearBuy"].daily,
            output_folder / f"{prefix}_fear_daily_{timestamp}.csv",
            index=False,
        )
        outputs[f"{prefix}_fear_trades"] = atomic_to_csv(
            results["MacroFearBuy"].trades,
            output_folder / f"{prefix}_fear_trades_{timestamp}.csv",
            index=False,
        )
        outputs[f"{prefix}_buy_hold_daily"] = atomic_to_csv(
            results["MonthlyBuyHold"].daily,
            output_folder / f"{prefix}_buy_hold_daily_{timestamp}.csv",
            index=False,
        )
    outputs["report"] = generate_contribution_report(
        output_folder,
        full_results=full_results,
        holdout_results=holdout_results,
        comparison=comparison,
        prediction_source=prediction_path,
        selected_params_source=selected_path,
        deployment_policy=deployment_policy,
    )
    outputs["manifest"] = _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "prediction_source": str(prediction_path),
            "selected_params_source": str(selected_path),
            "configuration": {
                "initial_lump_sum": contribution_config.initial_lump_sum,
                "monthly_contribution": contribution_config.monthly_contribution,
                "deployment_policy": {
                    "mild_fraction": deployment_policy.mild_fraction,
                    "fear_fraction": deployment_policy.fear_fraction,
                    "panic_fraction": deployment_policy.panic_fraction,
                    "cooldown_sessions": deployment_policy.cooldown_sessions,
                },
                "holdout_start": args.holdout_start,
            },
            "files": {name: str(path) for name, path in outputs.items()},
        },
        output_folder / f"manifest_{timestamp}.json",
    )

    print(comparison.to_string(index=False))
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
