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
    ContributionDeploymentPolicy,
    run_contribution_backtest,
)
from stock_research.macro_fear_buy_sp500.features import build_fear_features
from stock_research.macro_fear_buy_sp500.strategy import (
    generate_fear_buy_signals,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select monthly cash-deployment fractions on 2007-2016 only, "
            "then evaluate the frozen policy on 2017+."
        )
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--selected-params", type=Path)
    parser.add_argument("--initial", type=float, default=40_000.0)
    parser.add_argument("--monthly", type=float, default=4_000.0)
    parser.add_argument("--development-end", default="2016-12-30")
    parser.add_argument("--holdout-start", default="2017-01-03")
    parser.add_argument("--quick", action="store_true")
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


def _constant_signals(signals: pd.DataFrame) -> pd.DataFrame:
    benchmark = signals[["Date", "Open", "Close", "CashRate"]].copy()
    benchmark["TargetWeight"] = 1.0
    benchmark["SignalState"] = "BUY_HOLD"
    benchmark["TransitionReason"] = ""
    return benchmark


def _period_signals(
    features: pd.DataFrame,
    params: FearBuyParams,
    *,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    period = features.copy()
    if start:
        period = period[period["Date"] >= pd.Timestamp(start)]
    if end:
        period = period[period["Date"] <= pd.Timestamp(end)]
    return generate_fear_buy_signals(
        period.reset_index(drop=True),
        params,
    )


def _policy_grid(
    *,
    quick: bool,
) -> list[ContributionDeploymentPolicy]:
    mild_values = (0.0, 0.25, 1.0) if quick else (0.0, 0.1, 0.25, 0.5, 1.0)
    fear_values = (0.5, 1.0) if quick else (0.25, 0.5, 0.75, 1.0)
    panic_values = (1.0,) if quick else (0.75, 1.0)
    cooldown_values = (0, 21) if quick else (0, 21, 42, 63)
    policies: list[ContributionDeploymentPolicy] = []
    for mild in mild_values:
        for fear in fear_values:
            for panic in panic_values:
                if not mild <= fear <= panic:
                    continue
                for cooldown in cooldown_values:
                    policies.append(
                        ContributionDeploymentPolicy(
                            mild_fraction=mild,
                            fear_fraction=fear,
                            panic_fraction=panic,
                            cooldown_sessions=cooldown,
                        )
                    )
    return policies


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
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if bool(payload.get("quick_diagnostic")):
        raise ValueError("Refusing parameters from a quick diagnostic run.")
    params = FearBuyParams(**payload["strategy"])
    config = ContributionConfig(
        initial_lump_sum=args.initial,
        monthly_contribution=args.monthly,
        transaction_cost_bps=float(
            payload["research"]["transaction_cost_bps"]
        ),
        slippage_bps=float(payload["research"]["slippage_bps"]),
    )
    predictions = pd.read_csv(prediction_path, parse_dates=["Date"])
    features = build_fear_features(predictions, params)
    periods = {
        "Development": (None, args.development_end),
        "DevelopmentCrisis": (None, "2011-12-30"),
        "DevelopmentBull": ("2012-01-03", args.development_end),
        "Holdout": (args.holdout_start, None),
        "Full": (None, None),
    }
    signals = {
        name: _period_signals(
            features,
            params,
            start=start,
            end=end,
        )
        for name, (start, end) in periods.items()
    }
    benchmarks = {
        name: run_contribution_backtest(
            _constant_signals(period_signals),
            params,
            config,
            name=f"{name}BuyHold",
            core_weight_override=1.0,
            invest_contributions_without_signal=True,
        )
        for name, period_signals in signals.items()
    }

    rows: list[dict[str, object]] = []
    for policy in _policy_grid(quick=args.quick):
        row: dict[str, object] = {
            "MildFraction": policy.mild_fraction,
            "FearFraction": policy.fear_fraction,
            "PanicFraction": policy.panic_fraction,
            "CooldownSessions": policy.cooldown_sessions,
        }
        fold_ratios: list[float] = []
        for period in (
            "Development",
            "DevelopmentCrisis",
            "DevelopmentBull",
        ):
            result = run_contribution_backtest(
                signals[period],
                params,
                config,
                name="DevelopmentCandidate",
                deployment_policy=policy,
            )
            benchmark = benchmarks[period]
            ratio = result.summary.net_profit / benchmark.summary.net_profit
            row[f"{period}FinalValue"] = result.summary.final_value
            row[f"{period}ProfitRatio"] = ratio
            row[f"{period}XIRR(%)"] = (
                result.summary.money_weighted_return_percent
            )
            row[f"{period}MDD(%)"] = result.summary.max_drawdown_percent
            if period != "Development":
                fold_ratios.append(ratio)
        row["WorstDevelopmentFoldProfitRatio"] = min(fold_ratios)
        rows.append(row)
    candidates = pd.DataFrame(rows).sort_values(
        [
            "DevelopmentProfitRatio",
            "WorstDevelopmentFoldProfitRatio",
        ],
        ascending=False,
    )
    candidates = candidates.reset_index(drop=True)
    candidates.insert(0, "DevelopmentRank", candidates.index + 1)
    winner = candidates.iloc[0]
    selected_policy = ContributionDeploymentPolicy(
        mild_fraction=float(winner["MildFraction"]),
        fear_fraction=float(winner["FearFraction"]),
        panic_fraction=float(winner["PanicFraction"]),
        cooldown_sessions=int(winner["CooldownSessions"]),
    )

    evaluation_rows: list[dict[str, object]] = []
    for period in ("Development", "Holdout", "Full"):
        strategy = run_contribution_backtest(
            signals[period],
            params,
            config,
            name="SelectedContributionPolicy",
            deployment_policy=selected_policy,
        )
        benchmark = benchmarks[period]
        evaluation_rows.append(
            {
                "Period": period,
                "Portfolio": "SelectedContributionPolicy",
                "FinalValue": strategy.summary.final_value,
                "NetProfit": strategy.summary.net_profit,
                "ProfitRatioVsBuyHold": (
                    strategy.summary.net_profit
                    / benchmark.summary.net_profit
                ),
                "XIRR(%)": (
                    strategy.summary.money_weighted_return_percent
                ),
                "MDD(%)": strategy.summary.max_drawdown_percent,
                "Trades": strategy.summary.trade_count,
            }
        )
        evaluation_rows.append(
            {
                "Period": period,
                "Portfolio": "MonthlyBuyHold",
                "FinalValue": benchmark.summary.final_value,
                "NetProfit": benchmark.summary.net_profit,
                "ProfitRatioVsBuyHold": 1.0,
                "XIRR(%)": (
                    benchmark.summary.money_weighted_return_percent
                ),
                "MDD(%)": benchmark.summary.max_drawdown_percent,
                "Trades": benchmark.summary.trade_count,
            }
        )
    evaluation = pd.DataFrame(evaluation_rows)
    output_folder = (
        research_folder / "monthly_contributions" / "policy_optimization"
    )
    timestamp = datetime.now(UTC).astimezone().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    candidate_path = atomic_to_csv(
        candidates,
        output_folder / f"development_candidates_{timestamp}.csv",
        index=False,
    )
    evaluation_path = atomic_to_csv(
        evaluation,
        output_folder / f"frozen_policy_evaluation_{timestamp}.csv",
        index=False,
    )
    selected_path_output = _atomic_json(
        {
            "selected_on": f"Date <= {args.development_end}",
            "holdout_start": args.holdout_start,
            "quick_diagnostic": args.quick,
            "prediction_source": str(prediction_path),
            "signal_parameter_source": str(selected_path),
            "policy": {
                "mild_fraction": selected_policy.mild_fraction,
                "fear_fraction": selected_policy.fear_fraction,
                "panic_fraction": selected_policy.panic_fraction,
                "cooldown_sessions": selected_policy.cooldown_sessions,
            },
            "two_x_definition": (
                "strategy net profit / Buy Hold net profit >= 2.0"
            ),
            "files": {
                "candidates": str(candidate_path),
                "evaluation": str(evaluation_path),
            },
        },
        output_folder / f"selected_policy_{timestamp}.json",
    )
    print(f"Candidates={len(candidates)}")
    print(f"Selected={selected_policy}")
    print(evaluation.to_string(index=False))
    print(f"candidates={candidate_path}")
    print(f"evaluation={evaluation_path}")
    print(f"selected_policy={selected_path_output}")


if __name__ == "__main__":
    main()
