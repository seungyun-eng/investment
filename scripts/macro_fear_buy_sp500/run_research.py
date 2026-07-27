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
from stock_research.macro_fear_buy_sp500.config import load_fear_buy_config
from stock_research.macro_fear_buy_sp500.diagnostics import (
    evaluate_top_development_candidates_on_holdout,
    tactical_buy_forward_returns,
    tactical_trade_summary,
)
from stock_research.macro_fear_buy_sp500.features import build_fear_features
from stock_research.macro_fear_buy_sp500.optimization import (
    optimize_on_development_period,
)
from stock_research.macro_fear_buy_sp500.portfolio import (
    comparison_table,
    run_constant_weight_benchmark,
    run_fear_buy_backtest,
    segment_result,
)
from stock_research.macro_fear_buy_sp500.reporting import generate_fear_buy_report
from stock_research.macro_fear_buy_sp500.strategy import generate_fear_buy_signals
from stock_research.macro_momentum_sp500.robustness import (
    block_bootstrap_excess_return,
    yearly_portfolio_metrics,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize and evaluate a contrarian macro fear-buy SPY strategy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/macro_fear_buy_sp500/research.json"),
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small diagnostic grid; never label it as a final optimization.",
    )
    return parser.parse_args()


def _latest_predictions(folder: Path) -> Path:
    hits = list(folder.glob("oos_predictions_*.csv"))
    if not hits:
        raise FileNotFoundError(
            f"No strict-OOS prediction file found in {folder}. "
            "Run the macro_momentum_sp500 research workflow first."
        )
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    base_params, settings = load_fear_buy_config(args.config)
    prediction_path = args.predictions or _latest_predictions(
        paths.results / "SP500" / "macro_momentum_sp500"
    )
    prediction_path = prediction_path.resolve()

    print(f"Stage=load strict-OOS predictions Source={prediction_path}")
    predictions = pd.read_csv(prediction_path, parse_dates=["Date"])
    print(
        f"Rows={len(predictions)} "
        f"Range={predictions['Date'].min():%Y-%m-%d}.."
        f"{predictions['Date'].max():%Y-%m-%d}"
    )

    print(
        "Stage=development-only optimization "
        f"End={settings.development_end} Quick={args.quick}"
    )
    optimization = optimize_on_development_period(
        predictions,
        base_params,
        settings,
        quick=args.quick,
    )
    selected = optimization.selected_params
    print(f"Candidates={len(optimization.candidates)}")
    print(f"Selected={json.dumps(selected.as_dict(), default=str)}")

    print(
        "Stage=frozen-parameter full OOS and untouched-holdout evaluation "
        f"HoldoutStart={settings.holdout_start}"
    )
    portfolios = {
        "MacroFearBuy": run_fear_buy_backtest(
            predictions,
            selected,
            settings,
        ),
        "BuyHold": run_constant_weight_benchmark(
            predictions,
            selected,
            settings,
            weight=1.0,
            name="BuyHold",
        ),
        "Initial70Cash30": run_constant_weight_benchmark(
            predictions,
            selected,
            settings,
            weight=0.70,
            name="Initial70Cash30",
        ),
        "Initial80Cash20": run_constant_weight_benchmark(
            predictions,
            selected,
            settings,
            weight=0.80,
            name="Initial80Cash20",
        ),
    }
    comparison = comparison_table(portfolios, settings)
    yearly = yearly_portfolio_metrics(
        {name: result.daily for name, result in portfolios.items()}
    )
    full_bootstrap = block_bootstrap_excess_return(
        portfolios["MacroFearBuy"].daily,
        portfolios["BuyHold"].daily,
    )
    full_bootstrap.insert(0, "Period", "Full OOS")
    holdout_strategy = segment_result(
        portfolios["MacroFearBuy"],
        settings,
        start=settings.holdout_start,
    )
    holdout_benchmark = segment_result(
        portfolios["BuyHold"],
        settings,
        start=settings.holdout_start,
    )
    holdout_bootstrap = block_bootstrap_excess_return(
        holdout_strategy.daily,
        holdout_benchmark.daily,
    )
    holdout_bootstrap.insert(0, "Period", "Untouched holdout")
    bootstrap = pd.concat(
        [full_bootstrap, holdout_bootstrap],
        ignore_index=True,
    )
    holdout_stability = evaluate_top_development_candidates_on_holdout(
        predictions,
        optimization.candidates,
        base_params,
        settings,
        benchmark_holdout_cagr=holdout_benchmark.summary.cagr_percent,
    )
    buy_forward_returns = tactical_buy_forward_returns(
        portfolios["MacroFearBuy"]
    )
    tactical_summary = tactical_trade_summary(portfolios["MacroFearBuy"])
    features = build_fear_features(predictions, selected)
    signals = generate_fear_buy_signals(features, selected)

    output_folder = paths.results / "SP500" / "macro_fear_buy_sp500"
    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    outputs = {
        "optimization": atomic_to_csv(
            optimization.candidates,
            output_folder / f"development_optimization_{timestamp}.csv",
            index=False,
        ),
        "comparison": atomic_to_csv(
            comparison,
            output_folder / f"portfolio_comparison_{timestamp}.csv",
            index=False,
        ),
        "yearly": atomic_to_csv(
            yearly,
            output_folder / f"yearly_robustness_{timestamp}.csv",
            index=False,
        ),
        "bootstrap": atomic_to_csv(
            bootstrap,
            output_folder / f"block_bootstrap_{timestamp}.csv",
            index=False,
        ),
        "holdout_stability": atomic_to_csv(
            holdout_stability,
            output_folder / f"holdout_stability_{timestamp}.csv",
            index=False,
        ),
        "buy_forward_returns": atomic_to_csv(
            buy_forward_returns,
            output_folder / f"tactical_buy_forward_returns_{timestamp}.csv",
            index=False,
        ),
        "tactical_summary": atomic_to_csv(
            tactical_summary,
            output_folder / f"tactical_trade_summary_{timestamp}.csv",
            index=False,
        ),
        "signals": atomic_to_csv(
            signals,
            output_folder / f"fear_buy_signals_{timestamp}.csv",
            index=False,
        ),
        "daily": atomic_to_csv(
            portfolios["MacroFearBuy"].daily,
            output_folder / f"fear_buy_daily_{timestamp}.csv",
            index=False,
        ),
        "trades": atomic_to_csv(
            portfolios["MacroFearBuy"].trades,
            output_folder / f"fear_buy_trades_{timestamp}.csv",
            index=False,
        ),
    }
    outputs["selected_params"] = _atomic_json(
        {
            "selected_on": f"Date <= {settings.development_end}",
            "untouched_holdout_start": settings.holdout_start,
            "quick_diagnostic": args.quick,
            "strategy": selected.as_dict(),
            "research": settings.as_dict(),
        },
        output_folder / f"selected_params_{timestamp}.json",
    )
    outputs["report"] = generate_fear_buy_report(
        output_folder,
        portfolios=portfolios,
        comparison=comparison,
        optimization=optimization.candidates,
        holdout_stability=holdout_stability,
        buy_forward_returns=buy_forward_returns,
        tactical_summary=tactical_summary,
        yearly_metrics=yearly,
        bootstrap=bootstrap,
        selected_params=selected,
        settings=settings,
        prediction_source=prediction_path,
    )
    outputs["manifest"] = _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "prediction_source": str(prediction_path),
            "quick_diagnostic": args.quick,
            "files": {name: str(path) for name, path in outputs.items()},
        },
        output_folder / f"manifest_{timestamp}.json",
    )

    print(comparison.to_string(index=False))
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
