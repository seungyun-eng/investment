from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import load_paths
from stock_research.tsla_integrated.config import load_config
from stock_research.tsla_integrated.data import (
    load_macro_predictions,
    load_tsla_financials,
    load_tsla_prices,
)
from stock_research.tsla_integrated.downside import (
    add_strict_oos_downside_probability,
)
from stock_research.tsla_integrated.features import build_integrated_features
from stock_research.tsla_integrated.optimization import (
    buy_and_hold_params,
    evaluate,
    optimize_on_development,
)
from stock_research.tsla_integrated.portfolio import run_integrated_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize point-in-time TSLA financial/macro/technical signals."
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/tsla_integrated/research.json"),
    )
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--financials", type=Path)
    parser.add_argument("--macro-predictions", type=Path)
    parser.add_argument("--candidates", type=int)
    return parser.parse_args()


def _latest(folder: Path, pattern: str) -> Path:
    hits = list(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return max(hits, key=lambda path: path.stat().st_mtime)


def _atomic_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".json.tmp", prefix=path.stem + "_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    args = parse_args()
    paths = load_paths(args.stock_root)
    _, settings = load_config(args.config)
    price_path = args.prices or _latest(
        paths.processed, "Tesla*지표포함*.csv"
    )
    financial_path = (
        args.financials or paths.financial_raw / "TSLA_financials_Q.xlsx"
    )
    macro_path = args.macro_predictions or _latest(
        paths.results / "SP500" / "macro_momentum_sp500",
        "oos_predictions_*.csv",
    )
    prices = load_tsla_prices(price_path)
    features = build_integrated_features(
        prices,
        load_tsla_financials(financial_path),
        load_macro_predictions(macro_path),
        financial_release_lag_days=settings.financial_release_lag_days,
    )
    features = add_strict_oos_downside_probability(features)
    development = features[
        (features["Date"] >= settings.development_start)
        & (features["Date"] <= settings.development_end)
    ].reset_index(drop=True)
    holdout = features[
        features["Date"] >= settings.holdout_start
    ].reset_index(drop=True)
    backtest_kwargs = {
        "initial_capital": settings.initial_capital,
        "transaction_cost_bps": settings.transaction_cost_bps,
        "slippage_bps": settings.slippage_bps,
        "annual_short_borrow_bps": settings.annual_short_borrow_bps,
    }
    optimized = optimize_on_development(
        development,
        candidate_count=args.candidates or settings.optimization_candidates,
        seed=settings.seed,
        maximum_drawdown_percent=settings.maximum_drawdown_percent,
        **backtest_kwargs,
    )
    frozen = evaluate(
        holdout, optimized.selected_params, **backtest_kwargs
    )
    benchmark_signals = holdout.copy()
    benchmark_signals["CompositeScore"] = 1.0
    benchmark_signals["BuySignal"] = True
    benchmark_signals["SellSignal"] = False
    benchmark = run_integrated_backtest(
        benchmark_signals, buy_and_hold_params(), **backtest_kwargs
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    output = paths.results / "Tesla" / "integrated_signal"
    files = {
        "candidates": atomic_to_csv(
            optimized.candidates,
            output / f"development_candidates_{timestamp}.csv",
            index=False,
        ),
        "holdout_daily": atomic_to_csv(
            frozen.daily,
            output / f"holdout_daily_{timestamp}.csv",
            index=False,
        ),
        "holdout_trades": atomic_to_csv(
            frozen.trades,
            output / f"holdout_trades_{timestamp}.csv",
            index=False,
        ),
    }
    manifest = {
        "selected_on": (
            f"{settings.development_start} <= Date <= "
            f"{settings.development_end}"
        ),
        "untouched_holdout_start": settings.holdout_start,
        "financial_release_lag_days": settings.financial_release_lag_days,
        "inputs": {
            "prices": str(price_path.resolve()),
            "financials": str(financial_path.resolve()),
            "macro_predictions": str(macro_path.resolve()),
        },
        "selected_params": optimized.selected_params.as_dict(),
        "holdout": frozen.summary.__dict__,
        "buy_and_hold": benchmark.summary.__dict__,
        "outputs": {key: str(value) for key, value in files.items()},
    }
    manifest_path = _atomic_json(
        manifest, output / f"manifest_{timestamp}.json"
    )
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
