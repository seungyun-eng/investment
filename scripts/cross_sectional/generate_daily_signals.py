from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import (
    load_settings,
    settings_from_dict,
)
from stock_research.cross_sectional.research import (
    generate_latest_signals,
    load_model_status,
    load_selected_strategy,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate today's manual-trading signals without retraining."
    )
    parser.add_argument("--strategy")
    parser.add_argument(
        "--config",
        help=(
            "Optional settings JSON. If omitted, reuse the settings frozen "
            "inside selected_strategy.json."
        ),
    )
    parser.add_argument("--ticker-config", default="config/tickers.json")
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    strategy_path = (
        Path(args.strategy).expanduser().resolve()
        if args.strategy
        else _latest_strategy(paths.results)
    )
    if args.config:
        settings = load_settings(paths.repo_root / args.config)
    else:
        manifest = json.loads(strategy_path.read_text(encoding="utf-8"))
        settings = settings_from_dict(manifest["settings"])
    params = load_selected_strategy(strategy_path)
    model_status = load_model_status(strategy_path)
    result = generate_latest_signals(
        paths,
        settings,
        params,
        ticker_config_path=paths.repo_root / args.ticker_config,
        model_status=model_status,
    )
    print(f"Strategy: {strategy_path}")
    print(f"Signal date: {result['latest_date'].date()}")
    print(f"Output: {result['output_dir']}")
    print(
        result["latest_signals"][
            [
                "Ticker",
                "DailySignal",
                "TradeAction",
                "ModelSelected",
                "TargetWeight",
                "Rank",
                "AlphaScore",
            ]
        ].head(15).to_string(index=False)
    )


def _latest_strategy(results_root: Path) -> Path:
    candidates = list(
        (
            results_root / "Cross_Sectional" / "rank_signals"
        ).glob("*/selected_strategy.json")
    )
    if not candidates:
        raise FileNotFoundError(
            "No selected_strategy.json exists; run cross-sectional research first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
