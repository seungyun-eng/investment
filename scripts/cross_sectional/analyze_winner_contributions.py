from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import (
    load_settings,
    settings_from_dict,
)
from stock_research.cross_sectional.research import load_selected_strategy
from stock_research.cross_sectional.winner_attribution import (
    generate_winner_dependence_analysis,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose frozen V6-B PnL by ticker and rerun the unchanged "
            "strategy after ex-post top-winner exclusions."
        )
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional settings cross-check. It must equal the settings "
            "embedded in the frozen strategy JSON."
        ),
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Frozen V6-B selected_strategy.json.",
    )
    parser.add_argument(
        "--ticker-config",
        default="config/tickers.json",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--stock-root")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    ticker_config_path = _resolve_repo_path(
        paths.repo_root,
        args.ticker_config,
    )
    strategy_path = Path(args.strategy).expanduser().resolve()
    strategy_payload = json.loads(
        strategy_path.read_text(encoding="utf-8")
    )
    settings = settings_from_dict(strategy_payload["settings"])
    if args.config:
        config_path = _resolve_repo_path(paths.repo_root, args.config)
        if load_settings(config_path) != settings:
            raise ValueError(
                "Config differs from the settings embedded in the frozen "
                "strategy; refusing to change the V6-B baseline."
            )
    params = load_selected_strategy(strategy_path)
    artifacts = generate_winner_dependence_analysis(
        paths,
        settings,
        params,
        ticker_config_path=ticker_config_path,
        output_dir=args.output_dir,
    )
    print(f"Frozen strategy: {strategy_path}")
    print(f"Output: {artifacts.output_dir}")
    print(f"Ticker contributions: {artifacts.contributions_csv}")
    print(f"Excluded winners: {artifacts.excluded_winners_csv}")
    print(f"Leave-winner-out summary: {artifacts.scenario_summary_csv}")
    print(f"Continuous equity: {artifacts.scenario_equity_csv}")
    print(f"Daily attribution: {artifacts.daily_attribution_csv}")
    print(f"Manifest: {artifacts.manifest_csv}")


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else repo_root / candidate


if __name__ == "__main__":
    main()
