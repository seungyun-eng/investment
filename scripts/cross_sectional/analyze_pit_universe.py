from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.cross_sectional.config import (
    load_settings,
    settings_from_dict,
)
from stock_research.cross_sectional.pit_validation import (
    generate_pit_diagnostic,
)
from stock_research.cross_sectional.research import load_selected_strategy
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V6-B Task 2 with a causal weekly calendar and optional "
            "point-in-time membership plus delisting-return inputs."
        )
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument(
        "--ticker-config",
        required=True,
        help="Current 200-name ticker configuration used by the baseline.",
    )
    parser.add_argument(
        "--supplemental-ticker-config",
        action="append",
        default=[],
        help=(
            "Additional local ticker configuration. Repeat for multiple "
            "files."
        ),
    )
    parser.add_argument(
        "--pit-membership",
        help=(
            "CSV with AsOfDate,Ticker,Rank and optional audit columns. "
            "Omit to run only the explicitly non-PIT local proxy."
        ),
    )
    parser.add_argument(
        "--delistings",
        help=(
            "CSV with Ticker,EffectiveDate,DelistingReturn and optional "
            "DelistingCategory,Exchange."
        ),
    )
    parser.add_argument(
        "--missing-delisting-return-policy",
        choices=["ERROR", "ZERO", "TOTAL_LOSS", "EXCHANGE_HAIRCUT"],
        default="ERROR",
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional cross-check; must equal settings embedded in the "
            "frozen strategy JSON."
        ),
    )
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    strategy_path = Path(args.strategy).expanduser().resolve()
    strategy_payload = json.loads(
        strategy_path.read_text(encoding="utf-8")
    )
    settings = settings_from_dict(strategy_payload["settings"])
    if args.config:
        config_path = _resolve_path(paths.repo_root, args.config)
        if load_settings(config_path) != settings:
            raise ValueError(
                "Config differs from frozen strategy settings; refusing "
                "to alter the V6-B baseline."
            )
    params = load_selected_strategy(strategy_path)
    artifacts = generate_pit_diagnostic(
        paths,
        settings,
        params,
        current_ticker_config_path=_resolve_path(
            paths.repo_root,
            args.ticker_config,
        ),
        supplemental_ticker_config_paths=[
            _resolve_path(paths.repo_root, value)
            for value in args.supplemental_ticker_config
        ],
        pit_membership_path=(
            _resolve_path(paths.repo_root, args.pit_membership)
            if args.pit_membership
            else None
        ),
        delisting_events_path=(
            _resolve_path(paths.repo_root, args.delistings)
            if args.delistings
            else None
        ),
        missing_delisting_return_policy=(
            args.missing_delisting_return_policy
        ),
        output_dir=args.output_dir,
    )
    print(f"Frozen strategy: {strategy_path}")
    print(f"Output: {artifacts.output_dir}")
    print(f"Local universe audit: {artifacts.local_universe_audit_csv}")
    print(f"Requested comparison: {artifacts.requested_comparison_csv}")
    print(f"Scenario summary: {artifacts.scenario_summary_csv}")
    print(f"Annual membership: {artifacts.annual_membership_csv}")
    print(f"WDC audit: {artifacts.wdc_audit_csv}")
    print(
        "Signal-date comparison: "
        f"{artifacts.signal_date_comparison_csv}"
    )
    print(f"Scenario equity: {artifacts.scenario_equity_csv}")
    print(f"Data readiness: {artifacts.data_readiness_csv}")
    print(f"Delisting policy: {artifacts.delisting_policy_csv}")
    print(f"Manifest: {artifacts.manifest_json}")


def _resolve_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else repo_root / candidate


if __name__ == "__main__":
    main()
