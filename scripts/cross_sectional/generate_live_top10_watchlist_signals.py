from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.config import (
    StrategyParams,
    settings_from_dict,
)
from stock_research.cross_sectional.data import discover_universe
from stock_research.cross_sectional.dynamic_top_n import (
    build_sp500_top_n_membership,
)
from stock_research.cross_sectional.features import build_panel
from stock_research.cross_sectional.filing_signals import (
    add_filing_factors,
    merge_filing_features,
)
from stock_research.cross_sectional.filing_v7_optimization import (
    FilingV7Policy,
    generate_filing_v7_targets,
)
from stock_research.cross_sectional.live_top10_watchlist import (
    allocation_with_cash,
    apply_full_cash_market_gate,
    build_top_n_plus_watchlist_membership,
    new_account_allocation,
    watchlist_entries,
)
from stock_research.cross_sectional.pit_validation import (
    apply_membership_to_panel,
)
from stock_research.cross_sectional.signals import signal_day_panel
from stock_research.cross_sectional.v7_pit_evaluation import (
    normalize_change_membership,
)
from stock_research.cross_sectional.v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)
from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen-policy daily monitoring and Friday rebalance "
            "signals without re-optimizing parameters."
        )
    )
    parser.add_argument(
        "--config",
        default="config/cross_sectional/live_top10_watchlist.json",
    )
    parser.add_argument(
        "--optimization-config",
        default="config/cross_sectional/filing_v7_optimization.json",
    )
    parser.add_argument("--direct-rankings", required=True)
    parser.add_argument("--sp500-membership", required=True)
    parser.add_argument(
        "--ticker-config",
        default="config/cross_sectional/live_top10_watchlist_tickers.json",
    )
    parser.add_argument("--filing-features", required=True)
    parser.add_argument("--spy")
    parser.add_argument("--as-of")
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args()
    paths = load_paths(args.stock_root)
    config = _read_json(_resolve(paths.repo_root, args.config))
    optimization = _read_json(
        _resolve(paths.repo_root, args.optimization_config)
    )
    settings = settings_from_dict(config["research"])
    top_n = build_sp500_top_n_membership(
        pd.read_csv(_resolve(paths.repo_root, args.direct_rankings)),
        pd.read_csv(_resolve(paths.repo_root, args.sp500_membership)),
        top_n=int(config.get("top_n", 10)),
    )
    membership = build_top_n_plus_watchlist_membership(
        top_n,
        watchlist_entries(config.get("watchlist", [])),
        end_date=args.as_of or date.today(),
    )
    requested = set(membership["DataSymbol"])
    members, discovery_audit = discover_universe(
        paths, _resolve(paths.repo_root, args.ticker_config)
    )
    members = [member for member in members if member.ticker in requested]
    if len(members) < settings.minimum_cross_section_size:
        raise ValueError(f"Only {len(members)} requested tickers are model-ready")

    panel, data_audit = build_panel(members, settings)
    observed = add_v7_technical_observations(panel)
    pit = apply_membership_to_panel(
        observed, normalize_change_membership(membership), settings
    )
    technical = add_v7_technical_factors(pit, settings)
    variant_name = str(config["technical_variant"])
    variant = next(item for item in TECHNICAL_VARIANTS if item.name == variant_name)
    scoring_panel = scoring_panel_for_variant(technical, variant)
    merged = merge_filing_features(
        scoring_panel,
        pd.read_csv(_resolve(paths.repo_root, args.filing_features)),
    )
    factored = add_filing_factors(
        merged,
        minimum_cross_section_size=min(
            settings.minimum_cross_section_size,
            max(4, len(members) // 2),
        ),
    )
    spy_path = (
        _resolve(paths.repo_root, args.spy)
        if args.spy
        else paths.macro / "SPY Adjusted Historical Data.csv"
    )
    cash = dict(config["market_cash_gate"])
    factored = apply_full_cash_market_gate(
        factored,
        load_sp500_proxy(spy_path),
        slow_sessions=int(cash["slow_sessions"]),
        fast_sessions=int(cash["fast_sessions"]),
        band=float(cash["band"]),
    )
    latest_date = pd.Timestamp(factored["Date"].max())
    signal_days = signal_day_panel(
        factored,
        settings.train_start,
        str(latest_date.date()),
        settings.rebalance_weekday,
    )
    policy = FilingV7Policy(**dict(config["frozen_policy"]))
    base_params = StrategyParams.from_dict(
        dict(optimization["base_v7_params"])
    )
    scored, targets = generate_filing_v7_targets(
        signal_days, base_params, policy
    )
    latest_signal_date = pd.Timestamp(scored["Date"].max())
    latest_scores = scored.loc[scored["Date"].eq(latest_signal_date)].copy()
    existing = allocation_with_cash(targets, latest_signal_date)
    fresh = new_account_allocation(
        latest_scores, top_k=policy.top_k, date=latest_signal_date
    )

    destination = _destination(paths.results, args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(latest_scores, destination / "latest_scores.csv", index=False)
    atomic_to_csv(targets, destination / "signal_history.csv", index=False)
    atomic_to_csv(
        existing,
        destination / "existing_account_allocation_with_cash.csv",
        index=False,
    )
    atomic_to_csv(
        fresh,
        destination / "new_account_allocation_with_cash.csv",
        index=False,
    )
    readiness = discovery_audit.loc[
        discovery_audit["Ticker"].isin(requested)
    ].copy()
    atomic_to_csv(readiness, destination / "universe_readiness.csv", index=False)
    atomic_to_csv(data_audit, destination / "data_audit.csv", index=False)
    atomic_to_csv(
        membership,
        destination / "top10_plus_watchlist_membership.csv",
        index=False,
    )
    _atomic_json(
        destination / "manifest.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_status": config["model_status"],
            "mode": "FROZEN_POLICY_NO_REOPTIMIZATION",
            "latest_price_date": latest_date,
            "latest_signal_date": latest_signal_date,
            "policy": config["frozen_policy"],
            "policy_source": config["frozen_policy_source"],
            "market_cash_gate": cash,
            "daily_use": "monitor scores, filings, prices, and market regime",
            "trade_use": "Friday close signal; next market open execution",
            "warning": "Research-only model; strict annual-return constraints failed.",
        },
    )
    print(f"Output: {destination}")
    print(f"Price date: {latest_date.date()}")
    print(f"Signal date: {latest_signal_date.date()}")
    print(f"Model status: {config['model_status']}")
    print("New account:")
    print(fresh[["Ticker", "TargetWeight"]].to_string(index=False))
    print("Existing model account:")
    print(existing[["Ticker", "TargetWeight"]].to_string(index=False))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


def _destination(results: Path, value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return results / "Cross_Sectional" / "live_top10_watchlist_signals" / stamp


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
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


if __name__ == "__main__":
    main()
