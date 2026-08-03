from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.config import settings_from_dict
from stock_research.cross_sectional.data import discover_universe
from stock_research.cross_sectional.dynamic_top_n import (
    build_sp500_top_n_membership,
)
from stock_research.cross_sectional.filing_v7_optimization import (
    run_filing_v7_optimization,
)
from stock_research.cross_sectional.live_top10_watchlist import (
    allocation_with_cash,
    build_top_n_plus_watchlist_membership,
    new_account_allocation,
    watchlist_entries,
)
from stock_research.io_utils import atomic_to_csv
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the filing-aware V7 model on dated S&P Top 10 snapshots "
            "plus a user-maintained watchlist, with a full-cash risk gate."
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
    config = json.loads(
        _resolve(paths.repo_root, args.config).read_text(encoding="utf-8")
    )
    optimization = json.loads(
        _resolve(paths.repo_root, args.optimization_config).read_text(
            encoding="utf-8"
        )
    )
    direct = pd.read_csv(_resolve(paths.repo_root, args.direct_rankings))
    sp500 = pd.read_csv(_resolve(paths.repo_root, args.sp500_membership))
    top_n = build_sp500_top_n_membership(
        direct,
        sp500,
        top_n=int(config.get("top_n", 10)),
    )
    membership = build_top_n_plus_watchlist_membership(
        top_n,
        watchlist_entries(config.get("watchlist", [])),
        end_date=args.as_of or date.today(),
    )

    ticker_config = _resolve(paths.repo_root, args.ticker_config)
    discovered, discovery_audit = discover_universe(paths, ticker_config)
    requested = set(membership["DataSymbol"])
    ready_members = [member for member in discovered if member.ticker in requested]
    settings = settings_from_dict(config["research"])
    if len(ready_members) < settings.minimum_cross_section_size:
        raise ValueError(
            f"Only {len(ready_members)} Top-10/watchlist tickers are model-ready"
        )
    spy_path = (
        _resolve(paths.repo_root, args.spy)
        if args.spy
        else paths.macro / "SPY Adjusted Historical Data.csv"
    )
    artifacts = run_filing_v7_optimization(
        paths,
        settings,
        universe_config=config,
        optimization_config=optimization,
        filing_features_path=_resolve(paths.repo_root, args.filing_features),
        spy_path=spy_path,
        universe_members=ready_members,
        membership=membership,
        market_cash_gate=dict(config.get("market_cash_gate", {})),
        output_dir=args.output_dir,
    )

    membership_path = artifacts.output_dir / "top10_plus_watchlist_membership.csv"
    readiness_path = artifacts.output_dir / "universe_readiness.csv"
    allocation_path = artifacts.output_dir / "latest_allocation_with_cash.csv"
    new_account_path = (
        artifacts.output_dir / "latest_new_account_allocation_with_cash.csv"
    )
    atomic_to_csv(membership, membership_path, index=False)
    audit = discovery_audit.copy()
    audit["RequestedByLiveUniverse"] = audit["Ticker"].isin(requested)
    atomic_to_csv(
        audit.loc[audit["RequestedByLiveUniverse"]], readiness_path, index=False
    )
    signals = pd.read_csv(artifacts.selected_signals_csv)
    selected = signals.loc[
        signals["Series"].eq("V7_SEC_COMBINED_OPTIMIZED")
    ].copy()
    allocation = allocation_with_cash(selected)
    atomic_to_csv(allocation, allocation_path, index=False)
    latest_scores = pd.read_csv(artifacts.latest_scores_csv)
    selected_scores = latest_scores.loc[
        latest_scores["Series"].eq("V7_SEC_COMBINED_OPTIMIZED")
    ]
    selection = json.loads(artifacts.selection_json.read_text(encoding="utf-8"))
    new_account = new_account_allocation(
        selected_scores,
        top_k=int(selection["selected_combined"]["top_k"]),
    )
    atomic_to_csv(new_account, new_account_path, index=False)

    print(f"Output: {artifacts.output_dir}")
    print(f"Membership: {membership_path}")
    print(f"Readiness: {readiness_path}")
    print(f"Latest allocation: {allocation_path}")
    print(allocation[["Date", "Ticker", "TargetWeight"]].to_string(index=False))
    print(f"New-account allocation: {new_account_path}")
    print(new_account[["Date", "Ticker", "TargetWeight"]].to_string(index=False))


def _resolve(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


if __name__ == "__main__":
    main()
