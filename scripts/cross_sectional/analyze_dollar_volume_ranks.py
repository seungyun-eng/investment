from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.dollar_volume_ranking import (
    parse_historical_price_file,
    scan_backtest_folder,
    select_company_file,
)
from stock_research.io_utils import atomic_to_csv
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Back Test files and validate close-times-volume parsing "
            "before computing survivor-universe dollar-volume ranks."
        )
    )
    parser.add_argument("--stock-root")
    parser.add_argument("--backtest-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--ticker", default="WDC")
    parser.add_argument("--company", default="Western Digital")
    parser.add_argument("--validation-date", default="2026-07-28")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    backtest_root = (
        Path(args.backtest_root).expanduser().resolve()
        if args.backtest_root
        else paths.raw_prices
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else paths.results
        / "Cross_Sectional"
        / "dollar_volume_diagnostic"
        / _run_name()
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    inventory, scan_issues = scan_backtest_folder(backtest_root)
    company_file = select_company_file(inventory, args.company)
    parsed = parse_historical_price_file(
        company_file,
        ticker=args.ticker,
        company=args.company,
    )
    validation_date = pd.Timestamp(args.validation_date)
    validation = parsed.data.loc[parsed.data["Date"].eq(validation_date)].copy()
    if len(validation) != 1:
        raise ValueError(
            f"Expected one {args.ticker} row on {validation_date.date()}, "
            f"found {len(validation)}"
        )
    validation["DollarVolumeBillions"] = validation["DollarVolume"] / 1e9
    validation["CheckFormula"] = (
        validation["Close"].map(lambda value: f"{value:.6f}")
        + " * "
        + validation["Volume"].map(lambda value: f"{value:.0f}")
    )

    atomic_to_csv(inventory, output_dir / "scan_inventory.csv", index=False)
    atomic_to_csv(scan_issues, output_dir / "scan_issues.csv", index=False)
    atomic_to_csv(parsed.audit, output_dir / "wdc_parse_audit.csv", index=False)
    atomic_to_csv(
        parsed.data,
        output_dir / "wdc_normalized_prices.csv",
        index=False,
    )
    atomic_to_csv(
        validation,
        output_dir / "wdc_validation_2026_07_28.csv",
        index=False,
    )
    _atomic_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "folder scan and WDC single-file parsing validation",
            "ranking_scope": "not yet executed",
            "backtest_root": str(backtest_root),
            "company_folder_count": int(
                sum(path.is_dir() for path in backtest_root.iterdir())
            ),
            "csv_file_count": len(inventory),
            "scan_issue_count": len(scan_issues),
            "survivorship_warning": (
                "Future ranks are relative only to the frozen current-survivor "
                "universe, not actual historical whole-market ranks."
            ),
        },
        output_dir / "manifest.json",
    )

    row = validation.iloc[0]
    print(f"Output: {output_dir}")
    print(
        f"Scan: {len(inventory)} CSV files; "
        f"{len(scan_issues)} folder-level issue(s)"
    )
    print(
        f"{args.ticker} {validation_date.date()}: "
        f"Close={row['Close']:.6f}, Volume={row['Volume']:,.0f}, "
        f"DollarVolume=${row['DollarVolume']:,.2f} "
        f"(${row['DollarVolumeBillions']:.6f}B)"
    )
    print(
        "Scope warning: no rank has been computed yet; future ranks are "
        "current-survivor-relative, not whole-market historical ranks."
    )


def _run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}_survivor_dollar_volume_scan"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
