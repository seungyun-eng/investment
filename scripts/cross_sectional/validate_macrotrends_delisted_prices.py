from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.macrotrends_price_validation import (
    PriceProbeTarget,
    compare_with_local_price_file,
    probe_macrotrends_price_history,
    summarize_failed_fetch_log,
)
from stock_research.io_utils import atomic_to_csv
from stock_research.paths import load_paths

DEFAULT_TARGETS = (
    PriceProbeTarget("CELG", "celgene", "Celgene"),
    PriceProbeTarget("RTN", "raytheon", "Raytheon"),
    PriceProbeTarget("TIF", "tiffany", "Tiffany"),
)
CONTROL = PriceProbeTarget("WDC", "western-digital", "Western Digital")


def _parse_target(value: str) -> PriceProbeTarget:
    parts = value.split(":", maxsplit=2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("Use TICKER:company-slug[:Company Name].")
    company = parts[2] if len(parts) == 3 else ""
    return PriceProbeTarget(parts[0], parts[1], company).normalized()


def _latest_fetch_log(results_root: Path) -> Path | None:
    root = results_root / "Cross_Sectional" / "pit_universe_builds"
    matches = list(root.glob("*/fetch_log.csv"))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _find_control_file(raw_prices: Path, company: str) -> Path | None:
    folder = raw_prices / company
    matches = list(folder.glob("*.csv")) if folder.is_dir() else []
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description=(
            "Validate Macrotrends daily OHLCV availability for a small delisted "
            "sample. This script deliberately does not crawl all failed tickers."
        )
    )
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        help="TICKER:company-slug[:Company Name]; repeat for 2-3 validation names.",
    )
    parser.add_argument("--fetch-log", type=Path)
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    paths = load_paths()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output_folder.expanduser().resolve()
        if args.output_folder
        else paths.results
        / "Cross_Sectional"
        / "macrotrends_price_validation"
        / timestamp
    )
    output.mkdir(parents=True, exist_ok=True)

    targets = tuple(args.target) if args.target else DEFAULT_TARGETS
    if not 2 <= len(targets) <= 3:
        raise ValueError("Validation must use only 2 or 3 delisted/merged tickers.")

    results: list[dict[str, object]] = []
    control_result, control_daily = probe_macrotrends_price_history(
        CONTROL,
        timeout_seconds=args.timeout,
        retries=args.retries,
        request_pause_seconds=args.pause,
        retry_backoff_seconds=args.retry_backoff,
    )
    results.append({"SampleType": "ACTIVE_CONTROL", **control_result.to_record()})

    for target in targets:
        result, _ = probe_macrotrends_price_history(
            target,
            timeout_seconds=args.timeout,
            retries=args.retries,
            request_pause_seconds=args.pause,
            retry_backoff_seconds=args.retry_backoff,
        )
        results.append({"SampleType": "DELISTED_VALIDATION", **result.to_record()})

    probe_frame = pd.DataFrame(results)
    atomic_to_csv(probe_frame, output / "macrotrends_probe_results.csv", index=False)

    fetch_log = args.fetch_log or _latest_fetch_log(paths.results)
    if fetch_log and fetch_log.is_file():
        failed, failed_summary = summarize_failed_fetch_log(fetch_log)
        atomic_to_csv(failed, output / "source_fetch_failures.csv", index=False)
        atomic_to_csv(
            failed_summary,
            output / "source_fetch_failure_summary.csv",
            index=False,
        )
        print(f"Yahoo source failures: {len(failed)}")
        print(failed_summary.to_string(index=False))
    else:
        print("WARN: fetch_log.csv was not found; source-failure counts were skipped.")

    control_file = _find_control_file(paths.raw_prices, CONTROL.company)
    if control_result.price_data_available and control_file:
        overlap, comparison = compare_with_local_price_file(
            control_daily,
            control_file,
            ticker=CONTROL.ticker,
            company=CONTROL.company,
        )
        atomic_to_csv(
            comparison,
            output / "active_control_adjustment_summary.csv",
            index=False,
        )
        sample_dates = pd.to_datetime(
            ["2019-01-02", "2019-07-29", "2024-12-31", "2025-12-31", "2026-07-28"]
        )
        samples = overlap.loc[overlap["Date"].isin(sample_dates)]
        atomic_to_csv(
            samples,
            output / "active_control_adjustment_samples.csv",
            index=False,
        )

    validation = probe_frame.loc[
        probe_frame["SampleType"].eq("DELISTED_VALIDATION")
    ]
    successes = int(validation["price_data_available"].sum())
    print(probe_frame.to_string(index=False))
    print(f"Macrotrends delisted validation: {successes}/{len(validation)} succeeded.")
    if successes == 0:
        print(
            "STOP: the validation gate failed. The 62-name expansion was not run; "
            "use a delisting-inclusive source such as CRSP or Sharadar."
        )
    else:
        print(
            "At least one validation name succeeded. Review adjustment compatibility "
            "before implementing a separate, explicitly approved expansion stage."
        )
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
