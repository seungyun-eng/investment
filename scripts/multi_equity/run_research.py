from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.multi_equity import (
    EquitySpec,
    load_equity_specs,
    run_equity_research,
)
from stock_research.multi_equity.research import EquityResearchRun
from stock_research.paths import ProjectPaths, load_paths
from stock_research.tsla_integrated.config import (
    IntegratedSettings,
    load_config,
)
from stock_research.tsla_integrated.data import load_macro_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a fixed multi-equity universe on 2019-2025 and evaluate "
            "frozen parameters on 2026."
        )
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/multi_equity/research.json"),
    )
    parser.add_argument("--macro-predictions", type=Path)
    parser.add_argument("--candidates", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent ticker optimizations to run in parallel.",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        help="Limit the run to one or more configured tickers.",
    )
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


def _run_one(
    spec: EquitySpec,
    paths: ProjectPaths,
    macro: pd.DataFrame,
    settings: IntegratedSettings,
    candidate_count: int,
    timestamp: str,
) -> tuple[str, EquityResearchRun]:
    ticker = spec.ticker
    return (
        ticker,
        run_equity_research(
            spec,
            paths=paths,
            macro=macro,
            settings=settings,
            candidate_count=candidate_count,
            timestamp=timestamp,
        ),
    )


def main() -> None:
    args = parse_args()
    paths = load_paths(args.stock_root)
    _, settings = load_config(args.config)
    specs = load_equity_specs(args.config)
    if args.ticker:
        requested = {ticker.upper() for ticker in args.ticker}
        configured = {spec.ticker for spec in specs}
        unknown = requested - configured
        if unknown:
            raise ValueError(f"Unconfigured tickers: {sorted(unknown)}")
        specs = [spec for spec in specs if spec.ticker in requested]
    candidate_count = args.candidates or settings.optimization_candidates
    macro_path = args.macro_predictions or _latest(
        paths.results / "SP500" / "macro_momentum_sp500",
        "oos_predictions_*.csv",
    )
    macro = load_macro_predictions(macro_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    summaries_by_ticker: dict[str, dict[str, object]] = {}
    manifests: dict[str, object] = {}
    errors: dict[str, str] = {}
    worker_count = min(args.workers, len(specs))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for index, spec in enumerate(specs, start=1):
            print(
                f"[{index}/{len(specs)}] {spec.ticker}: queued "
                f"{candidate_count} candidates",
                flush=True,
            )
            future = executor.submit(
                _run_one,
                spec,
                paths,
                macro,
                settings,
                candidate_count,
                timestamp,
            )
            futures[future] = spec
        completed = 0
        for future in as_completed(futures):
            spec = futures[future]
            completed += 1
            try:
                _, run = future.result()
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                errors[spec.ticker] = f"{type(exc).__name__}: {exc}"
                summaries_by_ticker[spec.ticker] = (
                {
                    "Ticker": spec.ticker,
                    "Status": "FAILED",
                    "Error": errors[spec.ticker],
                }
                )
                print(
                    f"[{completed}/{len(specs)}] {spec.ticker}: FAILED {exc}",
                    flush=True,
                )
                continue
            summaries_by_ticker[spec.ticker] = run.summary
            manifests[spec.ticker] = run.manifest
            print(
                f"[{completed}/{len(specs)}] {spec.ticker}: "
                f"development ROI={run.summary['DevelopmentROI(%)']:.2f}%, "
                f"2026 ROI={run.summary['HoldoutROI(%)']:.2f}%",
                flush=True,
            )

    output = paths.results / "Multi_Equity" / "integrated_signal"
    summaries = [
        summaries_by_ticker[spec.ticker]
        for spec in specs
    ]
    summary_path = atomic_to_csv(
        pd.DataFrame(summaries),
        output / f"roi_summary_{timestamp}.csv",
        index=False,
    )
    manifest_payload = {
        "universe": [spec.ticker for spec in specs],
        "candidate_count_per_ticker": candidate_count,
        "macro_predictions": str(macro_path.resolve()),
        "research": settings.__dict__,
        "tickers": manifests,
        "errors": errors,
        "summary": str(summary_path),
    }
    manifest_path = _atomic_json(
        manifest_payload,
        output / f"manifest_{timestamp}.json",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "manifest": str(manifest_path),
                "errors": errors,
            },
            indent=2,
        ),
        flush=True,
    )
    if errors:
        raise RuntimeError(f"Some ticker runs failed: {errors}")


if __name__ == "__main__":
    main()
