from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_research.io_utils import atomic_to_csv, read_csv_fallback
from stock_research.market_data import update_one_price
from stock_research.paths import load_paths
from stock_research.tickers import load_tickers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the frozen current-survivor universe price histories "
            "from a requested start date without invoking financial crawlers."
        )
    )
    parser.add_argument("--stock-root")
    parser.add_argument("--ticker-config")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    ticker_config = (
        Path(args.ticker_config).expanduser().resolve()
        if args.ticker_config
        else _latest_automatic_ticker_config(paths.results)
    )
    tickers = load_tickers(ticker_config)
    requested = (
        {ticker.strip().upper() for ticker in args.ticker}
        if args.ticker
        else set(tickers)
    )
    missing = sorted(requested - set(tickers))
    if missing:
        raise ValueError(
            "Requested tickers are absent from the survivor config: "
            + ", ".join(missing)
        )
    selected = [
        ticker for ticker in tickers if ticker in requested
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    start = datetime.fromisoformat(args.start)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else paths.results
        / "Cross_Sectional"
        / "price_backfill_runs"
        / _run_name()
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "price_backfill_status.csv"
    manifest_path = output_dir / "manifest.json"

    rows: list[dict[str, object]] = []
    for index, ticker in enumerate(selected, start=1):
        config = tickers[ticker]
        print(f"[{index}/{len(selected)}] {ticker}: {config.display_name}")
        before = _price_file_status(paths.raw_prices, config.display_name)
        error = ""
        output: Path | None = None
        for attempt in range(1, args.retries + 1):
            try:
                output = update_one_price(
                    config,
                    paths.raw_prices,
                    initial_start=start,
                    refresh_start=start,
                    transport="yahoo_chart",
                    request_timeout_seconds=args.timeout_seconds,
                )
                if output is None:
                    raise ValueError("Yahoo returned no usable price rows")
                break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 10))
        after = _price_file_status(paths.raw_prices, config.display_name)
        success = output is not None and after["Valid"]
        rows.append(
            {
                "Ticker": ticker,
                "Company": config.display_name,
                "Success": success,
                "BeforeFile": before["File"],
                "BeforeRows": before["Rows"],
                "BeforeStart": before["Start"],
                "BeforeEnd": before["End"],
                "AfterFile": after["File"],
                "AfterRows": after["Rows"],
                "AfterStart": after["Start"],
                "AfterEnd": after["End"],
                "RequestedStart": start.date(),
                "Has2019Observation": bool(
                    success
                    and pd.notna(after["Start"])
                    and pd.Timestamp(after["Start"]).year <= 2019
                ),
                "ValidVolumeRows": after["ValidVolumeRows"],
                "Error": "" if success else error,
            }
        )
        if index % 10 == 0 or index == len(selected):
            atomic_to_csv(pd.DataFrame(rows), status_path, index=False)
        if index < len(selected):
            time.sleep(args.pause_seconds)

    status = pd.DataFrame(rows)
    atomic_to_csv(status, status_path, index=False)
    _atomic_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Yahoo Finance chart API via existing market_data.update_one_price",
            "ticker_config": str(ticker_config),
            "requested_start": args.start,
            "universe_definition": (
                "frozen current-survivor ticker config; not a historical "
                "point-in-time universe"
            ),
            "requested_tickers": len(selected),
            "successes": int(status["Success"].sum()),
            "failures": int((~status["Success"]).sum()),
            "has_2019_observation": int(status["Has2019Observation"].sum()),
            "later_listings_or_incomplete_history": int(
                (~status["Has2019Observation"]).sum()
            ),
            "status": str(status_path),
        },
        manifest_path,
    )
    print(f"Output: {output_dir}")
    print(
        f"Success: {int(status['Success'].sum())}/{len(status)}; "
        f"failure: {int((~status['Success']).sum())}"
    )
    print(
        "2019-or-earlier observation: "
        f"{int(status['Has2019Observation'].sum())}/{len(status)}"
    )
    failures = status.loc[
        ~status["Success"],
        ["Ticker", "Company", "Error"],
    ]
    if not failures.empty:
        print(failures.to_string(index=False))


def _latest_automatic_ticker_config(results_root: Path) -> Path:
    candidates = list(
        (results_root / "Cross_Sectional" / "backfill_runs").glob(
            "*/automatic_tickers.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "No automatic_tickers.json was found; pass --ticker-config"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _price_file_status(raw_root: Path, display_name: str) -> dict[str, object]:
    company_dir = raw_root / display_name
    candidates = sorted(
        company_dir.glob(f"*{display_name} Historical Data.csv")
    )
    if not candidates:
        return {
            "Valid": False,
            "File": "",
            "Rows": 0,
            "Start": pd.NaT,
            "End": pd.NaT,
            "ValidVolumeRows": 0,
        }
    path = candidates[-1]
    try:
        frame = read_csv_fallback(path, dtype=str)
        dates = pd.to_datetime(frame.get("Date"), errors="coerce")
        volumes = frame.get("Vol.", pd.Series(index=frame.index, dtype=object))
        valid_volume = volumes.fillna("").astype(str).str.strip().ne("")
        valid = (
            {"Date", "Price", "Vol."}.issubset(frame.columns)
            and dates.notna().any()
        )
        return {
            "Valid": bool(valid),
            "File": str(path),
            "Rows": len(frame),
            "Start": dates.min(),
            "End": dates.max(),
            "ValidVolumeRows": int(valid_volume.sum()),
        }
    except Exception:  # noqa: BLE001
        return {
            "Valid": False,
            "File": str(path),
            "Rows": 0,
            "Start": pd.NaT,
            "End": pd.NaT,
            "ValidVolumeRows": 0,
        }


def _run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}_survivor_prices_from_2019"


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
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
