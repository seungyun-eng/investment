from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import load_paths


def _prepare_ascii_certificate() -> None:
    import certifi

    target = Path(tempfile.gettempdir()) / "codex_spy_cacert.pem"
    shutil.copyfile(certifi.where(), target)
    os.environ["CURL_CA_BUNDLE"] = str(target)
    os.environ["REQUESTS_CA_BUNDLE"] = str(target)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end")
    args = parser.parse_args()

    _prepare_ascii_certificate()
    import yfinance as yf

    end = args.end or (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    raw = yf.download(
        "SPY",
        start=args.start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no SPY observations.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    factor = raw["Adj Close"] / raw["Close"]
    adjusted = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["Date"], errors="coerce"),
            "Adj Open": pd.to_numeric(raw["Open"], errors="coerce") * factor,
            "Adj Close": pd.to_numeric(raw["Adj Close"], errors="coerce"),
            "Volume": pd.to_numeric(raw["Volume"], errors="coerce"),
        }
    ).dropna()
    paths = load_paths()
    output = atomic_to_csv(
        adjusted,
        paths.macro / "SPY Adjusted Historical Data.csv",
        index=False,
    )
    print(f"Rows={len(adjusted)}")
    print(f"Start={adjusted['Date'].min().date()}")
    print(f"End={adjusted['Date'].max().date()}")
    print(f"Output={output}")


if __name__ == "__main__":
    main()
