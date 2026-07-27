from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import load_paths

FRED_SERIES = {
    "VIX3M": "VXVCLS",
    "T10Y2Y": "T10Y2Y",
    "T10Y3M": "T10Y3M",
    "NFCI": "NFCI",
    "GS10": "GS10",
    "TB3MS": "TB3MS",
    "BAA10Y": "BAA10Y",
    "HYOAS": "BAMLH0A0HYM2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update supplemental FRED series for macro-momentum SPY research."
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def fetch_research_series(start: str, end: str | None = None) -> pd.DataFrame:
    start_date = datetime.fromisoformat(start)
    end_date = (
        datetime.fromisoformat(end)
        if end
        else datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)
    )
    columns: list[pd.DataFrame] = []
    failures: list[str] = []
    for name, fred_code in FRED_SERIES.items():
        try:
            series = pdr.DataReader(fred_code, "fred", start_date, end_date)
            columns.append(series.rename(columns={fred_code: name}))
        except (OSError, ValueError, RemoteDataError) as exc:
            failures.append(f"{name}/{fred_code}: {exc}")
    if not columns:
        raise RuntimeError("All supplemental FRED downloads failed: " + "; ".join(failures))
    combined = pd.concat(columns, axis=1).sort_index().reset_index()
    combined = combined.rename(columns={combined.columns[0]: "Date"})
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    for name in FRED_SERIES:
        if name in combined:
            combined[name] = pd.to_numeric(combined[name], errors="coerce")
    combined = combined.dropna(subset=["Date"]).drop_duplicates("Date", keep="last")
    combined.attrs["failures"] = failures
    return combined


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    frame = fetch_research_series(args.start, args.end)
    first = frame["Date"].min().strftime("%Y.%m.%d")
    last = frame["Date"].max().strftime("%Y.%m.%d")
    output = paths.macro / f"{first}_{last} FRED Research Macro.csv"
    atomic_to_csv(frame, output, index=False)
    print(f"Saved={output}")
    print(f"Rows={len(frame)} Columns={','.join(frame.columns)}")
    for failure in frame.attrs.get("failures", []):
        print(f"WARN fetch failure: {failure}")
    if "HYOAS" in frame:
        first_oas = frame.loc[frame["HYOAS"].notna(), "Date"].min()
        print(f"HYOAS first available={first_oas:%Y-%m-%d}")


if __name__ == "__main__":
    main()
