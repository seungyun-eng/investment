from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas_datareader import data as pdr

from .io_utils import atomic_to_csv


MACRO_SERIES = {
    "CPI": "CPIAUCSL",
    "WTI": "MCOILWTICO",
    "VIX": "VIXCLS",
    "FedFundsRate": "FEDFUNDS",
    "Unemployment": "UNRATE",
    "HY_Spread": "BAMLH0A3HYCEY",
}
YIELD_CURVE_SERIES = ("GS10", "GS2")


def _existing_file(folder: Path, name: str) -> Path | None:
    hits = sorted(folder.glob(f"* {name}.csv"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _fetch_series(code: str, start: datetime, end: datetime) -> pd.DataFrame:
    frame = pdr.DataReader(code, "fred", start, end).reset_index()
    frame.columns = ["Date", "Value"]
    return frame


def update_macro_data(folder: Path, start_default: str = "1970-01-01") -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    series_to_run = dict(MACRO_SERIES)
    series_to_run["YieldCurve"] = YIELD_CURVE_SERIES

    for name, code in series_to_run.items():
        old_path = _existing_file(folder, name)
        if old_path:
            old = pd.read_csv(old_path, parse_dates=["Date"])
            start = old["Date"].max().to_pydatetime() + timedelta(days=1)
        else:
            old = pd.DataFrame(columns=["Date", "Value"])
            start = datetime.fromisoformat(start_default)
        end = datetime.today() + timedelta(days=1)

        try:
            if isinstance(code, tuple):
                left = pdr.DataReader(code[0], "fred", start, end)
                right = pdr.DataReader(code[1], "fred", start, end)
                new = (left[code[0]] - right[code[1]]).rename("Value").reset_index()
            else:
                new = _fetch_series(code, start, end)
        except Exception as exc:
            print(f"WARN {name}: fetch failed: {exc}")
            new = pd.DataFrame(columns=["Date", "Value"])

        combined = pd.concat([old, new], ignore_index=True)
        if combined.empty:
            continue
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined["Value"] = pd.to_numeric(combined["Value"], errors="coerce")
        combined = (
            combined.dropna(subset=["Date"])
            .drop_duplicates("Date", keep="last")
            .sort_values("Date")
            .reset_index(drop=True)
        )
        if name == "CPI":
            combined["YoY_%"] = combined["Value"].pct_change(12) * 100

        first = combined["Date"].min().strftime("%Y.%m.%d")
        last = combined["Date"].max().strftime("%Y.%m.%d")
        output = folder / f"{first}_{last} {name}.csv"
        atomic_to_csv(combined.sort_values("Date", ascending=False), output, index=False)
        if old_path and old_path != output:
            old_path.unlink(missing_ok=True)
        outputs.append(output)
    return outputs


def load_vix(folder: Path) -> pd.DataFrame:
    hits = sorted(folder.glob("*VIX*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise FileNotFoundError(f"No VIX CSV found in {folder}")
    frame = pd.read_csv(hits[0])
    date_col = frame.columns[0]
    value_col = "Value" if "Value" in frame.columns else frame.columns[1]
    frame = frame[[date_col, value_col]].rename(columns={date_col: "날짜", value_col: "VIX"})
    frame["날짜"] = pd.to_datetime(frame["날짜"], errors="coerce")
    frame["VIX"] = pd.to_numeric(frame["VIX"], errors="coerce")
    return frame.dropna(subset=["날짜"]).sort_values("날짜")
