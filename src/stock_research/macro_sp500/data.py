from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_research.indicators import parse_number
from stock_research.io_utils import read_csv_fallback


def find_sp500_proxy_file(macro_folder: Path) -> Path:
    patterns = ("*S&P500 Historical Data.csv", "*SPY*Historical Data.csv")
    hits: list[Path] = []
    for pattern in patterns:
        hits.extend(macro_folder.glob(pattern))
    unique = sorted(set(hits), key=lambda path: path.stat().st_mtime, reverse=True)
    if not unique:
        raise FileNotFoundError(f"No S&P 500 proxy CSV found in {macro_folder}")
    return unique[0]


def find_vix_file(macro_folder: Path) -> Path:
    hits = sorted(
        macro_folder.glob("*VIX*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError(f"No VIX CSV found in {macro_folder}")
    return hits[0]


def find_cash_rate_file(macro_folder: Path) -> Path:
    hits = sorted(
        macro_folder.glob("*FedFundsRate.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError(f"No Fed Funds rate CSV found in {macro_folder}")
    return hits[0]


def load_sp500_proxy(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = read_csv_fallback(path)
    if "Date" not in raw:
        raise ValueError(f"S&P proxy file has no Date column: {path}")
    close_column = next(
        (name for name in ("Adj Close", "Price", "Close") if name in raw),
        None,
    )
    if close_column is None:
        raise ValueError(f"S&P proxy file has no supported close column: {path}")
    open_column = (
        "Adj Open"
        if close_column == "Adj Close" and "Adj Open" in raw
        else "Open" if "Open" in raw else close_column
    )
    volume_column = next(
        (name for name in ("Volume", "Vol.") if name in raw),
        None,
    )
    if volume_column is None:
        raise ValueError(f"S&P proxy file has no Volume/Vol. column: {path}")

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["Date"], errors="coerce"),
            "Close": raw[close_column].map(parse_number),
            "Open": raw[open_column].map(parse_number),
            "Volume": raw[volume_column].map(parse_number),
        }
    )
    frame = (
        frame.dropna(subset=["Date", "Close", "Open", "Volume"])
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError(f"S&P proxy file contains no valid rows: {path}")
    if (frame[["Close", "Open", "Volume"]] <= 0).any().any():
        raise ValueError("S&P proxy prices and volume must be positive.")
    frame.attrs["source_path"] = str(path)
    frame.attrs["close_column"] = close_column
    frame.attrs["dividend_adjusted"] = close_column == "Adj Close"
    return frame


def load_vix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = read_csv_fallback(path)
    if len(raw.columns) < 2:
        raise ValueError(f"VIX file requires at least two columns: {path}")
    date_column = "Date" if "Date" in raw else raw.columns[0]
    value_column = "Value" if "Value" in raw else raw.columns[1]
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_column], errors="coerce"),
            "VIX": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    frame = (
        frame.dropna(subset=["Date", "VIX"])
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if frame.empty or (frame["VIX"] <= 0).any():
        raise ValueError("VIX data must contain positive observations.")
    frame.attrs["source_path"] = str(path)
    return frame


def load_cash_rate(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = read_csv_fallback(path)
    if "Date" not in raw or "Value" not in raw:
        raise ValueError(f"Cash-rate file requires Date and Value columns: {path}")
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["Date"], errors="coerce"),
            "CashRate": pd.to_numeric(raw["Value"], errors="coerce"),
        }
    )
    frame = (
        frame.dropna(subset=["Date", "CashRate"])
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if frame.empty or (frame["CashRate"] < 0).any():
        raise ValueError("Cash rates must contain nonnegative annual percentages.")
    frame.attrs["source_path"] = str(path)
    return frame


def load_macro_sp500_data(
    macro_folder: Path,
    *,
    price_file: str | Path | None = None,
    vix_file: str | Path | None = None,
    cash_rate_file: str | Path | None = None,
    require_cash_rate: bool = False,
) -> pd.DataFrame:
    price_path = Path(price_file) if price_file else find_sp500_proxy_file(macro_folder)
    vix_path = Path(vix_file) if vix_file else find_vix_file(macro_folder)
    price = load_sp500_proxy(price_path)
    vix = load_vix(vix_path)
    merged = price.merge(vix, on="Date", how="inner", validate="one_to_one")
    merged = merged.sort_values("Date").reset_index(drop=True)
    cash_path: Path | None
    try:
        cash_path = (
            Path(cash_rate_file)
            if cash_rate_file
            else find_cash_rate_file(macro_folder)
        )
    except FileNotFoundError:
        if require_cash_rate:
            raise
        cash_path = None
    if cash_path is not None:
        cash_rate = load_cash_rate(cash_path)
        merged = pd.merge_asof(
            merged,
            cash_rate,
            on="Date",
            direction="backward",
        )
    else:
        merged["CashRate"] = float("nan")
    if merged.empty:
        raise ValueError("S&P proxy and VIX have no overlapping dates.")
    merged.attrs.update(
        {
            "price_source": str(price_path),
            "vix_source": str(vix_path),
            "cash_rate_source": str(cash_path) if cash_path else "",
            "close_column": price.attrs["close_column"],
            "dividend_adjusted": price.attrs["dividend_adjusted"],
            "price_rows": len(price),
            "vix_rows": len(vix),
            "merged_rows": len(merged),
        }
    )
    return merged
