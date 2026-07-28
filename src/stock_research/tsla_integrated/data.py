from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_research.financial_analysis import load_financial_workbook
from stock_research.io_utils import read_csv_fallback


def load_equity_prices(path: Path) -> pd.DataFrame:
    """Load a processed equity CSV into stable English columns."""

    raw = read_csv_fallback(path)
    if len(raw.columns) < 5:
        raise ValueError(f"Equity price file has too few columns: {path}")
    positional = {
        raw.columns[0]: "Date",
        raw.columns[1]: "Close",
        raw.columns[2]: "Open",
        raw.columns[3]: "High",
        raw.columns[4]: "Low",
    }
    if len(raw.columns) > 5:
        positional[raw.columns[5]] = "Volume"
    frame = raw.rename(columns=positional)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["Date", "Open", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def load_equity_financials(path: Path) -> pd.DataFrame:
    return load_financial_workbook(path)


def load_tsla_prices(path: Path) -> pd.DataFrame:
    """Backward-compatible alias for the original TSLA workflow."""

    return load_equity_prices(path)


def load_tsla_financials(path: Path) -> pd.DataFrame:
    """Backward-compatible alias for the original TSLA workflow."""

    return load_equity_financials(path)


def load_macro_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return (
        frame.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
