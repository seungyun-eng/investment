from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.io_utils import read_csv_fallback

REQUIRED_PRICE_COLUMNS = {"Date", "Price", "Vol."}
MISSING_TEXT = {"", "-", "--", "N/A", "NA", "NAN", "NONE", "NULL"}
VOLUME_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?)\s*([KMB]?)\s*$",
    re.IGNORECASE,
)
VOLUME_MULTIPLIERS = {
    "": 1.0,
    "K": 1_000.0,
    "M": 1_000_000.0,
    "B": 1_000_000_000.0,
}


@dataclass(frozen=True)
class ParsedPriceFile:
    data: pd.DataFrame
    audit: pd.DataFrame


def scan_backtest_folder(backtest_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(backtest_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Back Test folder does not exist: {root}")

    inventory_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    for company_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        all_files = sorted(path for path in company_dir.rglob("*") if path.is_file())
        csv_files = [path for path in all_files if path.suffix.casefold() == ".csv"]
        if not all_files:
            issue_rows.append(
                {
                    "Company": company_dir.name,
                    "FilePath": "",
                    "Issue": "EMPTY_COMPANY_FOLDER",
                    "Detail": "No file was found under the company folder.",
                }
            )
            continue
        if not csv_files:
            issue_rows.append(
                {
                    "Company": company_dir.name,
                    "FilePath": "",
                    "Issue": "NO_CSV_FILE",
                    "Detail": f"Found {len(all_files)} non-CSV file(s).",
                }
            )
            continue
        if len(csv_files) > 1:
            issue_rows.append(
                {
                    "Company": company_dir.name,
                    "FilePath": "",
                    "Issue": "MULTIPLE_CSV_FILES",
                    "Detail": f"Found {len(csv_files)} CSV files.",
                }
            )
        for path in csv_files:
            inventory_rows.append(
                {
                    "Company": company_dir.name,
                    "FileName": path.name,
                    "FilePath": str(path),
                    "FileSizeBytes": path.stat().st_size,
                    "CsvFilesInCompanyFolder": len(csv_files),
                }
            )

    inventory = pd.DataFrame(
        inventory_rows,
        columns=[
            "Company",
            "FileName",
            "FilePath",
            "FileSizeBytes",
            "CsvFilesInCompanyFolder",
        ],
    )
    issues = pd.DataFrame(
        issue_rows,
        columns=["Company", "FilePath", "Issue", "Detail"],
    )
    return inventory, issues


def parse_abbreviated_volume(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text in MISSING_TEXT:
        return np.nan
    match = VOLUME_PATTERN.fullmatch(text)
    if match is None or not match.group(1):
        return np.nan
    number = match.group(1).replace(",", "")
    try:
        return float(number) * VOLUME_MULTIPLIERS[match.group(2).upper()]
    except (KeyError, ValueError):
        return np.nan


def parse_close(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "").replace("$", "")
    if text.upper() in MISSING_TEXT:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_historical_price_file(
    path: str | Path,
    *,
    ticker: str,
    company: str,
) -> ParsedPriceFile:
    source = Path(path).expanduser().resolve()
    raw = read_csv_fallback(source, dtype=str)
    raw.columns = [str(column).strip() for column in raw.columns]
    missing = sorted(REQUIRED_PRICE_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing}: {source}")

    normalized = pd.DataFrame(index=raw.index)
    normalized["Date"] = pd.to_datetime(
        raw["Date"].str.strip(),
        format="%m/%d/%Y",
        errors="coerce",
    )
    normalized["Ticker"] = ticker.strip().upper()
    normalized["Company"] = company.strip()
    normalized["Close"] = raw["Price"].map(parse_close)
    normalized["Volume"] = raw["Vol."].map(parse_abbreviated_volume)
    normalized["DollarVolume"] = normalized["Close"] * normalized["Volume"]

    invalid_date = normalized["Date"].isna()
    invalid_close = normalized["Close"].isna() | normalized["Close"].le(0)
    invalid_volume = normalized["Volume"].isna() | normalized["Volume"].lt(0)
    valid = ~(invalid_date | invalid_close | invalid_volume)
    duplicate_valid_dates = int(normalized.loc[valid, "Date"].duplicated().sum())
    result = (
        normalized.loc[valid]
        .drop_duplicates("Date", keep="first")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    audit = pd.DataFrame(
        [
            {
                "Ticker": ticker.strip().upper(),
                "Company": company.strip(),
                "FilePath": str(source),
                "InputRows": len(raw),
                "ValidRows": len(result),
                "InvalidDateRows": int(invalid_date.sum()),
                "InvalidCloseRows": int(invalid_close.sum()),
                "InvalidVolumeRows": int(invalid_volume.sum()),
                "DuplicateValidDatesDropped": duplicate_valid_dates,
                "FirstDate": result["Date"].min() if not result.empty else pd.NaT,
                "LastDate": result["Date"].max() if not result.empty else pd.NaT,
            }
        ]
    )
    return ParsedPriceFile(data=result, audit=audit)


def select_company_file(inventory: pd.DataFrame, company: str) -> Path:
    matches = inventory.loc[
        inventory["Company"].str.casefold().eq(company.casefold())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one CSV for {company!r}, found {len(matches)}"
        )
    return Path(matches.iloc[0]["FilePath"])
