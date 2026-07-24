from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import read_csv_fallback
from .macro_data import load_vix


def list_processed_companies(folder: Path) -> list[str]:
    suffix = "_지표포함.csv"
    return sorted(path.name[:-len(suffix)] for path in folder.glob(f"*{suffix}"))


def find_processed_file(folder: Path, company: str) -> Path:
    exact = folder / f"{company}_지표포함.csv"
    if exact.exists():
        return exact
    hits = sorted(
        folder.glob(f"{company}*_지표포함*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError(f"No processed CSV for {company} in {folder}")
    return hits[0]


def load_processed(
    folder: Path,
    company: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    frame = read_csv_fallback(find_processed_file(folder, company))
    frame["날짜"] = pd.to_datetime(frame["날짜"], errors="coerce")
    if start:
        frame = frame[frame["날짜"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["날짜"] <= pd.Timestamp(end)]
    return frame.sort_values("날짜").reset_index(drop=True)


def validate_actual_vix(frame: pd.DataFrame) -> pd.DataFrame:
    missing_count = int(frame["VIX"].isna().sum())
    valid = frame["VIX"].dropna()
    minimum = float(valid.min()) if not valid.empty else float("nan")
    maximum = float(valid.max()) if not valid.empty else float("nan")
    print(
        "Actual VIX validation: "
        f"min={minimum:.4g}, max={maximum:.4g}, missing={missing_count}"
    )
    if missing_count:
        dates = frame.loc[frame["VIX"].isna(), "날짜"].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "Missing actual daily VIX values after strict date merge: "
            + ", ".join(dates.tolist()[:10])
        )
    invalid = frame["VIX"] <= 0
    if invalid.any():
        dates = frame.loc[invalid, "날짜"].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "Actual daily VIX values must be greater than zero: "
            + ", ".join(dates.tolist()[:10])
        )
    frame.attrs["actual_vix_stats"] = {
        "minimum": minimum, "maximum": maximum, "missing_count": missing_count,
    }
    return frame


def load_processed_with_vix(
    processed_folder: Path,
    macro_folder: Path,
    company: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    stock = load_processed(processed_folder, company, start, end)
    vix = load_vix(macro_folder)
    merged = stock.merge(vix, on="날짜", how="left", validate="one_to_one")
    return validate_actual_vix(merged)
