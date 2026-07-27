from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_research.io_utils import read_csv_fallback
from stock_research.macro_sp500.data import (
    find_cash_rate_file,
    find_sp500_proxy_file,
    find_vix_file,
    load_sp500_proxy,
    load_vix,
)

from .config import ResearchConfig


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    pattern: str
    lag_days: int
    frequency: str
    required: bool = False


def _find_latest(folder: Path, pattern: str) -> Path | None:
    hits = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def load_value_series(
    path: str | Path,
    name: str,
    *,
    value_column: str = "Value",
) -> pd.DataFrame:
    raw = read_csv_fallback(path)
    if "Date" not in raw or value_column not in raw:
        raise ValueError(f"{path} requires Date and {value_column} columns.")
    frame = pd.DataFrame(
        {
            "ObservationDate": pd.to_datetime(raw["Date"], errors="coerce"),
            name: pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    return (
        frame.dropna(subset=["ObservationDate", name])
        .drop_duplicates("ObservationDate", keep="last")
        .sort_values("ObservationDate")
        .reset_index(drop=True)
    )


def merge_point_in_time(
    calendar: pd.DataFrame,
    series: pd.DataFrame,
    *,
    value_column: str,
    lag_days: int,
    tolerance_days: int | None = None,
) -> pd.DataFrame:
    """As-of merge a series using its conservative availability date.

    ``ObservationDate`` is retained beside the value so audits can prove which
    observation was available for every trading date.
    """

    left = calendar.sort_values("Date").copy()
    right = series[["ObservationDate", value_column]].copy()
    right["AvailableDate"] = right["ObservationDate"] + pd.to_timedelta(lag_days, unit="D")
    right = right.sort_values("AvailableDate")
    kwargs: dict[str, object] = {}
    if tolerance_days is not None:
        kwargs["tolerance"] = pd.Timedelta(days=tolerance_days)
    merged = pd.merge_asof(
        left,
        right,
        left_on="Date",
        right_on="AvailableDate",
        direction="backward",
        **kwargs,
    )
    return merged.drop(columns=["AvailableDate"]).rename(
        columns={"ObservationDate": f"{value_column}ObservationDate"}
    )


def _load_wide_fred(path: Path) -> dict[str, pd.DataFrame]:
    raw = read_csv_fallback(path)
    if "Date" not in raw:
        raise ValueError(f"{path} requires a Date column.")
    result: dict[str, pd.DataFrame] = {}
    dates = pd.to_datetime(raw["Date"], errors="coerce")
    for name in raw.columns:
        if name == "Date":
            continue
        frame = pd.DataFrame(
            {
                "ObservationDate": dates,
                name: pd.to_numeric(raw[name], errors="coerce"),
            }
        ).dropna()
        if not frame.empty:
            result[name] = frame.sort_values("ObservationDate").reset_index(drop=True)
    return result


def _fred_timing(name: str, config: ResearchConfig) -> tuple[int, int]:
    if name in {"GS10", "TB3MS"}:
        return config.monthly_release_lag_days, 120
    if name == "NFCI":
        return config.weekly_release_lag_days, 14
    return 0, 10


def _monthly_specs(config: ResearchConfig) -> tuple[SeriesSpec, ...]:
    lag = config.monthly_release_lag_days
    return (
        SeriesSpec("CPI", "* CPI.csv", lag, "monthly"),
        SeriesSpec("FedFundsKnown", "* FedFundsRate.csv", lag, "monthly"),
        SeriesSpec("Unemployment", "* Unemployment.csv", lag, "monthly"),
        SeriesSpec("WTI", "* WTI.csv", lag, "monthly"),
        SeriesSpec("YieldCurveLegacy", "* YieldCurve.csv", lag, "monthly"),
    )


def load_research_data(
    macro_folder: str | Path,
    config: ResearchConfig,
    *,
    price_file: str | Path | None = None,
    vix_file: str | Path | None = None,
    cash_rate_file: str | Path | None = None,
) -> pd.DataFrame:
    """Load adjusted SPY and macro observations with point-in-time availability."""

    macro_folder = Path(macro_folder)
    price_path = Path(price_file) if price_file else find_sp500_proxy_file(macro_folder)
    vix_path = Path(vix_file) if vix_file else find_vix_file(macro_folder)
    cash_path = Path(cash_rate_file) if cash_rate_file else find_cash_rate_file(macro_folder)

    price = load_sp500_proxy(price_path)
    if not price.attrs["dividend_adjusted"]:
        raise ValueError(
            "Macro-momentum research requires dividend-adjusted SPY data (Adj Close)."
        )
    result = price.sort_values("Date").reset_index(drop=True)

    vix = load_vix(vix_path).rename(columns={"Date": "ObservationDate"})
    result = merge_point_in_time(
        result,
        vix,
        value_column="VIX",
        lag_days=0,
        tolerance_days=7,
    )

    cash = load_value_series(cash_path, "CashRate")
    # CashRate is used only to calculate the return actually earned by cash. The
    # predictive FedFundsKnown feature below receives the conservative release lag.
    result = merge_point_in_time(
        result,
        cash,
        value_column="CashRate",
        lag_days=0,
        tolerance_days=75,
    )

    sources: dict[str, str] = {
        "price": str(price_path),
        "vix": str(vix_path),
        "cash": str(cash_path),
    }
    for spec in _monthly_specs(config):
        path = _find_latest(macro_folder, spec.pattern)
        if path is None:
            if spec.required:
                raise FileNotFoundError(f"Missing required macro series: {spec.pattern}")
            continue
        series = load_value_series(path, spec.name)
        result = merge_point_in_time(
            result,
            series,
            value_column=spec.name,
            lag_days=spec.lag_days,
            tolerance_days=120,
        )
        sources[spec.name] = str(path)

    # The legacy file is FRED BAMLH0A3HYCEY: a high-yield effective yield,
    # despite its historical filename. Keep the raw file untouched and expose a
    # truthful feature name.
    hy_path = _find_latest(macro_folder, "* HY_Spread.csv")
    if hy_path is not None:
        hy_yield = load_value_series(hy_path, "HYYield")
        result = merge_point_in_time(
            result,
            hy_yield,
            value_column="HYYield",
            lag_days=0,
            tolerance_days=7,
        )
        sources["HYYield"] = f"{hy_path} (FRED BAMLH0A3HYCEY; not an OAS spread)"

    fred_path = _find_latest(macro_folder, "*FRED Research Macro.csv")
    if fred_path is not None:
        for name, series in _load_wide_fred(fred_path).items():
            lag, tolerance = _fred_timing(name, config)
            result = merge_point_in_time(
                result,
                series,
                value_column=name,
                lag_days=lag,
                tolerance_days=tolerance,
            )
        sources["fred_research"] = str(fred_path)

    result = result.sort_values("Date").reset_index(drop=True)
    result.attrs.update(
        {
            "sources": sources,
            "dividend_adjusted": True,
            "close_column": price.attrs["close_column"],
            "monthly_release_lag_days": config.monthly_release_lag_days,
            "hy_semantics": "BAMLH0A3HYCEY effective yield; not OAS",
        }
    )
    return result
