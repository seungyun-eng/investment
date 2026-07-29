from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from stock_research.cross_sectional.dollar_volume_ranking import (
    parse_historical_price_file,
)
from stock_research.io_utils import read_csv_fallback

MACROTRENDS_PRICE_PAGE = (
    "https://www.macrotrends.net/stocks/charts/"
    "{ticker}/{slug}/stock-price-history"
)
MACROTRENDS_PRICE_CHART = (
    "https://www.macrotrends.net/production/stocks/desktop/"
    "PRODUCTION/stock_price_history.php"
)
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)
DATA_DAILY_PATTERN = re.compile(
    r"var\s+dataDaily\s*=\s*(\[.*?\]);",
    re.DOTALL,
)
REQUIRED_DAILY_FIELDS = {"d", "o", "h", "l", "c", "v"}
PAGE_NOT_FOUND_MARKERS = (
    "Error Code: 404",
    "Oops! Page not found",
    "We can't seem to find the page",
)


@dataclass(frozen=True)
class PriceProbeTarget:
    ticker: str
    company_slug: str
    company: str = ""

    def normalized(self) -> PriceProbeTarget:
        ticker = self.ticker.strip().upper()
        slug = self.company_slug.strip().strip("/").lower()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        if not slug:
            raise ValueError(f"Company slug cannot be empty for {ticker}.")
        return PriceProbeTarget(ticker, slug, self.company.strip())


@dataclass(frozen=True)
class HttpProbe:
    status_code: int | None
    final_url: str
    response_text: str
    error: str


@dataclass(frozen=True)
class PriceProbeResult:
    ticker: str
    company: str
    company_slug: str
    price_page_url: str
    price_page_status: int | None
    price_page_final_url: str
    price_page_not_found: bool
    chart_url: str
    chart_status: int | None
    daily_rows: int
    first_date: str
    last_date: str
    has_daily_ohlcv: bool
    price_data_available: bool
    failure_reason: str
    note: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def extract_macrotrends_daily(page_source: str) -> pd.DataFrame:
    """Extract the OHLCV array embedded in Macrotrends' price-chart iframe.

    Macrotrends encodes volume under ``v`` in millions of shares. The unit was
    cross-checked against local Back Test files before using the multiplier.
    """
    match = DATA_DAILY_PATTERN.search(page_source)
    if match is None:
        raise ValueError("Macrotrends dataDaily payload was not found.")
    payload = json.loads(match.group(1))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Macrotrends dataDaily payload was empty.")
    missing = REQUIRED_DAILY_FIELDS - set(payload[0])
    if missing:
        raise ValueError(f"Macrotrends dataDaily fields are missing: {sorted(missing)}")

    raw = pd.DataFrame(payload)
    normalized = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["d"], errors="coerce"),
            "Open": pd.to_numeric(raw["o"], errors="coerce"),
            "High": pd.to_numeric(raw["h"], errors="coerce"),
            "Low": pd.to_numeric(raw["l"], errors="coerce"),
            "Close": pd.to_numeric(raw["c"], errors="coerce"),
            "Volume": pd.to_numeric(raw["v"], errors="coerce") * 1_000_000.0,
        }
    )
    normalized = (
        normalized.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        .loc[lambda frame: frame["Close"].gt(0) & frame["Volume"].ge(0)]
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if normalized.empty:
        raise ValueError("Macrotrends dataDaily payload had no valid OHLCV rows.")
    return normalized


def _request_with_retry(
    url: str,
    *,
    params: dict[str, object] | None = None,
    timeout_seconds: float = 30.0,
    retries: int = 2,
    request_pause_seconds: float = 0.5,
    retry_backoff_seconds: float = 2.0,
    request_get: Callable[..., object] | None = None,
) -> HttpProbe:
    if request_pause_seconds < 0.5:
        raise ValueError("Macrotrends requests must be separated by at least 0.5 seconds.")
    if retries < 0:
        raise ValueError("retries must be non-negative.")
    if request_get is None:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Macrotrends validation requires requests.") from exc
        # Deliberately use the module-level helper. The existing financial
        # crawler found that short-lived Macrotrends sessions rate-limit less.
        request_get = requests.get

    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = request_get(
                url,
                params=params,
                headers={"User-Agent": HTTP_USER_AGENT},
                timeout=timeout_seconds,
            )
            status = int(response.status_code)
            probe = HttpProbe(
                status_code=status,
                final_url=str(response.url),
                response_text=str(response.text),
                error="",
            )
            time.sleep(request_pause_seconds)
            if status < 500 or attempt == retries:
                return probe
            time.sleep(retry_backoff_seconds * (2**attempt))
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(request_pause_seconds)
            if attempt < retries:
                time.sleep(retry_backoff_seconds * (2**attempt))
    return HttpProbe(
        status_code=None,
        final_url="",
        response_text="",
        error=last_error or "Unknown request failure.",
    )


def probe_macrotrends_price_history(
    target: PriceProbeTarget,
    *,
    timeout_seconds: float = 30.0,
    retries: int = 2,
    request_pause_seconds: float = 0.5,
    retry_backoff_seconds: float = 2.0,
    request_get: Callable[..., object] | None = None,
) -> tuple[PriceProbeResult, pd.DataFrame]:
    target = target.normalized()
    page_url = MACROTRENDS_PRICE_PAGE.format(
        ticker=target.ticker,
        slug=target.company_slug,
    )
    page = _request_with_retry(
        page_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        request_pause_seconds=request_pause_seconds,
        retry_backoff_seconds=retry_backoff_seconds,
        request_get=request_get,
    )
    page_not_found = (
        page.status_code == 404
        or any(marker.casefold() in page.response_text.casefold() for marker in PAGE_NOT_FOUND_MARKERS)
    )

    chart = _request_with_retry(
        MACROTRENDS_PRICE_CHART,
        params={"t": target.ticker, "yb": 15},
        timeout_seconds=timeout_seconds,
        retries=retries,
        request_pause_seconds=request_pause_seconds,
        retry_backoff_seconds=retry_backoff_seconds,
        request_get=request_get,
    )
    daily = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    parse_error = ""
    if chart.status_code == 200:
        try:
            daily = extract_macrotrends_daily(chart.response_text)
        except (ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)

    has_daily_ohlcv = not daily.empty
    failures: list[str] = []
    if page.error:
        failures.append(f"PRICE_PAGE_REQUEST_ERROR: {page.error}")
    elif page_not_found:
        failures.append("PRICE_PAGE_404")
    elif page.status_code is not None and page.status_code >= 400:
        failures.append(f"PRICE_PAGE_HTTP_{page.status_code}")
    if chart.error:
        failures.append(f"CHART_REQUEST_ERROR: {chart.error}")
    elif chart.status_code != 200:
        failures.append(f"CHART_HTTP_{chart.status_code}")
    elif parse_error:
        failures.append(f"CHART_PARSE_ERROR: {parse_error}")
    elif not has_daily_ohlcv:
        failures.append("CHART_NO_DAILY_OHLCV")

    first_date = daily["Date"].min().date().isoformat() if has_daily_ohlcv else ""
    last_date = daily["Date"].max().date().isoformat() if has_daily_ohlcv else ""
    result = PriceProbeResult(
        ticker=target.ticker,
        company=target.company,
        company_slug=target.company_slug,
        price_page_url=page_url,
        price_page_status=page.status_code,
        price_page_final_url=page.final_url,
        price_page_not_found=page_not_found,
        chart_url=f"{MACROTRENDS_PRICE_CHART}?t={target.ticker}&yb=15",
        chart_status=chart.status_code,
        daily_rows=len(daily),
        first_date=first_date,
        last_date=last_date,
        has_daily_ohlcv=has_daily_ohlcv,
        price_data_available=has_daily_ohlcv,
        failure_reason="; ".join(failures),
        note=(
            "Macrotrends prices are current-site reconstructions and do not "
            "provide point-in-time provenance."
        ),
    )
    return result, daily


def compare_with_local_price_file(
    macrotrends_daily: pd.DataFrame,
    local_path: str | Path,
    *,
    ticker: str,
    company: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parsed = parse_historical_price_file(
        local_path,
        ticker=ticker,
        company=company,
    ).data
    local = parsed[["Date", "Close", "Volume"]].rename(
        columns={"Close": "LocalClose", "Volume": "LocalVolume"}
    )
    macro = macrotrends_daily[["Date", "Close", "Volume"]].rename(
        columns={"Close": "MacrotrendsClose", "Volume": "MacrotrendsVolume"}
    )
    overlap = local.merge(macro, on="Date", how="inner", validate="one_to_one")
    overlap["CloseDifferencePct"] = (
        overlap["MacrotrendsClose"].div(overlap["LocalClose"]).sub(1.0).mul(100.0)
    )
    overlap["VolumeDifferencePct"] = (
        overlap["MacrotrendsVolume"].div(overlap["LocalVolume"]).sub(1.0).mul(100.0)
    )
    summary = pd.DataFrame(
        [
            {
                "Ticker": ticker.upper(),
                "MatchedRows": len(overlap),
                "FirstMatchedDate": (
                    overlap["Date"].min().date().isoformat() if not overlap.empty else ""
                ),
                "LastMatchedDate": (
                    overlap["Date"].max().date().isoformat() if not overlap.empty else ""
                ),
                "MedianAbsoluteCloseDifferencePct": (
                    overlap["CloseDifferencePct"].abs().median()
                    if not overlap.empty
                    else np.nan
                ),
                "MaxAbsoluteCloseDifferencePct": (
                    overlap["CloseDifferencePct"].abs().max()
                    if not overlap.empty
                    else np.nan
                ),
                "MedianAbsoluteVolumeDifferencePct": (
                    overlap["VolumeDifferencePct"].abs().median()
                    if not overlap.empty
                    else np.nan
                ),
                "MaxAbsoluteVolumeDifferencePct": (
                    overlap["VolumeDifferencePct"].abs().max()
                    if not overlap.empty
                    else np.nan
                ),
            }
        ]
    )
    return overlap, summary


def summarize_failed_fetch_log(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = read_csv_fallback(path, dtype=str).fillna("")
    failed = log.loc[log["Status"].str.upper().eq("FAILED")].copy()
    if failed.empty:
        return failed, pd.DataFrame(columns=["Reason", "Count"])
    detail = failed["Detail"].str.casefold()
    failed["Reason"] = np.select(
        [
            detail.str.contains("404", regex=False),
            detail.str.contains("no price", regex=False),
            detail.str.contains("timeout", regex=False),
        ],
        ["HTTP_404", "NO_PRICE_DATA", "TIMEOUT"],
        default="OTHER",
    )
    summary = (
        failed.groupby("Reason", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values(["Count", "Reason"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return failed, summary
