from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

from .io_utils import atomic_to_csv, read_csv_fallback
from .tickers import TickerConfig


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [col[0] for col in frame.columns]
    return frame


def _download_yfinance(
    ticker: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    data = _flatten_yfinance_columns(data)
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    date_col = "Date" if "Date" in data else data.columns[0]
    close_col = "Close" if "Close" in data else "Adj Close"
    out = pd.DataFrame({
        "Date": pd.to_datetime(data[date_col]),
        "Price": pd.to_numeric(data[close_col], errors="coerce"),
        "Open": pd.to_numeric(data.get("Open"), errors="coerce"),
        "High": pd.to_numeric(data.get("High"), errors="coerce"),
        "Low": pd.to_numeric(data.get("Low"), errors="coerce"),
        "Volume": pd.to_numeric(data.get("Volume"), errors="coerce"),
    })
    return out.dropna(subset=["Date", "Price"])


def _download_yahoo_chart(
    ticker: str,
    start: datetime,
    end: datetime,
    *,
    timeout_seconds: float = 30.0,
    session=None,
) -> pd.DataFrame:
    client = session or requests
    start_utc = start.replace(tzinfo=UTC) if start.tzinfo is None else start
    end_utc = end.replace(tzinfo=UTC) if end.tzinfo is None else end
    response = client.get(
        (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            + quote(ticker, safe="-")
        ),
        params={
            "period1": int(start_utc.timestamp()),
            "period2": int(end_utc.timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error"):
        raise ValueError(
            f"Yahoo chart error for {ticker}: {chart['error']}"
        )
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame()
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = result.get("indicators", {}).get("quote") or []
    if not timestamps or not quotes:
        return pd.DataFrame()
    quote_data = quotes[0]
    length = len(timestamps)

    def values(name: str) -> list[object]:
        raw = list(quote_data.get(name) or [])
        return (raw + [None] * length)[:length]

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                timestamps,
                unit="s",
                utc=True,
            ).tz_convert(None).normalize(),
            "Price": pd.to_numeric(values("close"), errors="coerce"),
            "Open": pd.to_numeric(values("open"), errors="coerce"),
            "High": pd.to_numeric(values("high"), errors="coerce"),
            "Low": pd.to_numeric(values("low"), errors="coerce"),
            "Volume": pd.to_numeric(values("volume"), errors="coerce"),
        }
    )
    return out.dropna(subset=["Date", "Price"])


def _download(
    ticker: str,
    start: datetime,
    end: datetime,
    *,
    transport: str = "yfinance",
    timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    if transport == "yfinance":
        return _download_yfinance(ticker, start, end)
    if transport == "yahoo_chart":
        return _download_yahoo_chart(
            ticker,
            start,
            end,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError("transport must be yfinance or yahoo_chart")


def _format_volume(value) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def _to_legacy_layout(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("Date")
    out["Vol."] = out["Volume"].map(_format_volume)
    out["Change %"] = (out["Price"].pct_change() * 100).map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}%"
    )
    return out[["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]]


def update_one_price(
    config: TickerConfig,
    raw_root: Path,
    *,
    initial_start: datetime | None = None,
    refresh_start: datetime | None = None,
    transport: str = "yfinance",
    request_timeout_seconds: float = 30.0,
) -> Path | None:
    company_dir = raw_root / config.display_name
    company_dir.mkdir(parents=True, exist_ok=True)
    managed = sorted(company_dir.glob(f"*{config.display_name} Historical Data.csv"))

    old = pd.DataFrame()
    if managed:
        old = read_csv_fallback(managed[-1])
        old["Date"] = pd.to_datetime(old["Date"], errors="coerce")
        start = (
            refresh_start
            if refresh_start is not None
            else old["Date"].max().to_pydatetime() + timedelta(days=1)
        )
    else:
        start = initial_start or datetime.fromisoformat("1970-01-01")
    end = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)

    new_raw = (
        _download(
            config.ticker,
            start,
            end,
            transport=transport,
            timeout_seconds=request_timeout_seconds,
        )
        if start < end
        else pd.DataFrame()
    )
    new = _to_legacy_layout(new_raw) if not new_raw.empty else pd.DataFrame()

    combined = pd.concat([old, new], ignore_index=True)
    if combined.empty:
        print(f"WARN {config.ticker}: no data")
        return None
    combined["Date"] = pd.to_datetime(
        combined["Date"],
        errors="coerce",
    ).dt.normalize()
    combined = (
        combined.dropna(subset=["Date"])
        .drop_duplicates("Date", keep="last")
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )
    first = combined["Date"].min().strftime("%Y.%m.%d")
    last = combined["Date"].max().strftime("%Y.%m.%d")
    output = company_dir / f"{first}_{last} {config.display_name} Historical Data.csv"
    atomic_to_csv(combined, output, index=False, date_format="%m/%d/%Y")

    # Only remove files managed by this script, never unrelated CSV files.
    for path in managed:
        if path != output:
            path.unlink(missing_ok=True)
    return output


def update_all_prices(
    tickers: dict[str, TickerConfig],
    raw_root: Path,
    selected: list[str] | None = None,
    *,
    initial_start: datetime | None = None,
    refresh_start: datetime | None = None,
    transport: str = "yfinance",
    request_timeout_seconds: float = 30.0,
) -> list[Path]:
    selected_set = {s.upper() for s in selected} if selected else None
    outputs: list[Path] = []
    for ticker, config in tickers.items():
        if selected_set and ticker not in selected_set:
            continue
        result = update_one_price(
            config,
            raw_root,
            initial_start=initial_start,
            refresh_start=refresh_start,
            transport=transport,
            request_timeout_seconds=request_timeout_seconds,
        )
        if result:
            outputs.append(result)
    return outputs
