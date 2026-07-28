from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .io_utils import atomic_to_excel
from .tickers import TickerConfig


STATEMENTS = {
    "Income Statement": "income-statement",
    "Balance Sheet": "balance-sheet",
    "Cash Flow Statement": "cash-flow-statement",
    "Key Financial Ratios": "financial-ratios",
}


def _statement_url(
    config: TickerConfig,
    statement_path: str,
    frequency: str,
) -> str:
    return (
        "https://www.macrotrends.net/stocks/charts/"
        f"{config.ticker}/{config.company_slug}/{statement_path}"
        f"?freq={frequency}"
    )


def _normalise_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Use stable metric/date labels before comparing or merging workbooks."""
    normalised = df.copy()
    normalised.index = normalised.index.map(str)
    columns: list[str] = []
    for column in normalised.columns:
        parsed = pd.to_datetime(column, errors="coerce")
        if pd.notna(parsed):
            columns.append(parsed.strftime("%Y-%m-%d"))
        else:
            columns.append(str(column))
    normalised.columns = columns
    return normalised


def _statement_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for column in df.columns:
        parsed = pd.to_datetime(column, errors="coerce")
        if pd.notna(parsed):
            dates.append(parsed)
    return dates


def _latest_statement_date(df: pd.DataFrame) -> pd.Timestamp:
    dates = _statement_dates(df)
    if not dates:
        raise ValueError("Statement contained no dated columns.")
    return max(dates)


def _merge_statement_history(
    fresh: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer freshly collected values while retaining older workbook history."""
    fresh = _normalise_statement(fresh)
    existing = _normalise_statement(existing)
    if _latest_statement_date(fresh) < _latest_statement_date(existing):
        raise ValueError("Fresh statement is older than the existing workbook.")

    metrics = list(dict.fromkeys([*fresh.index, *existing.index]))
    columns = list(dict.fromkeys([*fresh.columns, *existing.columns]))
    dated = [column for column in columns if pd.notna(pd.to_datetime(column, errors="coerce"))]
    undated = [column for column in columns if column not in dated]
    columns = sorted(dated, key=pd.Timestamp, reverse=True) + undated

    fresh = fresh.reindex(index=metrics, columns=columns)
    existing = existing.reindex(index=metrics, columns=columns)
    return fresh.combine_first(existing)


def _read_existing_workbook(path: Path) -> dict[str, pd.DataFrame]:
    with pd.ExcelFile(path) as workbook:
        return {
            sheet: pd.read_excel(workbook, sheet_name=sheet, index_col=0)
            for sheet in workbook.sheet_names
        }


def _extract_original_data(page_source: str) -> pd.DataFrame:
    soup = BeautifulSoup(page_source, "html.parser")
    scripts = soup.find_all("script")
    payload = None
    for script in scripts:
        text = script.string or script.get_text() or ""
        match = re.search(r"var\s+originalData\s*=\s*(\[.*?\]);", text, re.S)
        if match:
            payload = match.group(1)
            break
    if not payload:
        raise ValueError("Macrotrends originalData payload was not found.")

    data = json.loads(payload)
    rows: list[dict] = []
    for entry in data:
        row = {
            "Metric": BeautifulSoup(
                entry.get("field_name", ""), "html.parser"
            ).get_text(" ", strip=True)
        }
        for key, value in entry.items():
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", key):
                row[key] = value
        rows.append(row)
    if not rows:
        raise ValueError("Macrotrends payload contained no statement rows.")
    return pd.DataFrame(rows).set_index("Metric").sort_index(axis=1, ascending=False)


def create_driver(
    *,
    headless: bool = False,
    page_load_timeout_seconds: float = 30.0,
):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as exc:
        raise RuntimeError(
            "Financial crawling requires selenium and webdriver-manager."
        ) from exc

    options = Options()
    options.add_argument("--window-size=1920,1200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    if headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.set_page_load_timeout(page_load_timeout_seconds)
    return driver


def create_http_session():
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("HTTP financial crawling requires requests.") from exc

    # The module-level request helper creates a short-lived session per request.
    # Macrotrends rate-limits long-lived sessions much more aggressively.
    return requests


def fetch_statement(
    driver,
    config: TickerConfig,
    statement_path: str,
    *,
    frequency: str = "Q",
    wait_seconds: float = 5.0,
) -> pd.DataFrame:
    from selenium.common.exceptions import TimeoutException

    url = _statement_url(config, statement_path, frequency)
    try:
        driver.get(url)
    except TimeoutException:
        driver.execute_script("window.stop();")
    if f"/{config.ticker.lower()}/" not in driver.current_url.lower():
        raise ValueError(
            f"Unexpected Macrotrends redirect for {config.ticker}: {driver.current_url}"
        )
    time.sleep(wait_seconds)
    return _extract_original_data(driver.page_source)


def fetch_statement_http(
    session,
    config: TickerConfig,
    statement_path: str,
    *,
    frequency: str = "Q",
    timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    url = _statement_url(config, statement_path, frequency)
    response = session.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            )
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    if f"/{config.ticker.lower()}/" not in response.url.lower():
        raise ValueError(
            f"Unexpected Macrotrends redirect for {config.ticker}: {response.url}"
        )
    return _extract_original_data(response.text)


def scrape_financials(
    configs: dict[str, TickerConfig],
    output_folder: Path,
    *,
    selected: list[str] | None = None,
    frequency: str = "Q",
    headless: bool = False,
    wait_seconds: float = 5.0,
    retries: int = 2,
    page_load_timeout_seconds: float = 30.0,
    transport: str = "http",
    request_pause_seconds: float = 0.25,
    retry_backoff_seconds: float = 2.0,
) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    selected_set = {s.upper() for s in selected} if selected else None
    if transport not in {"http", "selenium"}:
        raise ValueError("transport must be 'http' or 'selenium'.")
    client = (
        create_http_session()
        if transport == "http"
        else create_driver(
            headless=headless,
            page_load_timeout_seconds=page_load_timeout_seconds,
        )
    )
    outputs: list[Path] = []
    try:
        for ticker, config in configs.items():
            if selected_set and ticker not in selected_set:
                continue
            sheets: dict[str, pd.DataFrame] = {}
            errors: list[str] = []
            for sheet_name, statement_path in STATEMENTS.items():
                for attempt in range(retries + 1):
                    try:
                        if transport == "http":
                            sheets[sheet_name] = fetch_statement_http(
                                client,
                                config,
                                statement_path,
                                frequency=frequency,
                                timeout_seconds=page_load_timeout_seconds,
                            )
                            time.sleep(request_pause_seconds)
                        else:
                            sheets[sheet_name] = fetch_statement(
                                client,
                                config,
                                statement_path,
                                frequency=frequency,
                                wait_seconds=wait_seconds,
                            )
                        break
                    except Exception as exc:
                        if attempt == retries:
                            errors.append(f"{sheet_name}: {exc}")
                        else:
                            print(
                                f"WARN {ticker} {sheet_name}: retry "
                                f"{attempt + 1}/{retries} after {exc}"
                            )
                            time.sleep(retry_backoff_seconds * (2**attempt))
                if sheet_name not in sheets:
                    break
            missing = set(STATEMENTS) - set(sheets)
            if errors or missing:
                details = "; ".join(errors) or f"missing sheets: {sorted(missing)}"
                print(f"ERROR {ticker}: incomplete workbook; existing file kept. {details}")
                continue

            output = output_folder / f"{ticker}_financials_{frequency}.xlsx"
            try:
                merged = {sheet: _normalise_statement(df) for sheet, df in sheets.items()}
                if output.exists():
                    existing = _read_existing_workbook(output)
                    merged = {
                        sheet: _merge_statement_history(df, existing[sheet])
                        if sheet in existing
                        else _normalise_statement(df)
                        for sheet, df in sheets.items()
                    }
                atomic_to_excel(merged, output, index=True)
                latest = min(_latest_statement_date(df) for df in merged.values())
                print(f"UPDATED {ticker}: complete through {latest.date()}")
                outputs.append(output)
            except Exception as exc:
                print(f"ERROR {ticker}: validation/write failed; existing file kept. {exc}")
    finally:
        if transport == "selenium":
            client.quit()
    return outputs
