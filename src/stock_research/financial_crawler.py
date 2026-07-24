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


def create_driver(*, headless: bool = False):
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
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )


def fetch_statement(
    driver,
    config: TickerConfig,
    statement_path: str,
    *,
    frequency: str = "Q",
    wait_seconds: float = 5.0,
) -> pd.DataFrame:
    url = (
        "https://www.macrotrends.net/stocks/charts/"
        f"{config.ticker}/{config.company_slug}/{statement_path}"
        f"?freq={frequency}"
    )
    driver.get(url)
    time.sleep(wait_seconds)
    return _extract_original_data(driver.page_source)


def scrape_financials(
    configs: dict[str, TickerConfig],
    output_folder: Path,
    *,
    selected: list[str] | None = None,
    frequency: str = "Q",
    headless: bool = False,
    wait_seconds: float = 5.0,
) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    selected_set = {s.upper() for s in selected} if selected else None
    driver = create_driver(headless=headless)
    outputs: list[Path] = []
    try:
        for ticker, config in configs.items():
            if selected_set and ticker not in selected_set:
                continue
            sheets: dict[str, pd.DataFrame] = {}
            for sheet_name, statement_path in STATEMENTS.items():
                try:
                    sheets[sheet_name] = fetch_statement(
                        driver,
                        config,
                        statement_path,
                        frequency=frequency,
                        wait_seconds=wait_seconds,
                    )
                except Exception as exc:
                    print(f"WARN {ticker} {sheet_name}: {exc}")
            if sheets:
                output = output_folder / f"{ticker}_financials_{frequency}.xlsx"
                atomic_to_excel(sheets, output, index=True)
                outputs.append(output)
    finally:
        driver.quit()
    return outputs
