from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_research.paths import ProjectPaths
from stock_research.tickers import TickerConfig, load_tickers
from stock_research.tsla_integrated.data import (
    load_equity_financials,
    load_equity_prices,
)


@dataclass(frozen=True)
class UniverseMember:
    ticker: str
    company: str
    price_path: Path
    financial_path: Path


def discover_universe(
    paths: ProjectPaths,
    ticker_config_path: str | Path,
) -> tuple[list[UniverseMember], pd.DataFrame]:
    """Discover stocks having both processed prices and quarterly financials."""

    ticker_configs = load_tickers(ticker_config_path)
    members: list[UniverseMember] = []
    audit_rows: list[dict[str, object]] = []
    for ticker, config in sorted(ticker_configs.items()):
        if _is_crypto(ticker):
            audit_rows.append(
                _audit_row(ticker, config, None, None, "EXCLUDED_CRYPTO")
            )
            continue
        price_matches = sorted(
            paths.processed.glob(f"{config.display_name}_*.csv"),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        price_path = price_matches[0] if price_matches else None
        financial_path = paths.financial_raw / f"{ticker}_financials_Q.xlsx"
        if price_path is None:
            status = "MISSING_PRICE"
        elif not financial_path.exists():
            status = "MISSING_FINANCIALS"
        else:
            status = "INCLUDED"
            members.append(
                UniverseMember(
                    ticker=ticker,
                    company=config.display_name,
                    price_path=price_path,
                    financial_path=financial_path,
                )
            )
        audit_rows.append(
            _audit_row(ticker, config, price_path, financial_path, status)
        )

    configured = set(ticker_configs)
    for path in sorted(paths.financial_raw.glob("*_financials_Q.xlsx")):
        ticker = path.name.removesuffix("_financials_Q.xlsx").upper()
        if ticker not in configured:
            audit_rows.append(
                {
                    "Ticker": ticker,
                    "Company": "",
                    "Status": "UNCONFIGURED_FINANCIALS",
                    "PriceFile": "",
                    "FinancialFile": path.name,
                }
            )
    return members, pd.DataFrame(audit_rows).sort_values(
        ["Status", "Ticker"]
    ).reset_index(drop=True)


def load_member_data(
    member: UniverseMember,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    prices = load_equity_prices(member.price_path)
    financials = load_equity_financials(member.financial_path)
    audit = {
        "Ticker": member.ticker,
        "Company": member.company,
        "PriceStart": prices["Date"].min(),
        "PriceEnd": prices["Date"].max(),
        "PriceRows": len(prices),
        "FinancialStart": financials["Date"].min(),
        "FinancialEnd": financials["Date"].max(),
        "FinancialRows": len(financials),
    }
    return prices, financials, audit


def _is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith("-USD")


def _audit_row(
    ticker: str,
    config: TickerConfig,
    price_path: Path | None,
    financial_path: Path | None,
    status: str,
) -> dict[str, object]:
    return {
        "Ticker": ticker,
        "Company": config.display_name,
        "Status": status,
        "PriceFile": price_path.name if price_path else "",
        "FinancialFile": (
            financial_path.name
            if financial_path is not None and financial_path.exists()
            else ""
        ),
    }
