from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_to_excel, read_csv_fallback
from .tickers import TickerConfig


@dataclass(frozen=True)
class DcfAssumptions:
    wacc: float = 0.10
    cost_of_equity: float = 0.10
    short_growth: float = 0.05
    projection_years: int = 5
    terminal_growth: float = 0.025


def safe_div(numerator, denominator, eps: float = 1e-9):
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")
    return n / d.where(d.abs() > eps)


def dcf_equity_value_from_fcf(
    fcf0: float,
    net_debt: float,
    rate: float,
    growth: float,
    years: int,
    terminal_growth: float,
) -> float:
    if pd.isna(fcf0) or fcf0 <= 0 or rate <= terminal_growth:
        return np.nan
    pv = sum(
        fcf0 * (1 + growth) ** year / (1 + rate) ** year
        for year in range(1, years + 1)
    )
    terminal_cash_flow = fcf0 * (1 + growth) ** years
    terminal_value = (
        terminal_cash_flow * (1 + terminal_growth)
        / (rate - terminal_growth)
    )
    return pv + terminal_value / (1 + rate) ** years - (
        net_debt if pd.notna(net_debt) else 0
    )


def pct_change_abs(series: pd.Series, periods: int = 1) -> pd.Series:
    previous = series.shift(periods)
    return (series - previous) / previous.abs() * 100


def _numeric_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().any():
                return series
    return pd.Series(np.nan, index=frame.index)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def load_financial_workbook(path: Path) -> pd.DataFrame:
    required = [
        "Income Statement",
        "Balance Sheet",
        "Cash Flow Statement",
        "Key Financial Ratios",
    ]
    frames = []
    for sheet in required:
        try:
            part = (
                pd.read_excel(path, sheet_name=sheet, index_col=0)
                .T.reset_index()
                .rename(columns={"index": "Date"})
            )
            part["Date"] = pd.to_datetime(part["Date"], errors="coerce")
            frames.append(part)
        except ValueError:
            continue
    if not frames:
        raise ValueError(f"No supported sheets found in {path}")
    merged = frames[0]
    for part in frames[1:]:
        merged = merged.merge(part, on="Date", how="outer")
    return merged.sort_values("Date").drop_duplicates("Date", keep="last")


def find_price_csv(raw_root: Path, config: TickerConfig) -> Path:
    folder = raw_root / config.display_name
    hits = sorted(
        folder.glob(f"*{config.display_name} Historical Data.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        hits = sorted(folder.glob("*Historical Data.csv"))
    if not hits:
        raise FileNotFoundError(f"No price history for {config.display_name}")
    return hits[0]


def find_processed_csv(processed_root: Path, config: TickerConfig) -> Path | None:
    hits = sorted(
        processed_root.glob(f"{config.display_name}*_지표포함*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


def _load_price_history(path: Path) -> pd.DataFrame:
    frame = read_csv_fallback(path)
    rename = {}
    for column in frame.columns:
        key = column.strip().lower().replace(" ", "")
        if key in {"date", "날짜"}:
            rename[column] = "Date"
        elif key in {"price", "close", "종가"}:
            rename[column] = "Price"
    frame = frame.rename(columns=rename)
    if not {"Date", "Price"}.issubset(frame.columns):
        raise ValueError(f"Price CSV lacks Date/Price columns: {path}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    return frame.dropna(subset=["Date", "Price"]).sort_values("Date")


def analyze_company(
    config: TickerConfig,
    *,
    raw_price_root: Path,
    processed_root: Path,
    financial_root: Path,
    output_root: Path,
    assumptions: DcfAssumptions = DcfAssumptions(),
    market_price_override: float | None = None,
) -> Path:
    financial_path = financial_root / f"{config.ticker}_financials_Q.xlsx"
    if not financial_path.exists():
        raise FileNotFoundError(financial_path)

    financial = load_financial_workbook(financial_path)
    price_full = _load_price_history(find_price_csv(raw_price_root, config))
    merged = pd.merge_asof(
        financial.sort_values("Date"),
        price_full[["Date", "Price"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )
    if market_price_override is not None:
        merged["Price"] = float(market_price_override)

    shares = _numeric_series(
        merged, ["Shares Outstanding", "Basic Shares Outstanding"]
    )
    net_income = _column(merged, "Net Income")
    eps = _numeric_series(merged, ["EPS - Earnings Per Share", "Basic EPS"])
    eps = eps.where(eps.notna(), safe_div(net_income, shares))
    price = _column(merged, "Price")
    liabilities = _column(merged, "Total Liabilities")
    cash = _column(merged, "Cash On Hand")
    current_assets = _column(merged, "Total Current Assets")
    total_assets = _column(merged, "Total Assets")
    equity = _column(merged, "Share Holder Equity")
    ebit = _column(merged, "EBIT")
    depreciation = _column(
        merged, "Total Depreciation And Amortization - Cash Flow"
    )
    capex_source = _column(
        merged, "Net Change In Property, Plant, And Equipment"
    )
    working_capital_source = _column(merged, "Total Change In Assets/Liabilities")
    dividends = _column(merged, "Common Stock Dividends Paid").abs()

    market_cap = price * shares
    ncav = current_assets - liabilities
    book_value_per_share = safe_div(equity, shares)
    graham = np.sqrt(
        22.5 * eps.clip(lower=0) * book_value_per_share.clip(lower=0)
    )
    fcff_q = ebit * 0.75 + depreciation + capex_source + working_capital_source
    owner_q = net_income + depreciation + capex_source + working_capital_source
    dividend_per_share = safe_div(dividends, shares)
    dividend_yield = safe_div(dividend_per_share, price) * 100
    ddm_price = (
        dividend_per_share * (1 + assumptions.short_growth)
        / (assumptions.cost_of_equity - assumptions.short_growth)
        if assumptions.cost_of_equity > assumptions.short_growth
        else pd.Series(np.nan, index=merged.index)
    )
    eps_growth_qoq = pct_change_abs(eps, 1)
    pe_q = safe_div(price, eps)
    ev_q = market_cap + liabilities - cash
    ev_ebit_q = safe_div(ev_q, ebit)

    quarterly = pd.DataFrame({
        "Date": merged["Date"],
        "Price": price,
        "Market Cap": market_cap,
        "NCAV": ncav,
        "Undervalued by NCAV": market_cap < ncav,
        "EPS_Q": eps,
        "EPS_Growth_QoQ_%": eps_growth_qoq,
        "PE_Q": pe_q,
        "Div_Yield_Q_%": dividend_yield,
        "PEG_Q": safe_div(pe_q, eps_growth_qoq),
        "PEGY_Q": safe_div(pe_q, eps_growth_qoq + dividend_yield),
        "EV/EBIT_Q": ev_ebit_q,
        "Graham Number": graham,
        "FCFF": fcff_q,
        "Owner Earnings": owner_q,
        "DDM Price": ddm_price,
    }).set_index("Date")

    q = merged.set_index("Date").sort_index()
    rolling = lambda series: series.rolling(4, min_periods=4).sum()
    net_ttm = rolling(_column(q, "Net Income"))
    ebit_ttm = rolling(_column(q, "EBIT"))
    dep_ttm = rolling(_column(q, "Total Depreciation And Amortization - Cash Flow"))
    capex_ttm = rolling(-_column(q, "Net Change In Property, Plant, And Equipment"))
    wc_ttm = rolling(-_column(q, "Total Change In Assets/Liabilities"))
    shares_q = _numeric_series(q, ["Shares Outstanding", "Basic Shares Outstanding"])
    price_q = _column(q, "Price")
    dividends_ttm = rolling(_column(q, "Common Stock Dividends Paid").abs())
    dividend_per_share_ttm = safe_div(dividends_ttm, shares_q)
    dividend_yield_ttm = safe_div(dividend_per_share_ttm, price_q) * 100
    liabilities_q = _column(q, "Total Liabilities")
    cash_q = _column(q, "Cash On Hand")
    current_assets_q = _column(q, "Total Current Assets")
    total_assets_q = _column(q, "Total Assets")
    market_cap_ttm = price_q * shares_q
    eps_ttm = safe_div(net_ttm, shares_q)
    pe_ttm = safe_div(price_q, eps_ttm)
    eps_yoy = pct_change_abs(eps_ttm, 4)
    net_debt = liabilities_q - cash_q
    fcff_ttm = ebit_ttm * 0.75 + dep_ttm - capex_ttm - wc_ttm
    owner_ttm = net_ttm + dep_ttm - capex_ttm - wc_ttm

    dcf_fcff = pd.Series(index=q.index, dtype=float)
    dcf_owner = pd.Series(index=q.index, dtype=float)
    for date in q.index:
        dcf_fcff.loc[date] = dcf_equity_value_from_fcf(
            fcff_ttm.loc[date],
            net_debt.loc[date],
            assumptions.wacc,
            assumptions.short_growth,
            assumptions.projection_years,
            assumptions.terminal_growth,
        )
        dcf_owner.loc[date] = dcf_equity_value_from_fcf(
            owner_ttm.loc[date],
            0.0,
            assumptions.cost_of_equity,
            assumptions.short_growth,
            assumptions.projection_years,
            assumptions.terminal_growth,
        )

    invested_capital = total_assets_q
    ev_ttm = market_cap_ttm + liabilities_q - cash_q
    retained_earnings = _column(q, "Retained Earnings (Accumulated Deficit)")
    legacy_altman_z = (
        1.2 * safe_div(current_assets_q - liabilities_q, total_assets_q)
        + 1.4 * safe_div(retained_earnings, total_assets_q)
        + 3.3 * safe_div(ebit_ttm, total_assets_q)
        + 0.6 * safe_div(market_cap_ttm, liabilities_q)
        + safe_div(net_ttm, total_assets_q)
    )
    graham_ttm = np.sqrt(
        22.5
        * eps_ttm.clip(lower=0)
        * safe_div(_column(q, "Share Holder Equity"), shares_q).clip(lower=0)
    )
    ddm_ttm_price = (
        dividend_per_share_ttm * (1 + assumptions.short_growth)
        / (assumptions.cost_of_equity - assumptions.short_growth)
        if assumptions.cost_of_equity > assumptions.short_growth
        else pd.Series(np.nan, index=q.index)
    )
    yearly_all = pd.DataFrame({
        "Market Cap": market_cap_ttm,
        "NCAV_TTM": current_assets_q - liabilities_q,
        "Undervalued by NCAV": market_cap_ttm < (current_assets_q - liabilities_q),
        "ROIC": safe_div(ebit_ttm, invested_capital),
        "EV/EBIT": safe_div(ev_ttm, ebit_ttm),
        "EPS_TTM": eps_ttm,
        "PE_TTM": pe_ttm,
        "EPS_Growth_YoY_%": eps_yoy,
        "Div_Yield_TTM_%": dividend_yield_ttm,
        "PEG_TTM": safe_div(pe_ttm, eps_yoy),
        "PEGY_TTM": safe_div(pe_ttm, eps_yoy + dividend_yield_ttm),
        "Graham Number": graham_ttm,
        "DDM Price": ddm_ttm_price,
        "DCF_FCFF_Price": safe_div(dcf_fcff, shares_q),
        "DCF_OwnerEarnings_Price": safe_div(dcf_owner, shares_q),
        "Altman Z (legacy notebook formula)": legacy_altman_z,
    })

    fiscal_frequency = {
        month: f"Q-{label}"
        for month, label in enumerate(
            ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
            start=1,
        )
    }[config.fiscal_year_end_month]
    quarter_number = yearly_all.index.to_period(fiscal_frequency).quarter
    yearly = yearly_all[quarter_number == 4]

    processed_path = find_processed_csv(processed_root, config)
    if processed_path:
        price_history = read_csv_fallback(processed_path)
    else:
        price_history = price_full

    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{config.ticker}_analysis_Q.xlsx"
    atomic_to_excel(
        {
            "Quarterly": quarterly.round(4),
            "Yearly_TTM": yearly.round(4),
            "Inputs": merged,
            "Price_History": price_history,
        },
        output,
        index={
            "Quarterly": True,
            "Yearly_TTM": True,
            "Inputs": False,
            "Price_History": False,
        },
    )
    return output
