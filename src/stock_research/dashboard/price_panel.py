from __future__ import annotations

"""Price-only cross-sectional panel builder for the dashboard.

Mirrors stock_research.cross_sectional.features.build_equity_features /
add_cross_sectional_factors, but drops every Macrotrends-financial-derived
column (PeTtm, EvEbitdaTtm, DcfPrice, GrowthFactor, QualityFactor, ...)
since dashboard-added tickers have no Macrotrends quarterly-financials
Excel file. Momentum/Trend/RiskControl are computed from price alone, exactly
as in the production engine. GrowthFactor/QualityFactor are left NaN here --
the caller must run reconstruct_growth_quality_pit_safe() afterward to fill
them from SEC filing data, same as the PIT-Reconstructed research pipeline
used elsewhere this session.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.cross_sectional.config import ResearchSettings
from stock_research.tsla_integrated.data import load_equity_prices


@dataclass(frozen=True)
class DashboardMember:
    ticker: str
    company: str
    price_path: Path


def build_price_only_panel(
    members: list[DashboardMember],
    settings: ResearchSettings,
    *,
    warmup_start: str | None = None,
) -> pd.DataFrame:
    """`warmup_start` truncates each ticker's raw price history before the
    rolling/rank computations run. Without it, a ticker like AAPL (priced
    back to 1980) drags every groupby(Date).rank() call in this module and
    in v7_technical.py/filing_signals.py/point_in_time.py across 45+ years
    of rows even when the caller only wants a 2-3 year backtest window --
    this made a 28-ticker dashboard request take minutes instead of
    seconds. 200+ extra trading sessions are kept before warmup_start so the
    200-session eligibility gate and rolling-200 trend window still have a
    full lookback at the caller's actual start date."""

    frames = []
    for member in members:
        prices = load_equity_prices(member.price_path)
        if warmup_start is not None:
            prices = prices.loc[prices["Date"].ge(pd.Timestamp(warmup_start))]
        frames.append(
            _build_equity_features(prices, ticker=member.ticker, company=member.company, settings=settings)
        )
    panel = pd.concat(frames, ignore_index=True)
    panel = _add_cross_sectional_factors(panel, settings)
    return panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def _build_equity_features(
    prices: pd.DataFrame,
    *,
    ticker: str,
    company: str,
    settings: ResearchSettings,
) -> pd.DataFrame:
    frame = prices.sort_values("Date").reset_index(drop=True).copy()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    frame["Ticker"] = ticker
    frame["Company"] = company
    frame["PriceHistorySessions"] = np.arange(1, len(frame) + 1)
    frame["Return21"] = close.pct_change(21, fill_method=None)
    frame["Return63"] = close.pct_change(63, fill_method=None)
    frame["Return126"] = close.pct_change(126, fill_method=None)
    frame["Trend50"] = close / close.rolling(50, min_periods=50).mean() - 1
    frame["Trend200"] = close / close.rolling(200, min_periods=200).mean() - 1
    frame["Volatility63"] = (
        close.pct_change(fill_method=None).rolling(63, min_periods=42).std() * np.sqrt(252)
    )
    frame["Drawdown126"] = close / close.rolling(126, min_periods=63).max() - 1
    frame["Eligible"] = (
        frame["PriceHistorySessions"].ge(settings.minimum_price_history_sessions)
        & close.gt(0)
        & frame[["Return126", "Trend200", "Volatility63"]].notna().all(axis=1)
    )
    return frame


def _add_cross_sectional_factors(panel: pd.DataFrame, settings: ResearchSettings) -> pd.DataFrame:
    frame = panel.copy()
    eligible = frame["Eligible"]
    ranked: dict[str, pd.Series] = {}
    for column in ("Return21", "Return63", "Return126", "Trend50", "Trend200"):
        ranked[column] = _centered_rank(frame, column, eligible, settings.minimum_cross_section_size)
    ranked["LowVolatility63"] = -_centered_rank(frame, "Volatility63", eligible, settings.minimum_cross_section_size)
    ranked["Drawdown126"] = _centered_rank(frame, "Drawdown126", eligible, settings.minimum_cross_section_size)

    frame["MomentumFactor"] = pd.concat(
        [ranked["Return21"], ranked["Return63"], ranked["Return126"]], axis=1
    ).mean(axis=1)
    frame["TrendFactor"] = pd.concat([ranked["Trend50"], ranked["Trend200"]], axis=1).mean(axis=1)
    frame["RiskControlFactor"] = pd.concat(
        [ranked["LowVolatility63"], ranked["Drawdown126"]], axis=1
    ).mean(axis=1)
    # Filled later by reconstruct_growth_quality_pit_safe() from SEC filing data.
    frame["GrowthFactor"] = np.nan
    frame["QualityFactor"] = np.nan

    counts = frame.loc[eligible].groupby("Date")["Ticker"].transform("count")
    frame["CrossSectionSize"] = 0
    frame.loc[eligible, "CrossSectionSize"] = counts.astype(int)
    frame["Eligible"] = frame["Eligible"] & frame["CrossSectionSize"].ge(settings.minimum_cross_section_size)
    factor_columns = ["MomentumFactor", "TrendFactor", "GrowthFactor", "QualityFactor", "RiskControlFactor"]
    frame.loc[~frame["Eligible"], factor_columns] = np.nan
    return frame


def _centered_rank(frame: pd.DataFrame, column: str, eligible: pd.Series, minimum_count: int) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").where(eligible)
    counts = values.notna().groupby(frame["Date"]).transform("sum")
    ranks = values.groupby(frame["Date"]).rank(pct=True, method="average") - 0.5
    return ranks.where(counts.ge(minimum_count))
