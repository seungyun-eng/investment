from __future__ import annotations

import numpy as np
import pandas as pd

TREND_WINDOWS = (
    10,
    20,
    30,
    40,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
)


def _numeric(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _rolling_percentile(series: pd.Series, window: int = 756) -> pd.Series:
    return series.rolling(window, min_periods=126).rank(pct=True)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).rolling(length, min_periods=length).mean()
    loss = -change.clip(upper=0).rolling(length, min_periods=length).mean()
    relative = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative)


def _financial_features(
    financials: pd.DataFrame,
    *,
    release_lag_days: int,
) -> pd.DataFrame:
    frame = financials.copy().sort_values("Date").reset_index(drop=True)
    revenue = _numeric(frame, ["Revenue"])
    gross_profit = _numeric(frame, ["Gross Profit"])
    operating_income = _numeric(
        frame, ["Operating Income", "EBIT", "Operating Income/Loss"]
    )
    net_income = _numeric(frame, ["Net Income", "Net Income/Loss"])
    cash = _numeric(frame, ["Cash On Hand"])
    liabilities = _numeric(frame, ["Total Liabilities"])
    operating_cash = _numeric(
        frame, ["Cash Flow From Operating Activities", "Operating Cash Flow"]
    )
    capex = _numeric(
        frame, ["Capital Expenditures", "Net Change In Property, Plant, And Equipment"]
    )
    result = pd.DataFrame(
        {
            "FinancialPeriodEnd": frame["Date"],
            "FinancialAvailableDate": (
                frame["Date"] + pd.to_timedelta(release_lag_days, unit="D")
            ),
            "RevenueGrowthYoY": revenue.pct_change(4, fill_method=None),
            "GrossMargin": gross_profit / revenue.replace(0, np.nan),
            "OperatingMargin": operating_income / revenue.replace(0, np.nan),
            "NetMargin": net_income / revenue.replace(0, np.nan),
            "FreeCashFlowMargin": (
                operating_cash + capex
            ) / revenue.replace(0, np.nan),
            "NetCashToRevenue": (
                cash - liabilities
            ) / revenue.rolling(4, min_periods=1).sum().replace(0, np.nan),
        }
    )
    return result.sort_values("FinancialAvailableDate")


def build_integrated_features(
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    financial_release_lag_days: int = 45,
) -> pd.DataFrame:
    """Build daily equity features using only information available that day."""

    frame = prices.copy().sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    frame["Return21"] = close.pct_change(21)
    frame["Return63"] = close.pct_change(63)
    for window in TREND_WINDOWS:
        frame[f"SMA{window}"] = close.rolling(
            window,
            min_periods=window,
        ).mean()
        frame[f"Trend{window}"] = close / frame[f"SMA{window}"] - 1
    frame["RSI14"] = _rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["MACD"] = ema12 - ema26
    frame["MACDSignal"] = frame["MACD"].ewm(span=9, adjust=False).mean()

    available = _financial_features(
        financials,
        release_lag_days=financial_release_lag_days,
    )
    frame = pd.merge_asof(
        frame,
        available,
        left_on="Date",
        right_on="FinancialAvailableDate",
        direction="backward",
    )
    macro_daily = macro.copy().sort_values("Date")
    keep = [
        column
        for column in (
            "Date",
            "CashRate",
            "VIX",
            "MacroConfirmationScore",
            "RiskProbability_63",
            "RiskProbability_126",
        )
        if column in macro_daily
    ]
    frame = pd.merge_asof(
        frame.sort_values("Date"),
        macro_daily[keep],
        on="Date",
        direction="backward",
    )
    frame["VixPercentile"] = _rolling_percentile(
        pd.to_numeric(frame.get("VIX"), errors="coerce")
    )
    risk63 = pd.to_numeric(frame.get("RiskProbability_63"), errors="coerce")
    risk126 = pd.to_numeric(frame.get("RiskProbability_126"), errors="coerce")
    frame["ModelRisk"] = 0.4 * risk63 + 0.6 * risk126
    return frame
