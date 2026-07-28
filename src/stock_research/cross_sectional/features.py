from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.financial_analysis import dcf_equity_value_from_fcf

from .config import ResearchSettings
from .data import UniverseMember, load_member_data

FINANCIAL_SIGNAL_COLUMNS = (
    "RevenueGrowthYoY",
    "EpsGrowthYoY",
    "OperatingMarginChangeYoY",
    "OperatingMargin",
    "FreeCashFlowMargin",
    "ReturnOnInvestment",
    "NetCashToAssets",
    "EpsTtm",
    "EpsTtmGrowthYoY",
    "EpsTtmGrowthAcceleration",
    "EbitdaTtm",
    "EbitdaTtmGrowthYoY",
    "EbitdaTtmGrowthAcceleration",
    "DcfPrice",
    "DcfPriceGrowthYoY",
    "DcfUpside",
    "PeTtm",
    "EvEbitdaTtm",
    "GrowthAdjustedPe",
    "GrowthAdjustedEvEbitda",
)

TTM_GROWTH_WEIGHTS = {
    "EpsTtmGrowthYoY": 0.30,
    "EpsTtmGrowthAcceleration": 0.10,
    "DcfPriceGrowthYoY": 0.25,
    "EbitdaTtmGrowthYoY": 0.25,
    "EbitdaTtmGrowthAcceleration": 0.10,
}

TTM_VALUE_WEIGHTS = {
    "DcfUpside": 0.30,
    "GrowthAdjustedPe": 0.25,
    "GrowthAdjustedEvEbitda": 0.25,
    "OperatingMargin": 0.10,
    "FreeCashFlowMargin": 0.10,
}


def build_panel(
    members: list[UniverseMember],
    settings: ResearchSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for member in members:
        prices, financials, audit = load_member_data(member)
        equity = build_equity_features(
            prices,
            financials,
            ticker=member.ticker,
            company=member.company,
            settings=settings,
        )
        train_mask = equity["Date"].between(
            settings.train_start,
            settings.train_end,
        )
        audit["TrainingSessions"] = int(train_mask.sum())
        audit["TrainingEligibleSessions"] = int(
            (train_mask & equity["Eligible"]).sum()
        )
        audit["FirstEligibleDate"] = equity.loc[
            equity["Eligible"], "Date"
        ].min()
        audits.append(audit)
        frames.append(equity)
    panel = pd.concat(frames, ignore_index=True)
    panel = add_cross_sectional_factors(panel, settings)
    return (
        panel.sort_values(["Date", "Ticker"]).reset_index(drop=True),
        pd.DataFrame(audits).sort_values("Ticker").reset_index(drop=True),
    )


def build_equity_features(
    prices: pd.DataFrame,
    financials: pd.DataFrame,
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
        close.pct_change(fill_method=None)
        .rolling(63, min_periods=42)
        .std()
        * np.sqrt(252)
    )
    frame["Drawdown126"] = (
        close / close.rolling(126, min_periods=63).max() - 1
    )

    available = _build_available_financials(
        financials,
        settings,
    )
    frame = pd.merge_asof(
        frame.sort_values("Date"),
        available,
        left_on="Date",
        right_on="FinancialAvailableDate",
        direction="backward",
    )
    frame["FinancialAgeDays"] = (
        frame["Date"] - frame["FinancialAvailableDate"]
    ).dt.days
    stale = frame["FinancialAgeDays"] > settings.max_financial_age_days
    frame["PeTtm"] = (
        close / frame["EpsTtm"].where(frame["EpsTtm"].gt(0))
    )
    enterprise_value = (
        close * frame["Shares"]
        + frame["TotalLiabilities"]
        - frame["Cash"]
    )
    frame["EvEbitdaTtm"] = (
        enterprise_value
        / frame["EbitdaTtm"].where(frame["EbitdaTtm"].gt(0))
    ).where(enterprise_value.gt(0))
    frame["DcfUpside"] = (
        frame["DcfPrice"].where(frame["DcfPrice"].gt(0)) / close - 1
    )
    frame["GrowthAdjustedPe"] = (
        frame["EpsTtmGrowthYoY"].where(frame["EpsTtmGrowthYoY"].gt(0))
        / frame["PeTtm"].where(frame["PeTtm"].gt(0))
    )
    frame["GrowthAdjustedEvEbitda"] = (
        frame["EbitdaTtmGrowthYoY"].where(
            frame["EbitdaTtmGrowthYoY"].gt(0)
        )
        / frame["EvEbitdaTtm"].where(frame["EvEbitdaTtm"].gt(0))
    )
    frame.loc[stale, list(FINANCIAL_SIGNAL_COLUMNS)] = np.nan
    frame["FinancialStale"] = stale | frame["FinancialAvailableDate"].isna()
    frame["Eligible"] = (
        frame["PriceHistorySessions"].ge(
            settings.minimum_price_history_sessions
        )
        & close.gt(0)
        & frame[["Return126", "Trend200", "Volatility63"]]
        .notna()
        .all(axis=1)
    )
    return frame


def add_cross_sectional_factors(
    panel: pd.DataFrame,
    settings: ResearchSettings,
) -> pd.DataFrame:
    frame = panel.copy()
    eligible = frame["Eligible"]
    ranked: dict[str, pd.Series] = {}
    positive_columns = (
        "Return21",
        "Return63",
        "Return126",
        "Trend50",
        "Trend200",
        "RevenueGrowthYoY",
        "EpsGrowthYoY",
        "OperatingMarginChangeYoY",
        "OperatingMargin",
        "FreeCashFlowMargin",
        "ReturnOnInvestment",
        "NetCashToAssets",
        *TTM_GROWTH_WEIGHTS,
        *TTM_VALUE_WEIGHTS,
        "Drawdown126",
    )
    for column in positive_columns:
        ranked[column] = _centered_rank(
            frame,
            column,
            eligible,
            settings.minimum_cross_section_size,
        )
    ranked["LowVolatility63"] = -_centered_rank(
        frame,
        "Volatility63",
        eligible,
        settings.minimum_cross_section_size,
    )
    if settings.financial_feature_mode == "ttm_value_momentum":
        for column in (*TTM_GROWTH_WEIGHTS, *TTM_VALUE_WEIGHTS):
            value = pd.to_numeric(frame[column], errors="coerce")
            ranked[column] = ranked[column].where(
                value.gt(0),
                ranked[column].clip(upper=0),
            )

    frame["MomentumFactor"] = _row_mean(
        ranked,
        ("Return21", "Return63", "Return126"),
    )
    frame["TrendFactor"] = _row_mean(
        ranked,
        ("Trend50", "Trend200"),
    )
    if settings.financial_feature_mode == "ttm_value_momentum":
        frame["GrowthFactor"] = _weighted_row_mean(
            ranked,
            TTM_GROWTH_WEIGHTS,
        )
        frame["QualityFactor"] = _weighted_row_mean(
            ranked,
            TTM_VALUE_WEIGHTS,
        )
    else:
        frame["GrowthFactor"] = _row_mean(
            ranked,
            (
                "RevenueGrowthYoY",
                "EpsGrowthYoY",
                "OperatingMarginChangeYoY",
            ),
        )
        frame["QualityFactor"] = _row_mean(
            ranked,
            (
                "OperatingMargin",
                "FreeCashFlowMargin",
                "ReturnOnInvestment",
                "NetCashToAssets",
            ),
        )
    frame["RiskControlFactor"] = _row_mean(
        ranked,
        ("LowVolatility63", "Drawdown126"),
    )
    counts = frame.loc[eligible].groupby("Date")["Ticker"].transform("count")
    frame["CrossSectionSize"] = 0
    frame.loc[eligible, "CrossSectionSize"] = counts.astype(int)
    frame["Eligible"] &= frame["CrossSectionSize"].ge(
        settings.minimum_cross_section_size
    )
    factor_columns = [
        "MomentumFactor",
        "TrendFactor",
        "GrowthFactor",
        "QualityFactor",
        "RiskControlFactor",
    ]
    frame.loc[~frame["Eligible"], factor_columns] = np.nan
    return frame


def _build_available_financials(
    financials: pd.DataFrame,
    settings: ResearchSettings,
) -> pd.DataFrame:
    frame = financials.sort_values("Date").reset_index(drop=True).copy()
    revenue = _numeric(frame, ("Revenue",))
    eps = _numeric(frame, ("EPS - Earnings Per Share", "Basic EPS"))
    operating_income = _numeric(
        frame,
        ("Operating Income", "EBIT", "Operating Income/Loss"),
    )
    operating_cash = _numeric(
        frame,
        ("Cash Flow From Operating Activities", "Operating Cash Flow"),
    )
    capex = _numeric(
        frame,
        (
            "Capital Expenditures",
            "Net Change In Property, Plant, And Equipment",
        ),
    )
    cash = _numeric(frame, ("Cash On Hand",))
    debt = _numeric(frame, ("Long Term Debt",))
    assets = _numeric(frame, ("Total Assets",))
    shares = _numeric(
        frame,
        ("Shares Outstanding", "Basic Shares Outstanding"),
    )
    net_income = _numeric(frame, ("Net Income", "Net Income/Loss"))
    ebit = _numeric(frame, ("EBIT", "Operating Income"))
    ebitda = _numeric(frame, ("EBITDA",))
    depreciation = _numeric(
        frame,
        ("Total Depreciation And Amortization - Cash Flow",),
    )
    liabilities = _numeric(frame, ("Total Liabilities",))
    capex_ttm = _numeric(
        frame,
        ("Net Change In Property, Plant, And Equipment",),
    ).rolling(4, min_periods=4).sum() * -1
    working_capital_ttm = _numeric(
        frame,
        ("Total Change In Assets/Liabilities",),
    ).rolling(4, min_periods=4).sum() * -1
    net_income_ttm = net_income.rolling(4, min_periods=4).sum()
    ebit_ttm = ebit.rolling(4, min_periods=4).sum()
    ebitda_ttm = ebitda.rolling(4, min_periods=4).sum()
    depreciation_ttm = depreciation.rolling(4, min_periods=4).sum()
    eps_ttm = net_income_ttm / shares.replace(0, np.nan)
    fcff_ttm = (
        ebit_ttm * 0.75
        + depreciation_ttm
        - capex_ttm
        - working_capital_ttm
    )
    owner_earnings_ttm = (
        net_income_ttm
        + depreciation_ttm
        - capex_ttm
        - working_capital_ttm
    )
    net_debt = liabilities - cash
    dcf_fcff = pd.Series(
        [
            dcf_equity_value_from_fcf(
                fcf,
                debt,
                settings.dcf_wacc,
                settings.dcf_short_growth,
                settings.dcf_projection_years,
                settings.dcf_terminal_growth,
            )
            for fcf, debt in zip(fcff_ttm, net_debt, strict=True)
        ],
        index=frame.index,
        dtype=float,
    )
    dcf_owner = pd.Series(
        [
            dcf_equity_value_from_fcf(
                fcf,
                0.0,
                settings.dcf_cost_of_equity,
                settings.dcf_short_growth,
                settings.dcf_projection_years,
                settings.dcf_terminal_growth,
            )
            for fcf in owner_earnings_ttm
        ],
        index=frame.index,
        dtype=float,
    )
    dcf_price = pd.concat(
        [
            (dcf_fcff / shares.replace(0, np.nan)).where(dcf_fcff.gt(0)),
            (dcf_owner / shares.replace(0, np.nan)).where(dcf_owner.gt(0)),
        ],
        axis=1,
    ).mean(axis=1)
    eps_growth = _growth_against_absolute(eps_ttm, 4)
    ebitda_growth = _growth_against_absolute(ebitda_ttm, 4)
    operating_margin = operating_income / revenue.replace(0, np.nan)
    return pd.DataFrame(
        {
            "FinancialPeriodEnd": frame["Date"],
            "FinancialAvailableDate": frame["Date"]
            + pd.to_timedelta(
                settings.financial_release_lag_days,
                unit="D",
            ),
            "RevenueGrowthYoY": revenue.pct_change(4, fill_method=None),
            "EpsGrowthYoY": _growth_against_absolute(eps, 4),
            "OperatingMarginChangeYoY": operating_margin.diff(4),
            "OperatingMargin": operating_margin,
            "FreeCashFlowMargin": (
                operating_cash + capex
            ) / revenue.replace(0, np.nan),
            "ReturnOnInvestment": _numeric(
                frame,
                ("ROI - Return On Investment",),
            ),
            "NetCashToAssets": (cash - debt) / assets.replace(0, np.nan),
            "Shares": shares,
            "Cash": cash,
            "TotalLiabilities": liabilities,
            "EpsTtm": eps_ttm,
            "EpsTtmGrowthYoY": eps_growth,
            "EpsTtmGrowthAcceleration": eps_growth.diff(),
            "EbitdaTtm": ebitda_ttm,
            "EbitdaTtmGrowthYoY": ebitda_growth,
            "EbitdaTtmGrowthAcceleration": ebitda_growth.diff(),
            "DcfPrice": dcf_price,
            "DcfPriceGrowthYoY": _positive_growth(dcf_price, 4),
        }
    ).sort_values("FinancialAvailableDate")


def _numeric(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in frame:
            result = pd.to_numeric(frame[name], errors="coerce")
            if result.notna().any():
                return result
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _growth_against_absolute(series: pd.Series, periods: int) -> pd.Series:
    prior = series.shift(periods)
    return (series - prior) / prior.abs().replace(0, np.nan)


def _positive_growth(series: pd.Series, periods: int) -> pd.Series:
    current = series.where(series.gt(0))
    prior = current.shift(periods)
    return current / prior.where(prior.gt(0)) - 1


def _centered_rank(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
    minimum_count: int,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").where(eligible)
    counts = values.notna().groupby(frame["Date"]).transform("sum")
    ranks = values.groupby(frame["Date"]).rank(
        pct=True,
        method="average",
    ) - 0.5
    return ranks.where(counts.ge(minimum_count))


def _row_mean(
    ranked: dict[str, pd.Series],
    names: tuple[str, ...],
) -> pd.Series:
    return pd.concat([ranked[name] for name in names], axis=1).mean(axis=1)


def _weighted_row_mean(
    ranked: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    values = pd.concat(
        [ranked[name].rename(name) for name in weights],
        axis=1,
    )
    available_weights = values.notna().mul(pd.Series(weights), axis=1)
    numerator = values.mul(pd.Series(weights), axis=1).sum(
        axis=1,
        min_count=1,
    )
    denominator = available_weights.sum(axis=1).replace(0, np.nan)
    return numerator / denominator
