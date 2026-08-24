from __future__ import annotations

"""Export weekly scores for the monthly-weight-reset dashboard model."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.signals import signal_day_panel
from stock_research.dashboard import engine
from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard.today import _atomic_json
from stock_research.paths import load_paths


_FILING_COVERAGE_FIELDS = {
    "FiledRevenueGrowthYoY": "revenue_growth",
    "FiledOperatingMargin": "operating_margin",
    "FiledFreeCashFlowMargin": "free_cash_flow_margin",
    "FiledNetDebtToAssets": "net_debt_to_assets",
    "FiledInterestCoverage": "interest_coverage",
    "FiledShareGrowthYoY": "share_growth",
}


def _add_rank_diagnostics(
    targets: pd.DataFrame,
    *,
    trend_floor: float,
    momentum_floor: float,
    minimum_filing_coverage: int,
) -> pd.DataFrame:
    frame = targets.copy()

    def missing_fields(row: pd.Series) -> list[str]:
        return [
            label
            for column, label in _FILING_COVERAGE_FIELDS.items()
            if pd.isna(row.get(column))
        ]

    def exclusion_reasons(row: pd.Series) -> list[str]:
        reasons: list[str] = []
        if not bool(row.get("Eligible", False)):
            reasons.append("PRICE_OR_TECHNICAL_DATA_INSUFFICIENT")
        trend = pd.to_numeric(row.get("Trend200"), errors="coerce")
        if pd.isna(trend) or float(trend) < trend_floor:
            reasons.append("TREND_200_BELOW_FLOOR")
        momentum = pd.to_numeric(row.get("Return126"), errors="coerce")
        if pd.isna(momentum) or float(momentum) < momentum_floor:
            reasons.append("RETURN_126_BELOW_FLOOR")
        if not bool(row.get("FilingCoveragePass", False)):
            reasons.append("SEC_COVERAGE_INSUFFICIENT")
        if not bool(row.get("FilingQualityPass", False)):
            reasons.append("SEC_QUALITY_BELOW_FLOOR")
        if not bool(row.get("FilingVetoPass", False)):
            reasons.append("SEC_CRITICAL_FLAG")
        return reasons

    frame["MissingSecFields"] = frame.apply(missing_fields, axis=1)
    frame["ExclusionReasons"] = frame.apply(exclusion_reasons, axis=1)
    frame["MinimumFilingCoverage"] = minimum_filing_coverage
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Alpha Desk weekly scores from a fixed start date.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--weeks", type=int, default=53)
    parser.add_argument("--stock-root")
    return parser.parse_args()


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return strict JSON-friendly records without NaN/Infinity values."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _latest_file(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _macro_payload(folder: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    series: dict[str, list[dict[str, object]]] = {}

    def add_value_file(name: str, pattern: str, *, weekly: bool = False) -> None:
        path = _latest_file(folder, pattern)
        if path is None:
            return
        frame = pd.read_csv(path)
        if not {"Date", "Value"} <= set(frame):
            return
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Value"]).sort_values("Date")
        frame = frame[frame["Date"].between(start, end)]
        if weekly:
            frame = frame.set_index("Date").resample("W-FRI").last().dropna().reset_index()
        frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
        series[name] = _records(frame[["Date", "Value"]])

    add_value_file("VIX", "* VIX.csv", weekly=True)
    add_value_file("FedFunds", "* FedFundsRate.csv")
    add_value_file("WTI", "* WTI.csv")
    add_value_file("YieldCurve", "* YieldCurve.csv")

    fred_path = _latest_file(folder, "*FRED Research Macro.csv")
    if fred_path is not None:
        fred = pd.read_csv(fred_path)
        fred["Date"] = pd.to_datetime(fred["Date"], errors="coerce")
        fred = fred[fred["Date"].between(start, end)].sort_values("Date")
        for source, target in (("HYOAS", "HighYieldSpread"), ("GS10", "Treasury10Y"), ("T10Y2Y", "YieldCurve"), ("NFCI", "FinancialConditions"), ("VIX3M", "VIX3M")):
            if source not in fred:
                continue
            compact = fred[["Date", source]].rename(columns={source: "Value"}).dropna()
            compact["Date"] = compact["Date"].dt.strftime("%Y-%m-%d")
            series[target] = _records(compact)

    spy_path = _latest_file(folder, "SPY Adjusted Historical Data.csv")
    if spy_path is not None:
        spy = pd.read_csv(spy_path)
        price_column = "Adj Close" if "Adj Close" in spy else "Close"
        if {"Date", price_column} <= set(spy):
            spy["Date"] = pd.to_datetime(spy["Date"], errors="coerce")
            spy[price_column] = pd.to_numeric(spy[price_column], errors="coerce")
            spy = spy.dropna(subset=["Date", price_column]).sort_values("Date")
            spy["Value"] = spy[price_column].pct_change().rolling(21).std() * (252 ** 0.5) * 100
            spy = spy[spy["Date"].between(start, end)].set_index("Date").resample("W-FRI").last().dropna().reset_index()
            spy["Date"] = spy["Date"].dt.strftime("%Y-%m-%d")
            series["RealizedVolatility"] = _records(spy[["Date", "Value"]])

    return {
        "as_of": max((rows[-1]["Date"] for rows in series.values() if rows), default=None),
        "series": series,
    }


def _attach_price_dependent_ratios(ticker_filings: pd.DataFrame, full_panel: pd.DataFrame) -> pd.DataFrame:
    """EV/EBIT, FCFF/EV, Altman Z-Score -- the ratios that need a market
    price on top of what sec_filings.py can compute from filings alone
    (DcfValue/EpsTtm/Fcff/Roic are price-independent and already computed
    upstream). Display-only; never fed into signals.py/portfolio.py.
    """
    if ticker_filings.empty:
        return ticker_filings
    result = ticker_filings.sort_values("AvailableDate").copy()
    prices = full_panel[["Date", "Close"]].dropna(subset=["Date"]).sort_values("Date")
    result = pd.merge_asof(
        result, prices, left_on="AvailableDate", right_on="Date", direction="backward"
    )
    market_cap = result["Close"] * result["SharesOutstandingSplitAdjusted"]
    enterprise_value = market_cap + result["TotalDebt"] - result["Cash"]
    result["EvEbit"] = (enterprise_value / result["EbitTtm"]).where(result["EbitTtm"] > 0)
    result["FcffToEv"] = (result["FcffTtm"] / enterprise_value).where(enterprise_value > 0)
    result["AltmanZ"] = (
        1.2 * (result["WorkingCapital"] / result["Assets"])
        + 1.4 * (result["RetainedEarnings"] / result["Assets"])
        + 3.3 * (result["EbitTtm"] / result["Assets"])
        + 0.6 * (market_cap / result["Liabilities"])
        + 1.0 * (result["RevenueTtm"] / result["Assets"])
    )
    return result.drop(columns=["Date", "Close"])


def _export_reports(
    output: Path,
    factored: pd.DataFrame,
    filings: pd.DataFrame,
    members: list[object],
    macro_folder: Path,
    history_start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    company_lookup = {
        str(getattr(item, "ticker", item)): str(getattr(item, "company", getattr(item, "ticker", item)))
        for item in members
    }
    report_payload: dict[str, object] = {
        "generated_on": date.today().isoformat(),
        "market_as_of": str(end.date()),
        "macro": _macro_payload(macro_folder, history_start, end),
        "reports": {},
    }
    reports: dict[str, object] = report_payload["reports"]  # type: ignore[assignment]
    price_columns = ["Date", "Close"]
    snapshot_columns = [
        "Close", "Return21", "Return63", "Return126", "RevenueGrowthYoY",
        "EpsGrowthYoY", "OperatingMargin", "FreeCashFlowMargin",
        "ReturnOnInvestment", "NetCashToAssets", "Cash", "TotalLiabilities",
        "EpsTtm", "EpsTtmGrowthYoY", "EbitdaTtm", "EbitdaTtmGrowthYoY",
        "PeTtm", "EvEbitdaTtm", "DcfPrice", "DcfUpside",
    ]
    financial_columns = [
        "FinancialPeriodEnd", "FinancialAvailableDate", "RevenueGrowthYoY",
        "EpsGrowthYoY", "OperatingMargin", "FreeCashFlowMargin",
        "ReturnOnInvestment", "NetCashToAssets", "Cash", "TotalLiabilities",
        "EpsTtm", "EpsTtmGrowthYoY", "EbitdaTtm", "EbitdaTtmGrowthYoY",
        "PeTtm", "EvEbitdaTtm", "DcfPrice", "DcfUpside",
    ]
    filing_columns = [
        "AvailableDate", "PeriodOfReport", "Form", "FiscalYear", "FiscalPeriod",
        "Revenue", "RevenueGrowthYoYFiled", "Ebitda", "EbitdaMargin",
        "EbitdaGrowthYoYFiled", "NetIncome", "NetIncomeGrowthYoYFiled",
        "GrossMargin", "OperatingMargin", "FreeCashFlow", "FreeCashFlowMargin",
        "Cash", "TotalDebt", "Liabilities", "DebtToEquity", "NetDebtToAssets",
        "InterestCoverage", "ResearchAndDevelopmentToRevenue",
        "StockCompensationToRevenue", "ShareGrowthYoYFiled",
        # Display-only valuation ratios (never fed into signals/scoring).
        # EpsTtm/Fcff/FcffTtm/Roic/DcfValue/DcfAssumed* and the growth
        # columns are pure-filing computations from sec_filings.py;
        # EvEbit/FcffToEv/AltmanZ are finished below since they
        # additionally need the market price.
        "EpsTtm", "Fcff", "FcffTtm", "FcffGrowthYoYFiled", "Roic",
        "DcfValue", "DcfAssumedGrowth", "DcfAssumedTerminalGrowth",
        "DcfAssumedDiscountRate", "EvEbit", "FcffToEv", "AltmanZ",
        "Sic", "SicDescription",
    ]
    factored = factored.copy()
    factored["Date"] = pd.to_datetime(factored["Date"])
    filings = filings.copy()
    filings["AvailableDate"] = pd.to_datetime(filings["AvailableDate"], errors="coerce")
    for ticker in company_lookup:
        full_panel = factored[factored["Ticker"] == ticker].sort_values("Date").copy()
        close_series = pd.to_numeric(full_panel["Close"], errors="coerce")
        delta = close_series.diff()
        average_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        relative_strength = average_gain / average_loss.replace(0, pd.NA)
        full_panel["RSI14"] = 100 - (100 / (1 + relative_strength))
        for window in (20, 50, 200):
            full_panel[f"MA{window}"] = close_series.rolling(window, min_periods=window).mean()
        volume = pd.to_numeric(full_panel.get("Volume"), errors="coerce").fillna(0)
        direction = close_series.diff().apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        full_panel["OBV"] = (direction * volume).cumsum()
        full_panel["OBV20Change"] = full_panel["OBV"].diff(20)
        panel = full_panel[full_panel["Date"].between(history_start, end)].copy()
        if panel.empty:
            continue
        latest = panel.iloc[-1]
        snapshot = {column: latest[column] for column in snapshot_columns if column in panel.columns}
        snapshot.update({
            column: latest.get(column)
            for column in ("RSI14", "MA20", "MA50", "MA200", "OBV", "OBV20Change")
        })
        snapshot["TrendVsMA200"] = (
            "ABOVE" if pd.notna(latest.get("MA200")) and latest["Close"] >= latest["MA200"] else "BELOW"
        )
        snapshot["Cross50x200"] = (
            "GOLDEN" if pd.notna(latest.get("MA50")) and pd.notna(latest.get("MA200")) and latest["MA50"] >= latest["MA200"] else "DEATH"
        )
        prices = panel[[column for column in price_columns if column in panel.columns]].copy()
        prices["Date"] = prices["Date"].dt.strftime("%Y-%m-%d")
        technical = panel[["Date", "Close", "RSI14", "MA20", "MA50", "MA200", "OBV"]].copy()
        technical = technical.iloc[::5].copy()
        if not technical.empty and technical.iloc[-1]["Date"] != panel.iloc[-1]["Date"]:
            technical = pd.concat([technical, panel.iloc[[-1]][technical.columns]], ignore_index=True)
        technical["Date"] = pd.to_datetime(technical["Date"]).dt.strftime("%Y-%m-%d")
        available_financial = [column for column in financial_columns if column in panel.columns]
        financial = panel[available_financial].copy()
        if "FinancialPeriodEnd" in financial:
            financial["FinancialPeriodEnd"] = pd.to_datetime(financial["FinancialPeriodEnd"], errors="coerce").dt.strftime("%Y-%m-%d")
            financial = financial.dropna(subset=["FinancialPeriodEnd"]).drop_duplicates("FinancialPeriodEnd", keep="last").tail(12)
        ticker_filings = filings[filings["Ticker"].astype(str) == ticker].sort_values("AvailableDate").copy()
        ticker_filings = _attach_price_dependent_ratios(ticker_filings, full_panel)
        available_filing = [column for column in filing_columns if column in ticker_filings.columns]
        ticker_filings = ticker_filings[available_filing].drop_duplicates(["AvailableDate", "Form"], keep="last").tail(12)
        for date_column in ("AvailableDate", "PeriodOfReport"):
            if date_column in ticker_filings:
                ticker_filings[date_column] = pd.to_datetime(ticker_filings[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
        raw_ticker_filings = filings[filings["Ticker"].astype(str) == ticker].sort_values("AvailableDate")
        if not raw_ticker_filings.empty:
            annual = raw_ticker_filings[raw_ticker_filings["PeriodKind"].astype(str) == "ANNUAL"]
            reference = annual.iloc[-1] if not annual.empty else raw_ticker_filings.iloc[-1]
            diluted_shares = pd.to_numeric(pd.Series([reference.get("DilutedShares")]), errors="coerce").iloc[0]
            net_income = pd.to_numeric(pd.Series([reference.get("NetIncome")]), errors="coerce").iloc[0]
            eps = net_income / diluted_shares if pd.notna(net_income) and pd.notna(diluted_shares) and diluted_shares else None
            close = pd.to_numeric(pd.Series([latest.get("Close")]), errors="coerce").iloc[0]
            snapshot.update({
            "ReferencePeriod": reference.get("PeriodOfReport"),
            "Revenue": reference.get("Revenue"),
            "RevenueGrowthYoY": reference.get("RevenueGrowthYoYFiled"),
            "Ebitda": reference.get("Ebitda"),
            "EbitdaGrowthYoY": reference.get("EbitdaGrowthYoYFiled"),
            "NetIncome": reference.get("NetIncome"),
            "NetIncomeGrowthYoY": reference.get("NetIncomeGrowthYoYFiled"),
            "Eps": eps,
            "Pe": close / eps if eps is not None and eps > 0 and pd.notna(close) else None,
            "GrossMargin": reference.get("GrossMargin"),
            "OperatingMargin": reference.get("OperatingMargin"),
            "FreeCashFlow": reference.get("FreeCashFlow"),
            "FreeCashFlowMargin": reference.get("FreeCashFlowMargin"),
            "Cash": reference.get("Cash"),
            "TotalDebt": reference.get("TotalDebt"),
            "Liabilities": reference.get("Liabilities"),
            "DebtToEquity": reference.get("DebtToEquity"),
            "InterestCoverage": reference.get("InterestCoverage"),
            })
        sic = ""
        sic_description = ""
        if not raw_ticker_filings.empty:
            sic = str(raw_ticker_filings.iloc[-1].get("Sic", "") or "")
            sic_description = str(raw_ticker_filings.iloc[-1].get("SicDescription", "") or "")
        reports[ticker] = {
            "ticker": ticker,
            "company": str(company_lookup.get(ticker, ticker)),
            "as_of": latest["Date"].strftime("%Y-%m-%d"),
            "sic": sic,
            "sic_description": sic_description,
            "latest": _records(pd.DataFrame([snapshot]))[0],
            "price_history": _records(prices),
            "technical_history": _records(technical),
            "financial_history": _records(financial),
            "filing_history": _records(ticker_filings),
        }
    # Peer group: every OTHER ticker in this universe sharing the same
    # 4-digit SIC industry code (SEC's own classification, already cached
    # on every submissions.json -- see sync_sec_filings). Display-only
    # grouping for the frontend's peer-comparison view, not used anywhere
    # in scoring.
    sic_by_ticker = {t: reports[t]["sic"] for t in reports if reports[t].get("sic")}
    for ticker, info in reports.items():
        info["peers"] = sorted(
            other for other, sic in sic_by_ticker.items()
            if other != ticker and sic == sic_by_ticker.get(ticker)
        )
    _atomic_json(output.resolve(), report_payload)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    state = dashboard_state.load_state(paths)
    end = pd.Timestamp(args.end)
    history_start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(weeks=args.weeks)
    warmup_start = history_start - pd.Timedelta(days=engine.WARMUP_BUFFER_DAYS)
    members = engine._members(paths, state)
    settings = engine._settings(
        str(warmup_start.date()),
        str((history_start - pd.Timedelta(days=1)).date()),
        str(history_start.date()),
        str(end.date()),
        min(8, max(2, len(members) // 2)),
    )
    filing_path = engine._ensure_filing_data(paths, state)
    filings = pd.read_csv(filing_path)
    factored = engine._build_factored_panel(
        members,
        settings,
        filings,
        warmup_start=str(warmup_start.date()),
    )
    weekly = signal_day_panel(factored, str(history_start.date()), str(end.date()), settings.rebalance_weekday)
    base_params = engine._base_params()
    policy = engine._policy(state.top_k)
    _, targets = generate_filing_v7_targets(weekly, base_params, policy)
    targets = _add_rank_diagnostics(
        targets,
        trend_floor=base_params.trend_floor,
        momentum_floor=base_params.momentum_floor,
        minimum_filing_coverage=policy.minimum_filing_coverage,
    )
    columns = {
        "Date": "date",
        "Ticker": "ticker",
        "AlphaScore": "final_score",
        "BaseV7Score": "v7_score",
        "FilingScore": "filing_score",
        "Rank": "rank",
        "TargetWeight": "target_weight",
        "Eligible": "eligible",
        "Trend200": "trend_200",
        "Return126": "return_126",
        "FilingCoverageCount": "filing_coverage_count",
        "MinimumFilingCoverage": "minimum_filing_coverage",
        "FilingCoveragePass": "filing_coverage_pass",
        "FilingQualityPass": "filing_quality_pass",
        "FilingVetoPass": "filing_veto_pass",
        "Qualified": "qualified",
        "MissingSecFields": "missing_sec_fields",
        "ExclusionReasons": "exclusion_reasons",
    }
    available = [column for column in columns if column in targets.columns]
    compact = targets[available].rename(columns=columns).copy()
    compact["date"] = pd.to_datetime(compact["date"]).dt.date.astype(str)
    compact = compact.sort_values(["ticker", "date"])
    records = json.loads(compact.to_json(orient="records"))
    payload = {
        "model": (
            "Filing V7-3 + SEC 25% · Top 5 · Weekly signals · "
            "Monthly first-signal weight reset"
        ),
        "generated_on": date.today().isoformat(),
        "start": str(history_start.date()),
        "end": str(end.date()),
        "universe_count": len(state.tickers),
        "universe": [
            {"ticker": item.ticker, "company": item.company, "source": item.source}
            for item in state.tickers
        ],
        "records": records,
    }
    _atomic_json(args.output.resolve(), payload)
    if args.report_output:
        _export_reports(args.report_output, factored, filings, members, paths.macro, history_start, end)
    print(json.dumps({"output": str(args.output.resolve()), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
