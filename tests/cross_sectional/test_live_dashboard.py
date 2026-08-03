from __future__ import annotations

import json
import re

import pandas as pd

from stock_research.cross_sectional.live_dashboard import (
    ROW_COLUMNS,
    build_dashboard_payload,
    render_dashboard_html,
)


def test_payload_keeps_cash_missing_tickers_and_scores() -> None:
    signals = pd.DataFrame(
        [
            {
                "Date": "2026-07-24",
                "Ticker": "AAA",
                "Company": "Alpha",
                "Close": 100,
                "MarketRiskOn": False,
                "AlphaScore": 0.2,
                "BaseV7Score": 0.18,
                "FilingScore": 0.58,
                "Qualified": True,
                "Rank": 1,
                "TargetWeight": 0.0,
                "ModelSelected": False,
                "TradeAction": "SELL",
                "ExitReason": "UNIVERSE_EXIT",
            },
            {
                "Date": "2026-07-31",
                "Ticker": "AAA",
                "Company": "Alpha",
                "Close": 105,
                "MarketRiskOn": True,
                "AlphaScore": 0.25,
                "BaseV7Score": 0.23,
                "FilingScore": 0.63,
                "Qualified": True,
                "Rank": 1,
                "TargetWeight": 1.0,
                "ModelSelected": True,
                "SignalReferenceReturn": 0.05,
                "TradeAction": "BUY",
            },
        ]
    )
    membership = pd.DataFrame(
        [
            {
                "AsOfDate": "2026-07-01",
                "DataSymbol": "AAA",
                "Company": "Alpha",
                "Selected": True,
                "MembershipBucket": "TOP_N",
            },
            {
                "AsOfDate": "2026-07-01",
                "DataSymbol": "MISS",
                "Company": "Missing Corp",
                "Selected": True,
                "MembershipBucket": "WATCHLIST",
            },
        ]
    )
    readiness = pd.DataFrame(
        [
            {"Ticker": "AAA", "Company": "Alpha", "Status": "INCLUDED"},
            {
                "Ticker": "MISS",
                "Company": "Missing Corp",
                "Status": "MISSING_FINANCIALS",
            },
        ]
    )
    equity = pd.DataFrame(
        [
            {"Series": "V7_SEC_COMBINED_OPTIMIZED", "Date": "2026-07-24", "Equity": 100000},
            {"Series": "SPY_BUY_HOLD", "Date": "2026-07-24", "Equity": 100000},
            {"Series": "V7_SEC_COMBINED_OPTIMIZED", "Date": "2026-07-31", "Equity": 105000},
            {"Series": "SPY_BUY_HOLD", "Date": "2026-07-31", "Equity": 102000},
        ]
    )
    payload = build_dashboard_payload(
        signals,
        equity,
        membership,
        readiness,
        {"policy": {"filing_weight": 0.05, "top_k": 1}},
    )

    assert len(payload["scoreDates"]) == 2
    assert len(payload["scoreDates"][0]["rows"]) == 2
    assert payload["scoreDates"][0]["riskOn"] is False
    row = dict(zip(ROW_COLUMNS, payload["scoreDates"][1]["rows"][0]))
    assert row["Alpha"] == 0.25
    missing = [
        dict(zip(ROW_COLUMNS, item))
        for item in payload["scoreDates"][1]["rows"]
        if item[0] == "MISS"
    ][0]
    assert missing["Ready"] == "MISSING_FINANCIALS"
    assert missing["Alpha"] is None


def test_rendered_html_contains_parseable_payload() -> None:
    payload = {
        "generatedAt": None,
        "modelStatus": "TEST",
        "warning": None,
        "strategySeries": "model",
        "benchmarkSeries": "spy",
        "initialCapital": 100000,
        "filingWeight": 0.05,
        "topK": 3,
        "rowColumns": ROW_COLUMNS,
        "scoreDates": [{"date": "2026-07-31", "riskOn": True, "rows": []}],
        "curve": [["2026-07-31", 100000, 100000]],
    }
    html = render_dashboard_html(payload)
    match = re.search(
        r'<script id="dashboard-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    assert json.loads(match.group(1))["topK"] == 3
    assert "__DASHBOARD_DATA__" not in html
    assert "현금" in html
