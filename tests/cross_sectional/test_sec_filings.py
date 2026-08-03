from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.sec_filings import (
    SecFilingSettings,
    add_derived_filing_metrics,
    analyze_filing_document,
    build_ticker_cik_map,
    filter_filings,
    select_filing_fact,
    submission_frame,
    validate_user_agent,
)


def test_sec_user_agent_requires_contact_email() -> None:
    with pytest.raises(ValueError, match="contact email"):
        validate_user_agent("anonymous bot")
    assert (
        validate_user_agent("Personal Research owner@example.com")
        == "Personal Research owner@example.com"
    )


def test_submission_filter_uses_next_day_and_excludes_amendments() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["A", "B", "C"],
                "filingDate": ["2025-02-01", "2025-02-02", "2018-01-01"],
                "reportDate": ["2024-12-31", "2024-12-31", "2017-12-31"],
                "acceptanceDateTime": [
                    "2025-02-01T21:01:00.000Z",
                    "2025-02-02T12:00:00.000Z",
                    "2018-01-01T12:00:00.000Z",
                ],
                "form": ["10-K", "10-K/A", "10-K"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                "primaryDocDescription": ["Annual", "Amended", "Old"],
            }
        }
    }
    frame = submission_frame(payload)
    selected = filter_filings(frame, SecFilingSettings(start_date="2019-01-01"))
    assert selected["AccessionNumber"].tolist() == ["A"]
    assert selected.iloc[0]["AvailableDate"] == pd.Timestamp("2025-02-02")


def test_select_filing_fact_prefers_report_end_and_quarter_duration() -> None:
    facts = pd.DataFrame(
        {
            "Value": [100.0, 300.0, 90.0],
            "End": pd.to_datetime(["2025-03-31", "2025-03-31", "2024-03-31"]),
            "DurationDays": [90.0, 180.0, 90.0],
            "ConceptPriority": [0, 0, 0],
            "FiledDate": pd.to_datetime(["2025-05-01"] * 3),
            "Concept": ["Revenue"] * 3,
            "FiscalYear": [2025] * 3,
            "FiscalPeriod": ["Q1"] * 3,
        }
    )
    selected = select_filing_fact(
        facts,
        metric="Revenue",
        period_end=pd.Timestamp("2025-03-31"),
        target_duration_days=91,
    )
    assert selected["Value"] == pytest.approx(100.0)


def test_derived_metrics_compute_growth_leverage_and_dilution() -> None:
    frame = pd.DataFrame(
        {
            "AvailableDate": pd.to_datetime(["2024-05-02", "2025-05-02"]),
            "PeriodKind": ["QUARTERLY", "QUARTERLY"],
            "FiscalPeriod": ["Q1", "Q1"],
            "FiscalYear": [2024, 2025],
            "Revenue": [100.0, 120.0],
            "GrossProfit": [50.0, 66.0],
            "OperatingIncome": [20.0, 30.0],
            "NetIncome": [10.0, 15.0],
            "OperatingCashFlow": [25.0, 36.0],
            "CapitalExpenditures": [5.0, 6.0],
            "Cash": [30.0, 40.0],
            "Assets": [200.0, 240.0],
            "Equity": [100.0, 120.0],
            "DebtCurrent": [5.0, 6.0],
            "DebtNoncurrent": [45.0, 42.0],
            "InterestExpense": [2.0, 2.0],
            "ResearchAndDevelopment": [10.0, 12.0],
            "ShareBasedCompensation": [3.0, 4.0],
            "SharesOutstanding": [100.0, 105.0],
            "OperatingCashFlowDurationDays": [90.0, 90.0],
            "RevenueDurationDays": [90.0, 90.0],
        }
    )
    result = add_derived_filing_metrics(frame)
    latest = result.iloc[-1]
    assert latest["RevenueGrowthYoYFiled"] == pytest.approx(0.20)
    assert latest["ShareGrowthYoYFiled"] == pytest.approx(0.05)
    assert latest["OperatingMargin"] == pytest.approx(0.25)
    assert latest["FreeCashFlowMargin"] == pytest.approx(0.25)
    assert latest["NetDebtToAssets"] == pytest.approx((48 - 40) / 240)


def test_text_features_are_auditable_counts_not_sentiment_labels() -> None:
    html = b"""
    <html><body><h1>Risk Factors</h1>
    We expect stronger demand, but competition creates uncertainty.
    We identified a material weakness. Stock-based compensation may cause dilution.
    </body></html>
    """
    features = analyze_filing_document(html)
    assert features["RiskTermsCount"] >= 2
    assert features["CompetitionTermsCount"] == 1
    assert features["DilutionTermsCount"] == 2
    assert bool(features["MaterialWeaknessFlag"])


def test_red_flags_exclude_negated_and_hypothetical_boilerplate() -> None:
    html = b"""
    <html><body>
    We did not identify any material weaknesses. Future control failures may
    result in a restatement of our financial statements. Conditions do not
    raise substantial doubt about our ability to continue as a going concern.
    </body></html>
    """
    features = analyze_filing_document(html)
    assert not bool(features["MaterialWeaknessFlag"])
    assert not bool(features["RestatementFlag"])
    assert not bool(features["GoingConcernFlag"])


def test_red_flags_detect_explicit_admissions() -> None:
    html = b"""
    <html><body>
    Management identified a material weakness in internal control. We are
    required to restate our previously issued financial statements. These
    conditions raise substantial doubt about our ability to continue as a
    going concern.
    </body></html>
    """
    features = analyze_filing_document(html)
    assert bool(features["MaterialWeaknessFlag"])
    assert bool(features["RestatementFlag"])
    assert bool(features["GoingConcernFlag"])


def test_ticker_map_normalizes_sec_share_class_separator() -> None:
    payload = {
        "0": {"ticker": "BRK.B", "cik_str": 1067983},
        "1": {"ticker": "AAPL", "cik_str": 320193},
    }
    result = build_ticker_cik_map(payload)
    assert result["BRK-B"] == "0001067983"
    assert result["AAPL"] == "0000320193"
