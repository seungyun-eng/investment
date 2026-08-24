from __future__ import annotations

import pandas as pd
import pytest

from stock_research.cross_sectional.sec_filings import (
    SecFilingSettings,
    add_derived_filing_metrics,
    analyze_filing_document,
    build_ticker_cik_map,
    cached_submission_cik,
    extract_standardized_facts,
    filter_filings,
    latest_filing_has_no_core_facts,
    select_filing_fact,
    split_adjust_share_series,
    submission_frame,
    validate_user_agent,
)


def test_latest_filing_without_core_facts_is_detected() -> None:
    metrics = pd.DataFrame(
        {
            "AvailableDate": ["2026-04-30", "2026-07-30"],
            "Revenue": [100.0, float("nan")],
            "OperatingIncome": [20.0, float("nan")],
            "NetIncome": [15.0, float("nan")],
            "Assets": [500.0, float("nan")],
        }
    )
    assert latest_filing_has_no_core_facts(metrics) is True


def test_latest_filing_with_any_core_fact_is_not_marked_empty() -> None:
    metrics = pd.DataFrame(
        {
            "AvailableDate": ["2026-07-30"],
            "Revenue": [60_801_000_000.0],
            "OperatingIncome": [float("nan")],
            "NetIncome": [float("nan")],
            "Assets": [float("nan")],
        }
    )
    assert latest_filing_has_no_core_facts(metrics) is False


def test_standard_capex_and_interest_tag_alternatives_are_extracted() -> None:
    observation = {
        "start": "2026-01-01",
        "end": "2026-03-31",
        "val": 10.0,
        "accn": "test-accession",
        "fy": 2026,
        "fp": "Q1",
        "form": "10-Q",
        "filed": "2026-05-01",
    }
    payload = {
        "facts": {
            "us-gaap": {
                "PaymentsToAcquireProductiveAssets": {
                    "units": {"USD": [observation]}
                },
                "PaymentsToAcquireOtherPropertyPlantAndEquipment": {
                    "units": {"USD": [{**observation, "val": 11.0}]}
                },
                "InterestExpenseNonoperating": {
                    "units": {"USD": [{**observation, "val": 2.0}]}
                },
            }
        }
    }

    result = extract_standardized_facts(payload)
    assert set(result.loc[result["Metric"].eq("CapitalExpenditures"), "Concept"]) == {
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    }
    assert result.loc[result["Metric"].eq("InterestExpense"), "Concept"].tolist() == [
        "InterestExpenseNonoperating"
    ]


def test_standard_debt_interest_and_basic_share_fallbacks_are_extracted() -> None:
    instant = {
        "end": "2026-03-31",
        "val": 10.0,
        "accn": "test-accession",
        "fy": 2026,
        "fp": "Q1",
        "form": "10-Q",
        "filed": "2026-05-01",
    }
    duration = {**instant, "start": "2026-01-01"}
    payload = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {"units": {"USD": [instant]}},
                "ConvertibleDebtNoncurrent": {
                    "units": {"USD": [{**instant, "val": 20.0}]}
                },
                "InterestExpenseDebt": {
                    "units": {"USD": [{**duration, "val": 2.0}]}
                },
                "WeightedAverageNumberOfSharesOutstandingBasic": {
                    "units": {"shares": [{**duration, "val": 100.0}]}
                },
            }
        }
    }

    result = extract_standardized_facts(payload)
    assert set(result["Metric"]) == {
        "DebtCurrent",
        "DebtNoncurrent",
        "InterestExpense",
        "SharesOutstanding",
    }


def test_ifrs_classified_capex_and_borrowing_tags_are_extracted() -> None:
    instant = {
        "end": "2025-12-31",
        "val": 10.0,
        "accn": "test-accession",
        "fy": 2025,
        "fp": "FY",
        "form": "20-F",
        "filed": "2026-02-04",
    }
    payload = {
        "facts": {
            "ifrs-full": {
                "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": {
                    "units": {"DKK": [{**instant, "start": "2025-01-01"}]}
                },
                "CurrentPortionOfLongtermBorrowings": {
                    "units": {"DKK": [instant]}
                },
                "LongtermBorrowings": {
                    "units": {"DKK": [{**instant, "val": 20.0}]}
                },
            }
        }
    }

    result = extract_standardized_facts(payload)
    assert set(result["Metric"]) == {
        "CapitalExpenditures",
        "DebtCurrent",
        "DebtNoncurrent",
    }


def test_share_growth_labels_weighted_average_basic_fallback() -> None:
    frame = pd.DataFrame(
        {
            "AvailableDate": pd.to_datetime(["2024-05-02", "2025-05-02"]),
            "PeriodKind": ["QUARTERLY", "QUARTERLY"],
            "FiscalPeriod": ["Q1", "Q1"],
            "FiscalYear": [2024, 2025],
            "Revenue": [100.0, 100.0],
            "GrossProfit": [50.0, 50.0],
            "OperatingIncome": [20.0, 20.0],
            "NetIncome": [10.0, 10.0],
            "OperatingCashFlow": [25.0, 25.0],
            "CapitalExpenditures": [5.0, 5.0],
            "Cash": [30.0, 30.0],
            "Assets": [200.0, 200.0],
            "Equity": [100.0, 100.0],
            "DebtCurrent": [5.0, 5.0],
            "DebtNoncurrent": [45.0, 45.0],
            "InterestExpense": [2.0, 2.0],
            "ResearchAndDevelopment": [10.0, 10.0],
            "ShareBasedCompensation": [3.0, 3.0],
            "SharesOutstanding": [100.0, 98.0],
            "SharesOutstandingConcept": [
                "WeightedAverageNumberOfSharesOutstandingBasic",
                "WeightedAverageNumberOfSharesOutstandingBasic",
            ],
            "OperatingCashFlowDurationDays": [90.0, 90.0],
            "RevenueDurationDays": [90.0, 90.0],
        }
    )

    result = add_derived_filing_metrics(frame)
    assert result.iloc[-1]["ShareGrowthYoYFiled"] == pytest.approx(-0.02)
    assert result.iloc[-1]["ShareGrowthBasis"] == "WEIGHTED_AVERAGE_BASIC"


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


def test_split_adjust_share_series_folds_a_forward_split_backward() -> None:
    # A clean 10-for-1 split between the 3rd and 4th observations, with no
    # other issuance/buybacks: everything before the split should be
    # rescaled onto the same (post-split) basis as the tail of the series.
    raw = pd.Series([95.0, 97.0, 100.0, 1000.0, 1010.0])
    adjusted = split_adjust_share_series(raw)
    assert adjusted.tolist() == pytest.approx([950.0, 970.0, 1000.0, 1000.0, 1010.0])


def test_split_adjust_share_series_leaves_normal_issuance_alone() -> None:
    # A 5% YoY issuance is nowhere near a split ratio and must pass through.
    raw = pd.Series([100.0, 105.0])
    assert split_adjust_share_series(raw).tolist() == pytest.approx([100.0, 105.0])


def test_derived_metrics_share_growth_ignores_a_stock_split() -> None:
    # Regression: a raw (unadjusted) 10-for-1 split used to read as +900%
    # "dilution" in ShareGrowthYoYFiled, wrongly tanking a ticker's
    # FilingBalanceSheetFactor for the quarter the split fell in even though
    # nothing about real share issuance changed.
    frame = pd.DataFrame(
        {
            "AvailableDate": pd.to_datetime(["2023-05-02", "2024-05-02"]),
            "PeriodKind": ["QUARTERLY", "QUARTERLY"],
            "FiscalPeriod": ["Q1", "Q1"],
            "FiscalYear": [2023, 2024],
            "Revenue": [100.0, 120.0],
            "GrossProfit": [50.0, 60.0],
            "OperatingIncome": [20.0, 24.0],
            "NetIncome": [10.0, 12.0],
            "OperatingCashFlow": [25.0, 30.0],
            "CapitalExpenditures": [5.0, 6.0],
            "Cash": [30.0, 35.0],
            "Assets": [200.0, 220.0],
            "Equity": [100.0, 110.0],
            "DebtCurrent": [5.0, 5.0],
            "DebtNoncurrent": [45.0, 45.0],
            "InterestExpense": [2.0, 2.0],
            "ResearchAndDevelopment": [10.0, 11.0],
            "ShareBasedCompensation": [3.0, 3.5],
            "SharesOutstanding": [100.0, 1000.0],  # pure 10-for-1 split, no real dilution
            "OperatingCashFlowDurationDays": [90.0, 90.0],
            "RevenueDurationDays": [90.0, 90.0],
        }
    )
    result = add_derived_filing_metrics(frame)
    latest = result.iloc[-1]
    assert latest["ShareGrowthYoYFiled"] == pytest.approx(0.0, abs=1e-9)


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


def test_cached_submission_cik_recovers_validated_missing_ticker(tmp_path) -> None:
    ticker_root = tmp_path / "AEP"
    ticker_root.mkdir()
    (ticker_root / "submissions.json").write_text(
        '{"cik":"0000004904","tickers":["AEP"]}', encoding="utf-8"
    )

    assert cached_submission_cik(ticker_root, "AEP") == "0000004904"


def test_cached_submission_cik_rejects_mismatched_ticker(tmp_path) -> None:
    ticker_root = tmp_path / "AEP"
    ticker_root.mkdir()
    (ticker_root / "submissions.json").write_text(
        '{"cik":"0000004904","tickers":["NOT-AEP"]}', encoding="utf-8"
    )

    assert cached_submission_cik(ticker_root, "AEP") is None
