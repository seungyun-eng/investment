from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSION_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC_COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)
SEC_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
)

DEFAULT_FORMS = ("10-K", "10-Q", "20-F", "40-F", "6-K")
TEXT_ANALYSIS_VERSION = "3"

# Ordered concepts: the first available standard tag is preferred. Extension
# concepts are intentionally excluded because their meaning is issuer-specific.
STANDARD_CONCEPTS: dict[str, tuple[tuple[str, str], ...]] = {
    "Revenue": (
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("ifrs-full", "Revenue"),
    ),
    "GrossProfit": (
        ("us-gaap", "GrossProfit"),
        ("ifrs-full", "GrossProfit"),
    ),
    "OperatingIncome": (
        ("us-gaap", "OperatingIncomeLoss"),
        ("ifrs-full", "ProfitLossFromOperatingActivities"),
    ),
    "NetIncome": (
        ("us-gaap", "NetIncomeLoss"),
        # Some GAAP filers (e.g. AVGO) tag bottom-line net income under
        # ProfitLoss instead of NetIncomeLoss -- without this fallback their
        # NetIncome (and everything derived from it) is silently mostly-NaN.
        ("us-gaap", "ProfitLoss"),
        ("ifrs-full", "ProfitLoss"),
    ),
    "OperatingCashFlow": (
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
    ),
    "CapitalExpenditures": (
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        # SEC filers use these standard US-GAAP alternatives for the same
        # cash-flow statement concept. AMZN and V use ProductiveAssets;
        # LLY uses OtherPropertyPlantAndEquipment in recent filings.
        ("us-gaap", "PaymentsToAcquireProductiveAssets"),
        ("us-gaap", "PaymentsToAcquireOtherPropertyPlantAndEquipment"),
        ("ifrs-full", "PurchaseOfPropertyPlantAndEquipment"),
        (
            "ifrs-full",
            "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        ),
    ),
    "Cash": (
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("ifrs-full", "CashAndCashEquivalents"),
    ),
    "Assets": (
        ("us-gaap", "Assets"),
        ("ifrs-full", "Assets"),
    ),
    "Liabilities": (
        ("us-gaap", "Liabilities"),
        ("ifrs-full", "Liabilities"),
    ),
    "Equity": (
        ("us-gaap", "StockholdersEquity"),
        (
            "us-gaap",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        ("ifrs-full", "Equity"),
    ),
    "DebtCurrent": (
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent"),
        ("us-gaap", "LongTermDebtCurrent"),
        ("us-gaap", "DebtCurrent"),
        ("us-gaap", "ConvertibleDebtCurrent"),
        ("ifrs-full", "CurrentBorrowings"),
        ("ifrs-full", "CurrentPortionOfLongtermBorrowings"),
        ("ifrs-full", "ShorttermBorrowings"),
    ),
    "DebtNoncurrent": (
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"),
        ("us-gaap", "LongTermDebtNoncurrent"),
        ("us-gaap", "ConvertibleDebtNoncurrent"),
        ("us-gaap", "LongTermDebt"),
        ("ifrs-full", "NoncurrentBorrowings"),
        ("ifrs-full", "LongtermBorrowings"),
    ),
    "InterestExpense": (
        # This is the SEC taxonomy's exact spelling (lowercase "o" in
        # Nonoperating). Keep the legacy variant below for cached/vendor
        # payloads that may have normalized the name differently.
        ("us-gaap", "InterestExpenseNonoperating"),
        ("us-gaap", "InterestExpenseNonOperating"),
        ("us-gaap", "InterestExpense"),
        ("us-gaap", "InterestExpenseDebt"),
        ("us-gaap", "InterestExpenseBorrowings"),
        ("ifrs-full", "FinanceCosts"),
    ),
    "ResearchAndDevelopment": (
        ("us-gaap", "ResearchAndDevelopmentExpense"),
        ("ifrs-full", "ResearchAndDevelopmentExpense"),
    ),
    "ShareBasedCompensation": (
        ("us-gaap", "ShareBasedCompensation"),
        ("ifrs-full", "ShareBasedPayment"),
    ),
    "SharesOutstanding": (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        # Some issuers do not publish a consolidated point-in-time share fact
        # because the XBRL is split by share class. For YoY dilution/buyback
        # measurement, basic weighted-average shares are the closest standard
        # non-diluted fallback. The selected basis is exposed below rather than
        # silently mixing the two definitions.
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
        ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"),
        ("ifrs-full", "WeightedAverageShares"),
    ),
    "DilutedShares": (
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("ifrs-full", "DilutedWeightedAverageShares"),
    ),
    "DepreciationAndAmortization": (
        ("us-gaap", "DepreciationDepletionAndAmortization"),
        ("us-gaap", "DepreciationAmortizationAndAccretionNet"),
        ("us-gaap", "DepreciationAndAmortization"),
        ("ifrs-full", "DepreciationAndAmortisationExpense"),
    ),
    # Many large issuers (AVGO, MSFT, GOOG, JPM, ...) never report a single
    # combined D&A concept and instead split depreciation and intangible
    # amortization into two separate line items -- both are needed to
    # reconstruct EBITDA for those companies (see add_derived_filing_metrics).
    "Depreciation": (
        ("us-gaap", "Depreciation"),
        ("us-gaap", "DepreciationNonproduction"),
    ),
    "AmortizationOfIntangibleAssets": (
        ("us-gaap", "AmortizationOfIntangibleAssets"),
        ("us-gaap", "AmortizationOfFiniteLivedIntangibleAssets"),
    ),
    # Added for display-only valuation ratios (EV/EBIT, FCFF, ROIC, Altman
    # Z-Score, DDM Value) -- see add_derived_filing_metrics. Not used by any
    # scoring/ranking signal.
    "CurrentAssets": (
        ("us-gaap", "AssetsCurrent"),
        ("ifrs-full", "CurrentAssets"),
    ),
    "CurrentLiabilities": (
        ("us-gaap", "LiabilitiesCurrent"),
        ("ifrs-full", "CurrentLiabilities"),
    ),
    "RetainedEarnings": (
        ("us-gaap", "RetainedEarningsAccumulatedDeficit"),
        ("ifrs-full", "RetainedEarnings"),
    ),
    "IncomeTaxExpense": (
        ("us-gaap", "IncomeTaxExpenseBenefit"),
        ("ifrs-full", "IncomeTaxExpenseContinuingOperations"),
    ),
}

INSTANT_METRICS = {
    "Cash",
    "Assets",
    "Liabilities",
    "Equity",
    "DebtCurrent",
    "DebtNoncurrent",
    "SharesOutstanding",
    "CurrentAssets",
    "CurrentLiabilities",
    "RetainedEarnings",
}

TEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "RiskTerms": (
        r"\brisk factors?\b",
        r"\bmaterial adverse\b",
        r"\buncertain(?:ty|ties)?\b",
        r"\bvolatil(?:e|ity)\b",
    ),
    "CompetitionTerms": (
        r"\bcompetition\b",
        r"\bcompetitive pressures?\b",
        r"\bmarket share\b",
        r"\bbarriers? to entry\b",
    ),
    "LeverageTerms": (
        r"\bindebtedness\b",
        r"\bdebt covenant(?:s)?\b",
        r"\bliquidity risk\b",
        r"\binterest expense\b",
    ),
    "DilutionTerms": (
        r"\bdilution\b",
        r"\bshare-based compensation\b",
        r"\bstock-based compensation\b",
        r"\bequity award(?:s)?\b",
    ),
    "CapitalAllocationTerms": (
        r"\bshare repurchase(?:s)?\b",
        r"\bcapital allocation\b",
        r"\breturn of capital\b",
        r"\bdividend(?:s)?\b",
    ),
    "OutlookTerms": (
        r"\boutlook\b",
        r"\bguidance\b",
        r"\bwe expect\b",
        r"\bwe anticipate\b",
    ),
}

RED_FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "GoingConcernFlag": (
        r"\braises? substantial doubt.{0,160}\bcontinue as a going concern\b",
        r"\bsubstantial doubt exists?.{0,160}\bcontinue as a going concern\b",
        r"\bthere is substantial doubt.{0,160}\bcontinue as a going concern\b",
    ),
    "MaterialWeaknessFlag": (
        r"\b(?:we|management) (?:have |has )?identified.{0,80}\bmaterial weakness(?:es)?\b",
        r"\bthere (?:is|are|was|were).{0,50}\bmaterial weakness(?:es)?\b",
        r"\binternal control over financial reporting was not effective\b",
    ),
    "RestatementFlag": (
        r"\bwe (?:will|must|have|had|are required to) restat(?:e|ed).{0,100}\bfinancial statements\b",
        r"\brestatement of (?:our )?previously issued financial statements\b",
        r"\bfinancial statements should no longer be relied upon\b",
    ),
}

RED_FLAG_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "GoingConcernFlag": (
        r"\b(?:do|does|did) not raise substantial doubt\b",
        r"\bno substantial doubt\b",
    ),
    "MaterialWeaknessFlag": (
        r"\b(?:did|have|has) not identif(?:y|ied).{0,60}\bmaterial weakness",
        r"\bno material weakness(?:es)?\b",
    ),
    "RestatementFlag": (
        r"\b(?:may|might|could) (?:result|require).{0,80}\brestatement\b",
        r"\bno restatement\b",
    ),
}


@dataclass(frozen=True)
class SecFilingSettings:
    start_date: str = "2019-01-01"
    end_date: str | None = None
    forms: tuple[str, ...] = DEFAULT_FORMS
    include_amendments: bool = False
    download_primary_documents: bool = True
    request_interval_seconds: float = 0.20
    timeout_seconds: float = 30.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        if self.request_interval_seconds < 0.10:
            raise ValueError(
                "request_interval_seconds must be at least 0.10 seconds"
            )
        if self.timeout_seconds <= 0 or self.max_retries < 1:
            raise ValueError("timeout_seconds and max_retries must be positive")
        if not self.forms:
            raise ValueError("at least one SEC form is required")


@dataclass(frozen=True)
class SecFilingArtifacts:
    output_dir: Path
    filing_index_csv: Path
    filing_metrics_csv: Path
    filing_text_features_csv: Path
    point_in_time_features_csv: Path
    download_audit_csv: Path
    manifest_json: Path


class SecEdgarClient:
    """Small cached SEC client that stays below the published fair-access cap."""

    def __init__(
        self,
        *,
        user_agent: str,
        settings: SecFilingSettings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.user_agent = validate_user_agent(user_agent)
        self.settings = settings
        self.session = session or requests.Session()
        self.sleep = sleep
        self.monotonic = monotonic
        self._last_request_at: float | None = None

    def cached_json(
        self,
        url: str,
        path: Path,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        payload = self._request(url).content
        _atomic_write_bytes(path, payload)
        return json.loads(payload.decode("utf-8"))

    def cached_document(
        self,
        url: str,
        path: Path,
        *,
        refresh: bool = False,
    ) -> bytes:
        if path.exists() and not refresh:
            return path.read_bytes()
        payload = self._request(url).content
        _atomic_write_bytes(path, payload)
        return payload

    def _request(self, url: str) -> requests.Response:
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            self._throttle()
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.settings.timeout_seconds,
                )
                self._last_request_at = self.monotonic()
                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"SEC returned HTTP {response.status_code} for {url}",
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (requests.RequestException, OSError) as exc:
                error = exc
                if attempt + 1 < self.settings.max_retries:
                    self.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"SEC request failed after retries: {url}") from error

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self.monotonic() - self._last_request_at
        remaining = self.settings.request_interval_seconds - elapsed
        if remaining > 0:
            self.sleep(remaining)


def validate_user_agent(value: str | None) -> str:
    user_agent = str(value or "").strip()
    if "@" not in user_agent or len(user_agent) < 8:
        raise ValueError(
            "SEC_USER_AGENT must identify the downloader and include a contact "
            "email, for example 'Personal Research name@example.com'."
        )
    return user_agent


def settings_from_dict(raw: dict[str, Any]) -> SecFilingSettings:
    return SecFilingSettings(
        start_date=str(raw.get("start_date", "2019-01-01")),
        end_date=(str(raw["end_date"]) if raw.get("end_date") else None),
        forms=tuple(str(value).upper() for value in raw.get("forms", DEFAULT_FORMS)),
        include_amendments=bool(raw.get("include_amendments", False)),
        download_primary_documents=bool(
            raw.get("download_primary_documents", True)
        ),
        request_interval_seconds=float(
            raw.get("request_interval_seconds", 0.20)
        ),
        timeout_seconds=float(raw.get("timeout_seconds", 30.0)),
        max_retries=int(raw.get("max_retries", 4)),
    )


def universe_tickers(config: dict[str, Any]) -> list[str]:
    values = config.get("universe", [])
    tickers = [str(row["ticker"]).strip().upper() for row in values]
    return list(dict.fromkeys(ticker for ticker in tickers if ticker))


def sync_sec_filings(
    paths: ProjectPaths,
    *,
    tickers: Iterable[str],
    settings: SecFilingSettings,
    user_agent: str,
    output_label: str,
    refresh_metadata: bool = False,
    refresh_documents: bool = False,
) -> SecFilingArtifacts:
    """Download, cache, and analyze point-in-time SEC periodic filings."""

    ticker_list = list(dict.fromkeys(str(value).upper() for value in tickers))
    if not ticker_list:
        raise ValueError("No tickers were supplied")
    raw_root = paths.stock_root / "SEC Filings"
    output_dir = paths.processed / "SEC Filings" / output_label
    raw_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = SecEdgarClient(user_agent=user_agent, settings=settings)
    text_feature_cache = _load_text_feature_cache(
        output_dir / "filing_text_features.csv"
    )

    ticker_payload = client.cached_json(
        SEC_TICKER_URL,
        raw_root / "company_tickers.json",
        refresh=refresh_metadata,
    )
    cik_map = build_ticker_cik_map(ticker_payload)
    filing_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for ticker in ticker_list:
        ticker_root = raw_root / ticker
        cik = cik_map.get(ticker) or cached_submission_cik(ticker_root, ticker)
        if cik is None:
            audits.append(_audit(ticker, "FAILED", "CIK_NOT_FOUND"))
            continue
        try:
            submissions = client.cached_json(
                SEC_SUBMISSIONS_URL.format(cik=cik),
                ticker_root / "submissions.json",
                refresh=refresh_metadata,
            )
            submission_frames = [submission_frame(submissions)]
            for file_row in submissions.get("filings", {}).get("files", []):
                name = str(file_row.get("name", "")).strip()
                if not name:
                    continue
                older = client.cached_json(
                    SEC_SUBMISSION_FILE_URL.format(name=quote(name)),
                    ticker_root / "submission_history" / name,
                    refresh=refresh_metadata,
                )
                submission_frames.append(submission_frame(older))
            filings = filter_filings(
                pd.concat(submission_frames, ignore_index=True), settings
            )
            companyfacts_url = SEC_COMPANYFACTS_URL.format(cik=cik)
            companyfacts_path = ticker_root / "companyfacts.json"
            companyfacts = client.cached_json(
                companyfacts_url,
                companyfacts_path,
                refresh=refresh_metadata,
            )
            standardized = extract_standardized_facts(companyfacts)
            ticker_metrics = derive_filing_metrics(
                filings, standardized, ticker=ticker
            )
            metadata_retry = False
            if (
                not refresh_metadata
                and latest_filing_has_no_core_facts(ticker_metrics)
            ):
                # A new submission can arrive before a previously cached
                # Company Facts response contains its XBRL observations.  Do
                # one forced refresh so the dashboard never accepts that stale
                # cache as a successfully populated filing.
                metadata_retry = True
                companyfacts = client.cached_json(
                    companyfacts_url,
                    companyfacts_path,
                    refresh=True,
                )
                standardized = extract_standardized_facts(companyfacts)
                ticker_metrics = derive_filing_metrics(
                    filings, standardized, ticker=ticker
                )
            incomplete_latest = latest_filing_has_no_core_facts(ticker_metrics)
            if not ticker_metrics.empty:
                # SEC's own SIC industry classification, already present on
                # every cached submissions.json response -- no extra fetch.
                # Used only for display-only peer-group grouping in the
                # dashboard (e.g. AEP/VST/NRG all SIC 4911 Electric
                # Services), never for scoring/ranking.
                ticker_metrics["Sic"] = str(submissions.get("sic", ""))
                ticker_metrics["SicDescription"] = str(
                    submissions.get("sicDescription", "")
                )
            metric_rows.extend(ticker_metrics.to_dict("records"))

            downloaded = 0
            for filing in filings.to_dict("records"):
                filing["Ticker"] = ticker
                filing["CIK"] = cik
                document_path: Path | None = None
                document_error = ""
                if settings.download_primary_documents:
                    try:
                        document_path, payload = download_primary_document(
                            client,
                            ticker_root=ticker_root,
                            cik=cik,
                            filing=filing,
                            refresh=refresh_documents,
                        )
                        identity = (ticker, str(filing["AccessionNumber"]))
                        digest = hashlib.sha256(payload).hexdigest()
                        cached = text_feature_cache.get(identity)
                        if (
                            cached is not None
                            and cached.get("DocumentSha256") == digest
                            and not refresh_documents
                        ):
                            analysis = cached
                        else:
                            analysis = analyze_filing_document(payload)
                        text_rows.append(
                            {
                                **analysis,
                                "Ticker": ticker,
                                "AccessionNumber": filing["AccessionNumber"],
                                "Form": filing["Form"],
                                "FiledDate": filing["FiledDate"],
                                "AcceptedAt": filing["AcceptedAt"],
                                "AvailableDate": filing["AvailableDate"],
                            }
                        )
                        downloaded += 1
                    except Exception as exc:  # noqa: BLE001 - resumable batch
                        document_error = f"{type(exc).__name__}: {exc}"
                filing["LocalDocumentPath"] = (
                    str(document_path) if document_path else ""
                )
                filing["DocumentError"] = document_error
                filing_rows.append(filing)
            audits.append(
                _audit(
                    ticker,
                    "PARTIAL" if incomplete_latest else "OK",
                    "LATEST_FILING_CORE_FACTS_MISSING" if incomplete_latest else "",
                    cik=cik,
                    filings=len(filings),
                    documents=downloaded,
                    facts=len(standardized),
                    metadata_retry=metadata_retry,
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate issuer failure
            audits.append(
                _audit(
                    ticker,
                    "FAILED",
                    f"{type(exc).__name__}: {exc}",
                    cik=cik,
                )
            )

    filing_index = pd.DataFrame(filing_rows)
    metrics = pd.DataFrame(metric_rows)
    text_features = pd.DataFrame(text_rows)
    point_in_time = build_point_in_time_features(metrics, text_features)
    audit = pd.DataFrame(audits)
    outputs = {
        "filing_index_csv": output_dir / "filing_index.csv",
        "filing_metrics_csv": output_dir / "filing_metrics.csv",
        "filing_text_features_csv": output_dir / "filing_text_features.csv",
        "point_in_time_features_csv": output_dir
        / "point_in_time_features.csv",
        "download_audit_csv": output_dir / "download_audit.csv",
        "manifest_json": output_dir / "manifest.json",
    }
    atomic_to_csv(filing_index, outputs["filing_index_csv"], index=False)
    atomic_to_csv(metrics, outputs["filing_metrics_csv"], index=False)
    atomic_to_csv(
        text_features, outputs["filing_text_features_csv"], index=False
    )
    atomic_to_csv(
        point_in_time, outputs["point_in_time_features_csv"], index=False
    )
    atomic_to_csv(audit, outputs["download_audit_csv"], index=False)
    _atomic_json(
        outputs["manifest_json"],
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "source": "SEC EDGAR public submissions, Company Facts, and archives",
            "settings": asdict(settings),
            "output_label": output_label,
            "tickers": ticker_list,
            "successful_tickers": int(audit["Status"].eq("OK").sum()),
            "partial_tickers": int(audit["Status"].eq("PARTIAL").sum()),
            "failed_tickers": int(audit["Status"].eq("FAILED").sum()),
            "filing_count": len(filing_index),
            "document_feature_count": len(text_features),
            "point_in_time_rule": (
                "Filing features become eligible on the calendar day after "
                "the SEC acceptance date to avoid same-day close lookahead."
            ),
            "raw_cache_root": str(raw_root),
            "outputs": {key: str(value) for key, value in outputs.items()},
            "limitations": [
                "Standard taxonomy facts only; issuer extension concepts are excluded.",
                "Text features are evidence counts, not a claim of management quality.",
                "6-K content varies and may not contain a complete quarterly report.",
            ],
        },
    )
    return SecFilingArtifacts(output_dir=output_dir, **outputs)


def build_ticker_cik_map(payload: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in payload.values():
        ticker = str(row.get("ticker", "")).strip().upper().replace(".", "-")
        cik = str(row.get("cik_str", "")).strip()
        if ticker and cik:
            result[ticker] = cik.zfill(10)
    return result


def cached_submission_cik(ticker_root: Path, ticker: str) -> str | None:
    """Recover a validated CIK when SEC's current ticker index omits a company."""

    submissions_path = ticker_root / "submissions.json"
    if not submissions_path.exists():
        return None
    try:
        payload = json.loads(submissions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    requested = ticker.strip().upper().replace(".", "-")
    payload_tickers = {
        str(value).strip().upper().replace(".", "-")
        for value in payload.get("tickers", [])
    }
    cik = str(payload.get("cik", "")).strip()
    if requested not in payload_tickers or not cik.isdigit():
        return None
    return cik.zfill(10)


def submission_frame(payload: dict[str, Any]) -> pd.DataFrame:
    recent = payload.get("filings", {}).get("recent", payload)
    if not isinstance(recent, dict) or not recent:
        return pd.DataFrame()
    lengths = [len(value) for value in recent.values() if isinstance(value, list)]
    if not lengths:
        return pd.DataFrame()
    length = min(lengths)
    values = {
        key: value[:length]
        for key, value in recent.items()
        if isinstance(value, list)
    }
    frame = pd.DataFrame(values)
    mapping = {
        "accessionNumber": "AccessionNumber",
        "filingDate": "FiledDate",
        "reportDate": "PeriodOfReport",
        "acceptanceDateTime": "AcceptedAt",
        "form": "Form",
        "primaryDocument": "PrimaryDocument",
        "primaryDocDescription": "PrimaryDocumentDescription",
    }
    frame = frame.rename(columns=mapping)
    for column in mapping.values():
        if column not in frame:
            frame[column] = ""
    return frame[list(mapping.values())]


def filter_filings(
    filings: pd.DataFrame,
    settings: SecFilingSettings,
) -> pd.DataFrame:
    if filings.empty:
        return filings.copy()
    frame = filings.copy()
    frame["Form"] = frame["Form"].astype(str).str.upper().str.strip()
    allowed = set(settings.forms)
    if settings.include_amendments:
        allowed |= {f"{form}/A" for form in settings.forms}
    frame = frame.loc[frame["Form"].isin(allowed)]
    frame["FiledDate"] = pd.to_datetime(frame["FiledDate"], errors="coerce")
    frame["PeriodOfReport"] = pd.to_datetime(
        frame["PeriodOfReport"], errors="coerce"
    )
    frame["AcceptedAt"] = pd.to_datetime(
        frame["AcceptedAt"], errors="coerce", utc=True
    )
    start = pd.Timestamp(settings.start_date)
    end = (
        pd.Timestamp(settings.end_date)
        if settings.end_date
        else pd.Timestamp.today().normalize()
    )
    frame = frame.loc[frame["FiledDate"].between(start, end)]
    accepted_date = frame["AcceptedAt"].dt.tz_convert(None).dt.normalize()
    accepted_date = accepted_date.fillna(frame["FiledDate"])
    frame["AvailableDate"] = accepted_date + pd.Timedelta(days=1)
    frame = frame.drop_duplicates("AccessionNumber", keep="first")
    return frame.sort_values(["FiledDate", "AccessionNumber"]).reset_index(
        drop=True
    )


def extract_standardized_facts(payload: dict[str, Any]) -> pd.DataFrame:
    facts = payload.get("facts", {})
    rows: list[dict[str, Any]] = []
    for metric, concepts in STANDARD_CONCEPTS.items():
        for priority, (taxonomy, concept) in enumerate(concepts):
            concept_payload = facts.get(taxonomy, {}).get(concept)
            if not concept_payload:
                continue
            for unit, observations in concept_payload.get("units", {}).items():
                for observation in observations:
                    rows.append(
                        {
                            "Metric": metric,
                            "ConceptPriority": priority,
                            "Taxonomy": taxonomy,
                            "Concept": concept,
                            "Unit": unit,
                            "Start": observation.get("start"),
                            "End": observation.get("end"),
                            "Value": observation.get("val"),
                            "AccessionNumber": observation.get("accn"),
                            "FiscalYear": observation.get("fy"),
                            "FiscalPeriod": observation.get("fp"),
                            "Form": observation.get("form"),
                            "FiledDate": observation.get("filed"),
                            "Frame": observation.get("frame"),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("Start", "End", "FiledDate"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame["DurationDays"] = (frame["End"] - frame["Start"]).dt.days
    return frame.loc[frame["Value"].notna()].reset_index(drop=True)


def derive_filing_metrics(
    filings: pd.DataFrame,
    facts: pd.DataFrame,
    *,
    ticker: str,
) -> pd.DataFrame:
    if filings.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for filing in filings.to_dict("records"):
        form = str(filing["Form"])
        annual = form.startswith(("10-K", "20-F", "40-F"))
        target_days = 365 if annual else 91
        period_end = pd.Timestamp(filing["PeriodOfReport"])
        accession = str(filing["AccessionNumber"])
        subset = facts.loc[facts["AccessionNumber"].eq(accession)].copy()
        row: dict[str, Any] = {
            "Ticker": ticker,
            "AccessionNumber": accession,
            "Form": form,
            "PeriodKind": "ANNUAL" if annual else "QUARTERLY",
            "FiledDate": filing["FiledDate"],
            "AcceptedAt": filing["AcceptedAt"],
            "AvailableDate": filing["AvailableDate"],
            "PeriodOfReport": period_end,
        }
        fiscal_years: list[float] = []
        fiscal_periods: list[str] = []
        for metric in STANDARD_CONCEPTS:
            selected = select_filing_fact(
                subset.loc[subset["Metric"].eq(metric)],
                metric=metric,
                period_end=period_end,
                target_duration_days=target_days,
            )
            row[metric] = selected.get("Value", np.nan)
            row[f"{metric}Concept"] = selected.get("Concept", "")
            row[f"{metric}DurationDays"] = selected.get(
                "DurationDays", np.nan
            )
            if pd.notna(selected.get("FiscalYear")):
                fiscal_years.append(float(selected["FiscalYear"]))
            if str(selected.get("FiscalPeriod", "")):
                fiscal_periods.append(str(selected["FiscalPeriod"]))
        row["FiscalYear"] = (
            int(pd.Series(fiscal_years).mode().iloc[0])
            if fiscal_years
            else period_end.year
        )
        row["FiscalPeriod"] = (
            pd.Series(fiscal_periods).mode().iloc[0]
            if fiscal_periods
            else ("FY" if annual else "")
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("AvailableDate").reset_index(
        drop=True
    )
    return add_derived_filing_metrics(frame)


CORE_FILING_FACTS = ("Revenue", "OperatingIncome", "NetIncome", "Assets")


def latest_filing_has_no_core_facts(metrics: pd.DataFrame) -> bool:
    """Return true when a discovered latest filing has no usable XBRL core facts."""

    if metrics.empty:
        return False
    latest = metrics.sort_values("AvailableDate").iloc[-1]
    return not any(
        pd.notna(latest.get(column)) for column in CORE_FILING_FACTS
    )


def select_filing_fact(
    facts: pd.DataFrame,
    *,
    metric: str,
    period_end: pd.Timestamp,
    target_duration_days: int,
) -> dict[str, Any]:
    if facts.empty:
        return {}
    frame = facts.copy()
    frame["EndGap"] = (frame["End"] - period_end).abs().dt.days
    frame = frame.loc[frame["EndGap"].le(45) | frame["EndGap"].isna()]
    if frame.empty:
        return {}
    if metric in INSTANT_METRICS:
        frame["DurationGap"] = frame["DurationDays"].notna().astype(int)
    else:
        frame = frame.loc[frame["DurationDays"].notna()]
        frame["DurationGap"] = (
            frame["DurationDays"] - target_duration_days
        ).abs()
    if frame.empty:
        return {}
    chosen = frame.sort_values(
        ["EndGap", "DurationGap", "ConceptPriority", "FiledDate"],
        ascending=[True, True, True, False],
    ).iloc[0]
    return chosen.to_dict()


def split_adjust_share_series(values: pd.Series) -> pd.Series:
    """Rescale a chronologically-sorted share-count series onto a single
    consistent basis by neutralizing discrete jumps consistent with a
    forward/reverse stock split.

    SEC filings report the actual share count as of that filing -- correctly
    unadjusted for later splits. A naive period-over-period share-count
    comparison therefore reads a 10-for-1 split as "1000% dilution" for the
    one period it falls in. This does not change what was actually filed;
    it only normalizes the series so period-over-period comparisons (e.g.
    ShareGrowthYoYFiled) reflect real issuance/buybacks rather than a
    split's mechanical share-count change. Input must already be sorted by
    date (ascending).
    """
    raw = values.to_numpy(dtype=float)
    n = len(raw)
    factors = np.ones(n)
    cumulative = 1.0
    for i in range(n - 1, 0, -1):
        prev, curr = raw[i - 1], raw[i]
        if np.isfinite(prev) and np.isfinite(curr) and prev > 0:
            ratio = curr / prev
            if ratio > 1.8 or ratio < 0.55:
                cumulative *= ratio
        factors[i - 1] = cumulative
    return pd.Series(raw * factors, index=values.index)


def _ttm_with_q4_derivation(frame: pd.DataFrame, column: str) -> pd.Series:
    """Trailing-12-month sum of a flow metric, aligned to `frame`'s index.

    Most US filers never file a standalone Q4 10-Q -- Q4 only appears
    bundled into the annual 10-K (FY total). A naive 4-quarter rolling sum
    over just the QUARTERLY rows therefore silently skips every Q4 and
    wraps into next year's Q1. Q4 is derived here as FY total minus the
    filed Q1+Q2+Q3 (only when all three are present), then a continuous
    4-quarter rolling sum is taken over PeriodOfReport order. Same pattern
    as dashboard.engine._trailing_twelve_month_eps, generalized to any flow
    column (NetIncome, OperatingIncome, Revenue, DividendPerShare, ...).

    Returns all-NaN when the period schema is absent. Every caller is in the
    display-only valuation block, so a missing TTM drops a displayed ratio and
    never reaches signals.py/portfolio.py -- the same graceful-degradation
    convention `add_derived_filing_metrics` already uses for
    SharesOutstandingConcept.
    """
    if not {"PeriodOfReport", "PeriodKind"}.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    ordered = frame[["PeriodOfReport", "PeriodKind"]].copy()
    ordered[column] = pd.to_numeric(frame[column], errors="coerce")
    quarterly = ordered.loc[ordered["PeriodKind"].eq("QUARTERLY"), ["PeriodOfReport", column]]
    annual = ordered.loc[ordered["PeriodKind"].eq("ANNUAL")]
    derived_q4 = []
    for _, arow in annual.iterrows():
        fy_end = arow["PeriodOfReport"]
        preceding = quarterly.loc[
            quarterly["PeriodOfReport"].between(
                fy_end - pd.Timedelta(days=280), fy_end - pd.Timedelta(days=1)
            )
        ]
        if len(preceding) == 3 and pd.notna(arow[column]) and preceding[column].notna().all():
            derived_q4.append(
                {"PeriodOfReport": fy_end, column: arow[column] - preceding[column].sum()}
            )
    all_quarters = (
        pd.concat([quarterly, pd.DataFrame(derived_q4)], ignore_index=True)
        if derived_q4
        else quarterly
    )
    all_quarters = all_quarters.sort_values("PeriodOfReport").drop_duplicates(
        "PeriodOfReport", keep="last"
    )
    all_quarters["Ttm"] = all_quarters[column].rolling(4, min_periods=4).sum()
    ttm_by_period = all_quarters.set_index("PeriodOfReport")["Ttm"]
    return ordered["PeriodOfReport"].map(ttm_by_period).set_axis(frame.index)


def _most_recent_annual(frame: pd.DataFrame, column: str) -> pd.Series:
    """As-of-`PeriodOfReport` lookup of the most recent 10-K's value of
    `column`, carried forward until the next 10-K. Always a clean,
    unambiguous full-fiscal-year figure regardless of how a filer tags
    interim-quarter durations. All-NaN without the period schema, same
    display-only degradation as `_ttm_with_q4_derivation`."""
    if not {"PeriodOfReport", "PeriodKind"}.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    annual = frame.loc[
        frame["PeriodKind"].eq("ANNUAL"), ["PeriodOfReport", column]
    ].dropna().sort_values("PeriodOfReport")
    if annual.empty:
        return pd.Series(np.nan, index=frame.index)
    lookup = pd.DataFrame({"PeriodOfReport": frame["PeriodOfReport"]}).sort_values(
        "PeriodOfReport"
    )
    matched = pd.merge_asof(
        lookup, annual.rename(columns={column: "_value"}),
        on="PeriodOfReport", direction="backward",
    )
    return matched.set_index(lookup.index)["_value"].reindex(frame.index)


def _capped_five_year_cagr(
    period_ends: pd.Series, ttm_values: pd.Series, *, cap: float
) -> pd.Series:
    """5-year CAGR of a TTM series, floored at 0% and capped at `cap`.

    Used as a growth-rate ASSUMPTION feeding the DCF fair-value model
    (see add_derived_filing_metrics) -- not a filed fact. Capping prevents
    a single noisy/lumpy period from producing an implausible growth
    input, and keeps the assumed growth safely below the assumed discount
    rate. Returns NaN wherever fewer than 5 years of TTM history exist.
    """
    lookup = pd.Series(ttm_values.to_numpy(), index=period_ends).dropna()
    lookup = lookup[~lookup.index.duplicated(keep="last")].sort_index()

    def _growth(period_end: pd.Timestamp) -> float:
        if pd.isna(period_end) or period_end not in lookup.index:
            return np.nan
        prior_candidates = lookup.loc[: period_end - pd.Timedelta(days=365 * 5)]
        if prior_candidates.empty:
            return np.nan
        prior_value = prior_candidates.iloc[-1]
        current_value = lookup.loc[period_end]
        if not (
            pd.notna(prior_value)
            and prior_value > 0
            and pd.notna(current_value)
            and current_value > 0
        ):
            return np.nan
        return min(cap, max(0.0, (current_value / prior_value) ** (1 / 5) - 1))

    return period_ends.map(_growth)


def add_derived_filing_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for optional_column in (
        "Depreciation",
        "AmortizationOfIntangibleAssets",
        "DepreciationAndAmortization",
    ):
        if optional_column not in result.columns:
            result[optional_column] = np.nan
    result["TotalDebt"] = result[["DebtCurrent", "DebtNoncurrent"]].sum(
        axis=1, min_count=1
    )
    result["FreeCashFlow"] = (
        result["OperatingCashFlow"] - result["CapitalExpenditures"]
    )
    result["GrossMargin"] = _safe_div(result["GrossProfit"], result["Revenue"])
    result["OperatingMargin"] = _safe_div(
        result["OperatingIncome"], result["Revenue"]
    )
    matching_fcf_duration = (
        result["OperatingCashFlowDurationDays"]
        - result["RevenueDurationDays"]
    ).abs().le(45)
    result["FreeCashFlowMargin"] = _safe_div(
        result["FreeCashFlow"], result["Revenue"]
    ).where(matching_fcf_duration)
    result["NetDebtToAssets"] = _safe_div(
        result["TotalDebt"] - result["Cash"], result["Assets"]
    )
    result["DebtToEquity"] = _safe_div(result["TotalDebt"], result["Equity"])
    result["InterestCoverage"] = _safe_div(
        result["OperatingIncome"], result["InterestExpense"].abs()
    )
    result["ResearchAndDevelopmentToRevenue"] = _safe_div(
        result["ResearchAndDevelopment"], result["Revenue"]
    )
    result["StockCompensationToRevenue"] = _safe_div(
        result["ShareBasedCompensation"], result["Revenue"]
    )
    result["RevenueGrowthYoYFiled"] = _same_period_growth(result, "Revenue")
    result["NetIncomeGrowthYoYFiled"] = _same_period_growth(
        result, "NetIncome"
    )
    result["SharesOutstandingSplitAdjusted"] = split_adjust_share_series(
        pd.to_numeric(result["SharesOutstanding"], errors="coerce")
    )
    share_concept = result.get(
        "SharesOutstandingConcept", pd.Series("", index=result.index)
    ).astype(str)
    result["ShareGrowthBasis"] = np.where(
        share_concept.isin(
            {
                "WeightedAverageNumberOfSharesOutstandingBasic",
                "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                "WeightedAverageShares",
            }
        ),
        "WEIGHTED_AVERAGE_BASIC",
        "PERIOD_END_OUTSTANDING",
    )
    result["ShareGrowthYoYFiled"] = _same_period_growth(
        result, "SharesOutstandingSplitAdjusted"
    )
    split_depreciation_and_amortization = result[
        ["Depreciation", "AmortizationOfIntangibleAssets"]
    ].sum(axis=1, min_count=1)
    depreciation_and_amortization = split_depreciation_and_amortization.where(
        split_depreciation_and_amortization.notna(),
        result["DepreciationAndAmortization"],
    )
    result["Ebitda"] = pd.concat(
        [result["OperatingIncome"], depreciation_and_amortization], axis=1
    ).sum(axis=1, min_count=2)
    result["EbitdaMargin"] = _safe_div(result["Ebitda"], result["Revenue"])
    result["EbitdaGrowthYoYFiled"] = _same_period_growth(result, "Ebitda")

    # -- Display-only valuation ratios (EPS/EV-EBIT/FCFF/ROIC/Altman
    # Z/DCF). None of this feeds signals.py/portfolio.py -- same
    # "display-only, not part of any trading signal" precedent as
    # dashboard.engine._trailing_twelve_month_eps/_attach_pe_peg. Anything
    # needing price (EV/EBIT, Altman Z's and FCFF/EV's market-value-of-
    # equity term) is finished downstream in export_score_history.py,
    # which has price data.
    # None of these inputs is guaranteed to be present: a filer can omit any
    # concept, and callers that only want the filed ratios above pass a frame
    # without the valuation inputs at all. Missing ones are materialised as NaN
    # once here so the block degrades to blank ratios instead of raising
    # KeyError partway through -- the same graceful-degradation convention used
    # for SharesOutstandingConcept above.
    for optional_input in (
        "NetIncome", "OperatingIncome", "Revenue", "IncomeTaxExpense",
        "CurrentAssets", "CurrentLiabilities", "OperatingCashFlow",
        "InterestExpense", "CapitalExpenditures", "TotalDebt", "Equity", "Cash",
    ):
        if optional_input not in result.columns:
            result[optional_input] = np.nan
    if "PeriodOfReport" not in result.columns:
        result["PeriodOfReport"] = pd.NaT

    # DilutedShares is not guaranteed: the same filers that force the basic
    # share fallback above (diluted counts split across share classes, or an
    # IFRS filer) never populate it, and reading it directly raised KeyError.
    # Fall back per-row to the split-adjusted outstanding count -- for a
    # company with no dilutive securities the two are equal anyway, and an
    # EpsTtm off basic shares is closer than no EpsTtm at all. Display-only,
    # same as everything else in this block.
    diluted_shares = split_adjust_share_series(
        pd.to_numeric(
            result.get("DilutedShares", pd.Series(np.nan, index=result.index)),
            errors="coerce",
        )
    )
    result["DilutedSharesSplitAdjusted"] = diluted_shares.where(
        diluted_shares.notna(), result["SharesOutstandingSplitAdjusted"]
    )
    result["NetIncomeTtm"] = _ttm_with_q4_derivation(result, "NetIncome")
    result["EbitTtm"] = _ttm_with_q4_derivation(result, "OperatingIncome")
    result["RevenueTtm"] = _ttm_with_q4_derivation(result, "Revenue")
    result["EpsTtm"] = _safe_div(
        result["NetIncomeTtm"], result["DilutedSharesSplitAdjusted"]
    )

    # Effective tax rate approximated per-filing as Tax / (NetIncome + Tax)
    # (i.e. Tax / Pretax Income, since no separate PretaxIncome concept is
    # extracted). Falls back to a 21% US statutory-rate assumption when the
    # filing doesn't tag IncomeTaxExpenseBenefit or when NetIncome+Tax<=0.
    pretax = result["NetIncome"] + result["IncomeTaxExpense"]
    effective_tax_rate = (result["IncomeTaxExpense"] / pretax).where(pretax > 0)
    result["EffectiveTaxRate"] = effective_tax_rate.clip(lower=0, upper=0.5).fillna(0.21)

    result["WorkingCapital"] = result["CurrentAssets"] - result["CurrentLiabilities"]
    result["Fcff"] = (
        result["OperatingCashFlow"]
        + result["InterestExpense"].fillna(0) * (1 - result["EffectiveTaxRate"])
        - result["CapitalExpenditures"]
    ).where(matching_fcf_duration)
    result["FcffGrowthYoYFiled"] = _same_period_growth(result, "Fcff")
    # Many filers (e.g. AEP) tag OperatingCashFlow/CapitalExpenditures as
    # cumulative year-to-date in interim 10-Qs rather than discrete-quarter
    # (visible as OperatingCashFlowDurationDays ~180/272 instead of ~91) --
    # summing those as if they were discrete quarters would double-count.
    # The Q4-derivation TTM is only trustworthy where Fcff's own quarters
    # are already discrete, so fall back to the most recent full fiscal
    # year's (10-K's) Fcff -- always a clean, unambiguous 365-day figure --
    # whenever the TTM reconstruction itself is unavailable.
    result["FcffTtm"] = _ttm_with_q4_derivation(result, "Fcff")
    result["FcffTtm"] = result["FcffTtm"].where(
        result["FcffTtm"].notna(), _most_recent_annual(result, "Fcff")
    )

    result["Nopat"] = result["EbitTtm"] * (1 - result["EffectiveTaxRate"])
    result["InvestedCapital"] = result["TotalDebt"] + result["Equity"] - result["Cash"]
    result["Roic"] = _safe_div(result["Nopat"], result["InvestedCapital"]).where(
        result["InvestedCapital"] > 0
    )

    # DCF (2-stage FCFF, no price needed -- unlike EV/EBIT and Altman Z's
    # market-value-of-equity term, which need price and are finished
    # downstream in export_score_history.py). 5 years of explicit FCFF
    # growth at g1 (the 5-year TTM revenue CAGR, capped at 6% -- FCFF
    # itself is too sparse/lumpy quarter-to-quarter for a stable CAGR, so
    # revenue growth is used as the proxy growth driver), then a Gordon
    # terminal value at a fixed 2.5% perpetual growth. Discount rate is a
    # flat 8% WACC assumption. All three (g1, terminal growth, discount
    # rate) are modeling ASSUMPTIONS, not filed facts -- flagged as such
    # wherever this is displayed. g1 only compounds the explicit 5-year
    # cash flows (a finite sum, never divided by r-g1), so it cannot blow
    # up the model on its own -- only the terminal growth rate could, and
    # that is a fixed 2.5% constant safely below the 8% discount rate, so
    # no additional gating is needed here (a prior version incorrectly
    # gated on g1 < discount rate, which blanked out DCF for any fast
    # grower like SPGI whose capped growth rate hit the cap exactly).
    dcf_discount_rate = 0.08
    dcf_terminal_growth = 0.025
    revenue_growth = _capped_five_year_cagr(
        result["PeriodOfReport"], result["RevenueTtm"], cap=0.06
    )
    result["DcfAssumedGrowth"] = revenue_growth
    result["DcfAssumedTerminalGrowth"] = dcf_terminal_growth
    result["DcfAssumedDiscountRate"] = dcf_discount_rate
    explicit_years = range(1, 6)
    discounted_explicit = sum(
        result["FcffTtm"]
        * (1 + revenue_growth) ** year
        / (1 + dcf_discount_rate) ** year
        for year in explicit_years
    )
    terminal_fcff = result["FcffTtm"] * (1 + revenue_growth) ** 5 * (1 + dcf_terminal_growth)
    discounted_terminal = (
        terminal_fcff / (dcf_discount_rate - dcf_terminal_growth)
    ) / (1 + dcf_discount_rate) ** 5
    enterprise_value_dcf = discounted_explicit + discounted_terminal
    equity_value_dcf = enterprise_value_dcf - result["TotalDebt"] + result["Cash"]
    dcf_value = _safe_div(equity_value_dcf, result["SharesOutstandingSplitAdjusted"])
    # A non-positive fair value isn't a meaningful "price" to show, even
    # though it's a mathematically valid model output -- e.g. a heavy
    # capex quarter (AMZN's AI-datacenter buildout) can make quarterly
    # FCFF, and therefore the whole projection, go negative. Treated the
    # same as any other "can't give a real answer" case: blank, not a
    # misleading negative number.
    result["DcfValue"] = dcf_value.where(dcf_value > 0)
    return result


def download_primary_document(
    client: SecEdgarClient,
    *,
    ticker_root: Path,
    cik: str,
    filing: dict[str, Any],
    refresh: bool,
) -> tuple[Path, bytes]:
    accession = str(filing["AccessionNumber"])
    primary_document = str(filing["PrimaryDocument"]).strip()
    if not accession or not primary_document:
        raise ValueError("filing has no accession or primary document")
    accession_plain = accession.replace("-", "")
    safe_form = re.sub(r"[^A-Z0-9-]+", "_", str(filing["Form"]).upper())
    folder = (
        ticker_root
        / "filings"
        / f"{pd.Timestamp(filing['FiledDate']).date()}_{safe_form}_{accession_plain}"
    )
    destination = folder / Path(primary_document).name
    url = SEC_ARCHIVE_URL.format(
        cik=int(cik),
        accession=accession_plain,
        document=quote(primary_document),
    )
    return destination, client.cached_document(
        url, destination, refresh=refresh
    )


def analyze_filing_document(payload: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(payload).hexdigest()
    soup = BeautifulSoup(payload, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip().lower()
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    scale = 10_000 / word_count if word_count else np.nan
    result: dict[str, Any] = {
        "TextAnalysisVersion": TEXT_ANALYSIS_VERSION,
        "DocumentSha256": digest,
        "DocumentBytes": len(payload),
        "WordCount": word_count,
    }
    for name, patterns in TEXT_PATTERNS.items():
        count = sum(len(re.findall(pattern, text)) for pattern in patterns)
        result[f"{name}Count"] = count
        result[f"{name}Per10kWords"] = count * scale
    for name, patterns in RED_FLAG_PATTERNS.items():
        result[name] = _contains_unnegated_red_flag(
            text,
            patterns,
            RED_FLAG_EXCLUSIONS.get(name, ()),
        )
    return result


def _contains_unnegated_red_flag(
    text: str,
    patterns: tuple[str, ...],
    exclusions: tuple[str, ...],
) -> bool:
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            context = text[max(0, match.start() - 120) : match.end() + 40]
            if not any(re.search(exclusion, context) for exclusion in exclusions):
                return True
    return False


def build_point_in_time_features(
    metrics: pd.DataFrame,
    text_features: pd.DataFrame,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    if text_features.empty:
        result = metrics.copy()
    else:
        shared = [
            "Ticker",
            "AccessionNumber",
            "Form",
            "FiledDate",
            "AcceptedAt",
            "AvailableDate",
        ]
        left = metrics.copy()
        right = text_features.copy()
        for column in ("FiledDate", "AvailableDate"):
            left[column] = pd.to_datetime(left[column], errors="coerce")
            right[column] = pd.to_datetime(right[column], errors="coerce")
        left["AcceptedAt"] = pd.to_datetime(
            left["AcceptedAt"], errors="coerce", utc=True
        )
        right["AcceptedAt"] = pd.to_datetime(
            right["AcceptedAt"], errors="coerce", utc=True
        )
        result = left.merge(
            right,
            on=shared,
            how="left",
            validate="one_to_one",
        )
    return result.sort_values(["Ticker", "AvailableDate"]).reset_index(
        drop=True
    )


def _load_text_feature_cache(
    path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return {}
    required = {
        "Ticker",
        "AccessionNumber",
        "DocumentSha256",
        "TextAnalysisVersion",
    }
    if not required.issubset(frame.columns):
        return {}
    frame = frame.loc[
        frame["TextAnalysisVersion"].astype(str).eq(TEXT_ANALYSIS_VERSION)
    ]
    return {
        (str(row["Ticker"]), str(row["AccessionNumber"])): row.to_dict()
        for _, row in frame.iterrows()
    }


def _same_period_growth(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    keys = frame[["PeriodKind", "FiscalPeriod"]].astype(str).agg("|".join, axis=1)
    prior_lookup: dict[tuple[str, int], float] = {}
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for index in frame.sort_values("AvailableDate").index:
        year = int(frame.at[index, "FiscalYear"])
        key = (keys.at[index], year)
        prior = prior_lookup.get((keys.at[index], year - 1), np.nan)
        current = values.at[index]
        if pd.notna(current) and pd.notna(prior) and prior != 0:
            output.at[index] = (current - prior) / abs(prior)
        if pd.notna(current):
            prior_lookup[key] = float(current)
    return output


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator, errors="coerce"
    ).replace(0, np.nan)


def _audit(
    ticker: str,
    status: str,
    error: str,
    *,
    cik: str = "",
    filings: int = 0,
    documents: int = 0,
    facts: int = 0,
    metadata_retry: bool = False,
) -> dict[str, Any]:
    return {
        "Ticker": ticker,
        "CIK": cik,
        "Status": status,
        "FilingCount": filings,
        "DocumentCount": documents,
        "StandardFactCount": facts,
        "MetadataRefreshRetried": metadata_retry,
        "Error": error,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(path, encoded)
