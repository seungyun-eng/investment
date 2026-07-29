from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths
from stock_research.tickers import load_tickers

HISTORICAL_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/"
    "main/sp_500_historical_components.csv"
)
TRADEFOMO_URL_TEMPLATE = "https://tradefomo.ai/marketcap-ranking/{year}"
YAHOO_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
)
YAHOO_SHARES_URL = (
    "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/"
    "v1/finance/timeseries/{ticker}"
)
HTTP_USER_AGENT = (
    "Mozilla/5.0 (compatible; V7-PIT-Universe-Research/0.1; "
    "+local-research-pipeline)"
)

TICKER_ALIASES = {
    "FB": "META",
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

YAHOO_TICKER_ALIASES = {
    "ABC": "COR",
    "ANTM": "ELV",
    "BK": "BNY",
    "BLL": "BALL",
    "FB": "META",
    "FISV": "FI",
    "FLT": "CPAY",
    "MMC": "MRSH",
    "NLOK": "GEN",
    "RE": "EG",
    "SYMC": "GEN",
    "TMK": "GL",
    "WLTW": "WTW",
}

KNOWN_TICKER_HISTORY = {
    "META": (
        (pd.Timestamp.min, pd.Timestamp("2022-06-09"), "FB"),
    ),
}


@dataclass(frozen=True)
class PitUniverseSettings:
    snapshot_years: tuple[int, ...] = tuple(range(2019, 2027))
    snapshot_month: int = 1
    snapshot_day: int = 1
    target_size: int = 100
    direct_source_limit: int = 50
    maximum_price_age_days: int = 10
    maximum_shares_age_days: int = 550
    request_timeout_seconds: float = 30.0
    request_pause_seconds: float = 0.05
    maximum_workers: int = 6
    historical_constituents_url: str = HISTORICAL_CONSTITUENTS_URL
    direct_ranking_url_template: str = TRADEFOMO_URL_TEMPLATE

    def __post_init__(self) -> None:
        if not self.snapshot_years:
            raise ValueError("snapshot_years cannot be empty")
        if self.target_size < 1:
            raise ValueError("target_size must be positive")
        if self.direct_source_limit < 1:
            raise ValueError("direct_source_limit must be positive")
        if self.maximum_price_age_days < 0:
            raise ValueError("maximum_price_age_days must be non-negative")
        if self.maximum_shares_age_days < 0:
            raise ValueError("maximum_shares_age_days must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.request_pause_seconds < 0:
            raise ValueError("request_pause_seconds must be non-negative")
        if self.maximum_workers < 1:
            raise ValueError("maximum_workers must be positive")
        for year in self.snapshot_years:
            datetime(year, self.snapshot_month, self.snapshot_day)


@dataclass(frozen=True)
class PitUniverseSampleArtifacts:
    output_dir: Path
    direct_rankings_csv: Path
    proxy_candidates_csv: Path
    proxy_snapshots_csv: Path
    hybrid_snapshots_csv: Path
    source_comparison_csv: Path
    coverage_csv: Path
    missing_local_csv: Path
    fetch_log_csv: Path
    manifest_json: Path


def load_pit_universe_settings(
    path: str | Path,
    *,
    years: Iterable[int] | None = None,
) -> PitUniverseSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    selected_years = (
        tuple(int(year) for year in years)
        if years is not None
        else tuple(int(year) for year in raw["snapshot_years"])
    )
    return PitUniverseSettings(
        snapshot_years=selected_years,
        snapshot_month=int(raw.get("snapshot_month", 1)),
        snapshot_day=int(raw.get("snapshot_day", 1)),
        target_size=int(raw.get("target_size", 100)),
        direct_source_limit=int(raw.get("direct_source_limit", 50)),
        maximum_price_age_days=int(
            raw.get("maximum_price_age_days", 10)
        ),
        maximum_shares_age_days=int(
            raw.get("maximum_shares_age_days", 550)
        ),
        request_timeout_seconds=float(
            raw.get("request_timeout_seconds", 30.0)
        ),
        request_pause_seconds=float(
            raw.get("request_pause_seconds", 0.05)
        ),
        maximum_workers=int(raw.get("maximum_workers", 6)),
        historical_constituents_url=str(
            raw.get(
                "historical_constituents_url",
                HISTORICAL_CONSTITUENTS_URL,
            )
        ),
        direct_ranking_url_template=str(
            raw.get(
                "direct_ranking_url_template",
                TRADEFOMO_URL_TEMPLATE,
            )
        ),
    )


def parse_tradefomo_ranking(html: str, year: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    for anchor in soup.select('a[href*="/stock-analysis-and-signals/"]'):
        text = " ".join(anchor.get_text(" ", strip=True).split())
        match = re.match(
            r"^(?P<rank>\d+)\s+(?P<ticker>\S+)\s+Name\s+"
            r"(?P<company>.*?)\s+Industry(?:\s+(?P<industry>.*?))?\s+"
            r"Market Cap\s+\$(?P<market_cap>[\d.,]+[TBMK]?)$",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        published_ticker = normalize_ticker(match.group("ticker"))
        as_of = pd.Timestamp(year=year, month=1, day=1)
        ticker = contemporaneous_ticker(published_ticker, as_of)
        rows.append(
            {
                "AsOfDate": f"{year:04d}-01-01",
                "Ticker": ticker,
                "PublishedTicker": published_ticker,
                "TickerHistoryStatus": (
                    "KNOWN_RENAME_CORRECTED"
                    if ticker != published_ticker
                    else "AS_PUBLISHED"
                ),
                "Company": match.group("company").strip(),
                "Industry": (match.group("industry") or "").strip(),
                "MarketCap": parse_abbreviated_number(
                    match.group("market_cap")
                ),
                "Rank": int(match.group("rank")),
                "RankSource": "TRADEFOMO_DIRECT_PUBLISHED",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            f"TradeFomo {year} page contained no parseable ranking rows"
        )
    return (
        frame.drop_duplicates(["AsOfDate", "Rank"])
        .sort_values("Rank")
        .reset_index(drop=True)
    )


def parse_abbreviated_number(value: object) -> float:
    text = str(value).strip().upper().replace(",", "")
    if not text:
        return math.nan
    multiplier = 1.0
    if text[-1:] in {"K", "M", "B", "T"}:
        multiplier = {
            "K": 1e3,
            "M": 1e6,
            "B": 1e9,
            "T": 1e12,
        }[text[-1]]
        text = text[:-1]
    return float(text) * multiplier


def historical_members_as_of(
    history: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> tuple[pd.Timestamp, list[str]]:
    required = {"date", "tickers"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            f"Historical constituents missing columns: {sorted(missing)}"
        )
    target = pd.Timestamp(as_of).normalize()
    dated = history.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    eligible = dated.loc[dated["date"].le(target)].sort_values("date")
    if eligible.empty:
        raise ValueError(f"No constituent snapshot at or before {target.date()}")
    row = eligible.iloc[-1]
    tickers = sorted(
        {
            normalize_ticker(value)
            for value in str(row["tickers"]).split(",")
            if str(value).strip()
        }
    )
    return pd.Timestamp(row["date"]).normalize(), tickers


def fetch_tradefomo_rankings(
    settings: PitUniverseSettings,
    *,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = session or requests.Session()
    frames: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []
    for year in settings.snapshot_years:
        url = settings.direct_ranking_url_template.format(year=year)
        try:
            response = client.get(
                url,
                headers={"User-Agent": HTTP_USER_AGENT},
                timeout=settings.request_timeout_seconds,
            )
            response.raise_for_status()
            frame = parse_tradefomo_ranking(response.text, year)
            frames.append(frame)
            logs.append(
                {
                    "Stage": "DIRECT_RANKING",
                    "Year": year,
                    "Ticker": "",
                    "Status": "SUCCESS",
                    "Rows": len(frame),
                    "Detail": url,
                }
            )
        except Exception as exc:
            logs.append(
                {
                    "Stage": "DIRECT_RANKING",
                    "Year": year,
                    "Ticker": "",
                    "Status": "FAILED",
                    "Rows": 0,
                    "Detail": f"{type(exc).__name__}: {exc}",
                }
            )
        time.sleep(settings.request_pause_seconds)
    rankings = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "AsOfDate",
                "Ticker",
                "Company",
                "Industry",
                "MarketCap",
                "Rank",
                "RankSource",
            ]
        )
    )
    return rankings, pd.DataFrame(logs)


def fetch_historical_constituents(
    settings: PitUniverseSettings,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    response = client.get(
        settings.historical_constituents_url,
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(response.text))


def fetch_yahoo_candidate_observations(
    tickers: Iterable[str],
    settings: PitUniverseSettings,
    *,
    cache_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    symbols = sorted({normalize_ticker(ticker) for ticker in tickers})
    rows: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    with ThreadPoolExecutor(
        max_workers=settings.maximum_workers
    ) as executor:
        futures = {
            executor.submit(
                _load_or_fetch_yahoo_candidate,
                ticker,
                settings,
                destination,
            ): ticker
            for ticker in symbols
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                payload = future.result()
                rows.extend(payload["observations"])
                logs.append(
                    {
                        "Stage": "YAHOO_MARKET_CAP_INPUTS",
                        "Year": "",
                        "Ticker": ticker,
                        "Status": payload["status"],
                        "Rows": len(payload["observations"]),
                        "Detail": payload.get("detail", ""),
                    }
                )
            except Exception as exc:
                logs.append(
                    {
                        "Stage": "YAHOO_MARKET_CAP_INPUTS",
                        "Year": "",
                        "Ticker": ticker,
                        "Status": "FAILED",
                        "Rows": 0,
                        "Detail": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(logs)


def rank_sp500_proxy(
    candidates: pd.DataFrame,
    *,
    target_size: int,
) -> pd.DataFrame:
    required = {
        "AsOfDate",
        "Ticker",
        "Company",
        "MarketCap",
        "PriceDataAvailable",
        "SharesDataAvailable",
        "EligibleForRanking",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate data missing columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    for as_of, group in candidates.groupby("AsOfDate", sort=True):
        eligible = group.loc[group["EligibleForRanking"]].copy()
        eligible["IssuerKey"] = eligible["Company"].map(issuer_key)
        eligible = (
            eligible.sort_values(
                ["MarketCap", "Ticker"],
                ascending=[False, True],
            )
            .drop_duplicates("IssuerKey", keep="first")
            .head(target_size)
            .reset_index(drop=True)
        )
        eligible["Rank"] = np.arange(1, len(eligible) + 1)
        eligible["RankSource"] = (
            "SP500_MEMBERSHIP_YAHOO_MARKET_CAP_PROXY"
        )
        eligible["UniverseMethod"] = (
            "Historical S&P 500 members; Yahoo close multiplied by "
            "historical shares outstanding; issuer deduplicated"
        )
        frames.append(eligible)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def build_hybrid_snapshot(
    direct: pd.DataFrame,
    proxy: pd.DataFrame,
    *,
    target_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dates = sorted(
        set(direct.get("AsOfDate", pd.Series(dtype=str)))
        | set(proxy.get("AsOfDate", pd.Series(dtype=str)))
    )
    for as_of in dates:
        published = direct.loc[direct["AsOfDate"].eq(as_of)].copy()
        published = published.sort_values("Rank")
        published["UniverseMethod"] = (
            "TradeFomo direct published Jan-1 ranking"
        )
        published["IssuerKey"] = published["Company"].map(issuer_key)
        used_tickers = set(published["Ticker"])
        used_issuers = set(published["IssuerKey"])
        supplement = proxy.loc[proxy["AsOfDate"].eq(as_of)].copy()
        supplement["IssuerKey"] = supplement["Company"].map(issuer_key)
        supplement = supplement.loc[
            ~supplement["Ticker"].isin(used_tickers)
            & ~supplement["IssuerKey"].isin(used_issuers)
        ].sort_values("MarketCap")
        needed = max(target_size - len(published), 0)
        supplement = supplement.tail(needed).sort_values(
            "MarketCap",
            ascending=False,
        )
        supplement["Rank"] = np.arange(
            len(published) + 1,
            len(published) + len(supplement) + 1,
        )
        supplement["RankSource"] = (
            "SP500_PROXY_FILL_NOT_ACTUAL_WHOLE_MARKET_RANK"
        )
        combined = pd.concat(
            [published, supplement],
            ignore_index=True,
            sort=False,
        )
        frames.append(combined.head(target_size))
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def attach_local_data_status(
    frame: pd.DataFrame,
    paths: ProjectPaths,
    ticker_config_paths: Iterable[str | Path],
) -> pd.DataFrame:
    ticker_to_file: dict[str, str] = {}
    for config_path in ticker_config_paths:
        for ticker, config in load_tickers(config_path).items():
            folder = paths.raw_prices / config.display_name
            files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
            if files:
                ticker_to_file[normalize_ticker(ticker)] = str(files[0])
    result = frame.copy()
    result["LocalLookupTicker"] = result["Ticker"].map(local_lookup_ticker)
    result["LocalPriceFile"] = result["LocalLookupTicker"].map(
        ticker_to_file
    ).fillna("")
    result["InLocalData"] = result["LocalPriceFile"].ne("")
    return result


def compare_direct_and_proxy(
    direct: pd.DataFrame,
    proxy: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for as_of, direct_year in direct.groupby("AsOfDate", sort=True):
        proxy_year = proxy.loc[proxy["AsOfDate"].eq(as_of)]
        direct_by_key = {
            yahoo_ticker(ticker): ticker for ticker in direct_year["Ticker"]
        }
        proxy_keys = {
            yahoo_ticker(ticker) for ticker in proxy_year["Ticker"]
        }
        overlap = set(direct_by_key) & proxy_keys
        rows.append(
            {
                "AsOfDate": as_of,
                "DirectPublishedRows": len(direct_year),
                "ProxyRows": len(proxy_year),
                "DirectCanonicalTickerOverlapCount": len(overlap),
                "DirectCanonicalTickerOverlapPct": (
                    len(overlap) / len(direct_by_key) * 100
                    if direct_by_key
                    else math.nan
                ),
                "DirectOnlyTickers": ",".join(
                    sorted(
                        direct_by_key[key]
                        for key in set(direct_by_key) - proxy_keys
                    )
                ),
                "ProxyMethod": (
                    "historical S&P 500 membership plus Yahoo market cap"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_coverage_report(
    direct: pd.DataFrame,
    proxy: pd.DataFrame,
    hybrid: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = sorted(
        set(direct.get("AsOfDate", pd.Series(dtype=str)))
        | set(proxy.get("AsOfDate", pd.Series(dtype=str)))
    )
    for as_of in dates:
        for label, frame in (
            ("TRADEFOMO_DIRECT", direct),
            ("SP500_MARKET_CAP_PROXY", proxy),
            ("HYBRID_DIRECT_PLUS_PROXY_FILL", hybrid),
        ):
            current = frame.loc[frame["AsOfDate"].eq(as_of)]
            local_count = (
                int(current["InLocalData"].sum())
                if "InLocalData" in current
                else 0
            )
            rows.append(
                {
                    "AsOfDate": as_of,
                    "SnapshotType": label,
                    "SnapshotRows": len(current),
                    "LocalPriceRows": local_count,
                    "LocalCoveragePct": (
                        local_count / len(current) * 100
                        if len(current)
                        else math.nan
                    ),
                    "MissingLocalRows": len(current) - local_count,
                    "IsActualWholeUSMarketTop100": False,
                }
            )
        candidate_year = candidates.loc[candidates["AsOfDate"].eq(as_of)]
        rows.append(
            {
                "AsOfDate": as_of,
                "SnapshotType": "SP500_PROXY_INPUT_AUDIT",
                "SnapshotRows": len(candidate_year),
                "LocalPriceRows": int(
                    candidate_year.get(
                        "InLocalData",
                        pd.Series(False, index=candidate_year.index),
                    ).sum()
                ),
                "LocalCoveragePct": math.nan,
                "MissingLocalRows": math.nan,
                "IsActualWholeUSMarketTop100": False,
                "PriceInputAvailable": int(
                    candidate_year["PriceDataAvailable"].sum()
                ),
                "SharesInputAvailable": int(
                    candidate_year["SharesDataAvailable"].sum()
                ),
                "RankEligible": int(
                    candidate_year["EligibleForRanking"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def generate_pit_universe_source_sample(
    paths: ProjectPaths,
    settings: PitUniverseSettings,
    *,
    ticker_config_paths: Iterable[str | Path],
    output_dir: str | Path | None = None,
) -> PitUniverseSampleArtifacts:
    generated_at = datetime.now(UTC)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "pit_universe_builds"
            / (
                generated_at.strftime("%Y%m%d_%H%M%S_%f")
                + "_v7_pit_source_sample"
            )
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "source_cache" / "yahoo"

    direct, direct_log = fetch_tradefomo_rankings(settings)
    history = fetch_historical_constituents(settings)
    membership_rows: list[dict[str, object]] = []
    all_tickers: set[str] = set()
    membership_logs: list[dict[str, object]] = []
    for year in settings.snapshot_years:
        as_of = pd.Timestamp(
            year=year,
            month=settings.snapshot_month,
            day=settings.snapshot_day,
        )
        source_date, tickers = historical_members_as_of(history, as_of)
        all_tickers.update(tickers)
        membership_logs.append(
            {
                "Stage": "HISTORICAL_SP500_MEMBERSHIP",
                "Year": year,
                "Ticker": "",
                "Status": "SUCCESS",
                "Rows": len(tickers),
                "Detail": (
                    f"requested={as_of.date()}; source={source_date.date()}"
                ),
            }
        )
        for ticker in tickers:
            membership_rows.append(
                {
                    "AsOfDate": as_of.date().isoformat(),
                    "MembershipSourceDate": source_date.date().isoformat(),
                    "Ticker": ticker,
                }
            )
    membership = pd.DataFrame(membership_rows)
    yahoo, yahoo_log = fetch_yahoo_candidate_observations(
        all_tickers,
        settings,
        cache_dir=cache_dir,
    )
    candidates = membership.merge(
        yahoo,
        on=["AsOfDate", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "PriceDataAvailable",
        "SharesDataAvailable",
        "EligibleForRanking",
    ):
        candidates[column] = candidates[column].eq(True)
    configs = [Path(path).expanduser().resolve() for path in ticker_config_paths]
    candidates = attach_local_data_status(candidates, paths, configs)
    proxy = rank_sp500_proxy(candidates, target_size=settings.target_size)
    proxy = attach_local_data_status(proxy, paths, configs)
    direct = attach_local_data_status(direct, paths, configs)
    hybrid = build_hybrid_snapshot(
        direct,
        proxy,
        target_size=settings.target_size,
    )
    hybrid = attach_local_data_status(hybrid, paths, configs)
    comparison = compare_direct_and_proxy(direct, proxy)
    coverage = build_coverage_report(
        direct,
        proxy,
        hybrid,
        candidates,
    )
    missing_local = hybrid.loc[
        ~hybrid["InLocalData"],
        [
            "AsOfDate",
            "Ticker",
            "Company",
            "MarketCap",
            "Rank",
            "RankSource",
        ],
    ].copy()
    fetch_log = pd.concat(
        [
            direct_log,
            pd.DataFrame(membership_logs),
            yahoo_log,
        ],
        ignore_index=True,
    )

    direct_path = destination / "tradefomo_direct_rankings.csv"
    candidate_path = destination / "sp500_proxy_candidates.csv"
    proxy_path = destination / "sp500_proxy_top100.csv"
    hybrid_path = destination / "hybrid_top100_sample.csv"
    comparison_path = destination / "source_comparison.csv"
    coverage_path = destination / "sample_coverage.csv"
    missing_path = destination / "missing_local_sample.csv"
    fetch_log_path = destination / "fetch_log.csv"
    history_path = destination / "historical_sp500_source.csv"
    manifest_path = destination / "manifest.json"
    atomic_to_csv(direct, direct_path, index=False)
    atomic_to_csv(candidates, candidate_path, index=False)
    atomic_to_csv(proxy, proxy_path, index=False)
    atomic_to_csv(hybrid, hybrid_path, index=False)
    atomic_to_csv(comparison, comparison_path, index=False)
    atomic_to_csv(coverage, coverage_path, index=False)
    atomic_to_csv(missing_local, missing_path, index=False)
    atomic_to_csv(fetch_log, fetch_log_path, index=False)
    atomic_to_csv(history, history_path, index=False)
    _atomic_json(
        {
            "generated_at": generated_at.isoformat(),
            "task": "V7 point-in-time universe source sample",
            "settings": asdict(settings),
            "source_decision": {
                "direct_published": (
                    "TradeFomo Jan-1 historical US-listed market-cap "
                    "ranking; public page exposes 50 rows, not 100"
                ),
                "free_proxy": (
                    "historical S&P 500 membership plus Yahoo close and "
                    "historical shares outstanding"
                ),
                "hybrid": (
                    "published top 50 plus S&P proxy fill; ranks after "
                    "the published range are not actual whole-market ranks"
                ),
            },
            "limitations": [
                (
                    "The S&P proxy can omit large US-listed securities "
                    "outside the S&P 500, including foreign ADRs."
                ),
                (
                    "Yahoo historical shares are not guaranteed to be "
                    "first-seen point-in-time fundamentals."
                ),
                (
                    "The free sample is not a CRSP/Sharadar-quality "
                    "delisting-inclusive whole-market top-100 universe."
                ),
                (
                    "No V6-B code or backtest was changed or executed."
                ),
            ],
            "output_dir": str(destination),
        },
        manifest_path,
    )
    return PitUniverseSampleArtifacts(
        output_dir=destination,
        direct_rankings_csv=direct_path,
        proxy_candidates_csv=candidate_path,
        proxy_snapshots_csv=proxy_path,
        hybrid_snapshots_csv=hybrid_path,
        source_comparison_csv=comparison_path,
        coverage_csv=coverage_path,
        missing_local_csv=missing_path,
        fetch_log_csv=fetch_log_path,
        manifest_json=manifest_path,
    )


def normalize_ticker(value: object) -> str:
    return str(value).strip().upper().replace("/", ".")


def contemporaneous_ticker(
    published_ticker: object,
    as_of: str | pd.Timestamp,
) -> str:
    ticker = normalize_ticker(published_ticker)
    date = pd.Timestamp(as_of).tz_localize(None).normalize()
    for start, end, historical_ticker in KNOWN_TICKER_HISTORY.get(
        ticker,
        (),
    ):
        if start <= date < end:
            return historical_ticker
    return ticker


def yahoo_ticker(value: object) -> str:
    ticker = normalize_ticker(value)
    return YAHOO_TICKER_ALIASES.get(ticker, ticker).replace(".", "-")


def local_lookup_ticker(value: object) -> str:
    ticker = normalize_ticker(value)
    return TICKER_ALIASES.get(ticker, ticker).replace(".", "-")


def issuer_key(value: object) -> str:
    text = re.sub(r"[^A-Z0-9]+", "", str(value).upper())
    for suffix in (
        "INCORPORATED",
        "CORPORATION",
        "COMPANY",
        "LIMITED",
        "HOLDINGS",
        "PLC",
        "INC",
        "CORP",
        "LTD",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _load_or_fetch_yahoo_candidate(
    ticker: str,
    settings: PitUniverseSettings,
    cache_dir: Path,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{_safe_filename(ticker)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = _fetch_yahoo_candidate(ticker, settings)
    _atomic_json(payload, cache_path)
    time.sleep(settings.request_pause_seconds)
    return payload


def _fetch_yahoo_candidate(
    ticker: str,
    settings: PitUniverseSettings,
) -> dict[str, Any]:
    symbol = yahoo_ticker(ticker)
    first_as_of = pd.Timestamp(
        year=min(settings.snapshot_years),
        month=settings.snapshot_month,
        day=settings.snapshot_day,
        tz="UTC",
    )
    last_as_of = pd.Timestamp(
        year=max(settings.snapshot_years),
        month=settings.snapshot_month,
        day=settings.snapshot_day,
        tz="UTC",
    )
    start = first_as_of - pd.Timedelta(
        days=max(settings.maximum_shares_age_days, 40)
    )
    end = max(
        last_as_of + pd.Timedelta(days=5),
        pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1),
    )
    headers = {"User-Agent": HTTP_USER_AGENT}
    chart_response = requests.get(
        YAHOO_CHART_URL.format(ticker=symbol),
        params={
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": "1d",
            "events": "history,splits",
        },
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    chart_response.raise_for_status()
    chart_payload = chart_response.json()
    chart_error = chart_payload.get("chart", {}).get("error")
    chart_results = chart_payload.get("chart", {}).get("result")
    if chart_error or not chart_results:
        return {
            "status": "NO_CHART_DATA",
            "detail": str(chart_error or "empty chart result"),
            "observations": [],
        }
    chart = chart_results[0]
    meta = chart.get("meta", {})
    price_frame = _parse_chart_prices(chart)
    splits = _parse_splits(chart)

    shares_response = requests.get(
        YAHOO_SHARES_URL.format(ticker=symbol),
        params={
            "symbol": symbol,
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
        },
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    shares_response.raise_for_status()
    shares = _parse_shares(shares_response.json())
    observations = [
        _candidate_observation(
            ticker=ticker,
            company=str(
                meta.get("longName")
                or meta.get("shortName")
                or ticker
            ),
            as_of=pd.Timestamp(
                year=year,
                month=settings.snapshot_month,
                day=settings.snapshot_day,
                tz="UTC",
            ),
            price_frame=price_frame,
            shares=shares,
            splits=splits,
            settings=settings,
        )
        for year in settings.snapshot_years
    ]
    usable = sum(
        bool(row["EligibleForRanking"]) for row in observations
    )
    return {
        "status": "SUCCESS" if usable else "NO_ELIGIBLE_OBSERVATION",
        "detail": (
            f"currency={meta.get('currency', '')}; "
            f"exchange={meta.get('fullExchangeName', '')}; "
            f"eligible_snapshots={usable}"
        ),
        "observations": observations,
    }


def _parse_chart_prices(chart: dict[str, Any]) -> pd.DataFrame:
    timestamps = chart.get("timestamp") or []
    quotes = chart.get("indicators", {}).get("quote") or []
    closes = quotes[0].get("close", []) if quotes else []
    length = min(len(timestamps), len(closes))
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                timestamps[:length],
                unit="s",
                utc=True,
            ),
            "SplitAdjustedClose": pd.to_numeric(
                closes[:length],
                errors="coerce",
            ),
        }
    )
    return frame.dropna().sort_values("Date").reset_index(drop=True)


def _parse_splits(chart: dict[str, Any]) -> pd.DataFrame:
    raw = chart.get("events", {}).get("splits", {})
    rows: list[dict[str, object]] = []
    for event in raw.values():
        numerator = event.get("numerator")
        denominator = event.get("denominator")
        factor = math.nan
        if numerator is not None and denominator not in {None, 0}:
            factor = float(numerator) / float(denominator)
        elif event.get("splitRatio"):
            left, right = str(event["splitRatio"]).split(":", maxsplit=1)
            factor = float(left) / float(right)
        rows.append(
            {
                "Date": pd.to_datetime(
                    int(event["date"]),
                    unit="s",
                    utc=True,
                ),
                "Factor": factor,
            }
        )
    return pd.DataFrame(rows, columns=["Date", "Factor"]).dropna()


def _parse_shares(payload: dict[str, Any]) -> pd.DataFrame:
    result = payload.get("timeseries", {}).get("result") or []
    if not result:
        return pd.DataFrame(columns=["Date", "SharesOutstanding"])
    series = result[0]
    timestamps = series.get("timestamp") or []
    values = series.get("shares_out") or []
    length = min(len(timestamps), len(values))
    return (
        pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    timestamps[:length],
                    unit="s",
                    utc=True,
                ),
                "SharesOutstanding": pd.to_numeric(
                    values[:length],
                    errors="coerce",
                ),
            }
        )
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _candidate_observation(
    *,
    ticker: str,
    company: str,
    as_of: pd.Timestamp,
    price_frame: pd.DataFrame,
    shares: pd.DataFrame,
    splits: pd.DataFrame,
    settings: PitUniverseSettings,
) -> dict[str, object]:
    price_rows = price_frame.loc[price_frame["Date"].lt(as_of)]
    share_rows = shares.loc[shares["Date"].lt(as_of)]
    price_available = not price_rows.empty
    shares_available = not share_rows.empty
    price_date = (
        pd.Timestamp(price_rows.iloc[-1]["Date"])
        if price_available
        else pd.NaT
    )
    shares_date = (
        pd.Timestamp(share_rows.iloc[-1]["Date"])
        if shares_available
        else pd.NaT
    )
    split_adjusted_close = (
        float(price_rows.iloc[-1]["SplitAdjustedClose"])
        if price_available
        else math.nan
    )
    future_splits = (
        splits.loc[splits["Date"].gt(price_date), "Factor"]
        if price_available and not splits.empty
        else pd.Series(dtype=float)
    )
    future_split_factor = (
        float(future_splits.prod()) if len(future_splits) else 1.0
    )
    raw_close = split_adjusted_close * future_split_factor
    shares_outstanding = (
        float(share_rows.iloc[-1]["SharesOutstanding"])
        if shares_available
        else math.nan
    )
    market_cap = raw_close * shares_outstanding
    price_age = (
        int((as_of.normalize() - price_date.normalize()).days)
        if price_available
        else math.nan
    )
    shares_age = (
        int((as_of.normalize() - shares_date.normalize()).days)
        if shares_available
        else math.nan
    )
    eligible = bool(
        price_available
        and shares_available
        and np.isfinite(market_cap)
        and market_cap > 0
        and price_age <= settings.maximum_price_age_days
        and shares_age <= settings.maximum_shares_age_days
    )
    reasons: list[str] = []
    if not price_available:
        reasons.append("MISSING_PRICE")
    elif price_age > settings.maximum_price_age_days:
        reasons.append("STALE_PRICE")
    if not shares_available:
        reasons.append("MISSING_SHARES")
    elif shares_age > settings.maximum_shares_age_days:
        reasons.append("STALE_SHARES")
    return {
        "AsOfDate": as_of.date().isoformat(),
        "Ticker": ticker,
        "YahooTicker": yahoo_ticker(ticker),
        "Company": company,
        "PriceDate": (
            price_date.date().isoformat() if price_available else ""
        ),
        "SplitAdjustedClose": split_adjusted_close,
        "FutureSplitFactor": future_split_factor,
        "RawClose": raw_close,
        "SharesDate": (
            shares_date.date().isoformat() if shares_available else ""
        ),
        "SharesOutstanding": shares_outstanding,
        "MarketCap": market_cap,
        "PriceAgeDays": price_age,
        "SharesAgeDays": shares_age,
        "PriceDataAvailable": price_available,
        "SharesDataAvailable": shares_available,
        "EligibleForRanking": eligible,
        "ExclusionReason": ";".join(reasons),
        "SharesPointInTimeQuality": (
            "YAHOO_HISTORICAL_NOT_FIRST_SEEN_PIT_GUARANTEED"
        ),
    }


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
