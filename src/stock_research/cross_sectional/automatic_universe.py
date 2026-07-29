from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths
from stock_research.tickers import load_tickers

NASDAQ_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)
OTHER_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)
NASDAQ_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=10000&offset=0&download=true"
)

EXCHANGE_NAMES = {
    "Q": "NASDAQ Global Select",
    "G": "NASDAQ Global Market",
    "S": "NASDAQ Capital Market",
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe",
    "V": "IEX",
}

EXCLUDED_SECURITY_PATTERN = re.compile(
    r"\b("
    r"warrants?|rights?|units?|preferred|preference|"
    r"notes?|bonds?|debentures?|etf|etn|fund|"
    r"closed[- ]end|contingent value|subscription receipt"
    r"|when[- ]issued"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AutomaticUniverseSettings:
    target_size: int = 200
    minimum_price: float = 5.0
    minimum_market_cap: float = 1_000_000_000.0
    minimum_dollar_volume: float = 10_000_000.0
    minimum_ipo_age_years: int = 2
    exclude_spacs: bool = True
    request_timeout_seconds: float = 30.0
    nasdaq_listed_url: str = NASDAQ_LISTED_URL
    other_listed_url: str = OTHER_LISTED_URL
    nasdaq_screener_url: str = NASDAQ_SCREENER_URL

    def __post_init__(self) -> None:
        if self.target_size < 1:
            raise ValueError("target_size must be positive")
        if self.minimum_price < 0:
            raise ValueError("minimum_price must be non-negative")
        if self.minimum_market_cap < 0:
            raise ValueError("minimum_market_cap must be non-negative")
        if self.minimum_dollar_volume < 0:
            raise ValueError("minimum_dollar_volume must be non-negative")
        if self.minimum_ipo_age_years < 0:
            raise ValueError("minimum_ipo_age_years must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")


@dataclass(frozen=True)
class AutomaticUniverseArtifacts:
    output_dir: Path
    selected_universe: Path
    backfill_queue: Path
    audit: Path
    manifest: Path


def load_automatic_universe_settings(
    path: str | Path,
) -> AutomaticUniverseSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AutomaticUniverseSettings(
        target_size=int(raw.get("target_size", 200)),
        minimum_price=float(raw.get("minimum_price", 5.0)),
        minimum_market_cap=float(
            raw.get("minimum_market_cap", 1_000_000_000.0)
        ),
        minimum_dollar_volume=float(
            raw.get("minimum_dollar_volume", 10_000_000.0)
        ),
        minimum_ipo_age_years=int(
            raw.get("minimum_ipo_age_years", 2)
        ),
        exclude_spacs=bool(raw.get("exclude_spacs", True)),
        request_timeout_seconds=float(
            raw.get("request_timeout_seconds", 30.0)
        ),
        nasdaq_listed_url=str(
            raw.get("nasdaq_listed_url", NASDAQ_LISTED_URL)
        ),
        other_listed_url=str(
            raw.get("other_listed_url", OTHER_LISTED_URL)
        ),
        nasdaq_screener_url=str(
            raw.get("nasdaq_screener_url", NASDAQ_SCREENER_URL)
        ),
    )


def fetch_automatic_universe_sources(
    settings: AutomaticUniverseSettings,
    *,
    session: requests.Session | None = None,
) -> tuple[str, str, dict[str, Any]]:
    client = session or requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
    }
    nasdaq = client.get(
        settings.nasdaq_listed_url,
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    nasdaq.raise_for_status()
    other = client.get(
        settings.other_listed_url,
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    other.raise_for_status()
    screener = client.get(
        settings.nasdaq_screener_url,
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    screener.raise_for_status()
    payload = screener.json()
    if not isinstance(payload.get("data"), dict):
        raise TypeError("Nasdaq screener response does not contain data")
    if not isinstance(payload["data"].get("rows"), list):
        raise TypeError("Nasdaq screener response does not contain rows")
    return nasdaq.text, other.text, payload


def build_automatic_universe(
    nasdaq_listed_text: str,
    other_listed_text: str,
    screener_payload: dict[str, Any],
    settings: AutomaticUniverseSettings,
    *,
    as_of: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot_time = as_of or datetime.now(UTC)
    listings = pd.concat(
        [
            _parse_nasdaq_listed(nasdaq_listed_text),
            _parse_other_listed(other_listed_text),
        ],
        ignore_index=True,
    )
    listings = (
        listings.sort_values(
            ["DataSymbol", "ListingPriority", "ListingSymbol"]
        )
        .drop_duplicates("DataSymbol", keep="first")
        .reset_index(drop=True)
    )
    screener = _parse_screener(screener_payload)
    frame = listings.merge(
        screener,
        on="DataSymbol",
        how="left",
        validate="one_to_one",
    )
    frame["DollarVolume"] = frame["LastSale"] * frame["Volume"]
    frame["IPOAgeYears"] = (
        snapshot_time.year - frame["IPOYear"]
    ).where(frame["IPOYear"].notna())
    frame["IssuerKey"] = frame["SecurityName"].map(_issuer_key)
    frame["ExclusionReasons"] = [
        ";".join(
            _exclusion_reasons(row, settings)
        )
        for _, row in frame.iterrows()
    ]
    frame["Eligible"] = frame["ExclusionReasons"].eq("")
    issuer_candidates = frame.loc[frame["Eligible"]].sort_values(
        ["DollarVolume", "MarketCap", "DataSymbol"],
        ascending=[False, False, True],
    )
    duplicate_classes = issuer_candidates.duplicated(
        "IssuerKey",
        keep="first",
    )
    duplicate_indices = issuer_candidates.index[duplicate_classes]
    frame.loc[duplicate_indices, "ExclusionReasons"] = (
        "DUPLICATE_ISSUER_SHARE_CLASS"
    )
    frame["Eligible"] = frame["ExclusionReasons"].eq("")
    eligible = frame.loc[frame["Eligible"]].sort_values(
        ["DollarVolume", "MarketCap", "DataSymbol"],
        ascending=[False, False, True],
    )
    rank_by_symbol = pd.Series(
        range(1, len(eligible) + 1),
        index=eligible["DataSymbol"],
        dtype="Int64",
    )
    frame["LiquidityRank"] = frame["DataSymbol"].map(rank_by_symbol)
    frame["Selected"] = (
        frame["Eligible"]
        & frame["LiquidityRank"].le(settings.target_size)
    )
    frame["Status"] = "EXCLUDED"
    frame.loc[frame["Eligible"], "Status"] = "ELIGIBLE_NOT_SELECTED"
    frame.loc[frame["Selected"], "Status"] = "SELECTED"
    frame["SnapshotDate"] = snapshot_time.date().isoformat()
    frame = frame.sort_values(
        ["Selected", "Eligible", "LiquidityRank", "DataSymbol"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    selected = frame.loc[frame["Selected"]].copy()
    return selected, frame


def generate_automatic_universe(
    paths: ProjectPaths,
    settings: AutomaticUniverseSettings,
    *,
    ticker_config_path: str | Path,
    output_dir: str | Path | None = None,
    session: requests.Session | None = None,
) -> AutomaticUniverseArtifacts:
    retrieved_at = datetime.now(UTC)
    nasdaq_text, other_text, screener_payload = (
        fetch_automatic_universe_sources(settings, session=session)
    )
    selected, audit = build_automatic_universe(
        nasdaq_text,
        other_text,
        screener_payload,
        settings,
        as_of=retrieved_at,
    )
    selected = _attach_local_data_status(
        selected,
        paths,
        ticker_config_path,
    )
    local_by_symbol = selected.set_index("DataSymbol")[
        [
            "HasTickerConfig",
            "HasLocalPrice",
            "HasLocalFinancials",
            "V6Ready",
            "LocalPriceFile",
            "LocalFinancialFile",
        ]
    ]
    for column in local_by_symbol.columns:
        audit[column] = audit["DataSymbol"].map(local_by_symbol[column])
    for column in (
        "HasTickerConfig",
        "HasLocalPrice",
        "HasLocalFinancials",
        "V6Ready",
    ):
        audit[column] = audit[column].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
    for column in ("LocalPriceFile", "LocalFinancialFile"):
        audit[column] = audit[column].fillna("")

    timestamp = retrieved_at.strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "universe_snapshots"
            / f"{timestamp}_automatic_universe"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    selected_path = destination / "automatic_universe.csv"
    backfill_path = destination / "automatic_universe_backfill_queue.csv"
    audit_path = destination / "automatic_universe_audit.csv"
    manifest_path = destination / "automatic_universe_manifest.json"
    backfill = selected.loc[~selected["V6Ready"]].copy()
    backfill["MissingComponents"] = [
        ";".join(
            component
            for component, available in (
                ("TICKER_CONFIG", row.HasTickerConfig),
                ("PRICE_HISTORY", row.HasLocalPrice),
                ("QUARTERLY_FINANCIALS", row.HasLocalFinancials),
            )
            if not bool(available)
        )
        for row in backfill.itertuples(index=False)
    ]
    atomic_to_csv(selected, selected_path, index=False)
    atomic_to_csv(backfill, backfill_path, index=False)
    atomic_to_csv(audit, audit_path, index=False)
    _atomic_json(
        {
            "generated_at": retrieved_at.isoformat(),
            "methodology": {
                "listing_source": "Nasdaq Trader Symbol Directory",
                "market_snapshot_source": "Nasdaq Stock Screener",
                "ranking": (
                    "eligible securities sorted by current dollar-volume "
                    "proxy, then market capitalization"
                ),
                "liquidity_caveat": (
                    "DollarVolume uses the screener's current-session volume, "
                    "not a 63-session median."
                ),
                "survivorship_caveat": (
                    "This is a current snapshot and must not be applied "
                    "retroactively to historical backtests."
                ),
            },
            "settings": asdict(settings),
            "counts": {
                "directory_rows": len(audit),
                "eligible_rows": int(audit["Eligible"].sum()),
                "selected_rows": int(audit["Selected"].sum()),
                "selected_v6_ready": int(selected["V6Ready"].sum()),
                "selected_backfill_required": len(backfill),
            },
            "source_urls": {
                "nasdaq_listed": settings.nasdaq_listed_url,
                "other_listed": settings.other_listed_url,
                "screener": settings.nasdaq_screener_url,
            },
            "output_dir": str(destination),
        },
        manifest_path,
    )
    return AutomaticUniverseArtifacts(
        output_dir=destination,
        selected_universe=selected_path,
        backfill_queue=backfill_path,
        audit=audit_path,
        manifest=manifest_path,
    )


def _parse_nasdaq_listed(text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text), sep="|", dtype=str)
    frame = frame.loc[frame["Symbol"].notna()].copy()
    frame = frame.loc[
        ~frame["Symbol"].str.startswith("File Creation Time", na=False)
    ]
    return pd.DataFrame(
        {
            "ListingSymbol": frame["Symbol"].str.strip(),
            "DataSymbol": frame["Symbol"].map(_normalize_symbol),
            "SecurityName": frame["Security Name"].fillna("").str.strip(),
            "ExchangeCode": frame["Market Category"].fillna("").str.strip(),
            "Exchange": "NASDAQ",
            "ETF": frame["ETF"].fillna("").str.strip(),
            "TestIssue": frame["Test Issue"].fillna("").str.strip(),
            "FinancialStatus": (
                frame["Financial Status"].fillna("").str.strip()
            ),
            "ListingPriority": 0,
        }
    )


def _parse_other_listed(text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text), sep="|", dtype=str)
    frame = frame.loc[frame["ACT Symbol"].notna()].copy()
    frame = frame.loc[
        ~frame["ACT Symbol"].str.startswith("File Creation Time", na=False)
    ]
    exchange_code = frame["Exchange"].fillna("").str.strip()
    return pd.DataFrame(
        {
            "ListingSymbol": frame["ACT Symbol"].str.strip(),
            "DataSymbol": frame["ACT Symbol"].map(_normalize_symbol),
            "SecurityName": frame["Security Name"].fillna("").str.strip(),
            "ExchangeCode": exchange_code,
            "Exchange": exchange_code.map(EXCHANGE_NAMES).fillna(
                "OTHER"
            ),
            "ETF": frame["ETF"].fillna("").str.strip(),
            "TestIssue": frame["Test Issue"].fillna("").str.strip(),
            "FinancialStatus": "N",
            "ListingPriority": 1,
        }
    )


def _parse_screener(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("data", {}).get("rows", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Nasdaq screener returned no rows")
    required = {
        "symbol",
        "name",
        "lastsale",
        "volume",
        "marketCap",
        "country",
        "ipoyear",
        "industry",
        "sector",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Nasdaq screener rows are missing columns: {missing}"
        )
    result = pd.DataFrame(
        {
            "DataSymbol": frame["symbol"].map(_normalize_symbol),
            "ScreenerName": frame["name"].fillna("").str.strip(),
            "LastSale": frame["lastsale"].map(_parse_number),
            "Volume": frame["volume"].map(_parse_number),
            "MarketCap": frame["marketCap"].map(_parse_number),
            "Country": frame["country"].fillna("").str.strip(),
            "IPOYear": frame["ipoyear"].map(_parse_number),
            "Industry": frame["industry"].fillna("").str.strip(),
            "Sector": frame["sector"].fillna("").str.strip(),
        }
    )
    return (
        result.sort_values(["DataSymbol", "MarketCap"], ascending=[True, False])
        .drop_duplicates("DataSymbol", keep="first")
        .reset_index(drop=True)
    )


def _exclusion_reasons(
    row: pd.Series,
    settings: AutomaticUniverseSettings,
) -> list[str]:
    reasons: list[str] = []
    name = str(row.get("SecurityName", "") or "")
    industry = str(row.get("Industry", "") or "")
    if str(row.get("ETF", "")).upper() == "Y":
        reasons.append("ETF")
    if str(row.get("TestIssue", "")).upper() == "Y":
        reasons.append("TEST_ISSUE")
    financial_status = str(row.get("FinancialStatus", "")).upper()
    if financial_status not in {"", "N"}:
        reasons.append("ABNORMAL_FINANCIAL_STATUS")
    if EXCLUDED_SECURITY_PATTERN.search(name):
        reasons.append("EXCLUDED_SECURITY_TYPE")
    if settings.exclude_spacs and (
        industry.casefold() == "blank checks"
        or (
            "acquisition" in name.casefold()
            and re.search(
                r"\b(corp|corporation|inc|ltd|plc|company)\b",
                name,
                flags=re.IGNORECASE,
            )
        )
    ):
        reasons.append("SPAC_OR_BLANK_CHECK")
    if pd.isna(row.get("LastSale")):
        reasons.append("MISSING_PRICE")
    elif float(row["LastSale"]) < settings.minimum_price:
        reasons.append("PRICE_BELOW_MINIMUM")
    if pd.isna(row.get("MarketCap")):
        reasons.append("MISSING_MARKET_CAP")
    elif float(row["MarketCap"]) < settings.minimum_market_cap:
        reasons.append("MARKET_CAP_BELOW_MINIMUM")
    if pd.isna(row.get("DollarVolume")):
        reasons.append("MISSING_DOLLAR_VOLUME")
    elif float(row["DollarVolume"]) < settings.minimum_dollar_volume:
        reasons.append("DOLLAR_VOLUME_BELOW_MINIMUM")
    ipo_age = row.get("IPOAgeYears")
    if pd.notna(ipo_age) and int(ipo_age) < settings.minimum_ipo_age_years:
        reasons.append("IPO_TOO_RECENT")
    return list(dict.fromkeys(reasons))


def _attach_local_data_status(
    selected: pd.DataFrame,
    paths: ProjectPaths,
    ticker_config_path: str | Path,
) -> pd.DataFrame:
    frame = selected.copy()
    configs = load_tickers(ticker_config_path)
    status: list[dict[str, object]] = []
    for ticker in frame["DataSymbol"].astype(str):
        config = configs.get(ticker)
        if config is None:
            price_path = None
        else:
            price_matches = sorted(
                paths.processed.glob(f"{config.display_name}_*.csv"),
                key=lambda path: (path.stat().st_mtime, path.name),
                reverse=True,
            )
            price_path = price_matches[0] if price_matches else None
        financial_path = paths.financial_raw / f"{ticker}_financials_Q.xlsx"
        has_financials = financial_path.exists()
        status.append(
            {
                "HasTickerConfig": config is not None,
                "HasLocalPrice": price_path is not None,
                "HasLocalFinancials": has_financials,
                "V6Ready": bool(
                    config is not None
                    and price_path is not None
                    and has_financials
                ),
                "LocalPriceFile": (
                    price_path.name if price_path is not None else ""
                ),
                "LocalFinancialFile": (
                    financial_path.name if has_financials else ""
                ),
            }
        )
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(status)],
        axis=1,
    )


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def _issuer_key(value: object) -> str:
    name = str(value or "").upper()
    name = re.split(r"\s+-\s+", name, maxsplit=1)[0]
    name = re.sub(
        r"\b(CLASS [A-Z0-9]+|COMMON STOCK|COMMON SHARES|"
        r"ORDINARY SHARES|CAPITAL STOCK|AMERICAN DEPOSITARY SHARES|"
        r"ADS)\b.*$",
        "",
        name,
    )
    normalized = re.sub(r"[^A-Z0-9]+", " ", name).strip()
    return normalized


def _parse_number(value: object) -> float:
    if value is None:
        return float("nan")
    cleaned = str(value).strip().replace("$", "").replace(",", "")
    if cleaned in {"", "--", "N/A", "NA", "None"}:
        return float("nan")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.stem + "_",
        dir=path.parent,
    )
    os.close(descriptor)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
