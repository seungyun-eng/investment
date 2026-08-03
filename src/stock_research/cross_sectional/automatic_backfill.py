from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from stock_research.financial_analysis import load_financial_workbook
from stock_research.financial_crawler import (
    create_http_session,
    resolve_company_slug_http,
    scrape_financials,
)
from stock_research.indicators import preprocess_company_dir
from stock_research.io_utils import atomic_to_csv
from stock_research.market_data import update_one_price
from stock_research.paths import ProjectPaths
from stock_research.tickers import TickerConfig, load_tickers
from stock_research.tsla_integrated.data import load_equity_prices

T = TypeVar("T")

REQUIRED_FINANCIAL_SHEETS = {
    "Income Statement",
    "Balance Sheet",
    "Cash Flow Statement",
    "Key Financial Ratios",
}

SECURITY_SUFFIX_PATTERN = re.compile(
    r"\b("
    r"class\s+[a-z0-9]+|common stock|ordinary shares?|"
    r"american depositary shares?|new york registry shares?|"
    r"capital stock|depositary shares?"
    r")\b.*$",
    flags=re.IGNORECASE,
)
CORPORATE_SUFFIX_PATTERN = re.compile(
    r"(?:\s+|^)(incorporated|inc|corporation|corp|company|"
    r"limited|ltd|plc|n\.?v\.?|s\.?a\.?)\.?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AutomaticBackfillSettings:
    initial_price_start: str = "2022-01-01"
    price_transport: str = "yahoo_chart"
    minimum_price_rows: int = 252
    maximum_price_age_days: int = 10
    minimum_financial_periods: int = 8
    financial_frequency: str = "Q"
    financial_transport: str = "http"
    price_request_pause_seconds: float = 0.25
    request_pause_seconds: float = 1.0
    ticker_pause_seconds: float = 10.0
    checkpoint_interval: int = 25
    retries: int = 2
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        datetime.fromisoformat(self.initial_price_start)
        if self.price_transport not in {"yfinance", "yahoo_chart"}:
            raise ValueError(
                "price_transport must be yfinance or yahoo_chart"
            )
        if self.minimum_price_rows < 1:
            raise ValueError("minimum_price_rows must be positive")
        if self.maximum_price_age_days < 0:
            raise ValueError("maximum_price_age_days must be non-negative")
        if self.minimum_financial_periods < 1:
            raise ValueError("minimum_financial_periods must be positive")
        if self.financial_frequency not in {"Q", "A"}:
            raise ValueError("financial_frequency must be Q or A")
        if self.financial_transport not in {"http", "selenium"}:
            raise ValueError("financial_transport must be http or selenium")
        if self.price_request_pause_seconds < 0:
            raise ValueError(
                "price_request_pause_seconds must be non-negative"
            )
        if self.request_pause_seconds < 0:
            raise ValueError("request_pause_seconds must be non-negative")
        if self.ticker_pause_seconds < 0:
            raise ValueError("ticker_pause_seconds must be non-negative")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")


@dataclass(frozen=True)
class AutomaticBackfillArtifacts:
    output_dir: Path
    ticker_config: Path
    status: Path
    manifest: Path


def load_automatic_backfill_settings(
    path: str | Path,
) -> AutomaticBackfillSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AutomaticBackfillSettings(
        initial_price_start=str(
            raw.get("initial_price_start", "2022-01-01")
        ),
        price_transport=str(
            raw.get("price_transport", "yahoo_chart")
        ).lower(),
        minimum_price_rows=int(raw.get("minimum_price_rows", 252)),
        maximum_price_age_days=int(
            raw.get("maximum_price_age_days", 10)
        ),
        minimum_financial_periods=int(
            raw.get("minimum_financial_periods", 8)
        ),
        financial_frequency=str(
            raw.get("financial_frequency", "Q")
        ).upper(),
        financial_transport=str(
            raw.get("financial_transport", "http")
        ).lower(),
        price_request_pause_seconds=float(
            raw.get("price_request_pause_seconds", 0.25)
        ),
        request_pause_seconds=float(
            raw.get("request_pause_seconds", 1.0)
        ),
        ticker_pause_seconds=float(
            raw.get("ticker_pause_seconds", 10.0)
        ),
        checkpoint_interval=int(raw.get("checkpoint_interval", 25)),
        retries=int(raw.get("retries", 2)),
        retry_backoff_seconds=float(
            raw.get("retry_backoff_seconds", 2.0)
        ),
        request_timeout_seconds=float(
            raw.get("request_timeout_seconds", 30.0)
        ),
    )


def find_latest_automatic_universe(paths: ProjectPaths) -> Path:
    root = paths.results / "Cross_Sectional" / "universe_snapshots"
    candidates = list(root.glob("*/automatic_universe.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No automatic_universe.csv found below {root}"
        )
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
    )


def build_automatic_ticker_configs(
    universe: pd.DataFrame,
    base_configs: dict[str, TickerConfig],
) -> dict[str, TickerConfig]:
    if "DataSymbol" not in universe:
        raise ValueError("Universe requires a DataSymbol column")
    configs: dict[str, TickerConfig] = {}
    used_display_names: set[str] = set()
    ordered = universe.sort_values(
        "LiquidityRank" if "LiquidityRank" in universe else "DataSymbol"
    )
    for row in ordered.to_dict(orient="records"):
        ticker = str(row["DataSymbol"]).strip().upper()
        if not ticker:
            continue
        base = base_configs.get(ticker)
        if base is not None:
            config = base
        else:
            company_name = _company_name(row)
            display_name = _safe_display_name(company_name, ticker)
            if display_name.casefold() in used_display_names:
                display_name = f"{display_name} {ticker}"
            config = TickerConfig(
                ticker=ticker,
                company_slug=_slugify_company(company_name, ticker),
                fiscal_year_end_month=12,
                display_name_override=display_name,
            )
        used_display_names.add(config.display_name.casefold())
        configs[ticker] = config
    return configs


def run_automatic_backfill(
    paths: ProjectPaths,
    settings: AutomaticBackfillSettings,
    *,
    universe_path: str | Path,
    base_ticker_config_path: str | Path,
    output_dir: str | Path | None = None,
    selected_tickers: list[str] | None = None,
    limit: int | None = None,
    stage: str = "all",
    force: bool = False,
    as_of: datetime | None = None,
) -> AutomaticBackfillArtifacts:
    if stage not in {"all", "price", "financial"}:
        raise ValueError("stage must be all, price, or financial")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    started_at = as_of or datetime.now(UTC)
    source_path = Path(universe_path).expanduser().resolve()
    universe = pd.read_csv(source_path)
    base_configs = load_tickers(base_ticker_config_path)
    configs = build_automatic_ticker_configs(universe, base_configs)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "backfill_runs"
            / (
                started_at.strftime("%Y%m%d_%H%M%S_%f")
                + "_automatic_universe_backfill"
            )
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "automatic_tickers.json"
    status_path = destination / "automatic_backfill_status.csv"
    manifest_path = destination / "automatic_backfill_manifest.json"
    _atomic_ticker_config(configs, config_path)

    ordered_symbols = [
        str(value).upper()
        for value in universe.sort_values(
            "LiquidityRank" if "LiquidityRank" in universe else "DataSymbol"
        )["DataSymbol"]
    ]
    block_by_ticker = (
        universe.set_index("DataSymbol")["CrawlBlockReason"]
        .fillna("")
        .astype(str)
        .to_dict()
        if "CrawlBlockReason" in universe
        else {}
    )
    blocked_symbols = {
        str(ticker).upper()
        for ticker, reason in block_by_ticker.items()
        if reason.strip()
    }
    requested = (
        {ticker.upper() for ticker in selected_tickers}
        if selected_tickers
        else None
    )
    if requested is not None:
        missing = requested - set(ordered_symbols)
        if missing:
            raise ValueError(
                "Requested tickers are not in the universe: "
                + ", ".join(sorted(missing))
            )
        if blocked := requested & blocked_symbols:
            raise ValueError(
                "Requested tickers are blocked from automatic crawling: "
                + ", ".join(sorted(blocked))
            )
        batch = [
            ticker for ticker in ordered_symbols if ticker in requested
        ]
    else:
        batch = [
            ticker
            for ticker in ordered_symbols
            if ticker not in blocked_symbols
            and (
                force
                or not _is_v6_ready(
                    paths,
                    configs[ticker],
                    settings,
                    as_of=started_at,
                )
            )
        ]
    if limit is not None:
        batch = batch[:limit]
    batch_set = set(batch)

    client = (
        create_http_session()
        if stage in {"all", "financial"}
        and settings.financial_transport == "http"
        else None
    )
    run_details: dict[str, dict[str, str]] = {}
    status: pd.DataFrame | None = None
    initial_start = datetime.fromisoformat(settings.initial_price_start)
    for batch_index, ticker in enumerate(batch, start=1):
        config = configs[ticker]
        details = {
            "PriceAction": "NOT_REQUESTED",
            "FinancialAction": "NOT_REQUESTED",
            "Errors": "",
        }
        errors: list[str] = []
        made_price_request = False
        made_financial_request = False
        print(f"BACKFILL {ticker}: {config.display_name}")

        if stage in {"all", "price"}:
            price_before = _price_status(
                paths,
                config,
                settings,
                as_of=started_at,
            )
            if price_before["Valid"] and not force:
                details["PriceAction"] = "SKIPPED_VALID"
            else:
                try:
                    made_price_request = True
                    raw_output = update_one_price(
                        config,
                        paths.raw_prices,
                        initial_start=initial_start,
                        refresh_start=initial_start if force else None,
                        transport=settings.price_transport,
                        request_timeout_seconds=(
                            settings.request_timeout_seconds
                        ),
                    )
                    if raw_output is None:
                        raise ValueError("price download returned no data")
                    processed = preprocess_company_dir(
                        paths.raw_prices / config.display_name,
                        paths.processed,
                    )
                    if processed is None:
                        raise ValueError("price preprocessing returned no data")
                    details["PriceAction"] = "UPDATED"
                except Exception as exc:  # noqa: BLE001
                    details["PriceAction"] = "FAILED"
                    errors.append(f"PRICE: {exc}")

        if stage in {"all", "financial"}:
            financial_before = _financial_status(
                paths,
                config,
                settings,
            )
            if financial_before["Valid"] and not force:
                details["FinancialAction"] = "SKIPPED_VALID"
            else:
                try:
                    made_financial_request = True
                    resolved = config
                    if client is not None:
                        resolved = _retry(
                            lambda config=config: resolve_company_slug_http(
                                client,
                                config,
                                timeout_seconds=(
                                    settings.request_timeout_seconds
                                ),
                            ),
                            attempts=settings.retries + 1,
                            backoff_seconds=settings.retry_backoff_seconds,
                        )
                        configs[ticker] = resolved
                        _atomic_ticker_config(configs, config_path)
                        time.sleep(settings.request_pause_seconds)
                    outputs = scrape_financials(
                        {ticker: resolved},
                        paths.financial_raw,
                        selected=[ticker],
                        frequency=settings.financial_frequency,
                        headless=True,
                        retries=settings.retries,
                        page_load_timeout_seconds=(
                            settings.request_timeout_seconds
                        ),
                        transport=settings.financial_transport,
                        request_pause_seconds=(
                            settings.request_pause_seconds
                        ),
                        retry_backoff_seconds=(
                            settings.retry_backoff_seconds
                        ),
                    )
                    if not outputs:
                        raise ValueError(
                            "financial crawler produced no validated workbook"
                        )
                    details["FinancialAction"] = "UPDATED"
                except Exception as exc:  # noqa: BLE001
                    details["FinancialAction"] = "FAILED"
                    errors.append(f"FINANCIAL: {exc}")

        details["Errors"] = "; ".join(errors)
        run_details[ticker] = details
        if (
            batch_index % settings.checkpoint_interval == 0
            or batch_index == len(batch)
        ):
            status = _build_status(
                paths,
                universe,
                configs,
                settings,
                batch_set=batch_set,
                run_details=run_details,
                as_of=started_at,
            )
            atomic_to_csv(status, status_path, index=False)
        if made_price_request:
            time.sleep(settings.price_request_pause_seconds)
        if made_financial_request:
            time.sleep(settings.ticker_pause_seconds)

    if status is None:
        status = _build_status(
            paths,
            universe,
            configs,
            settings,
            batch_set=batch_set,
            run_details=run_details,
            as_of=started_at,
        )
        atomic_to_csv(status, status_path, index=False)
    _atomic_ticker_config(configs, config_path)
    completed_at = datetime.now(UTC)
    _atomic_json(
        {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "source_universe": str(source_path),
            "settings": asdict(settings),
            "request": {
                "stage": stage,
                "force": force,
                "limit": limit,
                "selected_tickers": sorted(requested or []),
            },
            "counts": {
                "universe": len(status),
                "batch": len(batch),
                "price_valid": int(status["PriceValid"].sum()),
                "financial_valid": int(status["FinancialValid"].sum()),
                "v6_ready": int(status["V6Ready"].sum()),
                "batch_failures": int(
                    status.loc[status["RunSelected"], "Errors"].ne("").sum()
                ),
            },
            "caveats": {
                "financial_source": (
                    "Macrotrends pages collected through the existing "
                    "project crawler."
                ),
                "fiscal_year_end": (
                    "Automatically generated ticker configs default unknown "
                    "fiscal year ends to December; cross-sectional loading "
                    "uses reported statement dates."
                ),
                "point_in_time": (
                    "These workbooks contain statement period dates, not "
                    "original filing timestamps. The V6 release-lag rule "
                    "must remain enabled."
                ),
            },
            "outputs": {
                "ticker_config": str(config_path),
                "status": str(status_path),
            },
        },
        manifest_path,
    )
    return AutomaticBackfillArtifacts(
        output_dir=destination,
        ticker_config=config_path,
        status=status_path,
        manifest=manifest_path,
    )


def _company_name(row: dict[str, Any]) -> str:
    value = str(
        row.get("ScreenerName")
        or row.get("SecurityName")
        or row.get("DataSymbol")
        or ""
    ).strip()
    value = re.split(r"\s+-\s+", value, maxsplit=1)[0]
    value = SECURITY_SUFFIX_PATTERN.sub("", value).strip(" ,-")
    previous = None
    while previous != value:
        previous = value
        value = CORPORATE_SUFFIX_PATTERN.sub("", value).strip(" ,.-")
    return value or str(row.get("DataSymbol", "")).strip()


def _safe_display_name(company_name: str, ticker: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]+', " ", company_name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or ticker


def _slugify_company(company_name: str, ticker: str) -> str:
    normalised = unicodedata.normalize("NFKD", company_name)
    ascii_name = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("&", " and ")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or ticker.lower().replace(".", "-")


def _processed_path(
    paths: ProjectPaths,
    config: TickerConfig,
) -> Path | None:
    matches = sorted(
        paths.processed.glob(f"{config.display_name}_*.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return matches[0] if matches else None


def _price_status(
    paths: ProjectPaths,
    config: TickerConfig,
    settings: AutomaticBackfillSettings,
    *,
    as_of: datetime,
) -> dict[str, Any]:
    path = _processed_path(paths, config)
    result: dict[str, Any] = {
        "Valid": False,
        "Rows": 0,
        "Start": "",
        "End": "",
        "Path": str(path) if path else "",
        "Issue": "MISSING",
    }
    if path is None:
        return result
    try:
        frame = load_equity_prices(path)
        if frame.empty:
            result["Issue"] = "EMPTY"
            return result
        start = pd.Timestamp(frame["Date"].min())
        end = pd.Timestamp(frame["Date"].max())
        age_days = (as_of.date() - end.date()).days
        issues = []
        if len(frame) < settings.minimum_price_rows:
            issues.append("TOO_FEW_ROWS")
        if age_days > settings.maximum_price_age_days:
            issues.append("STALE")
        result.update(
            {
                "Valid": not issues,
                "Rows": len(frame),
                "Start": start.date().isoformat(),
                "End": end.date().isoformat(),
                "Issue": ";".join(issues),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["Issue"] = f"INVALID: {exc}"
    return result


def _financial_status(
    paths: ProjectPaths,
    config: TickerConfig,
    settings: AutomaticBackfillSettings,
) -> dict[str, Any]:
    path = (
        paths.financial_raw
        / f"{config.ticker}_financials_{settings.financial_frequency}.xlsx"
    )
    result: dict[str, Any] = {
        "Valid": False,
        "Periods": 0,
        "Start": "",
        "End": "",
        "Path": str(path) if path.exists() else "",
        "Issue": "MISSING",
    }
    if not path.exists():
        return result
    try:
        with pd.ExcelFile(path) as workbook:
            missing_sheets = REQUIRED_FINANCIAL_SHEETS - set(
                workbook.sheet_names
            )
        frame = load_financial_workbook(path)
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        issues = []
        if missing_sheets:
            issues.append(
                "MISSING_SHEETS:" + ",".join(sorted(missing_sheets))
            )
        if dates.nunique() < settings.minimum_financial_periods:
            issues.append("TOO_FEW_PERIODS")
        result.update(
            {
                "Valid": not issues,
                "Periods": int(dates.nunique()),
                "Start": (
                    dates.min().date().isoformat() if not dates.empty else ""
                ),
                "End": (
                    dates.max().date().isoformat() if not dates.empty else ""
                ),
                "Issue": ";".join(issues),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["Issue"] = f"INVALID: {exc}"
    return result


def _is_v6_ready(
    paths: ProjectPaths,
    config: TickerConfig,
    settings: AutomaticBackfillSettings,
    *,
    as_of: datetime,
) -> bool:
    return bool(
        _price_status(
            paths,
            config,
            settings,
            as_of=as_of,
        )["Valid"]
        and _financial_status(paths, config, settings)["Valid"]
    )


def _build_status(
    paths: ProjectPaths,
    universe: pd.DataFrame,
    configs: dict[str, TickerConfig],
    settings: AutomaticBackfillSettings,
    *,
    batch_set: set[str],
    run_details: dict[str, dict[str, str]],
    as_of: datetime,
) -> pd.DataFrame:
    rank_by_ticker = universe.set_index("DataSymbol").get(
        "LiquidityRank",
        pd.Series(dtype=float),
    )
    block_by_ticker = universe.set_index("DataSymbol").get(
        "CrawlBlockReason",
        pd.Series(dtype=object),
    )
    rows: list[dict[str, Any]] = []
    for ticker, config in configs.items():
        price = _price_status(
            paths,
            config,
            settings,
            as_of=as_of,
        )
        financial = _financial_status(paths, config, settings)
        details = run_details.get(
            ticker,
            {
                "PriceAction": (
                    "NOT_IN_BATCH" if ticker not in batch_set else "PENDING"
                ),
                "FinancialAction": (
                    "NOT_IN_BATCH" if ticker not in batch_set else "PENDING"
                ),
                "Errors": "",
            },
        )
        rows.append(
            {
                "Ticker": ticker,
                "LiquidityRank": rank_by_ticker.get(ticker, pd.NA),
                "Company": config.display_name,
                "CompanySlug": config.company_slug,
                "CrawlBlockReason": block_by_ticker.get(ticker, ""),
                "RunSelected": ticker in batch_set,
                **details,
                "PriceValid": bool(price["Valid"]),
                "PriceRows": int(price["Rows"]),
                "PriceStart": price["Start"],
                "PriceEnd": price["End"],
                "PriceIssue": price["Issue"],
                "PricePath": price["Path"],
                "FinancialValid": bool(financial["Valid"]),
                "FinancialPeriods": int(financial["Periods"]),
                "FinancialStart": financial["Start"],
                "FinancialEnd": financial["End"],
                "FinancialIssue": financial["Issue"],
                "FinancialPath": financial["Path"],
                "V6Ready": bool(price["Valid"] and financial["Valid"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["LiquidityRank", "Ticker"],
        na_position="last",
    ).reset_index(drop=True)


def _retry(
    operation: Callable[[], T],
    *,
    attempts: int,
    backoff_seconds: float,
) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(backoff_seconds * (2**attempt))
    assert last_error is not None
    raise last_error


def _atomic_ticker_config(
    configs: dict[str, TickerConfig],
    path: Path,
) -> None:
    payload: dict[str, dict[str, Any]] = {}
    for ticker, config in configs.items():
        payload[ticker] = {
            "company_slug": config.company_slug,
            "fiscal_year_end_month": config.fiscal_year_end_month,
            "display_name": config.display_name,
        }
    _atomic_json(payload, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.stem + "_",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(tmp_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
