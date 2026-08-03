from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths

HISTORICAL_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/"
    "main/sp_500_historical_components.csv"
)
WIKIPEDIA_SP500_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
HTTP_USER_AGENT = (
    "Mozilla/5.0 (compatible; stock-research-sp500-universe/0.1; "
    "+local-research-pipeline)"
)

# Only map ticker changes that preserve the same underlying operating company.
# Mergers and acquisitions intentionally remain separate historical securities.
DATA_TICKER_ALIASES = {
    "ABC": "COR",
    "ANTM": "ELV",
    "BK": "BNY",
    "BLL": "BALL",
    "CDAY": "DAY",
    "CTL": "LUMN",
    "FB": "META",
    "FBHS": "FBIN",
    "FI": "FISV",
    "FLT": "CPAY",
    "GPS": "GAP",
    "JEC": "J",
    "LB": "BBWI",
    "MMC": "MRSH",
    "NLOK": "GEN",
    "PKI": "RVTY",
    "RE": "EG",
    "SYMC": "GEN",
    "TMK": "GL",
    "WLTW": "WTW",
}

HISTORICAL_COMPANY_NAMES = {
    "BBT": "BB&T",
    "CBS": "CBS Corporation",
    "COTY": "Coty",
    "ETSY": "Etsy",
    "IBM": "International Business Machines",
    "MYL": "Mylan",
    "NOV": "NOV",
    "PARA": "Paramount Global",
    "PEAK": "Healthpeak Properties",
    "PKI": "PerkinElmer",
    "PVH": "PVH",
    "UBER": "Uber Technologies",
    "UTX": "United Technologies",
    "VIAC": "ViacomCBS",
    "WRK": "WestRock",
}

KNOWN_REUSED_TICKERS = {
    "APC",
    "NFX",
}


@dataclass(frozen=True)
class Sp500UniverseSettings:
    snapshot_years: tuple[int, ...] = tuple(range(2019, 2027))
    snapshot_month: int = 1
    snapshot_day: int = 1
    request_timeout_seconds: float = 30.0
    historical_constituents_url: str = HISTORICAL_CONSTITUENTS_URL
    wikipedia_url: str = WIKIPEDIA_SP500_URL

    def __post_init__(self) -> None:
        if not self.snapshot_years:
            raise ValueError("snapshot_years cannot be empty")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        for year in self.snapshot_years:
            datetime(
                year,
                self.snapshot_month,
                self.snapshot_day,
                tzinfo=UTC,
            )


@dataclass(frozen=True)
class Sp500UniverseArtifacts:
    output_dir: Path
    membership_csv: Path
    change_membership_csv: Path
    union_csv: Path
    historical_source_csv: Path
    wikipedia_current_csv: Path
    wikipedia_changes_csv: Path
    manifest_json: Path


def load_sp500_universe_settings(
    path: str | Path,
    *,
    years: Iterable[int] | None = None,
) -> Sp500UniverseSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    selected_years = (
        tuple(int(year) for year in years)
        if years is not None
        else tuple(int(year) for year in raw["snapshot_years"])
    )
    return Sp500UniverseSettings(
        snapshot_years=selected_years,
        snapshot_month=int(raw.get("snapshot_month", 1)),
        snapshot_day=int(raw.get("snapshot_day", 1)),
        request_timeout_seconds=float(
            raw.get("request_timeout_seconds", 30.0)
        ),
        historical_constituents_url=str(
            raw.get(
                "historical_constituents_url",
                HISTORICAL_CONSTITUENTS_URL,
            )
        ),
        wikipedia_url=str(
            raw.get("wikipedia_url", WIKIPEDIA_SP500_URL)
        ),
    )


def fetch_historical_sp500(
    settings: Sp500UniverseSettings,
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
    return pd.read_csv(StringIO(response.text))


def fetch_wikipedia_sp500(
    settings: Sp500UniverseSettings,
    *,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = session or requests.Session()
    response = client.get(
        settings.wikipedia_url,
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return parse_wikipedia_sp500_tables(tables)


def parse_wikipedia_sp500_tables(
    tables: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_source: pd.DataFrame | None = None
    changes_source: pd.DataFrame | None = None
    for table in tables:
        flattened = {_flatten_column(column) for column in table.columns}
        if {"symbol", "security"}.issubset(flattened):
            current_source = table
        if {
            "effective date effective date",
            "added ticker",
            "removed ticker",
        }.issubset(flattened):
            changes_source = table
    if current_source is None:
        raise ValueError("Wikipedia S&P 500 current-members table not found")
    if changes_source is None:
        raise ValueError("Wikipedia S&P 500 changes table not found")

    current_lookup = {
        _flatten_column(column): column for column in current_source.columns
    }
    current = pd.DataFrame(
        {
            "Ticker": current_source[current_lookup["symbol"]].map(
                normalize_sp500_ticker
            ),
            "Company": current_source[current_lookup["security"]]
            .astype(str)
            .str.strip(),
            "GicsSector": _optional_column(
                current_source,
                current_lookup,
                "gics sector",
            ),
            "GicsSubIndustry": _optional_column(
                current_source,
                current_lookup,
                "gics sub-industry",
            ),
            "Cik": _optional_column(
                current_source,
                current_lookup,
                "cik",
            ),
        }
    )
    current = (
        current.loc[current["Ticker"].ne("")]
        .drop_duplicates("Ticker")
        .sort_values("Ticker")
        .reset_index(drop=True)
    )

    changes_lookup = {
        _flatten_column(column): column for column in changes_source.columns
    }
    changes = pd.DataFrame(
        {
            "EffectiveDate": pd.to_datetime(
                changes_source[
                    changes_lookup["effective date effective date"]
                ],
                errors="coerce",
            ),
            "AddedTicker": changes_source[
                changes_lookup["added ticker"]
            ].map(normalize_sp500_ticker),
            "AddedCompany": changes_source[
                changes_lookup["added security"]
            ]
            .fillna("")
            .astype(str)
            .str.strip(),
            "RemovedTicker": changes_source[
                changes_lookup["removed ticker"]
            ].map(normalize_sp500_ticker),
            "RemovedCompany": changes_source[
                changes_lookup["removed security"]
            ]
            .fillna("")
            .astype(str)
            .str.strip(),
            "Reason": changes_source[changes_lookup["reason reason"]]
            .fillna("")
            .astype(str)
            .str.strip(),
        }
    )
    changes = (
        changes.dropna(subset=["EffectiveDate"])
        .sort_values(
            ["EffectiveDate", "AddedTicker", "RemovedTicker"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    return current, changes


def build_sp500_membership(
    history: pd.DataFrame,
    wikipedia_current: pd.DataFrame,
    wikipedia_changes: pd.DataFrame,
    settings: Sp500UniverseSettings,
    *,
    fetched_at: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    required_history = {"date", "tickers"}
    missing = required_history - set(history.columns)
    if missing:
        raise ValueError(
            f"Historical S&P source missing columns: {sorted(missing)}"
        )
    history_frame = history.copy()
    history_frame["date"] = pd.to_datetime(
        history_frame["date"],
        errors="coerce",
    )
    history_frame = history_frame.dropna(subset=["date"]).sort_values("date")
    if history_frame.empty:
        raise ValueError("Historical S&P source has no dated snapshots")
    history_max = pd.Timestamp(history_frame["date"].max()).normalize()

    current_required = {"Ticker", "Company"}
    if missing_current := current_required - set(wikipedia_current.columns):
        raise ValueError(
            "Wikipedia current members missing columns: "
            f"{sorted(missing_current)}"
        )
    changes_required = {
        "EffectiveDate",
        "AddedTicker",
        "AddedCompany",
        "RemovedTicker",
        "RemovedCompany",
    }
    if missing_changes := changes_required - set(wikipedia_changes.columns):
        raise ValueError(
            f"Wikipedia changes missing columns: {sorted(missing_changes)}"
        )

    fetch_date = pd.Timestamp(
        fetched_at if fetched_at is not None else datetime.now(UTC)
    )
    fetch_date = fetch_date.tz_localize(None).normalize()
    company_names = _company_name_map(
        wikipedia_current,
        wikipedia_changes,
    )
    rows: list[dict[str, object]] = []
    for year in settings.snapshot_years:
        target = pd.Timestamp(
            year=year,
            month=settings.snapshot_month,
            day=settings.snapshot_day,
        )
        if target <= history_max:
            source_date, historical_tickers = _historical_members_as_of(
                history_frame,
                target,
            )
            source = "HANS_HISTORICAL_COMPONENTS"
        else:
            if target > fetch_date:
                raise ValueError(
                    f"Cannot reconstruct future membership for {target.date()}"
                )
            source_date = fetch_date
            historical_tickers = _reverse_current_membership(
                wikipedia_current,
                wikipedia_changes,
                target,
                fetch_date,
            )
            source = "WIKIPEDIA_CURRENT_REVERSED_CHANGES"

        by_data_symbol: dict[str, set[str]] = {}
        for historical_ticker in historical_tickers:
            data_symbol = data_ticker(historical_ticker)
            by_data_symbol.setdefault(data_symbol, set()).add(
                historical_ticker
            )
        for rank, data_symbol in enumerate(
            sorted(by_data_symbol),
            start=1,
        ):
            historical_symbols = sorted(by_data_symbol[data_symbol])
            company = next(
                (
                    company_names.get(ticker, "")
                    for ticker in historical_symbols
                    if company_names.get(ticker, "")
                ),
                company_names.get(data_symbol, ""),
            )
            rows.append(
                {
                    "AsOfDate": target.date().isoformat(),
                    "Ticker": data_symbol,
                    "DataSymbol": data_symbol,
                    "HistoricalTickers": ",".join(historical_symbols),
                    "Company": company or data_symbol,
                    "Rank": rank,
                    "Selected": True,
                    "MembershipSource": source,
                    "MembershipSourceDate": source_date.date().isoformat(),
                    "PointInTimeStatus": (
                        "RETROSPECTIVE_PUBLIC_MEMBERSHIP_HISTORY"
                    ),
                }
            )
    membership = pd.DataFrame(rows)
    if membership.empty:
        raise ValueError("No S&P 500 membership snapshots were produced")
    if membership.duplicated(["AsOfDate", "Ticker"]).any():
        raise ValueError("Duplicate data ticker in S&P membership snapshot")
    return membership.sort_values(
        ["AsOfDate", "Rank", "Ticker"]
    ).reset_index(drop=True)


def build_sp500_change_membership(
    history: pd.DataFrame,
    wikipedia_current: pd.DataFrame,
    wikipedia_changes: pd.DataFrame,
    settings: Sp500UniverseSettings,
    *,
    fetched_at: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    required_history = {"date", "tickers"}
    if missing := required_history - set(history.columns):
        raise ValueError(
            f"Historical S&P source missing columns: {sorted(missing)}"
        )
    history_frame = history.copy()
    history_frame["date"] = pd.to_datetime(
        history_frame["date"],
        errors="coerce",
    )
    history_frame = history_frame.dropna(subset=["date"]).sort_values("date")
    if history_frame.empty:
        raise ValueError("Historical S&P source has no dated snapshots")

    start = pd.Timestamp(
        year=min(settings.snapshot_years),
        month=settings.snapshot_month,
        day=settings.snapshot_day,
    )
    fetch_date = pd.Timestamp(
        fetched_at if fetched_at is not None else datetime.now(UTC)
    )
    fetch_date = fetch_date.tz_localize(None).normalize()
    history_max = pd.Timestamp(history_frame["date"].max()).normalize()
    source_date, initial_tickers = _historical_members_as_of(
        history_frame,
        start,
    )
    company_names = _company_name_map(
        wikipedia_current,
        wikipedia_changes,
    )
    snapshots: list[tuple[pd.Timestamp, set[str], str, pd.Timestamp]] = []

    def append_snapshot(
        date: pd.Timestamp,
        members: Iterable[str],
        source: str,
        membership_source_date: pd.Timestamp,
    ) -> None:
        normalized = {
            normalize_sp500_ticker(value)
            for value in members
            if normalize_sp500_ticker(value)
        }
        snapshot = (
            pd.Timestamp(date).normalize(),
            normalized,
            source,
            pd.Timestamp(membership_source_date).normalize(),
        )
        if snapshots and snapshots[-1][0] == snapshot[0]:
            snapshots[-1] = snapshot
        elif not snapshots or snapshots[-1][1] != normalized:
            snapshots.append(snapshot)

    append_snapshot(
        start,
        initial_tickers,
        "HANS_HISTORICAL_COMPONENTS",
        source_date,
    )
    members = set(initial_tickers)
    historical_window = history_frame.loc[
        history_frame["date"].gt(start)
        & history_frame["date"].le(min(history_max, fetch_date))
    ]
    for row in historical_window.to_dict(orient="records"):
        row_members = {
            normalize_sp500_ticker(value)
            for value in str(row["tickers"]).split(",")
            if normalize_sp500_ticker(value)
        }
        append_snapshot(
            pd.Timestamp(row["date"]),
            row_members,
            "HANS_HISTORICAL_COMPONENTS",
            pd.Timestamp(row["date"]),
        )
        members = row_members

    if fetch_date > history_max:
        dated_changes = wikipedia_changes.copy()
        dated_changes["EffectiveDate"] = pd.to_datetime(
            dated_changes["EffectiveDate"],
            errors="coerce",
        )
        dated_changes = dated_changes.loc[
            dated_changes["EffectiveDate"].gt(history_max)
            & dated_changes["EffectiveDate"].le(fetch_date)
        ].sort_values("EffectiveDate")
        for effective_date, group in dated_changes.groupby(
            "EffectiveDate",
            sort=True,
        ):
            for row in group.to_dict(orient="records"):
                removed = normalize_sp500_ticker(row.get("RemovedTicker"))
                added = normalize_sp500_ticker(row.get("AddedTicker"))
                if removed:
                    members.discard(removed)
                if added:
                    members.add(added)
            append_snapshot(
                pd.Timestamp(effective_date),
                members,
                "WIKIPEDIA_MEMBERSHIP_CHANGES",
                pd.Timestamp(effective_date),
            )

        current_members = {
            normalize_sp500_ticker(value)
            for value in wikipedia_current["Ticker"]
            if normalize_sp500_ticker(value)
        }
        append_snapshot(
            fetch_date,
            current_members,
            "WIKIPEDIA_CURRENT",
            fetch_date,
        )

    rows: list[dict[str, object]] = []
    for date, historical_tickers, source, membership_source_date in snapshots:
        by_data_symbol: dict[str, set[str]] = {}
        for historical_ticker in historical_tickers:
            by_data_symbol.setdefault(
                data_ticker(historical_ticker),
                set(),
            ).add(historical_ticker)
        for rank, data_symbol in enumerate(sorted(by_data_symbol), start=1):
            historical_symbols = sorted(by_data_symbol[data_symbol])
            company = next(
                (
                    company_names.get(ticker, "")
                    for ticker in historical_symbols
                    if company_names.get(ticker, "")
                ),
                company_names.get(data_symbol, ""),
            )
            rows.append(
                {
                    "AsOfDate": date.date().isoformat(),
                    "Ticker": data_symbol,
                    "DataSymbol": data_symbol,
                    "HistoricalTickers": ",".join(historical_symbols),
                    "Company": company or data_symbol,
                    "Rank": rank,
                    "Selected": True,
                    "MembershipSource": source,
                    "MembershipSourceDate": (
                        membership_source_date.date().isoformat()
                    ),
                    "PointInTimeStatus": (
                        "RETROSPECTIVE_PUBLIC_MEMBERSHIP_HISTORY"
                    ),
                }
            )
    membership = pd.DataFrame(rows)
    if membership.empty:
        raise ValueError("No S&P 500 membership change snapshots were produced")
    if membership.duplicated(["AsOfDate", "Ticker"]).any():
        raise ValueError("Duplicate data ticker in S&P change snapshot")
    return membership.sort_values(
        ["AsOfDate", "Rank", "Ticker"]
    ).reset_index(drop=True)


def build_sp500_union(
    membership: pd.DataFrame,
    *,
    annual_membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"AsOfDate", "Ticker", "DataSymbol", "Company"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(
            f"S&P membership missing union columns: {sorted(missing)}"
        )
    frame = membership.copy()
    frame["AsOfDate"] = pd.to_datetime(frame["AsOfDate"], errors="raise")
    annual_summary: dict[str, tuple[int, str]] = {}
    if annual_membership is not None:
        annual = annual_membership.copy()
        annual["AsOfDate"] = pd.to_datetime(
            annual["AsOfDate"],
            errors="raise",
        )
        annual_summary = {
            str(data_symbol): (
                int(group["AsOfDate"].nunique()),
                ",".join(
                    str(year)
                    for year in sorted(group["AsOfDate"].dt.year.unique())
                ),
            )
            for data_symbol, group in annual.groupby("DataSymbol", sort=True)
        }
    rows: list[dict[str, object]] = []
    for data_symbol, group in frame.groupby("DataSymbol", sort=True):
        companies = [
            str(value).strip()
            for value in group["Company"]
            if str(value).strip()
        ]
        historical = sorted(
            {
                ticker
                for value in group.get(
                    "HistoricalTickers",
                    group["Ticker"],
                )
                for ticker in str(value).split(",")
                if ticker
            }
        )
        years = sorted(group["AsOfDate"].dt.year.unique())
        change_snapshot_count = int(group["AsOfDate"].nunique())
        annual_default = (
            (0, "")
            if annual_membership is not None
            else (
                change_snapshot_count,
                ",".join(str(year) for year in years),
            )
        )
        annual_count, annual_years = annual_summary.get(
            str(data_symbol),
            annual_default,
        )
        rows.append(
            {
                "DataSymbol": data_symbol,
                "ScreenerName": companies[0] if companies else data_symbol,
                "SecurityName": companies[0] if companies else data_symbol,
                "HistoricalTickers": ",".join(historical),
                "MembershipYears": ",".join(str(year) for year in years),
                "FirstMembershipDate": group["AsOfDate"]
                .min()
                .date()
                .isoformat(),
                "LastMembershipDate": group["AsOfDate"]
                .max()
                .date()
                .isoformat(),
                "SnapshotCount": annual_count,
                "AnnualMembershipYears": annual_years,
                "ChangeSnapshotCount": change_snapshot_count,
                "UniverseSource": "SP500_ANNUAL_MEMBERSHIP_UNION",
                "CrawlBlockReason": (
                    "HISTORICAL_SYMBOL_REUSED_BY_DIFFERENT_SECURITY"
                    if data_symbol in KNOWN_REUSED_TICKERS
                    else ""
                ),
            }
        )
    union = pd.DataFrame(rows).sort_values(
        ["SnapshotCount", "ChangeSnapshotCount", "DataSymbol"],
        ascending=[False, False, True],
    )
    union.insert(0, "LiquidityRank", range(1, len(union) + 1))
    return union.reset_index(drop=True)


def generate_sp500_universe(
    paths: ProjectPaths,
    settings: Sp500UniverseSettings,
    *,
    output_dir: str | Path | None = None,
    fetched_at: datetime | None = None,
) -> Sp500UniverseArtifacts:
    generated_at = fetched_at or datetime.now(UTC)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "sp500_universe"
            / (
                generated_at.strftime("%Y%m%d_%H%M%S_%f")
                + "_sp500_annual_membership"
            )
        )
    )
    destination.mkdir(parents=True, exist_ok=True)

    history = fetch_historical_sp500(settings)
    current, changes = fetch_wikipedia_sp500(settings)
    membership = build_sp500_membership(
        history,
        current,
        changes,
        settings,
        fetched_at=generated_at,
    )
    change_membership = build_sp500_change_membership(
        history,
        current,
        changes,
        settings,
        fetched_at=generated_at,
    )
    union = build_sp500_union(
        change_membership,
        annual_membership=membership,
    )

    membership_path = destination / "sp500_membership.csv"
    change_membership_path = destination / "sp500_membership_changes.csv"
    union_path = destination / "sp500_union.csv"
    history_path = destination / "historical_sp500_source.csv"
    current_path = destination / "wikipedia_current_sp500.csv"
    changes_path = destination / "wikipedia_sp500_changes.csv"
    manifest_path = destination / "manifest.json"
    atomic_to_csv(membership, membership_path, index=False)
    atomic_to_csv(
        change_membership,
        change_membership_path,
        index=False,
    )
    atomic_to_csv(union, union_path, index=False)
    atomic_to_csv(history, history_path, index=False)
    atomic_to_csv(current, current_path, index=False)
    atomic_to_csv(changes, changes_path, index=False)
    _atomic_json(
        {
            "generated_at": generated_at.isoformat(),
            "task": (
                "2019-2026 S&P 500 annual summaries, membership changes, "
                "and full-period crawl union"
            ),
            "settings": asdict(settings),
            "counts": {
                "union": len(union),
                "change_snapshots": int(
                    change_membership["AsOfDate"].nunique()
                ),
                "change_membership_rows": len(change_membership),
                "snapshots": {
                    str(date): int(count)
                    for date, count in membership.groupby(
                        "AsOfDate"
                    ).size().items()
                },
                "historical_source_rows": len(history),
                "wikipedia_current_rows": len(current),
                "wikipedia_change_rows": len(changes),
            },
            "source_coverage": {
                "historical_first_date": str(
                    pd.to_datetime(history["date"]).min().date()
                ),
                "historical_last_date": str(
                    pd.to_datetime(history["date"]).max().date()
                ),
                "live_extension_method": (
                    "Reverse Wikipedia changes after each requested date "
                    "from the fetched current membership."
                ),
            },
            "limitations": [
                (
                    "Public retrospective membership history can be "
                    "corrected after the fact and is not an official S&P "
                    "point-in-time vendor feed."
                ),
                (
                    "Ticker aliases are applied only to same-company ticker "
                    "changes; merger and acquisition securities remain "
                    "separate and may fail current web crawlers."
                ),
                (
                    "The union is a crawl queue. Backtests must apply "
                    "sp500_membership_changes.csv by date rather than treating "
                    "every union ticker as continuously eligible."
                ),
                (
                    "Price and financial availability are not implied by "
                    "index membership."
                ),
            ],
            "outputs": {
                "membership": str(membership_path),
                "change_membership": str(change_membership_path),
                "union": str(union_path),
            },
        },
        manifest_path,
    )
    return Sp500UniverseArtifacts(
        output_dir=destination,
        membership_csv=membership_path,
        change_membership_csv=change_membership_path,
        union_csv=union_path,
        historical_source_csv=history_path,
        wikipedia_current_csv=current_path,
        wikipedia_changes_csv=changes_path,
        manifest_json=manifest_path,
    )


def find_latest_sp500_union(paths: ProjectPaths) -> Path:
    root = paths.results / "Cross_Sectional" / "sp500_universe"
    candidates = list(root.glob("*/sp500_union.csv"))
    if not candidates:
        raise FileNotFoundError(f"No sp500_union.csv found below {root}")
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
    )


def normalize_sp500_ticker(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    ticker = str(value).strip().upper().split("(", maxsplit=1)[0].strip()
    if not ticker:
        return ""
    return (
        ticker.split(maxsplit=1)[0]
        .replace("/", "-")
        .replace(".", "-")
    )


def data_ticker(value: object) -> str:
    ticker = normalize_sp500_ticker(value)
    return DATA_TICKER_ALIASES.get(ticker, ticker)


def _historical_members_as_of(
    history: pd.DataFrame,
    target: pd.Timestamp,
) -> tuple[pd.Timestamp, list[str]]:
    eligible = history.loc[history["date"].le(target)].sort_values("date")
    if eligible.empty:
        raise ValueError(
            f"No historical S&P snapshot at or before {target.date()}"
        )
    row = eligible.iloc[-1]
    tickers = sorted(
        {
            normalize_sp500_ticker(value)
            for value in str(row["tickers"]).split(",")
            if normalize_sp500_ticker(value)
        }
    )
    return pd.Timestamp(row["date"]).normalize(), tickers


def _reverse_current_membership(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    target: pd.Timestamp,
    fetch_date: pd.Timestamp,
) -> list[str]:
    members = {
        normalize_sp500_ticker(value)
        for value in current["Ticker"]
        if normalize_sp500_ticker(value)
    }
    dated = changes.copy()
    dated["EffectiveDate"] = pd.to_datetime(
        dated["EffectiveDate"],
        errors="coerce",
    )
    future_changes = dated.loc[
        dated["EffectiveDate"].gt(target)
        & dated["EffectiveDate"].le(fetch_date)
    ].sort_values("EffectiveDate", ascending=False)
    for row in future_changes.to_dict(orient="records"):
        added = normalize_sp500_ticker(row.get("AddedTicker"))
        removed = normalize_sp500_ticker(row.get("RemovedTicker"))
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return sorted(members)


def _company_name_map(
    current: pd.DataFrame,
    changes: pd.DataFrame,
) -> dict[str, str]:
    result = dict(HISTORICAL_COMPANY_NAMES)
    for row in current.to_dict(orient="records"):
        ticker = normalize_sp500_ticker(row.get("Ticker"))
        company = str(row.get("Company") or "").strip()
        if ticker and company:
            result[ticker] = company
    for row in changes.to_dict(orient="records"):
        for ticker_key, company_key in (
            ("AddedTicker", "AddedCompany"),
            ("RemovedTicker", "RemovedCompany"),
        ):
            ticker = normalize_sp500_ticker(row.get(ticker_key))
            company = str(row.get(company_key) or "").strip()
            if ticker and company and ticker not in result:
                result[ticker] = company
    for old, new in DATA_TICKER_ALIASES.items():
        if new in result and old not in result:
            result[old] = result[new]
        if old in result and new not in result:
            result[new] = result[old]
    return result


def _flatten_column(column: object) -> str:
    if isinstance(column, tuple):
        values = [
            str(value).strip()
            for value in column
            if str(value).strip() and not str(value).startswith("Unnamed")
        ]
        return " ".join(values).lower()
    return str(column).strip().lower()


def _optional_column(
    frame: pd.DataFrame,
    lookup: dict[str, object],
    name: str,
) -> pd.Series:
    if name not in lookup:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[lookup[name]].fillna("").astype(str).str.strip()


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
