from __future__ import annotations

"""Automated review bridge between Alpha Desk D1 requests and V7.3.

The cloud app only records the user's request.  The scheduled weekly Python
publisher owns the financial decision because it can call the exact same
price, SEC, scoring, and signal functions used by the production model.
"""

import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stock_research.dashboard import data_collection, engine
from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard.forward_ledger import registry_dir
from stock_research.paths import ProjectPaths

ACTIONABLE_STATUSES = ("REQUESTED", "DEFERRED", "ERROR")
DEFAULT_RESEARCH_START = "2024-01-02"
D1_DATABASE = "alpha-desk-db"


def _wrangler_path(app_root: Path) -> Path:
    suffix = ".CMD" if os.name == "nt" else ""
    return app_root / "node_modules" / ".bin" / f"wrangler{suffix}"


@dataclass(frozen=True)
class TickerRequest:
    id: int
    ticker: str
    company_name: str
    reason: str = ""
    thesis: str = ""


@dataclass(frozen=True)
class ReviewDecision:
    request: TickerRequest
    status: str
    stage: str
    summary: str
    details: dict[str, Any]

    @property
    def approved(self) -> bool:
        return self.status == "APPROVED"


class WranglerD1TickerRequests:
    """Authenticated D1 access through the same local Wrangler used to deploy.

    No API token is stored in source or sent through the public app endpoint;
    the scheduled task runs under the already-authenticated Windows account.
    """

    def __init__(self, repo_root: Path) -> None:
        self.app_root = repo_root / "alpha-desk-cloud"
        self.wrangler = _wrangler_path(self.app_root)
        self.config = self.app_root / "wrangler.jsonc"
        if not self.wrangler.exists():
            raise FileNotFoundError(f"Wrangler executable not found: {self.wrangler}")

    def _execute(self, sql: str) -> list[dict[str, Any]]:
        completed = subprocess.run(
            [
                str(self.wrangler),
                "d1",
                "execute",
                D1_DATABASE,
                "--remote",
                "--config",
                str(self.config),
                "--command",
                sql,
                "--json",
            ],
            cwd=self.app_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(completed.stdout)
        if not payload or not payload[0].get("success", False):
            raise RuntimeError(f"D1 command failed: {payload}")
        return list(payload[0].get("results") or [])

    def fetch_actionable(self) -> list[TickerRequest]:
        statuses = ",".join(_sql_string(value) for value in ACTIONABLE_STATUSES)
        rows = self._execute(
            "SELECT id,ticker,company_name,reason,thesis "
            "FROM ticker_requests "
            f"WHERE status IN ({statuses}) ORDER BY id"
        )
        return [
            TickerRequest(
                id=int(row["id"]),
                ticker=str(row["ticker"]).upper(),
                company_name=str(row.get("company_name") or row["ticker"]),
                reason=str(row.get("reason") or ""),
                thesis=str(row.get("thesis") or ""),
            )
            for row in rows
        ]

    def mark_reviewing(self, requests: Iterable[TickerRequest]) -> None:
        ids = [request.id for request in requests]
        if not ids:
            return
        self._execute(
            "UPDATE ticker_requests SET status='REVIEWING',"
            "review_stage='PRICE_SEC_MODEL',"
            "review_summary='가격·SEC·모델 적격성 자동 검토 중' "
            f"WHERE id IN ({','.join(str(value) for value in ids)})"
        )

    def save_decisions(
        self,
        decisions: Iterable[ReviewDecision],
        *,
        effective_signal_date: str | None = None,
    ) -> None:
        reviewed_at = datetime.now(timezone.utc).isoformat()
        statements: list[str] = []
        for decision in decisions:
            status = decision.status
            stage = decision.stage
            summary = decision.summary
            effective = None
            if decision.approved:
                if effective_signal_date is None:
                    status = "REVIEWING"
                    stage = "SIGNAL"
                    summary = "검토 통과 · 이번 주 신호 계산 중"
                else:
                    stage = "COMPLETE"
                    summary = "검토 통과 · 이번 주 신호부터 유니버스 반영"
                    effective = effective_signal_date
            details = json.dumps(_json_safe(decision.details), ensure_ascii=False)
            statements.append(
                "UPDATE ticker_requests SET "
                f"status={_sql_string(status)},"
                f"review_stage={_sql_string(stage)},"
                f"review_summary={_sql_string(summary)},"
                f"review_details={_sql_string(details)},"
                f"reviewed_at={_sql_string(reviewed_at)},"
                f"effective_signal_date={_sql_string(effective)} "
                f"WHERE id={decision.request.id}"
            )
        if statements:
            self._execute(";".join(statements))

    def mark_errors(self, requests: Iterable[TickerRequest], error: Exception) -> None:
        message = str(error)[:500]
        decisions = [
            ReviewDecision(
                request=request,
                status="ERROR",
                stage="ERROR",
                summary="자동 검토 오류 · 다음 주 재시도",
                details={"error": message},
            )
            for request in requests
        ]
        self.save_decisions(decisions)


class WranglerD1UniverseRemovals:
    """Same authenticated-D1-through-local-Wrangler pattern as
    WranglerD1TickerRequests, for the simpler "remove this ticker" queue.
    Removal needs no price/SEC/model vetting (the ticker is already a
    validated, currently-held universe member) -- it is applied directly,
    unlike ticker additions which go through review_requests()."""

    def __init__(self, repo_root: Path) -> None:
        self.app_root = repo_root / "alpha-desk-cloud"
        self.wrangler = _wrangler_path(self.app_root)
        self.config = self.app_root / "wrangler.jsonc"
        if not self.wrangler.exists():
            raise FileNotFoundError(f"Wrangler executable not found: {self.wrangler}")

    def _execute(self, sql: str) -> list[dict[str, Any]]:
        completed = subprocess.run(
            [
                str(self.wrangler),
                "d1",
                "execute",
                D1_DATABASE,
                "--remote",
                "--config",
                str(self.config),
                "--command",
                sql,
                "--json",
            ],
            cwd=self.app_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(completed.stdout)
        if not payload or not payload[0].get("success", False):
            raise RuntimeError(f"D1 command failed: {payload}")
        return list(payload[0].get("results") or [])

    def fetch_requested(self) -> list[str]:
        rows = self._execute(
            "SELECT ticker FROM universe_removal_requests WHERE status='REQUESTED' ORDER BY id"
        )
        return [str(row["ticker"]).upper() for row in rows]

    def mark_applied(self, tickers: Iterable[str]) -> None:
        values = list(tickers)
        if not values:
            return
        applied_at = datetime.now(timezone.utc).isoformat()
        in_clause = ",".join(_sql_string(ticker) for ticker in values)
        self._execute(
            "UPDATE universe_removal_requests SET status='APPLIED',"
            f"applied_at={_sql_string(applied_at)} WHERE ticker IN ({in_clause})"
        )


def _sql_string(value: object) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, date, datetime)):
        return value.isoformat()
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _truthy(value: object) -> bool:
    return False if pd.isna(value) else bool(value)


def _integer(value: object) -> int:
    return 0 if pd.isna(value) else int(value)


def _copy_state(state: dashboard_state.DashboardState) -> dashboard_state.DashboardState:
    return dashboard_state.state_from_dict(state.as_dict())


def _candidate_entry(
    request: TickerRequest,
    *,
    added_date: str,
) -> dashboard_state.TickerEntry:
    return dashboard_state.TickerEntry(
        ticker=request.ticker,
        company=request.company_name or request.ticker,
        price_path=f"{data_collection.DASHBOARD_PRICE_DIR}/{request.ticker}.csv",
        source="ticker_request_review",
        added_date=added_date,
    )


def review_requests(
    paths: ProjectPaths,
    requests: list[TickerRequest],
    *,
    as_of: str | None = None,
    max_workers: int = 4,
) -> tuple[list[ReviewDecision], dashboard_state.DashboardState]:
    """Collect and score candidates without mutating the live universe."""

    original = dashboard_state.load_state(paths)
    candidate_state = _copy_state(original)
    existing = {entry.ticker for entry in original.tickers}
    review_date = as_of or date.today().isoformat()
    decisions: dict[int, ReviewDecision] = {}
    price_summaries: dict[str, dict[str, object]] = {}

    candidates = [request for request in requests if request.ticker not in existing]
    for request in requests:
        if request.ticker in existing:
            decisions[request.id] = ReviewDecision(
                request=request,
                status="APPROVED",
                stage="COMPLETE",
                summary="이미 현재 유니버스에 포함",
                details={"already_in_universe": True},
            )

    def download(request: TickerRequest) -> dict[str, object]:
        destination = (
            paths.stock_root
            / data_collection.DASHBOARD_PRICE_DIR
            / f"{request.ticker}.csv"
        )
        return data_collection.download_price_history(request.ticker, destination)

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(candidates)))) as executor:
        futures = {executor.submit(download, request): request for request in candidates}
        for future in as_completed(futures):
            request = futures[future]
            try:
                summary = future.result()
                price_summaries[request.ticker] = summary
                candidate_state.tickers.append(
                    _candidate_entry(request, added_date=review_date)
                )
            except Exception as error:  # noqa: BLE001 - shown in request status
                decisions[request.id] = ReviewDecision(
                    request=request,
                    status="ERROR",
                    stage="PRICE",
                    summary="가격 데이터 수집 실패 · 다음 주 재시도",
                    details={"price_error": str(error)},
                )

    scored = pd.DataFrame()
    scorable = [request for request in candidates if request.ticker in price_summaries]
    if scorable:
        try:
            candidate_artifacts = data_collection.sync_filings_for_candidates(
                paths,
                candidate_state,
                [request.ticker for request in scorable],
                refresh_metadata=True,
            )
            live_filing_path = (
                paths.processed
                / "SEC Filings"
                / data_collection.DASHBOARD_FILING_LABEL
                / "point_in_time_features.csv"
            )
            filing_frames = [pd.read_csv(candidate_artifacts.point_in_time_features_csv)]
            if live_filing_path.exists():
                filing_frames.insert(0, pd.read_csv(live_filing_path))
            filing_features = pd.concat(filing_frames, ignore_index=True)
            filing_features = filing_features.drop_duplicates(
                ["Ticker", "AvailableDate", "AccessionNumber"], keep="last"
            )
            scored = engine.build_scored_panel(
                paths,
                candidate_state,
                start=DEFAULT_RESEARCH_START,
                end=review_date,
                top_k=candidate_state.top_k,
                filing_features_override=filing_features,
            ).scored
        except Exception as error:  # noqa: BLE001 - retained for automatic retry
            for request in scorable:
                decisions[request.id] = ReviewDecision(
                    request=request,
                    status="ERROR",
                    stage="SEC_MODEL",
                    summary="SEC 또는 모델 검토 오류 · 다음 주 재시도",
                    details={"review_error": str(error)},
                )

    params = engine._base_params()
    policy = engine._policy(candidate_state.top_k)
    for request in scorable:
        if request.id in decisions:
            continue
        rows = scored.loc[scored["Ticker"].astype(str).str.upper().eq(request.ticker)]
        if rows.empty:
            decisions[request.id] = ReviewDecision(
                request=request,
                status="DEFERRED",
                stage="MODEL",
                summary="모델 평가 행 없음 · 다음 주 재검토",
                details={"reason": "NO_MODEL_ROW"},
            )
            continue
        row = rows.sort_values("Date").iloc[-1]
        price_rows = int(price_summaries[request.ticker].get("rows") or 0)
        eligible = _truthy(row.get("Eligible", False))
        coverage = _integer(row.get("FilingCoverageCount", 0))
        coverage_pass = _truthy(row.get("FilingCoveragePass", False))
        quality_pass = _truthy(row.get("FilingQualityPass", False))
        veto_pass = _truthy(row.get("FilingVetoPass", False))
        qualified = _truthy(row.get("Qualified", False))
        reasons: list[str] = []
        if price_rows < engine.MINIMUM_PRICE_HISTORY_SESSIONS:
            reasons.append(
                f"가격 이력 {price_rows}/{engine.MINIMUM_PRICE_HISTORY_SESSIONS}거래일"
            )
        if not eligible:
            reasons.append("200거래일 기반 기술지표 준비 미완료")
        if not coverage_pass:
            reasons.append(
                f"SEC 핵심수치 {coverage}/{policy.minimum_filing_coverage}개"
            )
        if not quality_pass:
            reasons.append("SEC 재무품질 기준 미충족")
        if not veto_pass:
            reasons.append("SEC 중대 경고 감지")
        if eligible and coverage_pass and quality_pass and veto_pass and not qualified:
            reasons.append("현재 추세·모멘텀 진입 기준 미충족")

        details = {
            "signal_date": row.get("Date"),
            "price_history_sessions": price_rows,
            "price_start": price_summaries[request.ticker].get("start_date"),
            "price_end": price_summaries[request.ticker].get("end_date"),
            "eligible": eligible,
            "filing_coverage_count": coverage,
            "minimum_filing_coverage": policy.minimum_filing_coverage,
            "filing_coverage_pass": coverage_pass,
            "filing_quality_pass": quality_pass,
            "filing_veto_pass": veto_pass,
            "trend_200": row.get("Trend200"),
            "trend_floor": params.trend_floor,
            "return_126": row.get("Return126"),
            "momentum_floor": params.momentum_floor,
            "qualified": qualified,
            "rank": row.get("Rank"),
            "alpha_score": row.get("AlphaScore"),
            "reasons": reasons,
        }
        decisions[request.id] = ReviewDecision(
            request=request,
            status="APPROVED" if qualified else "DEFERRED",
            stage="SIGNAL" if qualified else "MODEL",
            summary=(
                "검토 통과 · 이번 주 신호 계산에 반영 대기"
                if qualified
                else "자동 검토 보류 · 다음 주 재검토: " + ", ".join(reasons)
            ),
            details=details,
        )

    ordered = [decisions[request.id] for request in requests]
    approved_tickers = {
        decision.request.ticker
        for decision in ordered
        if decision.approved and decision.request.ticker not in existing
    }
    approved_state = _copy_state(original)
    approved_state.tickers.extend(
        entry
        for entry in candidate_state.tickers
        if entry.ticker in approved_tickers
    )
    return ordered, approved_state


def apply_approved_state(
    paths: ProjectPaths,
    state: dashboard_state.DashboardState,
) -> None:
    dashboard_state.save_state(paths, state)


def record_universe_change(
    paths: ProjectPaths,
    state: dashboard_state.DashboardState,
    *,
    added: list[str],
    change_date: str,
    removed: list[str] | None = None,
) -> dict[str, Any] | None:
    removed = removed or []
    if not added and not removed:
        return None
    root = registry_dir(paths)
    snapshots = root / "universe_snapshots"
    snapshot_path = snapshots / f"{change_date}.json"
    _atomic_json(snapshot_path, state.as_dict())

    note_parts = []
    if added:
        note_parts.append("자동 검토 통과 종목을 같은 주 신호부터 반영")
    if removed:
        note_parts.append("사용자 요청으로 종목 제외")
    note = " · ".join(note_parts)

    registry_path = root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    changes = registry["universe_changes"]
    if changes and changes[-1]["date"] == change_date:
        prior = changes[-1]
        entry = {
            **prior,
            "snapshot": snapshot_path.name,
            "added": sorted(set(prior.get("added", [])) | set(added)),
            "removed": sorted(set(prior.get("removed", [])) | set(removed)),
            "note": note or prior.get("note", ""),
        }
        changes[-1] = entry
    else:
        entry = {
            "date": change_date,
            "snapshot": snapshot_path.name,
            "universe_version": f"U{len(changes) + 1:03d}",
            "added": sorted(set(added)),
            "removed": sorted(set(removed)),
            "note": note,
        }
        changes.append(entry)
    _atomic_json(registry_path, registry)
    return entry


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=path.stem + "_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
