from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stock_research.dashboard import data_collection, engine
from stock_research.dashboard.state import DashboardState, TickerEntry, save_state
from stock_research.dashboard.ticker_request_review import (
    TickerRequest,
    record_universe_change,
    review_requests,
)
from stock_research.paths import ProjectPaths


def _paths(root: Path) -> ProjectPaths:
    paths = ProjectPaths(
        repo_root=root / "repo",
        stock_root=root,
        raw_prices=root / "Back Test",
        processed=root / "Processed Data",
        macro=root / "Macro Data",
        financial_raw=root / "Financial_Data_real",
        financial_legacy=root / "Financial Data",
        results=root / "Results",
        parameters=root / "Results" / "Parameters",
        transformer_results=root / "Results" / "Transformer",
    )
    paths.ensure_output_dirs()
    paths.repo_root.mkdir(parents=True, exist_ok=True)
    save_state(
        paths,
        DashboardState(
            top_k=5,
            sec_user_agent="test@example.com",
            tickers=[TickerEntry("BASE", "Base", "prices/base.csv", "test", "2026-01-01")],
        ),
    )
    return paths


def test_review_approves_only_the_candidate_that_passes_the_frozen_model_gates(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = _paths(tmp_path)
    requests = [
        TickerRequest(1, "PASS", "Passing"),
        TickerRequest(2, "WAIT", "Waiting"),
    ]

    def fake_download(ticker: str, destination: Path) -> dict[str, object]:
        return {
            "ticker": ticker,
            "rows": 300,
            "start_date": "2024-01-01",
            "end_date": "2026-08-14",
            "path": str(destination),
        }

    scored = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-08-14"), "Ticker": "PASS", "Eligible": True,
                "FilingCoverageCount": 8, "FilingCoveragePass": True,
                "FilingQualityPass": True, "FilingVetoPass": True, "Qualified": True,
                "Trend200": 0.2, "Return126": 0.3, "Rank": 2.0, "AlphaScore": 0.7,
            },
            {
                "Date": pd.Timestamp("2026-08-14"), "Ticker": "WAIT", "Eligible": True,
                "FilingCoverageCount": 4, "FilingCoveragePass": False,
                "FilingQualityPass": True, "FilingVetoPass": True, "Qualified": False,
                "Trend200": 0.2, "Return126": 0.3, "Rank": None, "AlphaScore": 0.4,
            },
        ]
    )
    candidate_filing_path = tmp_path / "candidate_filings.csv"
    pd.DataFrame(
        {
            "Ticker": ["PASS", "WAIT"],
            "AvailableDate": ["2026-08-01", "2026-08-01"],
            "AccessionNumber": ["one", "two"],
        }
    ).to_csv(candidate_filing_path, index=False)
    monkeypatch.setattr(data_collection, "download_price_history", fake_download)
    monkeypatch.setattr(
        data_collection,
        "sync_filings_for_candidates",
        lambda *a, **k: SimpleNamespace(point_in_time_features_csv=candidate_filing_path),
    )
    monkeypatch.setattr(engine, "build_scored_panel", lambda *a, **k: SimpleNamespace(scored=scored))
    monkeypatch.setattr(engine, "_base_params", lambda: SimpleNamespace(trend_floor=0.0, momentum_floor=0.0))
    monkeypatch.setattr(engine, "_policy", lambda top_k: SimpleNamespace(minimum_filing_coverage=6))

    decisions, approved_state = review_requests(paths, requests, as_of="2026-08-15")

    assert [decision.status for decision in decisions] == ["APPROVED", "DEFERRED"]
    assert [entry.ticker for entry in approved_state.tickers] == ["BASE", "PASS"]
    assert decisions[1].details["reasons"] == ["SEC 핵심수치 4/6개"]


def test_review_keeps_a_price_failure_retryable(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    request = TickerRequest(1, "BAD", "Bad")
    monkeypatch.setattr(
        data_collection,
        "download_price_history",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Yahoo unavailable")),
    )

    decisions, approved_state = review_requests(paths, [request], as_of="2026-08-15")

    assert decisions[0].status == "ERROR"
    assert decisions[0].stage == "PRICE"
    assert [entry.ticker for entry in approved_state.tickers] == ["BASE"]


def test_universe_change_merges_multiple_approvals_on_the_same_day(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    root = paths.repo_root / "config" / "dashboard_model_registry"
    root.mkdir(parents=True)
    (root / "registry.json").write_text(
        json.dumps(
            {
                "active_model_version": "V7.3",
                "forward_ledger_start": "2026-08-12",
                "universe_changes": [
                    {
                        "date": "2026-08-15", "snapshot": "2026-08-15.json",
                        "universe_version": "U002", "added": ["ONE"], "removed": [],
                        "note": "prior",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = DashboardState(
        top_k=5,
        sec_user_agent="test@example.com",
        tickers=[TickerEntry("TWO", "Two", "prices/two.csv", "test", "2026-08-15")],
    )

    entry = record_universe_change(
        paths, state, added=["TWO"], change_date="2026-08-15"
    )

    assert entry is not None
    assert entry["universe_version"] == "U002"
    assert entry["added"] == ["ONE", "TWO"]
    saved = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    assert saved["universe_changes"][-1]["added"] == ["ONE", "TWO"]
