from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_research.dashboard import data_collection
from stock_research.dashboard import today as today_module
from stock_research.dashboard.state import DashboardState, TickerEntry
from stock_research.dashboard.today import (
    _preserve_published_tsla_card,
    build_tsla_card,
    compose_today_payload,
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
    return paths


def _state(price_path: str = "Dashboard Data/Prices/TSLA.csv") -> DashboardState:
    return DashboardState(
        top_k=5,
        sec_user_agent="test@example.com",
        tickers=[
            TickerEntry(
                ticker="TSLA",
                company="Tesla",
                price_path=price_path,
                source="test",
                added_date="2026-08-01",
            )
        ],
    )


def test_tsla_card_uses_persisted_position_and_targets_v73_v2_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_tsla_card composes the persisted position (real shares/cash,
    independent of which strategy justified the purchase) against today's
    V7.3+V2 target weight. compute_today_signal itself needs the full
    frozen-model data pipeline, so it is stubbed here -- this test is only
    about build_tsla_card's own position/recommendation composition."""
    paths = _paths(tmp_path)

    account_path = paths.results / "tsla_cycle_live_signal" / "account_state.json"
    account_path.parent.mkdir(parents=True)
    account_path.write_text(
        json.dumps(
            {
                "cash": 12_000.0,
                "shares": 26.0,
                "average_cost": 316.36,
                "last_buy_date": "2026-07-15",
            }
        ),
        encoding="utf-8",
    )

    stub_signal = {
        "Date": pd.Timestamp("2026-08-14"),
        "Close": 340.0,
        "BaseState": "LONG",
        "V2Exposure": 0.8,
        "V2AsOf": "2026-08-14",
        "TargetWeight": 0.8,
    }
    monkeypatch.setattr(today_module.live_v73_v2_signal, "compute_today_signal", lambda paths: stub_signal)

    card = build_tsla_card(paths, _state(), maximum_missing_business_days=10_000)
    assert card["strategy"] == "V73_TSLA_PLUS_V2"
    assert card["position"]["shares"] == 26.0
    assert card["position"]["cash"] == 12_000.0
    assert card["position"]["last_buy_date"] == "2026-07-15"
    assert card["recommendation"]["Composition"]["BaseState"] == "LONG"
    assert card["recommendation"]["TargetWeight"] == pytest.approx(0.8)
    assert card["recommendation"]["Action"] == "BUY_TO_TARGET_NEXT_OPEN"


def test_today_payload_keeps_specialized_tsla_separate_from_top_k() -> None:
    cross = {
        "meta": {
            "latest_signal_date": "2026-08-07",
            "top_k": 3,
            "tickers": ["A", "B", "TSLA"],
            "execution_policy": "WEEKLY_SIGNAL_MONTHLY_FIRST_WEIGHT_RESET",
            "signal_frequency": "WEEKLY",
            "weight_reset_frequency": "MONTHLY_FIRST_SIGNAL",
            "immediate_membership_changes": True,
            "latest_signal_execution_reason": "MONTHLY_WEIGHT_RESET",
        },
        "top15_latest": [
            {"rank": 1, "ticker": "A", "target_weight": 0.5},
            {"rank": 2, "ticker": "B", "target_weight": 0.5},
            {"rank": 3, "ticker": "TSLA", "target_weight": None},
        ],
        "trades": [
            {"date": "2026-08-07", "ticker": "A", "action": "BUY"},
            {"date": "2026-08-01", "ticker": "B", "action": "SELL"},
        ],
        "summary": {"ROI": 20.0},
        "ticker_summary": [{"ticker": "A", "net_pnl": 20.0}],
    }
    tsla = {
        "recommendation": {"Action": "HOLD_MISSING_LAST_BUY_DATE"},
    }
    payload = compose_today_payload(cross, tsla, refresh=None)
    assert [row["ticker"] for row in payload["top_picks"]] == ["A", "B"]
    assert payload["rotation_actions"] == [
        {"date": "2026-08-07", "ticker": "A", "action": "BUY"}
    ]
    assert payload["action_count"] == 1
    assert payload["tsla"] is tsla


def test_cloud_publish_preserves_but_blocks_tsla_when_research_panel_is_absent(
    tmp_path,
) -> None:
    repo_root = tmp_path / "investment"
    published = repo_root / "alpha-desk-cloud" / "public" / "data" / "latest_today.json"
    published.parent.mkdir(parents=True)
    published.write_text(
        json.dumps(
            {
                "tsla": {
                    "data_fresh": True,
                    "position": {"shares": 1.0},
                    "recommendation": {
                        "Action": "BUY",
                        "OrderSide": "BUY",
                        "OrderEquityFraction": 0.2,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    paths = SimpleNamespace(repo_root=repo_root, results=tmp_path / "results")

    card = _preserve_published_tsla_card(paths, FileNotFoundError("panel.csv"))

    assert card["data_fresh"] is False
    assert card["refresh_error"] == "panel.csv"
    assert card["recommendation"]["UnderlyingActionBeforeFreshnessBlock"] == "BUY"
    assert card["recommendation"]["Action"] == "NO_ACTION_TSLA_DATA_UNAVAILABLE"
    assert card["recommendation"]["OrderSide"] is None
    assert payload["ticker_summary"] == [{"ticker": "A", "net_pnl": 20.0}]
    assert payload["execution_policy"] == {
        "name": "WEEKLY_SIGNAL_MONTHLY_FIRST_WEIGHT_RESET",
        "signal_frequency": "WEEKLY",
        "weight_reset_frequency": "MONTHLY_FIRST_SIGNAL",
        "immediate_membership_changes": True,
        "latest_signal_execution_reason": "MONTHLY_WEIGHT_RESET",
    }


def test_refresh_all_prices_switches_only_successful_tickers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    state = DashboardState(
        top_k=3,
        sec_user_agent="test@example.com",
        tickers=[
            TickerEntry("GOOD", "Good", "Back Test/good.csv", "seed", "2026-01-01"),
            TickerEntry("BAD", "Bad", "Back Test/bad.csv", "seed", "2026-01-01"),
        ],
    )

    def fake_download(ticker: str, destination: Path) -> dict[str, object]:
        if ticker == "BAD":
            raise RuntimeError("network failed")
        return {
            "ticker": ticker,
            "rows": 250,
            "start_date": "2025-01-01",
            "end_date": "2026-08-07",
            "path": str(destination),
        }

    monkeypatch.setattr(data_collection, "download_price_history", fake_download)
    result = data_collection.refresh_all_prices(paths, state, max_workers=2)
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    entries = {entry.ticker: entry for entry in state.tickers}
    assert entries["GOOD"].price_path == "Dashboard Data/Prices/GOOD.csv"
    assert entries["BAD"].price_path == "Back Test/bad.csv"


def test_sync_filings_can_force_sec_metadata_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    captured: dict[str, object] = {}

    def fake_sync(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "artifacts"

    monkeypatch.setattr(data_collection, "sync_sec_filings", fake_sync)
    result = data_collection.sync_filings_for_dashboard(
        paths, _state(), refresh_metadata=True
    )

    assert result == "artifacts"
    assert captured["refresh_metadata"] is True


def test_candidate_filing_sync_only_requests_candidate_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    captured: dict[str, object] = {}

    def fake_sync(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "candidate-artifacts"

    monkeypatch.setattr(data_collection, "sync_sec_filings", fake_sync)
    result = data_collection.sync_filings_for_candidates(
        paths, _state(), ["aep", "AEP"], refresh_metadata=True
    )

    assert result == "candidate-artifacts"
    assert captured["tickers"] == ["AEP"]
    assert captured["output_label"] == data_collection.DASHBOARD_CANDIDATE_FILING_LABEL
    assert captured["refresh_metadata"] is True
