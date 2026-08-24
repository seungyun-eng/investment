import json
from datetime import date
from types import SimpleNamespace

from scripts.dashboard.publish_alpha_desk import (
    _most_recent_friday,
    _prepare_forward_ledger_history,
    _write_monitor_universe,
)
from stock_research.dashboard.state import DashboardState, TickerEntry


def test_weekly_publisher_targets_the_current_or_prior_friday() -> None:
    assert _most_recent_friday(date(2026, 8, 14)) == date(2026, 8, 14)
    assert _most_recent_friday(date(2026, 8, 15)) == date(2026, 8, 14)
    assert _most_recent_friday(date(2026, 8, 17)) == date(2026, 8, 14)


def test_monitor_universe_includes_validated_cached_cik(tmp_path) -> None:
    ticker_root = tmp_path / "SEC Filings" / "AEP"
    ticker_root.mkdir(parents=True)
    (ticker_root / "submissions.json").write_text(
        '{"cik":"4904","tickers":["AEP"]}', encoding="utf-8"
    )
    state = DashboardState(
        top_k=5,
        sec_user_agent="test@example.com",
        tickers=[TickerEntry("AEP", "American Electric Power", "AEP.csv", "test", "2026-08-15")],
    )
    output = tmp_path / "monitor_universe.json"

    _write_monitor_universe(
        SimpleNamespace(stock_root=tmp_path), state, output, signal_as_of="2026-08-14"
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["signalAsOf"] == "2026-08-14"
    assert payload["entries"] == [
        {"ticker": "AEP", "company": "American Electric Power", "cik": "0000004904"}
    ]


def test_forward_ledger_history_collects_removed_snapshot_members(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "investment"
    snapshots = repo_root / "config" / "dashboard_model_registry" / "universe_snapshots"
    snapshots.mkdir(parents=True)
    snapshots.joinpath("2026-08-12.json").write_text(
        json.dumps(
            {
                "top_k": 5,
                "sec_user_agent": "test@example.com",
                "tickers": [
                    {"ticker": "JNJ", "company": "JNJ", "price_path": "old/JNJ.csv", "source": "snapshot", "added_date": "2026-08-12"}
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshots.joinpath("2026-08-14.json").write_text(
        json.dumps(
            {
                "top_k": 5,
                "sec_user_agent": "test@example.com",
                "tickers": [
                    {"ticker": "NVDA", "company": "NVIDIA", "price_path": "new/NVDA.csv", "source": "snapshot", "added_date": "2026-08-14"}
                ],
            }
        ),
        encoding="utf-8",
    )
    state_path = repo_root / "config" / "cross_sectional" / "dashboard_universe.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(snapshots.joinpath("2026-08-14.json").read_text(), encoding="utf-8")
    paths = SimpleNamespace(repo_root=repo_root, stock_root=tmp_path / "stock")
    downloaded = []
    synced = []

    def fake_download(ticker, destination):
        downloaded.append(ticker)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("Date,Close\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dashboard.publish_alpha_desk.data_collection.download_price_history",
        fake_download,
    )
    monkeypatch.setattr(
        "scripts.dashboard.publish_alpha_desk.data_collection.sync_filings_for_dashboard",
        lambda paths, state, refresh_metadata: synced.extend(entry.ticker for entry in state.tickers),
    )

    _prepare_forward_ledger_history(paths)

    assert downloaded == ["JNJ", "NVDA"]
    assert synced == ["JNJ", "NVDA"]
