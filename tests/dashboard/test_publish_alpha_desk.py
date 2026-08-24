import json
from datetime import date
from types import SimpleNamespace

from scripts.dashboard.publish_alpha_desk import _most_recent_friday, _write_monitor_universe
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
