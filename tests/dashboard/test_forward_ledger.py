from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_research.dashboard import forward_ledger
from stock_research.dashboard.forward_ledger import benchmark_series, build_segment_if_signaled, chain_segments
from stock_research.dashboard.state import DashboardState, TickerEntry, state_from_snapshot


def _segment(start_equity: float, end_equity: float, dates: list[str]) -> dict[str, object]:
    span = len(dates) - 1
    curve = [
        {"date": date, "equity": start_equity + (end_equity - start_equity) * index / span}
        for index, date in enumerate(dates)
    ]
    return {"equity_curve": curve}


def test_single_segment_tracks_its_own_growth_factor() -> None:
    segment = _segment(100_000.0, 110_000.0, ["2026-08-12", "2026-08-13", "2026-08-14"])
    series = chain_segments([segment], initial_nav=100_000.0)
    assert series[0] == {"date": "2026-08-12", "nav": 100_000.0}
    assert series[-1]["date"] == "2026-08-14"
    assert series[-1]["nav"] == pytest.approx(110_000.0)  # +10% growth factor applied directly to the starting NAV


def test_second_segment_starts_from_first_segments_ending_nav_minus_reentry_cost() -> None:
    segment_a = _segment(100_000.0, 108_000.0, ["2026-08-12", "2026-09-15"])
    segment_b = _segment(200.0, 220.0, ["2026-09-15", "2026-10-20"])  # unrelated nominal scale
    series = chain_segments([segment_a, segment_b], initial_nav=100_000.0, reentry_cost_bps=10.0)

    ending_nav_a = 108_000.0
    expected_start_b = ending_nav_a * (1 - 10.0 / 10_000.0)
    expected_end_b = expected_start_b * (220.0 / 200.0)

    assert series[1]["nav"] == pytest.approx(ending_nav_a)  # last row of segment A, no cost applied yet
    assert series[2]["nav"] == pytest.approx(expected_start_b)  # first row of segment B, after the re-entry cost
    assert series[-1]["nav"] == pytest.approx(expected_end_b)
    # Segment A's rows are never touched by segment B's re-entry cost or growth factor.
    assert series[0]["nav"] == pytest.approx(100_000.0)


def test_chain_segments_skips_empty_segments_without_erroring() -> None:
    segment = _segment(100_000.0, 105_000.0, ["2026-08-12", "2026-08-13"])
    series = chain_segments([{"equity_curve": []}, segment], initial_nav=100_000.0)
    assert len(series) == 2
    assert series[0]["nav"] == 100_000.0


def test_segment_without_a_completed_weekly_signal_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_signal(*args, **kwargs):
        raise ValueError("No trading data in range 2026-08-12..2026-08-13")

    monkeypatch.setattr(forward_ledger, "build_segment", no_signal)
    assert build_segment_if_signaled(None, Path("snapshot.json"), start="2026-08-12", end="2026-08-13", top_k=5, universe_version="U001") is None


def test_benchmark_series_starts_at_initial_nav_on_first_covered_date_and_skips_gaps() -> None:
    prices = {"2026-08-12": 500.0, "2026-08-14": 550.0}  # 2026-08-13 missing (holiday)
    series = benchmark_series(prices, ["2026-08-12", "2026-08-13", "2026-08-14"], initial_nav=100_000.0)
    assert [row["date"] for row in series] == ["2026-08-12", "2026-08-14"]
    assert series[0]["nav"] == pytest.approx(100_000.0)
    assert series[1]["nav"] == pytest.approx(100_000.0 * 550.0 / 500.0)


def test_benchmark_series_returns_empty_list_when_nothing_covered() -> None:
    assert benchmark_series({}, ["2026-08-12"]) == []


def test_state_from_snapshot_round_trips_a_universe_snapshot_file(tmp_path: Path) -> None:
    state = DashboardState(
        top_k=5,
        sec_user_agent="test@example.com",
        tickers=[
            TickerEntry(ticker="AAPL", company="Apple", price_path="Dashboard Data/Prices/AAPL.csv", source="seed", added_date="2026-08-12"),
            TickerEntry(ticker="MSFT", company="Microsoft", price_path="Dashboard Data/Prices/MSFT.csv", source="seed", added_date="2026-08-12"),
        ],
    )
    snapshot_path = tmp_path / "2026-08-12.json"
    snapshot_path.write_text(json.dumps(state.as_dict()), encoding="utf-8")

    loaded = state_from_snapshot(snapshot_path)
    assert loaded.top_k == 5
    assert loaded.sec_user_agent == "test@example.com"
    assert [t.ticker for t in loaded.tickers] == ["AAPL", "MSFT"]
    assert loaded.tickers[0].price_path == "Dashboard Data/Prices/AAPL.csv"
