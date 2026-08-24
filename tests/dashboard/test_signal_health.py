from __future__ import annotations

import pandas as pd
import pytest

from stock_research.dashboard.signal_health import (
    alpha_vs_universe_ew_series,
    build_signal_health,
    next_holding_period_returns,
    rank_ic_series,
    selection_rate,
    spearman_ic,
    top_k_spread_series,
    universe_equal_weight_curve_gross,
)


def test_spearman_ic_is_perfect_for_a_monotonic_relationship() -> None:
    assert spearman_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ic_returns_none_for_constant_input_instead_of_nan() -> None:
    assert spearman_ic([1, 1, 1, 1], [10, 20, 30, 40]) is None
    assert spearman_ic([1, 2, 3, 4], [5, 5, 5, 5]) is None


def _daily(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """rows: (date, ticker, open)."""
    frame = pd.DataFrame(rows, columns=["Date", "Ticker", "Open"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def _weekly(rows: list[tuple[str, str, float]], extra: dict | None = None) -> pd.DataFrame:
    """rows: (date, ticker, score). Adds Qualified=True by default."""
    frame = pd.DataFrame(rows, columns=["Date", "Ticker", "AlphaScore"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Qualified"] = True
    if extra:
        for key, value in extra.items():
            frame[key] = value
    return frame


# Two weekly signals (Fri 08-01, Fri 08-08), each executing at the next
# trading session's open (Mon 08-04, Mon 08-11) -- Friday-close is
# deliberately NOT used anywhere so a gap there can't leak into the return.
_DAILY = _daily([
    ("2026-07-31", "AAA", 999.0), ("2026-08-01", "AAA", 999.0),
    ("2026-08-04", "AAA", 100.0), ("2026-08-07", "AAA", 999.0), ("2026-08-08", "AAA", 999.0),
    ("2026-08-11", "AAA", 120.0), ("2026-08-14", "AAA", 999.0), ("2026-08-15", "AAA", 999.0),
    ("2026-08-18", "AAA", 130.0),
    ("2026-07-31", "BBB", 999.0), ("2026-08-01", "BBB", 999.0),
    ("2026-08-04", "BBB", 100.0), ("2026-08-07", "BBB", 999.0), ("2026-08-08", "BBB", 999.0),
    ("2026-08-11", "BBB", 90.0), ("2026-08-14", "BBB", 999.0), ("2026-08-15", "BBB", 999.0),
    ("2026-08-18", "BBB", 80.0),
    ("2026-07-31", "CCC", 999.0), ("2026-08-01", "CCC", 999.0),
    ("2026-08-04", "CCC", 100.0), ("2026-08-07", "CCC", 999.0), ("2026-08-08", "CCC", 999.0),
    ("2026-08-11", "CCC", 105.0), ("2026-08-14", "CCC", 999.0), ("2026-08-15", "CCC", 999.0),
    ("2026-08-18", "CCC", 100.0),
])
_SIGNAL_DATES = ["2026-08-01", "2026-08-08", "2026-08-15"]


def test_next_holding_period_returns_uses_execution_open_not_signal_close() -> None:
    scored = _weekly([(d, t, 0.0) for d in _SIGNAL_DATES for t in ("AAA", "BBB")])
    result = next_holding_period_returns(scored, _DAILY).set_index(["Ticker", "Date"])
    # AAA: entry open at 08-04 (=100), exit open at 08-11 (=120) -> +20%.
    assert result.loc[("AAA", "2026-08-01"), "ForwardReturn"] == pytest.approx(0.20)
    assert result.loc[("AAA", "2026-08-01"), "Status"] == "FINAL"
    # BBB: 100 -> 90 -> -10%.
    assert result.loc[("BBB", "2026-08-01"), "ForwardReturn"] == pytest.approx(-0.10)
    # Second signal (08-08): entry open 08-11, exit open 08-18.
    assert result.loc[("AAA", "2026-08-08"), "ForwardReturn"] == pytest.approx(130.0 / 120.0 - 1.0)
    assert result.loc[("AAA", "2026-08-08"), "Status"] == "FINAL"
    # Most recent signal date (08-15) has no next signal yet -> PENDING, no return.
    assert pd.isna(result.loc[("AAA", "2026-08-15"), "ForwardReturn"])
    assert result.loc[("AAA", "2026-08-15"), "Status"] == "PENDING"


def test_rank_ic_series_marks_the_latest_date_pending_not_final() -> None:
    scored = _weekly([(d, t, s) for d in _SIGNAL_DATES for t, s in (("AAA", 4.0), ("CCC", 2.0), ("BBB", 1.0))])
    frame = next_holding_period_returns(scored, _DAILY)
    series = rank_ic_series(frame, "AlphaScore")
    by_date = {row["date"]: row for row in series}
    assert by_date["2026-08-01"]["status"] == "FINAL"
    # AAA(4, +20%) > CCC(2, +5%) > BBB(1, -10%) -- perfect rank agreement.
    assert by_date["2026-08-01"]["ic"] == pytest.approx(1.0)
    assert by_date["2026-08-15"]["status"] == "PENDING"
    assert by_date["2026-08-15"]["ic"] is None


def test_rolling_average_withholds_a_window_until_enough_final_weeks_exist() -> None:
    from stock_research.dashboard.signal_health import rolling_average

    rows = [{"ic": 0.1, "status": "FINAL"} for _ in range(12)] + [{"ic": 0.2, "status": "FINAL"}] + [{"ic": None, "status": "PENDING"}]
    result = rolling_average(rows, "ic", (13, 26))
    assert result["13W"] == pytest.approx((0.1 * 12 + 0.2) / 13)
    assert result["26W"] is None  # only 13 FINAL weeks exist, not 26 -- must not report a partial average


def test_top_k_spread_series_uses_holding_period_return_and_marks_pending() -> None:
    scored = _weekly([
        ("2026-08-01", "AAA", 4.0), ("2026-08-01", "BBB", 1.0),
        ("2026-08-08", "AAA", 4.0), ("2026-08-08", "BBB", 1.0),
        ("2026-08-15", "AAA", 4.0), ("2026-08-15", "BBB", 1.0),
    ])
    frame = next_holding_period_returns(scored, _DAILY)
    spread = top_k_spread_series(frame, k=1)
    by_date = {row["date"]: row for row in spread}
    assert by_date["2026-08-01"]["top_k_return"] == pytest.approx(0.20)  # AAA only
    assert by_date["2026-08-01"]["universe_ew_return"] == pytest.approx((0.20 - 0.10) / 2)
    assert by_date["2026-08-15"]["status"] == "PENDING"
    assert by_date["2026-08-15"]["spread"] is None


def test_universe_equal_weight_curve_grows_by_the_average_holding_period_return() -> None:
    scored = _weekly([(d, t, 0.0) for d in _SIGNAL_DATES for t in ("AAA", "BBB")])
    frame = next_holding_period_returns(scored, _DAILY)
    curve = universe_equal_weight_curve_gross(frame, initial_nav=100_000.0)
    assert curve[0] == {"date": "2026-08-01", "nav": 100_000.0}
    # AAA +20%, BBB -10% -> average +5%.
    assert curve[1]["nav"] == pytest.approx(105_000.0)


def test_alpha_vs_universe_ew_series_shows_the_spread_widening_over_time() -> None:
    model_curve = [{"date": "2026-08-01", "nav": 100_000.0}, {"date": "2026-08-08", "nav": 110_000.0}, {"date": "2026-08-15", "nav": 121_000.0}]
    ew_curve = [{"date": "2026-08-01", "nav": 100_000.0}, {"date": "2026-08-08", "nav": 102_000.0}, {"date": "2026-08-15", "nav": 104_040.0}]
    series = alpha_vs_universe_ew_series(model_curve, ew_curve)
    assert series[0] == {"date": "2026-08-01", "model_return": 0.0, "universe_ew_return": 0.0, "alpha": 0.0}
    assert series[1]["model_return"] == pytest.approx(0.10)
    assert series[1]["universe_ew_return"] == pytest.approx(0.02)
    assert series[1]["alpha"] == pytest.approx(0.08)
    assert series[2]["alpha"] == pytest.approx(0.21 - 0.0404, abs=1e-6)  # spread keeps widening, not just the endpoint


def test_alpha_vs_universe_ew_series_drops_dates_not_shared_by_both_curves() -> None:
    model_curve = [{"date": "2026-08-01", "nav": 100_000.0}, {"date": "2026-08-08", "nav": 110_000.0}]
    ew_curve = [{"date": "2026-08-01", "nav": 100_000.0}]  # 08-08 missing from this side
    series = alpha_vs_universe_ew_series(model_curve, ew_curve)
    assert [row["date"] for row in series] == ["2026-08-01"]


def test_alpha_vs_universe_ew_series_returns_empty_for_an_empty_curve() -> None:
    assert alpha_vs_universe_ew_series([], [{"date": "2026-08-01", "nav": 100.0}]) == []
    assert alpha_vs_universe_ew_series([{"date": "2026-08-01", "nav": 100.0}], []) == []


def test_selection_rate() -> None:
    assert selection_rate(5, 31) == pytest.approx(5 / 31)
    assert selection_rate(5, 0) == 0.0


def test_build_signal_health_returns_the_full_payload_shape() -> None:
    scored = _weekly([(d, t, 0.0) for d in _SIGNAL_DATES for t in ("AAA", "BBB")])
    payload = build_signal_health(scored, _DAILY, k=1)
    assert payload["universe_size"] == 2
    assert payload["top_k"] == 1
    assert payload["selection_rate"] == pytest.approx(0.5)
    assert payload["completed_weeks"] == 2  # 2 of the 3 signal dates matured to FINAL
    assert len(payload["portfolio_ic"]) == 3
    assert payload["portfolio_ic_rolling"]["13W"] is None  # nowhere near 13 FINAL weeks
    assert payload["factor_ic"] == {}  # no factor columns in this synthetic panel
    assert len(payload["top_k_spread"]) == 3
    assert len(payload["universe_equal_weight_curve_gross"]) == 3
