from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_research.dashboard.forward_shadow import (
    COST_BPS,
    MODEL_MEGA,
    MODEL_SYSTEMIC,
    MODEL_V73,
    append_immutable,
    comparison_rows,
    run_accounts,
)


def test_append_immutable_accepts_new_keys_and_same_replay(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"
    first = pd.DataFrame({"model_id": ["A"], "Date": [pd.Timestamp("2026-09-11")], "nav": [100_000.0]})
    append_immutable(path, first, ["model_id", "Date"])
    combined = pd.concat(
        [first, pd.DataFrame({"model_id": ["A"], "Date": [pd.Timestamp("2026-09-14")], "nav": [100_250.0]})],
        ignore_index=True,
    )
    actual = append_immutable(path, combined, ["model_id", "Date"])
    assert actual.Date.tolist() == [pd.Timestamp("2026-09-11"), pd.Timestamp("2026-09-14")]
    replay = append_immutable(path, combined, ["model_id", "Date"])
    pd.testing.assert_frame_equal(actual, replay)


def test_append_immutable_rejects_rewriting_a_past_value(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"
    first = pd.DataFrame({"model_id": ["A"], "Date": [pd.Timestamp("2026-09-11")], "nav": [100_000.0]})
    append_immutable(path, first, ["model_id", "Date"])
    changed = first.assign(nav=99_999.0)
    with pytest.raises(RuntimeError, match="immutable shadow row changed"):
        append_immutable(path, changed, ["model_id", "Date"])


def test_account_executes_next_open_sells_first_and_marks_daily() -> None:
    model = "TEST_MODEL"
    recommendations = pd.DataFrame(
        [
            {
                "model_id": model,
                "signal_date": pd.Timestamp("2026-09-11"),
                "effective_trade_date": pd.Timestamp("2026-09-14"),
                "Ticker": "A",
                "target_weight": 1.0,
            },
            {
                "model_id": model,
                "signal_date": pd.Timestamp("2026-09-14"),
                "effective_trade_date": pd.Timestamp("2026-09-15"),
                "Ticker": "B",
                "target_weight": 1.0,
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {"Date": date, "Ticker": ticker, "AdjOpen": price, "AdjClose": price}
            for date, values in {
                pd.Timestamp("2026-09-11"): {"A": 100.0, "B": 50.0},
                pd.Timestamp("2026-09-14"): {"A": 100.0, "B": 50.0},
                pd.Timestamp("2026-09-15"): {"A": 110.0, "B": 55.0},
            }.items()
            for ticker, price in values.items()
        ]
    )
    holdings, trades, nav = run_accounts(
        recommendations, prices, model_ids=[model]
    )

    assert trades.trade_date.tolist()[:1] == [pd.Timestamp("2026-09-14")]
    second_day = trades.loc[trades.trade_date.eq(pd.Timestamp("2026-09-15"))]
    assert second_day.sort_values("execution_sequence").side.tolist() == ["SELL", "BUY"]
    assert np.allclose(
        trades.transaction_cost,
        trades.notional * COST_BPS / 10_000,
        rtol=0,
        atol=1e-9,
    )
    marked = holdings.groupby(["model_id", "Date"]).market_value.sum()
    ledger = nav.set_index(["model_id", "Date"])
    assert (marked.reindex(ledger.index, fill_value=0) + ledger.cash).to_numpy() == pytest.approx(
        ledger.nav.to_numpy()
    )


def test_model_comparison_includes_held_names_outside_latest_targets() -> None:
    recommendations = pd.DataFrame(
        [
            {"model_id": model_id, "signal_date": pd.Timestamp("2026-09-18"), "Ticker": ticker, "target_weight": 1.0}
            for model_id, ticker in (
                (MODEL_SYSTEMIC, "A"),
                (MODEL_MEGA, "A"),
                (MODEL_V73, "B"),
            )
        ]
    )
    holdings = pd.DataFrame(
        [
            {"model_id": MODEL_SYSTEMIC, "Date": pd.Timestamp("2026-09-18"), "Ticker": "OLD"},
            {"model_id": MODEL_MEGA, "Date": pd.Timestamp("2026-09-18"), "Ticker": "OLD"},
            {"model_id": MODEL_V73, "Date": pd.Timestamp("2026-09-18"), "Ticker": "B"},
        ]
    )

    comparison = comparison_rows(recommendations, holdings).set_index("Ticker")

    assert "OLD" in comparison.index
    assert comparison.loc["OLD", "model_overlap"] == "0/3"
    assert comparison.loc["OLD", "holdings_overlap_count"] == 2
