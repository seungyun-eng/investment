import json
from pathlib import Path

import pandas as pd
import pytest

from stock_research import data_loading
from stock_research.optimization import optimize_vix, run_vix_backtest
from stock_research.strategies.vix import VixParams, load_vix_rule_config, vix_buy_signal


def _cycle_data(cycles: int = 2) -> pd.DataFrame:
    rows = []
    for cycle in range(cycles):
        rows.extend([
            {
                "날짜": pd.Timestamp("2024-01-01") + pd.Timedelta(days=cycle * 2),
                "종가": 90.0, "VIX": 30.0, "RSI (14일)": 10.0,
                "볼린저밴드 하단": 100.0,
                "볼린저밴드 상단": 110.0,
                "MACD": 2.0, "MACD 시그널": 1.0,
            },
            {
                "날짜": pd.Timestamp("2024-01-02") + pd.Timedelta(days=cycle * 2),
                "종가": 120.0, "VIX": 10.0, "RSI (14일)": 90.0,
                "볼린저밴드 하단": 100.0,
                "볼린저밴드 상단": 110.0,
                "MACD": 0.0, "MACD 시그널": 1.0,
            },
        ])
    return pd.DataFrame(rows)


def test_actual_vix_is_strictly_merged_by_date(monkeypatch):
    stock = pd.DataFrame({
        "날짜": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "종가": [100.0, 101.0],
    })
    vix = pd.DataFrame({
        "날짜": pd.to_datetime(["2024-01-03", "2024-01-02"]),
        "VIX": [19.0, 31.0],
    })
    monkeypatch.setattr(data_loading, "load_processed", lambda *args, **kwargs: stock)
    monkeypatch.setattr(data_loading, "load_vix", lambda *args, **kwargs: vix)
    merged = data_loading.load_processed_with_vix(
        Path("processed"), Path("macro"), "Example"
    )
    assert merged["VIX"].tolist() == [31.0, 19.0]
    assert merged.attrs["actual_vix_stats"] == {
        "minimum": 19.0, "maximum": 31.0, "missing_count": 0,
    }


def test_missing_and_nonpositive_actual_vix_are_rejected():
    frame = pd.DataFrame({
        "날짜": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "VIX": [20.0, None],
    })
    with pytest.raises(ValueError, match="Missing actual daily VIX"):
        data_loading.validate_actual_vix(frame)
    frame["VIX"] = [20.0, 0.0]
    with pytest.raises(ValueError, match="greater than zero"):
        data_loading.validate_actual_vix(frame)


def test_vix_thresholds_come_from_config_and_are_not_optimized(tmp_path):
    config_path = tmp_path / "vix.json"
    config_path.write_text(json.dumps({
        "vix_buy_level": 29.0, "vix_sell_level": 11.0,
        "source": "actual_daily_vix", "rule_type": "fixed_levels",
    }), encoding="utf-8")
    rules = load_vix_rule_config(config_path)
    params, _, importance = optimize_vix(
        _cycle_data(), rules=rules, tpe_trials=2, cma_trials=0, seed=1
    )
    assert params.vix_buy_level == 29.0
    assert params.vix_sell_level == 11.0
    assert set(importance) <= {"rsi_buy_th", "rsi_sell_th", "boll_buffer"}
    assert 20.0 <= params.rsi_buy_th <= 45.0
    assert 55.0 <= params.rsi_sell_th <= 80.0
    assert params.rsi_sell_th >= params.rsi_buy_th + 15.0
    assert 0.0 <= params.boll_buffer <= 0.03


def test_actual_daily_vix_controls_eligibility():
    params = VixParams(35.0, 65.0, 0.0, 25.0, 20.0)
    row = _cycle_data(1).iloc[0].copy()
    row["VIX"] = 24.99
    assert not vix_buy_signal(row, params)
    row["VIX"] = 25.0
    assert vix_buy_signal(row, params)


def test_buy_plus_final_liquidation_trial_is_rejected():
    only_buy = _cycle_data(1).iloc[[0]].copy()
    with pytest.raises(RuntimeError, match="No valid VIX trial"):
        optimize_vix(only_buy, tpe_trials=1, cma_trials=0, seed=1)


def test_vix_trade_logs_have_required_context_and_reason():
    params = VixParams(35.0, 65.0, 0.0, 25.0, 20.0)
    result = run_vix_backtest(_cycle_data(2), params)
    required = {
        "Date", "Action", "StockPrice", "ActualVIX", "VixBuyLevel",
        "VixSellLevel", "RSI", "BollingerLower", "BollingerUpper", "MACD",
        "MACDSignal", "Reason", "Cash", "Shares", "TotalValue", "ROI",
    }
    assert required <= set(result.trades.columns)
    assert result.trades["Reason"].str.contains("ActualVIX").all()
    assert result.summary.completed_trades == 2
