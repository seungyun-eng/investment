from pathlib import Path

import pandas as pd

from stock_research.reporting import generate_simulation_report


def test_simulation_report_contains_chart_actions_and_tables(tmp_path: Path):
    data = pd.DataFrame({
        "날짜": pd.date_range("2024-01-01", periods=3),
        "종가": [100.0, 110.0, 105.0],
    })
    strategy = pd.DataFrame([
        {
            "Date": pd.Timestamp("2024-01-01"), "Action": "BUY",
            "StockPrice": 100.0, "ROI": 0.0, "Reason": "ActualVIX buy rule",
        },
        {
            "Date": pd.Timestamp("2024-01-02"), "Action": "SELL",
            "StockPrice": 110.0, "ROI": 10.0, "Reason": "ActualVIX sell rule",
        },
    ])
    buy_hold = pd.DataFrame([
        {
            "날짜": pd.Timestamp("2024-01-01"), "액션": "BUY_AND_HOLD_BUY",
            "가격": 100.0, "ROI(%)": 0.0,
        },
        {
            "날짜": pd.Timestamp("2024-01-03"), "액션": "BUY_AND_HOLD_END",
            "가격": 105.0, "ROI(%)": 5.0,
        },
    ])
    report = generate_simulation_report(
        data, {"once": strategy, "buy_and_hold": buy_hold}, tmp_path,
        company="Tesla", strategy="vix", parameter_index=4,
        start="2024-01-01", end="2024-01-03",
    )
    content = report.read_text(encoding="utf-8")
    assert report.suffix == ".html"
    assert "Tesla 시뮬레이션 통합 리포트" in content
    assert "ActualVIX buy rule" in content
    assert "BUY_AND_HOLD_END" in content
    assert "plotly" in content.lower()
