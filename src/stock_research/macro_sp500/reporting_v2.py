from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .optimization_v2 import WalkForwardV2Result
from .reporting import _atomic_write_html, _document, _figure


def generate_v2_optimization_report(
    result: WalkForwardV2Result,
    output_folder: Path,
    *,
    source_note: str,
) -> Path:
    figure = _figure(
        result.oos_strategy.daily,
        result.buy_hold.daily,
        result.oos_strategy.trades,
    )
    figure.data[0].name = "Macro SP500 V2"
    static_daily = result.static_70_30.daily
    static_base = float(static_daily["TotalValue"].iloc[0])
    figure.add_trace(
        go.Scatter(
            x=static_daily["Date"],
            y=static_daily["TotalValue"] / static_base * 100,
            name="Static 70/30",
            mode="lines",
            line={"dash": "dot"},
        ),
        row=1,
        col=1,
    )
    summaries = []
    for name, summary in (
        ("Macro SP500 V2", result.oos_strategy.summary),
        ("Buy & Hold", result.buy_hold.summary),
        ("Static 70/30", result.static_70_30.summary),
    ):
        summaries.append(
            {
                "Portfolio": name,
                "ROI(%)": summary.roi_percent,
                "CAGR(%)": summary.cagr_percent,
                "MDD(%)": summary.max_drawdown_percent,
                "Calmar": summary.calmar_ratio,
                "AverageExposure(%)": summary.average_exposure_percent,
                "Turnover": summary.turnover_multiple,
                "Rebalances": summary.rebalance_count,
            }
        )
    parameters = pd.DataFrame(
        [
            {"Parameter": key, "Value": value}
            for key, value in asdict(result.latest_params).items()
        ]
    )
    details = (
        "<div class='panel'><h2>Walk-forward folds</h2>"
        "<div class='table-wrap'>"
        f"{result.folds.to_html(index=False, float_format=lambda value: f'{value:.4f}')}"
        "</div></div>"
    )
    document = _document(
        title="Macro SP500 V2 walk-forward optimization",
        summary=pd.DataFrame(summaries),
        parameters=parameters,
        figure=figure,
        details=details,
        source_note=source_note,
    )
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return _atomic_write_html(
        document,
        output_folder / f"macro_sp500_v2_report_{timestamp}.html",
    )
