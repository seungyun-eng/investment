from __future__ import annotations

import html
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import MacroSp500Params
from .optimization import WalkForwardResult
from .portfolio import PortfolioResult


def _atomic_write_html(content: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=path.stem + "_",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _summary_frame(
    strategy: PortfolioResult | WalkForwardResult,
    benchmark: PortfolioResult | None = None,
) -> pd.DataFrame:
    if isinstance(strategy, WalkForwardResult):
        strategy_summary = strategy.oos_summary
        benchmark_summary = strategy.benchmark_summary
    else:
        if benchmark is None:
            raise ValueError("A benchmark result is required for simulation reporting.")
        strategy_summary = strategy.summary
        benchmark_summary = benchmark.summary
    rows = []
    for label, summary in (
        ("Macro SP500", strategy_summary),
        ("Buy & Hold", benchmark_summary),
    ):
        rows.append(
            {
                "Portfolio": label,
                "ROI(%)": summary.roi_percent,
                "CAGR(%)": summary.cagr_percent,
                "MDD(%)": summary.max_drawdown_percent,
                "Calmar": summary.calmar_ratio,
                "Sharpe": summary.sharpe_ratio,
                "AverageExposure(%)": summary.average_exposure_percent,
                "Rebalances": summary.rebalance_count,
            }
        )
    return pd.DataFrame(rows)


def _parameter_table(params: MacroSp500Params) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Parameter": name, "Value": value} for name, value in asdict(params).items()]
    )


def _figure(
    strategy_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    trades: pd.DataFrame,
) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Out-of-sample portfolio value",
            "S&P 500 proxy and rebalances",
            "VIX and trailing percentile",
            "Actual equity exposure",
        ),
        vertical_spacing=0.06,
    )
    strategy_base = float(strategy_daily["TotalValue"].iloc[0])
    benchmark_base = float(benchmark_daily["TotalValue"].iloc[0])
    figure.add_trace(
        go.Scatter(
            x=strategy_daily["Date"],
            y=strategy_daily["TotalValue"] / strategy_base * 100,
            name="Macro SP500",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=benchmark_daily["Date"],
            y=benchmark_daily["TotalValue"] / benchmark_base * 100,
            name="Buy & Hold",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=strategy_daily["Date"],
            y=strategy_daily["Close"],
            name="S&P proxy",
            mode="lines",
        ),
        row=2,
        col=1,
    )
    if not trades.empty:
        colors = {"BUY": "#16a34a", "SELL": "#dc2626"}
        symbols = {"BUY": "triangle-up", "SELL": "triangle-down"}
        for action in ("BUY", "SELL"):
            points = trades[trades["Action"] == action]
            if points.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=points["Date"],
                    y=points["ExecutionPrice"],
                    name=action,
                    mode="markers",
                    marker={
                        "color": colors[action],
                        "symbol": symbols[action],
                        "size": 8,
                    },
                    customdata=points[["TargetWeight", "Reason"]].to_numpy(),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>"
                        "Price=%{y:.2f}<br>"
                        "Target=%{customdata[0]:.0%}<br>"
                        "%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
    figure.add_trace(
        go.Scatter(
            x=strategy_daily["Date"],
            y=strategy_daily["VIX"],
            name="VIX",
            mode="lines",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=strategy_daily["Date"],
            y=strategy_daily["VixPercentile"] * 100,
            name="VIX percentile",
            mode="lines",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=strategy_daily["Date"],
            y=strategy_daily["ActualWeight"] * 100,
            name="Equity exposure",
            mode="lines",
            fill="tozeroy",
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="Index=100", row=1, col=1)
    figure.update_yaxes(title_text="Price", row=2, col=1)
    figure.update_yaxes(title_text="Level / %", row=3, col=1)
    figure.update_yaxes(title_text="Exposure %", range=[0, 105], row=4, col=1)
    figure.update_xaxes(title_text="Date", row=4, col=1)
    figure.update_layout(
        height=1150,
        hovermode="x unified",
        template="plotly_white",
        title="Macro SP500 strategy report",
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    return figure


def _document(
    *,
    title: str,
    summary: pd.DataFrame,
    parameters: pd.DataFrame,
    figure: go.Figure,
    details: str,
    source_note: str,
) -> str:
    chart_html = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    generated_at = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: "Segoe UI", sans-serif; margin: 24px; color: #172033;
       background: #f7f8fb; }}
main {{ max-width: 1500px; margin: 0 auto; }}
.panel {{ background: white; border: 1px solid #dce1e8; border-radius: 10px;
          padding: 20px; margin-bottom: 20px; }}
.warning {{ background: #fff8e6; border-color: #f0c36d; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border-bottom: 1px solid #e5e8ed; padding: 8px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f0f3f8; }}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<p>Generated {html.escape(generated_at)}</p>
<div class="panel warning">{html.escape(source_note)}</div>
<div class="panel"><h2>Performance</h2>
<div class="table-wrap">{summary.to_html(index=False, float_format=lambda value: f"{value:.3f}")}</div>
</div>
<div class="panel"><h2>Parameters</h2>
<div class="table-wrap">{parameters.to_html(index=False)}</div>
</div>
<div class="panel">{chart_html}</div>
{details}
</main></body></html>"""


def generate_optimization_report(
    result: WalkForwardResult,
    output_folder: Path,
    *,
    source_note: str,
) -> Path:
    figure = _figure(
        result.oos_daily,
        result.benchmark_daily,
        result.oos_trades,
    )
    folds = result.folds.copy()
    details = (
        "<div class='panel'><h2>Walk-forward folds</h2>"
        "<div class='table-wrap'>"
        f"{folds.to_html(index=False, float_format=lambda value: f'{value:.4f}')}"
        "</div>"
        "</div>"
    )
    document = _document(
        title="Macro SP500 first-pass walk-forward optimization",
        summary=_summary_frame(result),
        parameters=_parameter_table(result.latest_params),
        figure=figure,
        details=details,
        source_note=source_note,
    )
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return _atomic_write_html(
        document,
        output_folder / f"macro_sp500_optimization_report_{timestamp}.html",
    )


def generate_simulation_report(
    strategy: PortfolioResult,
    benchmark: PortfolioResult,
    params: MacroSp500Params,
    output_folder: Path,
    *,
    source_note: str,
    start: str,
    end: str,
) -> Path:
    figure = _figure(strategy.daily, benchmark.daily, strategy.trades)
    trade_table = strategy.trades.copy()
    details = (
        "<div class='panel'><h2>Rebalances</h2>"
        "<div class='table-wrap'>"
        f"{trade_table.to_html(index=False, float_format=lambda value: f'{value:.4f}')}"
        "</div>"
        "</div>"
    )
    document = _document(
        title=f"Macro SP500 simulation {start} to {end}",
        summary=_summary_frame(strategy, benchmark),
        parameters=_parameter_table(params),
        figure=figure,
        details=details,
        source_note=source_note,
    )
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return _atomic_write_html(
        document,
        output_folder / f"macro_sp500_simulation_report_{timestamp}.html",
    )
