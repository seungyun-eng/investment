from __future__ import annotations

import html
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_research.macro_sp500.reporting import _atomic_write_html

from .config import FearBuyParams, FearBuySettings
from .portfolio import PortfolioResult


def _table(frame: pd.DataFrame, *, limit: int | None = None) -> str:
    shown = frame.head(limit) if limit is not None else frame
    return shown.to_html(index=False, float_format=lambda value: f"{value:.4f}")


def _report_figure(
    portfolios: dict[str, PortfolioResult],
) -> go.Figure:
    strategy = portfolios["MacroFearBuy"]
    daily = strategy.daily
    trades = strategy.trades
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=(
            "Strict-OOS portfolio value (start = 100)",
            "Adjusted SPY price, actual trades, and equity exposure",
            "Point-in-time fear, euphoria, and VIX percentile",
            "Portfolio drawdown",
        ),
        specs=[[{}], [{"secondary_y": True}], [{}], [{}]],
    )
    for name, result in portfolios.items():
        values = result.daily["TotalValue"]
        figure.add_trace(
            go.Scatter(
                x=result.daily["Date"],
                y=values / float(values.iloc[0]) * 100.0,
                name=name,
                mode="lines",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Close"],
            name="Adjusted SPY",
            mode="lines",
            line={"color": "#475569", "width": 1.5},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["ActualWeight"] * 100.0,
            name="Actual equity exposure %",
            mode="lines",
            fill="tozeroy",
            opacity=0.35,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    if not trades.empty:
        tactical = trades[trades["Sleeve"] == "TACTICAL"]
        for action, symbol, color in (
            ("BUY", "triangle-up", "#15803d"),
            ("SELL", "triangle-down", "#dc2626"),
        ):
            shown = tactical[tactical["Action"] == action]
            if shown.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=shown["Date"],
                    y=shown["ExecutionPrice"],
                    name=f"Tactical {action}",
                    mode="markers",
                    marker={"symbol": symbol, "size": 11, "color": color},
                    customdata=shown[
                        [
                            "SignalDate",
                            "TargetWeight",
                            "FearScore",
                            "EuphoriaScore",
                            "Reason",
                        ]
                    ],
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>Execution=%{y:.2f}"
                        "<br>Signal=%{customdata[0]|%Y-%m-%d}"
                        "<br>Target=%{customdata[1]:.0%}"
                        "<br>Fear=%{customdata[2]:.3f}"
                        "<br>Euphoria=%{customdata[3]:.3f}"
                        "<br>%{customdata[4]}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
    for column, name in (
        ("FearScore", "Fear score"),
        ("EuphoriaScore", "Euphoria score"),
        ("VixPercentile", "VIX trailing percentile"),
    ):
        figure.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily[column] * 100.0,
                name=name,
                mode="lines",
            ),
            row=3,
            col=1,
        )
    for name, result in portfolios.items():
        values = result.daily["TotalValue"]
        drawdown = (values / values.cummax() - 1.0) * 100.0
        figure.add_trace(
            go.Scatter(
                x=result.daily["Date"],
                y=drawdown,
                name=f"{name} drawdown",
                mode="lines",
            ),
            row=4,
            col=1,
        )
    figure.update_yaxes(title_text="Index", row=1, col=1)
    figure.update_yaxes(title_text="SPY price", row=2, col=1, secondary_y=False)
    figure.update_yaxes(
        title_text="Exposure %",
        range=[0, 105],
        row=2,
        col=1,
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Score / percentile", row=3, col=1)
    figure.update_yaxes(title_text="Drawdown %", row=4, col=1)
    figure.update_xaxes(rangeslider={"visible": True}, row=4, col=1)
    figure.update_layout(
        height=1250,
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
        title="Macro Fear Buy: buy more in fear, rebuild cash in euphoria",
        margin={"l": 70, "r": 70, "t": 90, "b": 50},
    )
    return figure


def generate_fear_buy_report(
    output_folder: Path,
    *,
    portfolios: dict[str, PortfolioResult],
    comparison: pd.DataFrame,
    optimization: pd.DataFrame,
    holdout_stability: pd.DataFrame,
    buy_forward_returns: pd.DataFrame,
    tactical_summary: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selected_params: FearBuyParams,
    settings: FearBuySettings,
    prediction_source: Path,
) -> Path:
    figure_html = _report_figure(portfolios).to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )
    params = pd.DataFrame(
        [{"Parameter": key, "Value": value} for key, value in asdict(selected_params).items()]
    )
    strategy = portfolios["MacroFearBuy"]
    trade_table = strategy.trades.copy()
    if not trade_table.empty:
        trade_table["Date"] = pd.to_datetime(trade_table["Date"]).dt.strftime(
            "%Y-%m-%d"
        )
        trade_table["SignalDate"] = pd.to_datetime(
            trade_table["SignalDate"]
        ).dt.strftime("%Y-%m-%d")
    generated = datetime.now(UTC).astimezone()
    content = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Macro Fear Buy SP500 report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
h1, h2 {{ margin-top: 30px; }}
.note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px; }}
.warning {{ background: #fff7ed; border-left: 4px solid #ea580c; padding: 12px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #d8dee9; padding: 7px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>Macro Fear Buy SP500 — strict-OOS research</h1>
<p>Generated {generated:%Y-%m-%d %H:%M:%S %Z}</p>
<div class="note">
The strategy holds an initial core sleeve, deploys tactical cash in weekly
tranches as fear deepens, and trims only the tactical sleeve after a minimum
holding period, a profit gate, and a separate euphoria signal.
</div>
<div class="warning">
Parameters were selected only on data through
{html.escape(settings.development_end)}. Results from
{html.escape(settings.holdout_start)} onward were not used for selection.
This is research, not a guarantee of beating Buy &amp; Hold.
</div>
{figure_html}
<h2>Development and untouched-holdout comparison</h2>
<div class="table-wrap">{_table(comparison)}</div>
<h2>Block-bootstrap excess-return uncertainty</h2>
<div class="table-wrap">{_table(bootstrap)}</div>
<h2>Tactical trade diagnostics</h2>
<div class="table-wrap">{_table(tactical_summary)}</div>
<div class="table-wrap">{_table(buy_forward_returns)}</div>
<h2>Top development candidates on holdout (diagnostic only)</h2>
<div class="warning">
These holdout columns were computed only after parameter selection and did not
change the selected strategy.
</div>
<div class="table-wrap">{_table(holdout_stability)}</div>
<h2>Year-by-year robustness</h2>
<div class="table-wrap">{_table(yearly_metrics)}</div>
<h2>Selected production parameters</h2>
<div class="table-wrap">{_table(params)}</div>
<h2>All executed trades</h2>
<div class="table-wrap">{_table(trade_table)}</div>
<h2>Top development-period candidates</h2>
<div class="table-wrap">{_table(optimization, limit=30)}</div>
<h2>Method and provenance</h2>
<ul>
<li>Signals use only information available at each close and execute at the next open.</li>
<li>VIX and model-risk percentiles use a trailing five-year window with a one-year warmup.</li>
<li>Transaction costs and slippage are each {settings.transaction_cost_bps:.1f} bps.</li>
<li>Cash earns the point-in-time CashRate contained in the OOS prediction file.</li>
<li>ROI = (final value / total injected − 1) × 100.</li>
<li>Prediction source: {html.escape(str(prediction_source))}</li>
</ul>
</body>
</html>
"""
    timestamp = generated.strftime("%Y%m%d_%H%M%S_%f")
    return _atomic_write_html(
        content,
        output_folder / f"macro_fear_buy_report_{timestamp}.html",
    )
