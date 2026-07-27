from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_research.macro_sp500.reporting import _atomic_write_html

from .contributions import (
    ContributionDeploymentPolicy,
    ContributionResult,
)


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, float_format=lambda value: f"{value:.4f}")


def _add_period_value_traces(
    figure: go.Figure,
    results: dict[str, ContributionResult],
    *,
    row: int,
) -> None:
    for name, result in results.items():
        figure.add_trace(
            go.Scatter(
                x=result.daily["Date"],
                y=result.daily["TotalValue"],
                name=f"{name} value",
                mode="lines",
            ),
            row=row,
            col=1,
        )
    first = next(iter(results.values()))
    figure.add_trace(
        go.Scatter(
            x=first.daily["Date"],
            y=first.daily["TotalInjected"],
            name="Total cash injected",
            mode="lines",
            line={"dash": "dash", "color": "#64748b"},
        ),
        row=row,
        col=1,
    )


def _contribution_figure(
    full_results: dict[str, ContributionResult],
    holdout_results: dict[str, ContributionResult],
) -> go.Figure:
    strategy = full_results["MacroFearBuy"]
    daily = strategy.daily
    trades = strategy.trades
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.065,
        subplot_titles=(
            "2007–2026: portfolio value and cumulative deposits",
            "Fresh 2017 start: portfolio value and cumulative deposits",
            "Adjusted SPY and fear/euphoria allocation changes",
            "Macro Fear Buy equity exposure",
        ),
        specs=[[{}], [{}], [{"secondary_y": True}], [{}]],
    )
    _add_period_value_traces(figure, full_results, row=1)
    _add_period_value_traces(figure, holdout_results, row=2)
    figure.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Close"],
            name="Adjusted SPY",
            mode="lines",
            line={"color": "#475569", "width": 1.5},
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    signal_trades = trades[
        trades["Reason"].str.contains(
            "FEAR|PANIC|EUPHORIA",
            case=False,
            na=False,
        )
    ]
    for action, symbol, color in (
        ("BUY", "triangle-up", "#15803d"),
        ("SELL", "triangle-down", "#dc2626"),
    ):
        shown = signal_trades[signal_trades["Action"] == action]
        if shown.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=shown["Date"],
                y=shown["ExecutionPrice"],
                name=f"Signal {action}",
                mode="markers",
                marker={"symbol": symbol, "size": 11, "color": color},
                customdata=shown[
                    [
                        "DeploymentTargetWeight",
                        "Notional",
                        "FearScore",
                        "EuphoriaScore",
                        "Reason",
                    ]
                ],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Execution=%{y:.2f}"
                    "<br>Deployment target=%{customdata[0]:.0%}"
                    "<br>Invested=$%{customdata[1]:,.0f}"
                    "<br>Fear=%{customdata[2]:.3f}"
                    "<br>Euphoria=%{customdata[3]:.3f}"
                    "<br>%{customdata[4]}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["ActualWeight"] * 100.0,
            name="Full-history exposure %",
            mode="lines",
            fill="tozeroy",
            opacity=0.45,
        ),
        row=4,
        col=1,
    )
    holdout_daily = holdout_results["MacroFearBuy"].daily
    figure.add_trace(
        go.Scatter(
            x=holdout_daily["Date"],
            y=holdout_daily["ActualWeight"] * 100.0,
            name="Fresh-2017 exposure %",
            mode="lines",
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="USD", row=1, col=1)
    figure.update_yaxes(title_text="USD", row=2, col=1)
    figure.update_yaxes(title_text="SPY price", row=3, col=1)
    figure.update_yaxes(
        title_text="Exposure %",
        range=[0, 105],
        row=4,
        col=1,
    )
    figure.update_xaxes(rangeslider={"visible": True}, row=4, col=1)
    figure.update_layout(
        height=1250,
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
        title="$40,000 initial + $4,000 monthly cash accumulation scenario",
        margin={"l": 70, "r": 60, "t": 90, "b": 50},
    )
    return figure


def generate_contribution_report(
    output_folder: Path,
    *,
    full_results: dict[str, ContributionResult],
    holdout_results: dict[str, ContributionResult],
    comparison: pd.DataFrame,
    prediction_source: Path,
    selected_params_source: Path,
    deployment_policy: ContributionDeploymentPolicy,
) -> Path:
    figure_html = _contribution_figure(
        full_results,
        holdout_results,
    ).to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )
    generated = datetime.now(UTC).astimezone()
    policy_description = (
        f"MILD_FEAR {deployment_policy.mild_fraction:.0%}, "
        f"FEAR {deployment_policy.fear_fraction:.0%}, "
        f"PANIC {deployment_policy.panic_fraction:.0%} of pending cash; "
        f"{deployment_policy.cooldown_sessions}-session cooldown"
    )
    content = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Macro Fear Buy monthly contribution report</title>
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
<h1>$40,000 initial + $4,000 monthly cash accumulation</h1>
<p>Generated {generated:%Y-%m-%d %H:%M:%S %Z}</p>
<div class="note">
The first $40,000 is invested at the starting allocation. Beginning with the
next calendar month, $4,000 enters cash before the first available session open.
Macro Fear Buy leaves those deposits in cash until a weekly fear signal.
Deployment policy: {html.escape(policy_description)}. Its original lump-sum
reserve still follows the 80%/90%/100% allocation targets. Buy &amp; Hold is
the comparison portfolio and invests each monthly deposit immediately.
</div>
<div class="warning">
ROI uses final value divided by total cash injected. TWR CAGR removes the effect
of deposits, while XIRR is the investor's money-weighted annual return.
</div>
{figure_html}
<h2>Comparison</h2>
<div class="table-wrap">{_table(comparison)}</div>
<h2>Provenance</h2>
<ul>
<li>Prediction source: {html.escape(str(prediction_source))}</li>
<li>Selected parameters: {html.escape(str(selected_params_source))}</li>
<li>Signals execute at the next open with configured costs and slippage.</li>
</ul>
</body>
</html>
"""
    timestamp = generated.strftime("%Y%m%d_%H%M%S_%f")
    return _atomic_write_html(
        content,
        output_folder / f"monthly_contribution_report_{timestamp}.html",
    )
