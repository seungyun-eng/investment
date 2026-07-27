from __future__ import annotations

import html
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_research.macro_sp500.reporting import _atomic_write_html

from .config import ResearchConfig
from .evaluation import WalkForwardResult
from .portfolio import AllocationResult, performance_table, trade_cycle_diagnostics


def _table(frame: pd.DataFrame, *, limit: int | None = None) -> str:
    shown = frame.head(limit) if limit else frame
    return shown.to_html(index=False, float_format=lambda value: f"{value:.4f}")


def _research_figure(
    walk_forward: WalkForwardResult,
    portfolios: dict[str, AllocationResult],
    config: ResearchConfig,
) -> go.Figure:
    predictions = walk_forward.predictions
    strategy = portfolios.get("StatefulMacro", portfolios["MacroMomentum"]).daily
    strategy_trades = portfolios.get(
        "StatefulMacro",
        portfolios["MacroMomentum"],
    ).trades
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Strict out-of-sample portfolio value (normalized)",
            "Adjusted SPY price, stateful trades, and equity exposure",
            "Smoothed model risk and point-in-time macro confirmation",
            "Predicted vs realized forward excess return",
        ),
        specs=[[{}], [{"secondary_y": True}], [{}], [{}]],
    )
    for name, result in portfolios.items():
        base = float(result.daily["TotalValue"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=result.daily["Date"],
                y=result.daily["TotalValue"] / base * 100,
                name=name,
                mode="lines",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Close"],
            name="Adjusted SPY",
            mode="lines",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=strategy["Date"],
            y=strategy["ActualWeight"] * 100,
            name="Stateful exposure %",
            mode="lines",
            fill="tozeroy",
            opacity=0.45,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    if not strategy_trades.empty:
        for action, symbol, color in (
            ("BUY", "triangle-up", "#15803d"),
            ("SELL", "triangle-down", "#dc2626"),
        ):
            shown = strategy_trades[strategy_trades["Action"] == action]
            if shown.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=shown["Date"],
                    y=shown["ExecutionPrice"],
                    name=f"Stateful {action}",
                    mode="markers",
                    marker={"symbol": symbol, "size": 10, "color": color},
                    customdata=shown[
                        [
                            "SignalDate",
                            "TargetWeight",
                            "State",
                            "TransitionReason",
                        ]
                    ],
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>Price=%{y:.2f}"
                        "<br>Signal=%{customdata[0]|%Y-%m-%d}"
                        "<br>Target=%{customdata[1]:.0%}"
                        "<br>State=%{customdata[2]}"
                        "<br>Reason=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
    figure.add_trace(
        go.Scatter(
            x=strategy["Date"],
            y=strategy["RiskScore"] * 100,
            name="Smoothed 63/126d risk %",
            mode="lines",
        ),
        row=3,
        col=1,
    )
    figure.add_hline(
        y=config.state_caution_entry_risk * 100,
        line_dash="dash",
        line_color="#dc2626",
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=strategy["Date"],
            y=strategy["MacroScore"] * 100,
            name="Smoothed macro confirmation %",
            mode="lines",
        ),
        row=3,
        col=1,
    )
    return_horizon = config.primary_return_horizon
    figure.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions[f"PredictedExcessReturn_{return_horizon}"] * 100,
            name=f"Predicted {return_horizon}d excess return %",
            mode="lines",
        ),
        row=4,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions[f"ExcessReturn_{return_horizon}"] * 100,
            name=f"Realized {return_horizon}d excess return %",
            mode="lines",
            opacity=0.45,
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="Index=100", row=1, col=1)
    figure.update_yaxes(title_text="Price", row=2, col=1, secondary_y=False)
    figure.update_yaxes(
        title_text="Exposure %", range=[0, 105], row=2, col=1, secondary_y=True
    )
    figure.update_yaxes(title_text="Probability %", row=3, col=1)
    figure.update_yaxes(title_text="Return %", row=4, col=1)
    figure.update_layout(
        height=1200,
        template="plotly_white",
        hovermode="x unified",
        title="Macro + momentum SP500 point-in-time research",
        margin={"l": 70, "r": 60, "t": 90, "b": 60},
    )
    return figure


def generate_research_report(
    walk_forward: WalkForwardResult,
    portfolios: dict[str, AllocationResult],
    config: ResearchConfig,
    output_folder: Path,
    *,
    sensitivity: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sources: dict[str, str],
) -> Path:
    figure_html = _research_figure(walk_forward, portfolios, config).to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    selection_frequency = (
        walk_forward.selections.groupby(
            ["Task", "Family", "FeatureGroup"], as_index=False
        )
        .size()
        .sort_values(["Task", "size"], ascending=[True, False])
    )
    top_features = pd.DataFrame()
    if not walk_forward.feature_importance.empty:
        top_features = (
            walk_forward.feature_importance.groupby(
                ["Task", "Feature"],
                as_index=False,
            )
            .agg(
                MeanNormalizedImportance=("NormalizedImportance", "mean"),
                MedianNormalizedImportance=("NormalizedImportance", "median"),
                MeanSignedEffect=("SignedEffect", "mean"),
                OuterFolds=("OuterYear", "nunique"),
            )
            .sort_values(
                ["Task", "MeanNormalizedImportance"],
                ascending=[True, False],
            )
            .groupby("Task", as_index=False)
            .head(20)
        )
    all_metrics = walk_forward.metrics[walk_forward.metrics["Period"] == "All"]
    candidate_evaluations = int(
        walk_forward.candidate_scores["InnerFoldCount"].sum()
    )
    config_table = pd.DataFrame(
        [{"Parameter": key, "Value": value} for key, value in asdict(config).items()]
    )
    source_table = pd.DataFrame(
        [{"Series": key, "Source": value} for key, value in sources.items()]
    )
    generated_at = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    stateful_trades = portfolios.get("StatefulMacro", portfolios["MacroMomentum"]).trades
    stateful_cycles = trade_cycle_diagnostics(
        portfolios.get("StatefulMacro", portfolios["MacroMomentum"])
    )
    state_transitions = (
        stateful_trades[
            stateful_trades.get("TransitionReason", pd.Series(dtype=str)).fillna("").ne("")
        ]
        if not stateful_trades.empty and "TransitionReason" in stateful_trades
        else stateful_trades
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Macro + momentum SP500 research</title>
<style>
body {{ font-family: "Segoe UI", sans-serif; margin: 24px; color: #172033;
background:#f7f8fb; }} main {{ max-width:1500px; margin:auto; }}
.panel {{ background:white; border:1px solid #dce1e8; border-radius:10px;
padding:20px; margin-bottom:20px; }} .warning {{ background:#fff8e6;
border-color:#f0c36d; }} .good {{ background:#edf9f0; border-color:#94d3a2; }}
.table-wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; }}
th,td {{ border-bottom:1px solid #e5e8ed; padding:8px 10px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }} th {{ background:#f0f3f8; }}
code {{ background:#edf0f4; padding:2px 5px; border-radius:4px; }}
</style></head><body><main>
<h1>Macro + momentum SP500 research</h1>
<p>Generated {html.escape(generated_at)}</p>
<div class="panel warning"><strong>Interpretation:</strong> This is a predictive
research report, not evidence of a deployable trading edge. Every charted prediction
is strict out-of-sample. Monthly releases use a {config.monthly_release_lag_days}-day
lag, labels overlapping a validation boundary are purged, and trades use the next
session's open. The legacy <code>HY_Spread.csv</code> is treated as high-yield
effective yield, not OAS.</div>
<div class="panel good"><strong>Stateful allocation:</strong> Allocation decisions
are evaluated at the final trading session of each week. The 63/126-day risk
probabilities are smoothed and require point-in-time macro confirmation from
volatility, credit, financial conditions, labor, and the yield curve. Separate
entry/exit thresholds, confirmation counts, minimum hold, recovery, cooldown,
and a stronger loss-exit gate reduce one-day threshold reversals. The weak
expected-return regression remains diagnostic and does not drive this state
machine.</div>
<div class="panel good"><strong>Search coverage:</strong> {candidate_evaluations:,}
inner time-fold candidate evaluations across annual outer folds. Model family,
regularization, tree complexity, and feature group are reselected using past data
only.</div>
<div class="panel"><h2>Portfolio comparison</h2><div class="table-wrap">
{_table(performance_table(portfolios))}</div></div>
<div class="panel">{figure_html}</div>
<div class="panel"><h2>Stateful macro transitions and trades</h2><div class="table-wrap">
{_table(state_transitions)}</div></div>
<div class="panel"><h2>Stateful exposure-cycle diagnostics</h2>
<p>Execution-price pairing is diagnostic for partial allocation cycles, not
tax-lot realized P&amp;L. <code>Adverse=true</code> means a normal exposure was
reduced below its latest restoration price or defensive cash was reinvested at
a higher price.</p><div class="table-wrap">
{_table(stateful_cycles)}</div></div>
<div class="panel"><h2>Aggregate predictive metrics</h2><div class="table-wrap">
{_table(all_metrics)}</div></div>
<div class="panel"><h2>Risk calibration</h2><div class="table-wrap">
{_table(walk_forward.calibration)}</div></div>
<div class="panel"><h2>Selected model frequency</h2><div class="table-wrap">
{_table(selection_frequency)}</div></div>
<div class="panel"><h2>Strict OOS feature importance</h2>
<p>Linear models use standardized coefficient magnitude. Nonlinear models use
permutation importance measured only on that year's outer test sample. Importance
is diagnostic and never feeds model selection.</p><div class="table-wrap">
{_table(top_features)}</div></div>
<div class="panel"><h2>Stateful macro bootstrap vs buy & hold</h2><div class="table-wrap">
{_table(bootstrap)}</div></div>
<div class="panel"><h2>Year-by-year portfolio behavior</h2><div class="table-wrap">
{_table(yearly_metrics)}</div></div>
<div class="panel"><h2>Allocation and cost sensitivity</h2><div class="table-wrap">
{_table(sensitivity)}</div></div>
<div class="panel"><h2>Configuration</h2><div class="table-wrap">
{_table(config_table)}</div></div>
<div class="panel"><h2>Data sources</h2><div class="table-wrap">
{_table(source_table)}</div></div>
</main></body></html>"""
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return _atomic_write_html(
        document,
        output_folder / f"macro_momentum_sp500_report_{timestamp}.html",
    )
