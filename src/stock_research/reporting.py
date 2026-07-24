from __future__ import annotations

import html
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPORT_LABELS = {
    "once": "VIX 전략",
    "extra": "VIX 전략 (신호마다 추가 투자)",
    "buy_and_hold": "Buy & Hold",
    "perfect_foresight_DIAGNOSTIC": "Perfect Foresight (진단 전용)",
    "daily_dca": "Daily DCA",
}


def _column(frame: pd.DataFrame, english: str, korean: str) -> str | None:
    if english in frame.columns:
        return english
    if korean in frame.columns:
        return korean
    return None


def _normalized_trades(frame: pd.DataFrame) -> pd.DataFrame:
    date_col = _column(frame, "Date", "날짜")
    action_col = _column(frame, "Action", "액션")
    price_col = _column(frame, "StockPrice", "가격")
    roi_col = _column(frame, "ROI", "ROI(%)")
    if not date_col or not action_col or not price_col:
        return pd.DataFrame(columns=["Date", "Action", "StockPrice", "ROI", "Reason"])
    normalized = pd.DataFrame({
        "Date": pd.to_datetime(frame[date_col], errors="coerce"),
        "Action": frame[action_col].astype(str),
        "StockPrice": pd.to_numeric(frame[price_col], errors="coerce"),
        "ROI": (
            pd.to_numeric(frame[roi_col], errors="coerce")
            if roi_col else float("nan")
        ),
        "Reason": (
            frame["Reason"].fillna("").astype(str)
            if "Reason" in frame else ""
        ),
    })
    return normalized.dropna(subset=["Date", "StockPrice"]).reset_index(drop=True)


def _action_type(action: str) -> str:
    upper = action.upper()
    if "LIQUIDATE" in upper or upper.endswith("_END"):
        return "LIQUIDATE"
    if "SELL" in upper:
        return "SELL"
    if "BUY" in upper:
        return "BUY"
    return "OTHER"


def _roi(frame: pd.DataFrame) -> float:
    roi_col = _column(frame, "ROI", "ROI(%)")
    if not roi_col or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[roi_col], errors="coerce").iloc[-1])


def _atomic_write_html(content: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", prefix=path.stem + "_",
        dir=path.parent, delete=False, encoding="utf-8", newline="",
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def generate_simulation_report(
    data: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    output_folder: Path,
    *,
    company: str,
    strategy: str,
    parameter_index: int,
    start: str,
    end: str,
) -> Path:
    price_data = (
        data[["날짜", "종가"]].copy()
        .dropna(subset=["날짜", "종가"])
        .sort_values("날짜")
    )
    price_data["날짜"] = pd.to_datetime(price_data["날짜"])
    labels = [label for label, frame in outputs.items() if not frame.empty]
    figure = make_subplots(
        rows=len(labels), cols=1, shared_xaxes=True,
        subplot_titles=[REPORT_LABELS.get(label, label) for label in labels],
        vertical_spacing=min(0.08, 0.25 / max(len(labels), 1)),
    )
    colors = {"BUY": "#16a34a", "SELL": "#dc2626", "LIQUIDATE": "#f59e0b"}
    symbols = {"BUY": "triangle-up", "SELL": "triangle-down", "LIQUIDATE": "x"}

    normalized: dict[str, pd.DataFrame] = {}
    for row_number, label in enumerate(labels, start=1):
        figure.add_trace(
            go.Scatter(
                x=price_data["날짜"], y=price_data["종가"], mode="lines",
                name="Tesla 주가", line={"color": "#2563eb", "width": 1.5},
                legendgroup="price", showlegend=row_number == 1,
                hovertemplate="%{x|%Y-%m-%d}<br>종가: %{y:,.2f}<extra></extra>",
            ),
            row=row_number, col=1,
        )
        trades = _normalized_trades(outputs[label])
        trades["Type"] = trades["Action"].map(_action_type)
        normalized[label] = trades
        for action_type in ("BUY", "SELL", "LIQUIDATE"):
            points = trades[trades["Type"] == action_type]
            if points.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=points["Date"], y=points["StockPrice"], mode="markers",
                    name=action_type,
                    marker={
                        "color": colors[action_type], "symbol": symbols[action_type],
                        "size": 9 if label != "daily_dca" else 5,
                        "opacity": 0.8,
                    },
                    legendgroup=action_type, showlegend=row_number == 1,
                    customdata=points[["Action", "ROI"]].to_numpy(),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>%{customdata[0]}<br>"
                        "가격: %{y:,.2f}<br>ROI: %{customdata[1]:.2f}%<extra></extra>"
                    ),
                ),
                row=row_number, col=1,
            )
        figure.update_yaxes(title_text="주가", row=row_number, col=1)

    figure.update_layout(
        height=max(420, 300 * len(labels)),
        title=f"{company} 시뮬레이션 — 주가와 매수·매도 시점",
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    figure.update_xaxes(title_text="날짜", row=len(labels), col=1)
    chart_html = figure.to_html(
        full_html=False, include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )

    summary_rows = []
    sections = []
    for label in labels:
        trades = normalized[label]
        display_name = REPORT_LABELS.get(label, label)
        summary_rows.append({
            "전략": display_name,
            "ROI(%)": _roi(outputs[label]),
            "BUY": int((trades["Type"] == "BUY").sum()),
            "SELL": int((trades["Type"] == "SELL").sum()),
            "LIQUIDATE": int((trades["Type"] == "LIQUIDATE").sum()),
        })
        table = trades[["Date", "Action", "StockPrice", "ROI", "Reason"]].copy()
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        table["StockPrice"] = table["StockPrice"].map(lambda value: f"{value:,.2f}")
        table["ROI"] = table["ROI"].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}%"
        )
        table.columns = ["날짜", "행동", "주가", "ROI", "사유"]
        sections.append(
            f"<section><h2>{html.escape(display_name)}</h2>"
            f"<div class='table-wrap'>{table.to_html(index=False, escape=True)}</div>"
            "</section>"
        )

    summary = pd.DataFrame(summary_rows)
    summary["ROI(%)"] = summary["ROI(%)"].map(
        lambda value: "" if pd.isna(value) else f"{value:.2f}%"
    )
    now = datetime.now(UTC).astimezone()
    generated_at = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(company)} 시뮬레이션 리포트</title>
<style>
body {{ font-family: "Segoe UI", "Malgun Gothic", sans-serif; margin: 24px;
       color: #172033; background: #f7f8fb; }}
main {{ max-width: 1500px; margin: 0 auto; }}
h1, h2 {{ font-weight: 600; }}
.meta {{ color: #566176; margin-bottom: 20px; }}
.panel {{ background: white; border: 1px solid #dce1e8; border-radius: 10px;
          padding: 20px; margin-bottom: 20px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border-bottom: 1px solid #e5e8ed; padding: 8px 10px; text-align: left; }}
th {{ background: #f0f3f8; position: sticky; top: 0; }}
td:nth-child(3), td:nth-child(4) {{ text-align: right; }}
section {{ margin-top: 32px; }}
</style>
</head>
<body><main>
<h1>{html.escape(company)} 시뮬레이션 통합 리포트</h1>
<div class="meta">전략: {html.escape(strategy)} · 파라미터 Index:
{parameter_index} · 기간: {html.escape(start)} ~ {html.escape(end)} · 생성:
{generated_at}</div>
<div class="panel"><h2>전략별 요약</h2>{summary.to_html(index=False, escape=True)}</div>
<div class="panel">{chart_html}</div>
{''.join(sections)}
</main></body></html>"""

    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"{parameter_index}_{company}_{strategy}_simulation_report_"
        f"{start}_{end}_{timestamp}.html"
    )
    return _atomic_write_html(document, output_folder / filename)
