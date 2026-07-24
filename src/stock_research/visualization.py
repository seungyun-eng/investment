from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_financial_charts(summary_workbook: Path, output_folder: Path) -> list[Path]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install plotly to create financial charts.") from exc

    output_folder.mkdir(parents=True, exist_ok=True)
    quarterly = pd.read_excel(summary_workbook, sheet_name="Quarterly")
    yearly = pd.read_excel(summary_workbook, sheet_name="Yearly_TTM")
    price = pd.read_excel(summary_workbook, sheet_name="Price_History")

    for frame in (quarterly, yearly, price):
        date_col = "Date" if "Date" in frame else "날짜"
        if date_col in frame:
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
            if date_col != "Date":
                frame.rename(columns={date_col: "Date"}, inplace=True)

    price_col = next(
        (c for c in ("Price", "종가", "Close") if c in price.columns),
        None,
    )
    outputs: list[Path] = []

    def save(fig, name: str):
        path = output_folder / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        outputs.append(path)

    if {"Date", "EPS_TTM"}.issubset(yearly.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly["Date"], y=yearly["EPS_TTM"], name="EPS_TTM"))
        if price_col:
            fig.add_trace(go.Scatter(
                x=price["Date"], y=price[price_col], name="Price", yaxis="y2"
            ))
        fig.update_layout(
            title="EPS TTM and Price",
            yaxis={"title": "EPS"},
            yaxis2={"title": "Price", "overlaying": "y", "side": "right"},
        )
        save(fig, "01_eps_ttm_price")

    dcf_columns = [
        c for c in ("DCF_FCFF_Price", "DCF_OwnerEarnings_Price")
        if c in yearly.columns
    ]
    if dcf_columns:
        fig = go.Figure()
        for column in dcf_columns:
            fig.add_trace(go.Scatter(x=yearly["Date"], y=yearly[column], name=column))
        if price_col:
            fig.add_trace(go.Scatter(
                x=price["Date"], y=price[price_col], name="Price", yaxis="y2"
            ))
        fig.update_layout(
            title="DCF Estimates and Price",
            yaxis={"title": "DCF"},
            yaxis2={"title": "Price", "overlaying": "y", "side": "right"},
        )
        save(fig, "02_dcf_price")

    multiples = [c for c in ("PE_TTM", "ROIC", "EV/EBIT") if c in yearly.columns]
    if multiples:
        fig = go.Figure()
        for column in multiples:
            fig.add_trace(go.Scatter(x=yearly["Date"], y=yearly[column], name=column))
        save(fig, "03_valuation_metrics")
    return outputs
