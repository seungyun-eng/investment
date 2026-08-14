from __future__ import annotations

"""Point-in-time macro early-warning diagnostics, not a trading strategy.

The thresholds are declared before examining the named drawdowns.  This asks
only whether macro deterioration was observable *before* major SPY declines;
it does not alter Alpha Desk, tune a portfolio, or claim crash prediction.
"""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths


OUTPUT_PATH = Path("artifacts/macro_early_warning_diagnostics/results.json")
EVENTS = {
    "covid": {"peak": "2020-02-19", "trough": "2020-03-23"},
    "rate_credit_bear_2022": {"peak": "2021-12-31", "trough": "2022-10-12"},
    "spring_drawdown_2025": {"peak": "2024-12-06", "trough": "2025-04-08"},
    "q1_drawdown_2026": {"peak": "2026-01-02", "trough": "2026-03-30"},
}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _value(row: pd.Series, name: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
    return None if not np.isfinite(value) else float(value)


def _warning_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Predeclared broad risk flags; each measures a distinct transmission path."""

    result = pd.DataFrame({"Date": frame["Date"]})
    result["Labor"] = (
        frame["InitialJoblessClaims_Change21_Z252"].ge(1.0)
        | frame["ContinuingJoblessClaims_Change21_Z252"].ge(1.0)
    )
    result["Rates"] = (
        frame["Treasury2Y_Change21_Z252"].ge(1.0)
        | frame["RealYield5Y_Change21_Z252"].ge(1.0)
    )
    result["Inflation"] = (
        frame["CoreCPI_Momentum3M_Z252"].ge(1.0)
        | frame["CorePCE_Momentum3M_Z252"].ge(1.0)
    )
    result["CreditConditions"] = (
        frame["HYYield_Change21_Z252"].ge(1.0)
        | frame["NFCI_Change21_Z252"].ge(1.0)
    )
    result["Volatility"] = frame["VIX_Change21_Z252"].ge(1.0)
    categories = ["Labor", "Rates", "Inflation", "CreditConditions", "Volatility"]
    result["WarningBreadth"] = result[categories].sum(axis=1)
    result["EarlyWarning"] = result["WarningBreadth"].ge(2)
    return result


def _asof_row(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    available = frame.loc[frame["Date"].le(cutoff)]
    if available.empty:
        raise ValueError(f"No data at or before {cutoff:%Y-%m-%d}")
    return available.iloc[-1]


def main() -> None:
    paths = load_paths()
    config = load_research_config("config/macro_momentum_sp500/research.json")
    data = load_research_data(paths.macro, config)
    features = build_features(data, config)
    warnings = _warning_components(features)
    frame = features.merge(warnings, on="Date", how="left")

    event_rows: dict[str, object] = {}
    fields = (
        "VIX", "InitialJoblessClaims_Change21_Z252",
        "ContinuingJoblessClaims_Change21_Z252", "Treasury2Y_Change21_Z252",
        "RealYield5Y_Change21_Z252", "CoreCPI_Momentum3M_Z252",
        "CorePCE_Momentum3M_Z252", "HYYield_Change21_Z252",
        "NFCI_Change21_Z252", "VIX_Change21_Z252",
    )
    for name, event in EVENTS.items():
        peak = pd.Timestamp(event["peak"])
        trough = pd.Timestamp(event["trough"])
        peak_row = _asof_row(frame, peak)
        trough_row = _asof_row(frame, trough)
        peak_price = _value(peak_row, "Close")
        trough_price = _value(trough_row, "Close")
        observations: dict[str, object] = {}
        for business_days in (60, 20, 5, 0):
            cutoff = peak - pd.offsets.BDay(business_days)
            row = _asof_row(frame, cutoff)
            observations[f"{business_days}bd_before_peak"] = {
                "asof": str(row["Date"].date()),
                "warning_breadth": int(row["WarningBreadth"]),
                "early_warning": bool(row["EarlyWarning"]),
                "components": {
                    component: bool(row[component])
                    for component in ("Labor", "Rates", "Inflation", "CreditConditions", "Volatility")
                },
                "values": {field: _value(row, field) for field in fields},
            }
        event_rows[name] = {
            "peak": event["peak"],
            "trough": event["trough"],
            "spy_drawdown_percent": (
                None if peak_price is None or trough_price is None
                else (trough_price / peak_price - 1) * 100
            ),
            "observations": observations,
        }

    close = pd.to_numeric(frame["Close"], errors="coerce")
    future_minimum = pd.concat(
        [close.shift(-offset) for offset in range(1, 64)], axis=1
    ).min(axis=1)
    frame["Forward63Drawdown"] = future_minimum / close - 1
    usable = frame.dropna(subset=["Forward63Drawdown", "EarlyWarning"])
    diagnostics = {}
    for label, mask in {
        "early_warning": usable["EarlyWarning"],
        "no_early_warning": ~usable["EarlyWarning"],
    }.items():
        subset = usable.loc[mask]
        diagnostics[label] = {
            "days": int(len(subset)),
            "mean_forward_63d_drawdown_percent": float(subset["Forward63Drawdown"].mean() * 100),
            "rate_forward_63d_drawdown_below_minus_10pct": float(
                subset["Forward63Drawdown"].le(-0.10).mean() * 100
            ),
        }
    payload = {
        "research_only": True,
        "generated_at": date.today().isoformat(),
        "data_range": {
            "start": str(frame["Date"].min().date()),
            "end": str(frame["Date"].max().date()),
        },
        "point_in_time_rules": {
            "jobless_claims": "7-calendar-day availability lag",
            "core_cpi_core_pce": "45-calendar-day availability lag; 3M uses 63 trading sessions",
            "treasury_2y_and_real_yield_5y": "close-of-day availability; next-session eligibility",
        },
        "important_limit": (
            "This is an early-warning diagnostic, not a fitted forecasting model. "
            "The listed events are illustrative and the sample of crises is small."
        ),
        "predeclared_warning_rule": "At least two of Labor, Rates, Inflation, CreditConditions, Volatility have a 21-day (or Core inflation 3M) trailing z-score at or above +1.",
        "events": event_rows,
        "forward_63_trading_day_diagnostic": diagnostics,
    }
    _atomic_json(OUTPUT_PATH, payload)
    print(json.dumps({"output": str(OUTPUT_PATH), "events": len(event_rows)}, indent=2))


if __name__ == "__main__":
    main()
