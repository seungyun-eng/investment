from __future__ import annotations

"""Event study for point-in-time macro exit timing.

This intentionally does not construct or optimize a portfolio.  For each
predeclared SPY drawdown it asks when a simple, already-declared macro warning
would have been actionable at the next session's open, and how much additional
peak-to-trough loss followed.  No re-entry rule is assumed.
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


OUTPUT_PATH = Path("artifacts/macro_exit_timing_study/results.json")
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


def _warning_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Same predeclared broad warning rule used in the early-warning study."""

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
    # One-day spikes are not treated as an exit signal.  A warning must be
    # present on 3 of the previous 5 observed market days.
    result["PersistentEarlyWarning"] = (
        result["EarlyWarning"].rolling(5, min_periods=5).sum().ge(3)
    )
    return result


def _asof_index(frame: pd.DataFrame, target: str) -> int:
    indexes = frame.index[frame["Date"].le(pd.Timestamp(target))]
    if len(indexes) == 0:
        raise ValueError(f"No price observation at or before {target}")
    return int(indexes[-1])


def _event_timing(frame: pd.DataFrame, name: str, event: dict[str, str]) -> dict[str, object]:
    peak_index = _asof_index(frame, event["peak"])
    trough_index = _asof_index(frame, event["trough"])
    peak_close = float(frame.loc[peak_index, "Close"])
    trough_close = float(frame.loc[trough_index, "Close"])
    start = max(0, peak_index - 63)
    observations = frame.loc[start:trough_index].copy()
    persistent = observations.index[observations["PersistentEarlyWarning"].fillna(False)]
    alert_index = int(persistent[0]) if len(persistent) else None
    result: dict[str, object] = {
        "event": name,
        "peak_date": str(frame.loc[peak_index, "Date"].date()),
        "trough_date": str(frame.loc[trough_index, "Date"].date()),
        "peak_to_trough_drawdown_percent": (trough_close / peak_close - 1) * 100,
        "first_persistent_warning_in_prior_63_business_days": None,
    }
    if alert_index is None:
        return result
    execution_index = min(alert_index + 1, len(frame) - 1)
    execution_price = float(frame.loc[execution_index, "Open"])
    result["first_persistent_warning_in_prior_63_business_days"] = {
        "signal_date": str(frame.loc[alert_index, "Date"].date()),
        "next_open_execution_date": str(frame.loc[execution_index, "Date"].date()),
        "next_open_execution_price": execution_price,
        "warning_breadth": int(frame.loc[alert_index, "WarningBreadth"]),
        "components": [
            item for item in ("Labor", "Rates", "Inflation", "CreditConditions", "Volatility")
            if bool(frame.loc[alert_index, item])
        ],
        "signal_relative_to_peak_business_days": int(alert_index - peak_index),
        "drawdown_at_next_open_percent": (execution_price / peak_close - 1) * 100,
        "additional_decline_after_next_open_percent": (trough_close / execution_price - 1) * 100,
        "note": (
            "Illustrative exit-only calculation. It assumes cash after the next "
            "open and has no re-entry rule, so it is not a portfolio return."
        ),
    }
    return result


def main() -> None:
    paths = load_paths()
    config = load_research_config("config/macro_momentum_sp500/research.json")
    features = build_features(load_research_data(paths.macro, config), config)
    warning = _warning_components(features)
    frame = features.merge(warning, on="Date", how="left").reset_index(drop=True)
    events = {name: _event_timing(frame, name, event) for name, event in EVENTS.items()}
    close = pd.to_numeric(frame["Close"], errors="coerce")
    forward_minimum = pd.concat(
        [close.shift(-offset) for offset in range(1, 64)], axis=1
    ).min(axis=1)
    frame["Forward63Drawdown"] = forward_minimum / close - 1
    usable = frame.dropna(subset=["Forward63Drawdown", "PersistentEarlyWarning"])
    warning_mask = usable["PersistentEarlyWarning"].astype(bool)
    onset_mask = warning_mask & ~warning_mask.shift(1, fill_value=False)
    alert_quality = {}
    for label, mask in {
        "persistent_warning_days": warning_mask,
        "no_persistent_warning_days": ~warning_mask,
        "persistent_warning_onsets": onset_mask,
    }.items():
        subset = usable.loc[mask]
        alert_quality[label] = {
            "observations": int(len(subset)),
            "mean_forward_63_business_day_drawdown_percent": float(
                subset["Forward63Drawdown"].mean() * 100
            ),
            "rate_forward_63_business_day_drawdown_below_minus_10pct": float(
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
            "claims": "Seven-calendar-day availability lag",
            "core_inflation": "45-calendar-day availability lag",
            "market_data": "Signal at close; theoretical execution at next session open",
        },
        "warning_rule": (
            "At least two of Labor, Rates, Inflation, CreditConditions, Volatility "
            "are stressed, sustained for at least three of five market days."
        ),
        "events": events,
        "warning_quality_across_all_days": alert_quality,
        "limitation": (
            "This study measures alert timing only. It neither selects thresholds on "
            "these events nor includes a re-entry rule, taxes, or opportunity cost."
        ),
    }
    _atomic_json(OUTPUT_PATH, payload)
    print(json.dumps({"output": str(OUTPUT_PATH), "events": len(events)}, indent=2))


if __name__ == "__main__":
    main()
