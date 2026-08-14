from __future__ import annotations

"""Year-by-year audit of fixed macro persistence/breadth warnings.

This is a diagnostic, not a trading strategy.  The rules were declared before
this script was run, but the motivating crises were already inspected, so the
result is labelled historical validation rather than fresh out-of-sample proof.
All macro inputs use the point-in-time availability rules in data.py.
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


OUTPUT_PATH = Path("artifacts/macro_persistence_breadth_validation/results.json")
START_YEAR = 2007
FORWARD_SESSIONS = 63
DRAWDOWN_THRESHOLD = -0.10
CATEGORIES = ("Labor", "Rates", "Inflation", "CreditConditions", "Volatility")
RULES = {
    "B2_15_OF_20": {"minimum_breadth": 2, "required_days": 15, "window": 20},
    "B3_15_OF_20": {"minimum_breadth": 3, "required_days": 15, "window": 20},
    "B4_10_OF_20": {"minimum_breadth": 4, "required_days": 10, "window": 20},
    "B4_15_OF_20": {"minimum_breadth": 4, "required_days": 15, "window": 20},
}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem, suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _warning_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": features["Date"], "Close": features["Close"]})
    frame["Labor"] = (
        features["InitialJoblessClaims_Change21_Z252"].ge(1.0)
        | features["ContinuingJoblessClaims_Change21_Z252"].ge(1.0)
    )
    frame["Rates"] = (
        features["Treasury2Y_Change21_Z252"].ge(1.0)
        | features["RealYield5Y_Change21_Z252"].ge(1.0)
    )
    frame["Inflation"] = (
        features["CoreCPI_Momentum3M_Z252"].ge(1.0)
        | features["CorePCE_Momentum3M_Z252"].ge(1.0)
    )
    frame["CreditConditions"] = (
        features["HYYield_Change21_Z252"].ge(1.0)
        | features["NFCI_Change21_Z252"].ge(1.0)
    )
    frame["Volatility"] = features["VIX_Change21_Z252"].ge(1.0)
    frame["WarningBreadth"] = frame[list(CATEGORIES)].sum(axis=1)
    for name, rule in RULES.items():
        stressed = frame["WarningBreadth"].ge(rule["minimum_breadth"]).astype(int)
        frame[name] = stressed.rolling(
            rule["window"], min_periods=rule["window"]
        ).sum().ge(rule["required_days"])
    close = pd.to_numeric(frame["Close"], errors="coerce")
    future_minimum = pd.concat(
        [close.shift(-offset) for offset in range(1, FORWARD_SESSIONS + 1)],
        axis=1,
    ).min(axis=1)
    frame["ForwardDrawdown"] = future_minimum / close - 1
    frame["LargeDrawdownAhead"] = frame["ForwardDrawdown"].le(DRAWDOWN_THRESHOLD)
    return frame


def _weekly_observations(frame: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        frame.assign(Week=frame["Date"].dt.to_period("W-FRI"))
        .groupby("Week", sort=True, as_index=False)
        .tail(1)
        .drop(columns="Week")
    )
    return weekly.loc[
        weekly["Date"].dt.year.ge(START_YEAR) & weekly["ForwardDrawdown"].notna()
    ].reset_index(drop=True)


def _metrics(frame: pd.DataFrame, rule: str) -> dict[str, float | int | None]:
    signal = frame[rule].fillna(False).astype(bool)
    actual = frame["LargeDrawdownAhead"].astype(bool)
    tp = int((signal & actual).sum())
    fp = int((signal & ~actual).sum())
    fn = int((~signal & actual).sum())
    tn = int((~signal & ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    base_rate = float(actual.mean()) if len(frame) else None
    signal_rate = float(signal.mean()) if len(frame) else None
    no_signal_bad_rate = fn / (fn + tn) if fn + tn else None
    lift = precision / base_rate if precision is not None and base_rate else None
    return {
        "observations": int(len(frame)),
        "signals": int(signal.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "base_rate_percent": None if base_rate is None else base_rate * 100,
        "signal_rate_percent": None if signal_rate is None else signal_rate * 100,
        "precision_percent": None if precision is None else precision * 100,
        "recall_percent": None if recall is None else recall * 100,
        "no_signal_bad_rate_percent": (
            None if no_signal_bad_rate is None else no_signal_bad_rate * 100
        ),
        "precision_lift_vs_base": lift,
        "mean_forward_drawdown_when_signaled_percent": (
            None if not signal.any() else float(frame.loc[signal, "ForwardDrawdown"].mean() * 100)
        ),
    }


def _episode_rows(daily: pd.DataFrame, rule: str) -> list[dict[str, object]]:
    signal = daily[rule].fillna(False).astype(bool)
    onset = signal & ~signal.shift(1, fill_value=False)
    rows: list[dict[str, object]] = []
    for index in daily.index[onset]:
        row = daily.loc[index]
        rows.append(
            {
                "signal_date": str(row["Date"].date()),
                "year": int(row["Date"].year),
                "warning_breadth": int(row["WarningBreadth"]),
                "forward_63_session_drawdown_percent": (
                    None if pd.isna(row["ForwardDrawdown"])
                    else float(row["ForwardDrawdown"] * 100)
                ),
                "large_drawdown_ahead": bool(row["LargeDrawdownAhead"]),
            }
        )
    return rows


def main() -> None:
    paths = load_paths()
    config = load_research_config("config/macro_momentum_sp500/research.json")
    features = build_features(load_research_data(paths.macro, config), config)
    daily = _warning_frame(features)
    weekly = _weekly_observations(daily)
    yearly: dict[str, dict[str, object]] = {}
    for year, year_frame in weekly.groupby(weekly["Date"].dt.year, sort=True):
        yearly[str(int(year))] = {
            "base_rate_percent": float(year_frame["LargeDrawdownAhead"].mean() * 100),
            "rules": {rule: _metrics(year_frame, rule) for rule in RULES},
        }
    aggregate = {rule: _metrics(weekly, rule) for rule in RULES}
    segment_masks = {
        "2007_2019": weekly["Date"].dt.year.le(2019),
        "2020_2026": weekly["Date"].dt.year.ge(2020),
        "excluding_2008_and_2022": ~weekly["Date"].dt.year.isin([2008, 2022]),
        "only_2008_and_2022": weekly["Date"].dt.year.isin([2008, 2022]),
    }
    segment_robustness = {
        segment: {
            "observations": int(mask.sum()),
            "base_rate_percent": float(weekly.loc[mask, "LargeDrawdownAhead"].mean() * 100),
            "rules": {rule: _metrics(weekly.loc[mask], rule) for rule in RULES},
        }
        for segment, mask in segment_masks.items()
    }
    consistency = {}
    for rule in RULES:
        eligible_years = []
        positive_lift_years = []
        for year, row in yearly.items():
            metrics = row["rules"][rule]
            if metrics["signals"] == 0 or metrics["base_rate_percent"] == 0:
                continue
            eligible_years.append(year)
            if metrics["precision_lift_vs_base"] is not None and metrics["precision_lift_vs_base"] > 1:
                positive_lift_years.append(year)
        consistency[rule] = {
            "years_with_signals_and_drawdowns": len(eligible_years),
            "years_with_precision_above_base_rate": len(positive_lift_years),
            "eligible_years": eligible_years,
            "positive_lift_years": positive_lift_years,
        }
    payload = {
        "research_only": True,
        "validation_status": "HISTORICAL_YEAR_BY_YEAR_AUDIT_NOT_FRESH_OOS",
        "generated_at": date.today().isoformat(),
        "evaluation": {
            "start_year": START_YEAR,
            "end_date": str(weekly["Date"].max().date()),
            "sampling": "Last available market session of each W-FRI week",
            "outcome": "Minimum SPY close over the next 63 sessions is at least 10% below signal-date close",
            "important_caveat": (
                "The crisis examples motivated these rules before this audit. "
                "Yearly stability can reject a fragile rule but cannot create a fresh holdout."
            ),
        },
        "point_in_time": {
            "claims_lag_days": config.weekly_release_lag_days,
            "core_inflation_lag_days": config.monthly_release_lag_days,
            "features": "Trailing-only z-scores; no future macro values",
        },
        "rules": RULES,
        "aggregate": aggregate,
        "segment_robustness": segment_robustness,
        "yearly": yearly,
        "consistency": consistency,
        "episodes": {rule: _episode_rows(daily, rule) for rule in RULES},
    }
    _atomic_json(OUTPUT_PATH, payload)
    print(json.dumps({"output": str(OUTPUT_PATH), "weekly_rows": len(weekly)}, indent=2))


if __name__ == "__main__":
    main()
