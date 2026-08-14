from __future__ import annotations

"""Exploratory (not committed): does a Rates+Inflation-only warning -- excluding
Labor, CreditConditions, Volatility, which can move fast on sudden shocks --
give a stricter, lower-false-positive early warning for slow-building crises?

Same evaluation methodology as scripts/macro_momentum_sp500/run_persistence_breadth_yearly_validation.py:
weekly sampling, forward-63-session -10% drawdown outcome, point-in-time features.
"""

import json

import numpy as np
import pandas as pd

from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

START_YEAR = 2007
FORWARD_SESSIONS = 63
DRAWDOWN_THRESHOLD = -0.10

# Bounded grid, declared before inspection: only breadth>=2 (both rates AND
# inflation stressed, not just one) makes sense here since there are only 2
# domains -- breadth>=1 would just be "either", too loose to call strict.
RULES = {
    "RI_B2_10_OF_20": {"minimum_breadth": 2, "required_days": 10, "window": 20},
    "RI_B2_15_OF_20": {"minimum_breadth": 2, "required_days": 15, "window": 20},
    "RI_B2_18_OF_20": {"minimum_breadth": 2, "required_days": 18, "window": 20},
    "RI_B2_20_OF_20": {"minimum_breadth": 2, "required_days": 20, "window": 20},
}


def _warning_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": features["Date"], "Close": features["Close"]})
    frame["Rates"] = (
        features["Treasury2Y_Change21_Z252"].ge(1.0)
        | features["RealYield5Y_Change21_Z252"].ge(1.0)
    )
    frame["Inflation"] = (
        features["CoreCPI_Momentum3M_Z252"].ge(1.0)
        | features["CorePCE_Momentum3M_Z252"].ge(1.0)
    )
    frame["WarningBreadth"] = frame[["Rates", "Inflation"]].sum(axis=1)
    for name, rule in RULES.items():
        stressed = frame["WarningBreadth"].ge(rule["minimum_breadth"]).astype(int)
        frame[name] = stressed.rolling(
            rule["window"], min_periods=rule["window"]
        ).sum().ge(rule["required_days"])
    close = pd.to_numeric(frame["Close"], errors="coerce")
    future_minimum = pd.concat(
        [close.shift(-offset) for offset in range(1, FORWARD_SESSIONS + 1)], axis=1
    ).min(axis=1)
    frame["ForwardDrawdown"] = future_minimum / close - 1
    frame["LargeDrawdownAhead"] = frame["ForwardDrawdown"].le(DRAWDOWN_THRESHOLD)
    return frame


def _weekly(frame: pd.DataFrame) -> pd.DataFrame:
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
    lift = precision / base_rate if precision is not None and base_rate else None
    return {
        "observations": int(len(frame)), "signals": int(signal.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_percent": None if precision is None else round(precision * 100, 1),
        "recall_percent": None if recall is None else round(recall * 100, 1),
        "false_positive_rate_percent": None if (fp + tn) == 0 else round(fp / (fp + tn) * 100, 1),
        "precision_lift_vs_base": None if lift is None else round(lift, 2),
    }


def _episodes(daily: pd.DataFrame, rule: str) -> list[dict[str, object]]:
    signal = daily[rule].fillna(False).astype(bool)
    onset = signal & ~signal.shift(1, fill_value=False)
    rows = []
    for index in daily.index[onset]:
        row = daily.loc[index]
        rows.append({
            "signal_date": str(row["Date"].date()),
            "forward_63d_drawdown_percent": None if pd.isna(row["ForwardDrawdown"]) else round(float(row["ForwardDrawdown"] * 100), 1),
            "large_drawdown_ahead": bool(row["LargeDrawdownAhead"]),
        })
    return rows


def main() -> None:
    paths = load_paths()
    config = load_research_config("config/macro_momentum_sp500/research.json")
    features = build_features(load_research_data(paths.macro, config), config)
    daily = _warning_frame(features)
    weekly = _weekly(daily)

    print("=== Aggregate (2007-present, weekly obs) ===")
    for rule in RULES:
        print(rule, json.dumps(_metrics(weekly, rule)))

    print()
    print("=== Yearly recall for slow-building crisis years ===")
    slow_years = [2007, 2008, 2011, 2015, 2018, 2022]
    for year in slow_years:
        year_frame = weekly.loc[weekly["Date"].dt.year.eq(year)]
        if year_frame.empty:
            continue
        base = year_frame["LargeDrawdownAhead"].mean() * 100
        line = f"{year} (base={base:.1f}%): "
        for rule in RULES:
            m = _metrics(year_frame, rule)
            line += f"{rule}=R{m['recall_percent']}/P{m['precision_percent']} "
        print(line)

    print()
    print("=== False positives by non-crisis year (should be near 0 for a strict rule) ===")
    quiet_years = [2009, 2010, 2012, 2013, 2014, 2016, 2017, 2019, 2023, 2024]
    for year in quiet_years:
        year_frame = weekly.loc[weekly["Date"].dt.year.eq(year)]
        if year_frame.empty:
            continue
        line = f"{year}: "
        for rule in RULES:
            signal = year_frame[rule].fillna(False)
            line += f"{rule}_signals={int(signal.sum())} "
        print(line)

    print()
    print("=== Episodes for strictest rule (RI_B2_20_OF_20) ===")
    print(json.dumps(_episodes(daily, "RI_B2_20_OF_20"), indent=2))
    print()
    print("=== Episodes for RI_B2_15_OF_20 ===")
    print(json.dumps(_episodes(daily, "RI_B2_15_OF_20"), indent=2))


if __name__ == "__main__":
    main()
