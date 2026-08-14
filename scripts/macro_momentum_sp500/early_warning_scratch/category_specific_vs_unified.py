from __future__ import annotations

"""Exploratory (not committed): does giving each crisis-type its own
single-category strict persistence rule, OR'd together, produce more false
noise than the unified breadth>=2-of-5-domains rule?

Same evaluation methodology as the earlier scratch scripts: weekly sampling
from 2007, forward-63-session -10% drawdown outcome, point-in-time features.
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
PERSIST_WINDOW = 20
PERSIST_DAYS = 15  # strict: 15 of 20, same bar as the earlier B*_15_OF_20 rules


def _category_flags(features: pd.DataFrame) -> pd.DataFrame:
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
    return frame


CATEGORIES = ("Labor", "Rates", "Inflation", "CreditConditions", "Volatility")


def _add_rules(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # (a) each category alone, strict persistence -- the "specialist per crisis
    # type" approach.
    for category in CATEGORIES:
        frame[f"Persist_{category}"] = (
            frame[category].astype(int)
            .rolling(PERSIST_WINDOW, min_periods=PERSIST_WINDOW).sum()
            .ge(PERSIST_DAYS)
        )
    frame["OR_AnySpecialist"] = frame[[f"Persist_{c}" for c in CATEGORIES]].any(axis=1)
    # (b) unified: >=2 of 5 categories stressed on the SAME day, that
    # combined breadth sustained (the rule already validated earlier).
    frame["WarningBreadth"] = frame[list(CATEGORIES)].sum(axis=1)
    frame["Unified_B2"] = (
        frame["WarningBreadth"].ge(2).astype(int)
        .rolling(PERSIST_WINDOW, min_periods=PERSIST_WINDOW).sum()
        .ge(PERSIST_DAYS)
    )
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
    return {
        "signals": int(signal.sum()), "tp": tp, "fp": fp,
        "precision_percent": None if precision is None else round(precision * 100, 1),
        "recall_percent": None if recall is None else round(recall * 100, 1),
        "false_positive_rate_percent": None if (fp + tn) == 0 else round(fp / (fp + tn) * 100, 1),
    }


def main() -> None:
    paths = load_paths()
    config = load_research_config("config/macro_momentum_sp500/research.json")
    features = build_features(load_research_data(paths.macro, config), config)
    daily = _add_rules(_category_flags(features))
    weekly = _weekly(daily)

    rule_names = [f"Persist_{c}" for c in CATEGORIES] + ["OR_AnySpecialist", "Unified_B2"]

    print("=== Aggregate 2007-present ===")
    for rule in rule_names:
        print(f"{rule:25s} {json.dumps(_metrics(weekly, rule))}")

    print()
    print("=== Yearly recall (slow-building crisis years) ===")
    slow_years = [2007, 2008, 2011, 2015, 2018, 2022]
    for year in slow_years:
        yf = weekly.loc[weekly["Date"].dt.year.eq(year)]
        if yf.empty:
            continue
        base = yf["LargeDrawdownAhead"].mean() * 100
        line = f"{year} (base={base:.1f}%): "
        for rule in ["OR_AnySpecialist", "Unified_B2"]:
            m = _metrics(yf, rule)
            line += f"{rule}=R{m['recall_percent']}/P{m['precision_percent']} "
        print(line)
        # which specialist(s) fired that year
        fired = [c for c in CATEGORIES if daily.loc[daily["Date"].dt.year.eq(year), f"Persist_{c}"].any()]
        print(f"    specialists that fired in {year}: {fired}")

    print()
    print("=== False positives by quiet/non-crisis year ===")
    quiet_years = [2009, 2010, 2012, 2013, 2014, 2016, 2017, 2019, 2023, 2024]
    for year in quiet_years:
        yf = weekly.loc[weekly["Date"].dt.year.eq(year)]
        if yf.empty:
            continue
        line = f"{year}: "
        for rule in ["OR_AnySpecialist", "Unified_B2"]:
            signal = yf[rule].fillna(False)
            line += f"{rule}_signals={int(signal.sum())} "
        print(line)

    print()
    print("=== Per-category false-positive-only years (which specialist is noisiest) ===")
    for category in CATEGORIES:
        rule = f"Persist_{category}"
        noisy_years = {}
        for year in quiet_years:
            yf = weekly.loc[weekly["Date"].dt.year.eq(year)]
            if yf.empty:
                continue
            count = int(yf[rule].fillna(False).sum())
            if count:
                noisy_years[year] = count
        print(f"{category}: false-positive weeks in quiet years = {noisy_years}")


if __name__ == "__main__":
    main()
