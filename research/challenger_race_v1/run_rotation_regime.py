"""The 2026 failure is a rotation-speed failure. Can the speed be timed?

Established so far:
  - 2026's picks made +11.1% held passively; the model's own trading made -3.0%.
  - Widening the exit-rank buffer (rotate LESS) is the only change that turns
    2026 positive (+5.8%), and it is exactly the change that hurts 2024 (76.7 ->
    56.8).  Fast rotation paid in 2024 and punished in 2026.

So the question is not "which fixed speed is right" -- it is "can you tell, at
the time, which regime you are in".

Detector tested here: trailing realised rank IC of AlphaScore against 1-week
forward returns.  At signal date t only weeks that have already resolved are
used (t-1 and earlier), so it is point-in-time.  When trailing IC is negative
the cross-section is not rewarding the ranking, so the rule freezes rotation --
the portfolio simply does not trade that week and drifts.

Shuffle control: the same NUMBER of frozen weeks placed at random, 500 draws.
If the real rule does not land in the tail of that distribution, then "trade
less" is doing the work and the detector is worthless.

Research only.  Nothing committed, nothing deployed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.filing_v7_optimization import generate_filing_v7_targets
from stock_research.cross_sectional.portfolio import prepare_market
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import OUT, atomic_csv, atomic_json, frozen_signal_days, load_panel
from engine_bits import backtest_nav, stats

TRAILING_WEEKS = 13
SEED = 20260820


def weekly_ic(scored: pd.DataFrame, factored: pd.DataFrame) -> pd.DataFrame:
    """Rank IC of AlphaScore vs the return realised over the FOLLOWING week."""
    wide = factored.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    frame = scored.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    dates = sorted(frame["Date"].unique())
    rows = []
    for current, following in zip(dates, dates[1:]):
        group = frame.loc[frame["Date"].eq(current), ["Ticker", "AlphaScore"]].dropna()
        if len(group) < 8:
            continue
        try:
            start_px = wide.loc[current]
            end_px = wide.loc[following]
        except KeyError:
            continue
        rets = (end_px / start_px - 1.0).rename("Fwd")
        pair = group.set_index("Ticker").join(rets, how="inner").dropna()
        if len(pair) < 8 or pair["AlphaScore"].nunique() < 2:
            continue
        rows.append({"Date": current, "ResolvedOn": following,
                     "IC": float(pair["AlphaScore"].corr(pair["Fwd"], method="spearman")),
                     "RankPersistence": np.nan})
    ic = pd.DataFrame(rows)
    # rank persistence: how much this week's ranking looks like last week's
    prev = None
    persistence = []
    for date in dates:
        group = frame.loc[frame["Date"].eq(date), ["Ticker", "AlphaScore"]].dropna().set_index("Ticker")["AlphaScore"]
        if prev is not None:
            joined = pd.concat([prev.rename("prev"), group.rename("now")], axis=1).dropna()
            persistence.append({"Date": date,
                                "RankPersistence": float(joined["prev"].corr(joined["now"], method="spearman"))
                                if len(joined) >= 8 else np.nan})
        prev = group
    pers = pd.DataFrame(persistence)
    return ic.drop(columns=["RankPersistence"]).merge(pers, on="Date", how="left")


def freeze_flags(ic: pd.DataFrame, dates: list) -> pd.Series:
    """PIT: at date t, average the IC of weeks that had already resolved by t."""
    resolved = ic.set_index("ResolvedOn")["IC"].sort_index()
    flags = {}
    for date in dates:
        history = resolved.loc[resolved.index <= pd.Timestamp(date)]
        window = history.tail(TRAILING_WEEKS)
        flags[date] = bool(len(window) >= TRAILING_WEEKS and window.mean() < 0)
    return pd.Series(flags)


def run_with_frozen(execution: pd.DataFrame, frozen_dates: set, panel, start, end, capital, market):
    kept = execution.loc[~execution["Date"].isin(frozen_dates)]
    return backtest_nav(kept, panel.factored, start, end, capital, market)


def main():
    bundle = load_panel()
    panel, start, end = bundle["panel"], bundle["start"], bundle["end"]
    capital = panel.settings.initial_capital
    market = prepare_market(panel.factored, start=start, end=end)
    signal_days = frozen_signal_days(panel.factored, start, end, panel.settings.rebalance_weekday)
    scored, targets = generate_filing_v7_targets(signal_days, engine._base_params(), engine._policy(5))
    targets["Date"] = pd.to_datetime(targets["Date"])
    execution = monthly_weight_reset_targets(targets)

    ic = weekly_ic(scored, panel.factored)
    ic["Year"] = pd.to_datetime(ic["Date"]).dt.year
    atomic_csv(OUT / "diag_weekly_ic.csv", ic)

    pd.set_option("display.width", 220)
    print("=== detector inputs by year ===")
    print(ic.groupby("Year")[["IC", "RankPersistence"]].agg(["mean", "std", "count"]).round(3).to_string())

    exec_dates = sorted(execution["Date"].unique())
    flags = freeze_flags(ic, exec_dates)
    frozen = {d for d, f in flags.items() if f}
    print()
    print("execution weeks: %d, frozen by the rule: %d" % (len(exec_dates), len(frozen)))
    by_year = pd.Series({str(pd.Timestamp(d).year): 0 for d in exec_dates})
    counts = pd.Series([pd.Timestamp(d).year for d in frozen]).value_counts().sort_index()
    print("frozen weeks by year:", counts.to_dict())

    years = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"),
             "2026_OOS": ("2026-01-02", end)}
    rows = []
    for label, frozen_set in (("baseline (never freeze)", set()), ("IC-gated freeze", frozen)):
        nav, extra = run_with_frozen(execution, frozen_set, panel, start, end, capital, market)
        row = {"variant": label, "frozen_weeks": len(frozen_set),
               **{k: round(v, 2) for k, v in stats(nav).items()},
               "turnover": round(extra["AnnualizedTurnover"], 1)}
        for name, (a, b) in years.items():
            piece, _ = run_with_frozen(execution, frozen_set, panel, a, b, capital,
                                       prepare_market(panel.factored, start=a, end=b))
            row[name] = round(stats(piece)["ROI"], 1)
        rows.append(row)

    # shuffle control
    rng = np.random.default_rng(SEED)
    eligible = [d for d in exec_dates]
    sharpes, rois_2026 = [], []
    for _ in range(500):
        draw = set(rng.choice(eligible, size=len(frozen), replace=False)) if frozen else set()
        nav, _ = run_with_frozen(execution, draw, panel, start, end, capital, market)
        sharpes.append(stats(nav)["Sharpe"])
        piece, _ = run_with_frozen(execution, draw, panel, "2026-01-02", end, capital,
                                   prepare_market(panel.factored, start="2026-01-02", end=end))
        rois_2026.append(stats(piece)["ROI"])

    real = rows[1]
    control = {
        "shuffle_draws": 500,
        "real_full_Sharpe": real["Sharpe"],
        "shuffle_Sharpe_mean": float(np.mean(sharpes)),
        "shuffle_Sharpe_p95": float(np.percentile(sharpes, 95)),
        "real_Sharpe_percentile": float((np.array(sharpes) < real["Sharpe"]).mean() * 100),
        "real_2026_ROI": real["2026_OOS"],
        "shuffle_2026_ROI_mean": float(np.mean(rois_2026)),
        "shuffle_2026_ROI_p95": float(np.percentile(rois_2026, 95)),
        "real_2026_percentile": float((np.array(rois_2026) < real["2026_OOS"]).mean() * 100),
    }
    table = pd.DataFrame(rows)
    atomic_csv(OUT / "diag_rotation_regime.csv", table)
    atomic_json(OUT / "diag_rotation_regime_control.json", control)
    print()
    print(table.to_string(index=False))
    print()
    print("=== shuffle control: same number of frozen weeks, placed at random ===")
    for key, value in control.items():
        print("  %-26s %s" % (key, round(value, 2) if isinstance(value, float) else value))


if __name__ == "__main__":
    main()
