"""Why the vol target wins on SPY but loses on the book -- and the fix.

Hypothesis: a top-5 book's trailing volatility is dominated by single-name
noise, not by market regime.  When one holding gets jumpy the book's vol rises
and the rule cuts exposure, but nothing about the market has changed -- so it
sells winners for no reason.  SPY's volatility is a genuine regime signal.

Test: drive the same vol target off SPY's volatility instead of the book's,
and apply it to the book.  If the hypothesis holds, the SPY-driven version
should beat the book-driven one at matched exposure.

Also reports turnover honestly (dollars traded / average equity), since the
engine's own AnnualizedTurnover divides by *initial* capital and is unreadable
over an 18-year run.

Research only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from stock_research.cross_sectional.portfolio import (prepare_market, run_portfolio_backtest)
from stock_research.cross_sectional.signals import monthly_weight_reset_targets
from stock_research.dashboard import engine

from common import COST_BPS, OUT, atomic_csv
from engine_bits import V2_BASELINE, backtest_nav, spy_close, stats, v2_weights
from run_long import (END, START, WINDOWS, build_long_panel, daily_returns, dip_targets,
                      v7_price_targets)
from run_long_defense import apply_exposure, vol_target_exposure


def spy_driven_exposure(index: pd.DatetimeIndex, target: float, deadband: float = 0.10) -> pd.Series:
    close = spy_close()
    realised = close.pct_change().rolling(60).std(ddof=1) * np.sqrt(252)
    raw = (target / realised).clip(upper=1.0).shift(1).reindex(index).ffill()
    out, current = np.ones(len(index)), 1.0
    for i, proposed in enumerate(raw.to_numpy()):
        if not np.isnan(proposed) and abs(proposed - current) >= deadband:
            current = float(proposed)
        out[i] = current
    return pd.Series(out, index=index)


def honest_turnover(targets: pd.DataFrame, panel: pd.DataFrame, market) -> float:
    result = run_portfolio_backtest({}, targets, start=START, end=END,
                                    initial_capital=engine.INITIAL_CAPITAL,
                                    transaction_cost_bps=COST_BPS, prepared_market=market)
    daily = result.daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    average_equity = float(daily["Equity"].mean())
    dollars = result.summary.turnover_multiple * engine.INITIAL_CAPITAL
    years = (daily["Date"].iloc[-1] - daily["Date"].iloc[0]).days / 365.25
    return dollars / average_equity / years


def main():
    settings = engine._settings(START, "2019-12-31", "2020-01-02", END, 8)
    panel = build_long_panel()
    scored, targets = v7_price_targets(panel, settings)
    market = prepare_market(panel, start=START, end=END)
    nav, _ = backtest_nav(monthly_weight_reset_targets(targets), panel, START, END,
                          engine.INITIAL_CAPITAL, market)

    close = spy_close()
    v2 = pd.Series(v2_weights(close.to_numpy(), V2_BASELINE), index=close.index)
    v2 = v2.reindex(nav.index).ffill().fillna(1.0)

    rules = {"V2 (MA150 on SPY)": v2}
    for target in (0.12, 0.15, 0.18, 0.22):
        rules["voltgt%d driven by SPY vol" % int(target * 100)] = spy_driven_exposure(nav.index, target)
    for target in (0.25, 0.30):
        rules["voltgt%d driven by BOOK vol" % int(target * 100)] = vol_target_exposure(nav, target)
    rules["min(V2, SPY-vol voltgt15)"] = pd.concat(
        [v2, spy_driven_exposure(nav.index, 0.15)], axis=1).min(axis=1)

    rows = [{"rule": "no defense", "MeanExposure": 1.0,
             **{k: round(v, 3) for k, v in stats(nav).items()}}]
    for label, (a, b) in WINDOWS.items():
        piece = nav.loc[a:b]
        if len(piece) > 30:
            rows[0][label] = round(stats(piece)["ROI"], 1)

    for name, weights in rules.items():
        defended = apply_exposure(nav, weights)
        flat = apply_exposure(nav, pd.Series(float(weights.mean()), index=nav.index))
        s, f = stats(defended), stats(flat)
        row = {"rule": name, "MeanExposure": round(float(weights.mean()), 3),
               **{k: round(v, 3) for k, v in s.items()},
               "Sharpe_gain_vs_flat": round(s["Sharpe"] - f["Sharpe"], 3),
               "Calmar_gain_vs_flat": round(s["Calmar"] - f["Calmar"], 3)}
        for label, (a, b) in WINDOWS.items():
            piece = defended.loc[a:b]
            if len(piece) > 30:
                row[label] = round(stats(piece)["ROI"], 1)
        rows.append(row)

    table = pd.DataFrame(rows)
    atomic_csv(OUT / "long_addendum_results.csv", table)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(table.to_string(index=False))

    rets = daily_returns(panel)
    print()
    print("honest turnover (dollars traded / average equity / year)")
    print("  V7-price top5 : %.2fx" % honest_turnover(monthly_weight_reset_targets(targets), panel, market))
    print("  B dip sleeve  : %.2fx" % honest_turnover(dip_targets(scored, rets), panel, market))


if __name__ == "__main__":
    main()
