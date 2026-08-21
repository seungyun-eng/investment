"""Backtest helpers shared by the A and B challenger runs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.cross_sectional.portfolio import prepare_market, run_portfolio_backtest

from common import ROOT, COST_BPS

V2_SRC = ROOT / "scripts/macro_momentum_sp500/early_warning_scratch/tune_v2_overlay.py"
SPY_CSV = ROOT.parent / "Macro Data/SPY Adjusted Historical Data.csv"
SPY_TAIL = ROOT.parent / "Dashboard Data/Benchmarks/SPY.csv"
QQQ_CSV = ROOT.parent / "Dashboard Data/Benchmarks/QQQ.csv"

V2_BASELINE = dict(ma_window=150, sell_confirm=3, fallback_confirm=3,
                   rsi_hi=35, rsi_hi_w=0.30, rsi_lo=30, rsi_lo_w=0.80,
                   qr_depth=-0.03, qr_window=3)
V2_TUNED = dict(V2_BASELINE, fallback_confirm=2, qr_depth=-0.05, qr_window=5)


def _load_v2():
    spec = importlib.util.spec_from_file_location("tune_v2_overlay", V2_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.v2_weights


v2_weights = _load_v2()


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def stats(nav: pd.Series) -> dict:
    """CAGR/MDD/Sharpe/Calmar off a daily NAV series. ROI follows repo policy:
    (final / injected - 1) * 100 with a single up-front injection."""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0
    mdd = float((nav / nav.cummax() - 1.0).min())
    sd = float(ret.std(ddof=1))
    sharpe = float(ret.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    return {"ROI": total * 100, "CAGR": cagr * 100, "MDD": mdd * 100,
            "Sharpe": sharpe, "Calmar": (cagr / abs(mdd)) if mdd < 0 else float("nan")}


def backtest_nav(targets: pd.DataFrame, factored: pd.DataFrame, start: str, end: str,
                 capital: float, market=None) -> tuple[pd.Series, dict]:
    market = market or prepare_market(factored, start=start, end=end)
    result = run_portfolio_backtest({}, targets, start=start, end=end,
                                    initial_capital=capital,
                                    transaction_cost_bps=COST_BPS,
                                    prepared_market=market)
    daily = result.daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    nav = daily.set_index("Date")["Equity"]
    extra = {"AnnualizedTurnover": result.summary.annualized_turnover,
             "TickerTrades": result.summary.ticker_trades,
             "Rebalances": result.summary.rebalance_count}
    return nav, extra


# --------------------------------------------------------------------------- #
# benchmark prices
# --------------------------------------------------------------------------- #
def spy_close(history_start: str = "1993-01-01") -> pd.Series:
    """Full adjusted SPY history.  history_start only trims the head; keep it
    early enough that the 150-day MA is warm before any window under test."""
    spy = pd.read_csv(SPY_CSV, parse_dates=["Date"]).sort_values("Date")
    spy = spy[spy["Date"] >= history_start].reset_index(drop=True)
    tail = pd.read_csv(SPY_TAIL, parse_dates=["Date"]).sort_values("Date")
    cutoff = spy["Date"].iloc[-1]
    anchor = tail.loc[tail["Date"] == cutoff, "Close"]
    extra = tail.loc[tail["Date"] > cutoff, ["Date", "Close"]].copy()
    if not anchor.empty and not extra.empty:
        scale = spy["Adj Close"].iloc[-1] / float(anchor.iloc[0])
        extra["Adj Close"] = extra["Close"] * scale
        spy = pd.concat([spy[["Date", "Adj Close"]], extra[["Date", "Adj Close"]]],
                        ignore_index=True).sort_values("Date").reset_index(drop=True)
    return spy.set_index("Date")["Adj Close"]


def qqq_close() -> pd.Series:
    qqq = pd.read_csv(QQQ_CSV, parse_dates=["Date"]).sort_values("Date")
    return qqq.set_index("Date")["Close"]


def buy_and_hold(close: pd.Series, index: pd.DatetimeIndex, capital: float) -> pd.Series:
    series = close.reindex(index).ffill()
    return capital * series / series.iloc[0]


# --------------------------------------------------------------------------- #
# V2 overlay, "de-risk" form (the ACCEPTED variant)
# --------------------------------------------------------------------------- #
def v2_overlay_nav(base_nav: pd.Series, params: dict, charge_cost: bool = True) -> tuple[pd.Series, dict]:
    """Hold the same book at V2's staged weight w, rest in 0%-yield cash.

    charge_cost applies COST_BPS to |dw| on every weight change, which the
    original NAV-level study did not do.  A challenger that runs through the
    real execution engine pays its costs, so the champion should too."""
    close = spy_close()
    weights = pd.Series(v2_weights(close.to_numpy(), params), index=close.index)
    w = weights.reindex(base_nav.index).ffill().fillna(1.0)
    base_ret = base_nav.pct_change().fillna(0.0)
    # v2_weights already lags: it reads day i-1 indicators and sets weight[i], so
    # w applies to day i's return as-is. Shifting again here delayed both the exit
    # and the re-entry by an extra session and understated the overlay -- it made
    # the standalone-SPY cross-check read 195% against the 232% in
    # artifacts/v7_v2_overlay/tuning_results.json, whose spy_buy_hold reference
    # (163.41%) reproduces exactly, isolating the error to this line.
    ret = w * base_ret
    if charge_cost:
        ret = ret - w.diff().abs().fillna(0.0) * (COST_BPS / 10_000)
    nav = base_nav.iloc[0] * (1 + ret).cumprod()
    below = (w < 1.0).to_numpy()
    eps = int(np.sum(below[1:] & ~below[:-1]) + (1 if below[0] else 0))
    return nav, {"DefensiveDays": int(below.sum()), "DefensiveEpisodes": eps,
                 "MeanWeightWhenDefensive": float(w[w < 1.0].mean()) if below.any() else 1.0}
