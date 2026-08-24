from __future__ import annotations

"""Signal-quality diagnostics for the Filing V7-3 cross-sectional model:
Rank IC (does a high score actually predict the return the strategy can
*actually capture*?), per-factor IC (which of Quality/Trend/Momentum/Growth/
Filing is still predictive?), and a Universe Equal-Weight benchmark (did the
*selector* add value beyond just holding the whole universe?).

Every function here is pure and takes an already-scored weekly cross-section
(the `scored` frame from
`stock_research.dashboard.engine.build_scored_panel`) plus the full daily
`factored` panel it was built from -- one row per (Date, Ticker) for every
ticker in the universe, not just the ones selected into the portfolio.

Two point-in-time invariants this module is built around:

1. Holding-period return, not close-to-close. The model scores on a signal
   date's close but only trades at the *next trading session's open*
   (matches `engine.py`'s "next-session-open execution"). Using signal-date
   close to next-signal-date close would silently credit/blame the model for
   the Friday-close-to-Monday-open gap it never actually holds through.
2. FINAL vs PENDING maturation, not recomputation. A signal date's IC can
   only be computed once its *next* signal date's execution price exists.
   The most recent signal date is therefore always PENDING (no verdict yet),
   maturing to FINAL exactly once, the following week -- never recomputed
   after that. Rolling-window averages only ever consume FINAL
   observations, and a window is reported as unavailable (None) rather than
   as a mislabeled partial average until enough FINAL weeks exist.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FACTOR_COLUMNS = ("MomentumFactor", "TrendFactor", "GrowthFactor", "QualityFactor", "FilingFundamentalFactor")
SCORE_COLUMN = "AlphaScore"
DEFAULT_ROLLING_WINDOWS = (13, 26, 52)


def spearman_ic(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float | None:
    """Rank correlation between realized outcome and predicted score. None
    (not NaN) when either side has fewer than 2 distinct values -- there is
    no meaningful correlation to report, and downstream rolling averages
    should skip these rather than propagate a NaN."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if len(np.unique(actual_values[np.isfinite(actual_values)])) < 2:
        return None
    if len(np.unique(predicted_values[np.isfinite(predicted_values)])) < 2:
        return None
    result = spearmanr(actual_values, predicted_values, nan_policy="omit")
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def _next_trading_date_map(factored: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    all_dates = sorted(pd.Timestamp(d) for d in factored["Date"].unique())
    return {all_dates[i]: all_dates[i + 1] for i in range(len(all_dates) - 1)}


def next_holding_period_returns(scored: pd.DataFrame, factored: pd.DataFrame) -> pd.DataFrame:
    """Adds `ForwardReturn` and `Status` (FINAL/PENDING) to `scored`:
    ForwardReturn is Open[execution date of the NEXT signal] /
    Open[execution date of THIS signal] - 1, where an "execution date" is
    the next trading session after a signal date (the session the model can
    actually transact at). Requires `Open` on `factored` (present -- see
    `tsla_integrated.data.load_equity_prices`)."""

    next_trading_date = _next_trading_date_map(factored)
    opens = factored.set_index(["Date", "Ticker"])["Open"]

    signal_dates = sorted(pd.Timestamp(d) for d in scored["Date"].unique())
    next_signal_date = {signal_dates[i]: signal_dates[i + 1] for i in range(len(signal_dates) - 1)}
    latest_signal_date = signal_dates[-1] if signal_dates else None

    def _open_at_execution(signal_date: pd.Timestamp, ticker: str) -> float:
        execution_date = next_trading_date.get(signal_date)
        if execution_date is None:
            return np.nan
        try:
            return float(opens.loc[(execution_date, ticker)])
        except KeyError:
            return np.nan

    def _forward_return(row: pd.Series) -> float:
        signal_date = pd.Timestamp(row["Date"])
        exit_signal_date = next_signal_date.get(signal_date)
        if exit_signal_date is None:
            return np.nan
        entry_open = _open_at_execution(signal_date, row["Ticker"])
        exit_open = _open_at_execution(exit_signal_date, row["Ticker"])
        if not entry_open or not np.isfinite(entry_open) or not np.isfinite(exit_open):
            return np.nan
        return exit_open / entry_open - 1.0

    frame = scored.copy()
    frame["ForwardReturn"] = frame.apply(_forward_return, axis=1)
    frame["Status"] = np.where(frame["Date"].eq(latest_signal_date), "PENDING", "FINAL")
    return frame


def _qualified(frame: pd.DataFrame, *, qualified_only: bool) -> pd.DataFrame:
    if qualified_only and "Qualified" in frame.columns:
        return frame[frame["Qualified"].fillna(False)]
    return frame


def rank_ic_series(frame_with_returns: pd.DataFrame, score_column: str, *, qualified_only: bool = True) -> list[dict]:
    working = _qualified(frame_with_returns, qualified_only=qualified_only)
    series = []
    for date, group in working.groupby("Date"):
        status = group["Status"].iloc[0]
        valid = group.dropna(subset=[score_column, "ForwardReturn"])
        ic = spearman_ic(valid["ForwardReturn"], valid[score_column]) if (status == "FINAL" and len(valid) >= 3) else None
        series.append({"date": str(pd.Timestamp(date).date()), "ic": ic, "n": int(len(valid)), "status": status})
    return series


def rolling_average(rows: list[dict], key: str, windows: tuple[int, ...]) -> dict[str, float | None]:
    """Only FINAL observations count. A window is None until that many
    FINAL weeks actually exist -- never a partial average passed off as the
    full window."""

    final_values = [row[key] for row in rows if row.get("status") == "FINAL" and row.get(key) is not None]
    result: dict[str, float | None] = {}
    for window in windows:
        result[f"{window}W"] = (sum(final_values[-window:]) / window) if len(final_values) >= window else None
    return result


def top_k_spread_series(frame_with_returns: pd.DataFrame, *, k: int, score_column: str = SCORE_COLUMN, qualified_only: bool = True) -> list[dict]:
    working = _qualified(frame_with_returns, qualified_only=qualified_only)
    series = []
    for date, group in working.groupby("Date"):
        status = group["Status"].iloc[0]
        valid = group.dropna(subset=[score_column, "ForwardReturn"]) if status == "FINAL" else group.iloc[0:0]
        if valid.empty:
            series.append({"date": str(pd.Timestamp(date).date()), "top_k_return": None, "universe_ew_return": None, "spread": None, "status": status})
            continue
        top_k_return = float(valid.nlargest(k, score_column)["ForwardReturn"].mean())
        universe_ew_return = float(valid["ForwardReturn"].mean())
        series.append({
            "date": str(pd.Timestamp(date).date()),
            "top_k_return": top_k_return,
            "universe_ew_return": universe_ew_return,
            "spread": top_k_return - universe_ew_return,
            "status": status,
        })
    return series


def universe_equal_weight_curve_gross(frame_with_returns: pd.DataFrame, *, initial_nav: float = 100_000.0, qualified_only: bool = True) -> list[dict]:
    """Buy-and-hold-equal-weight-across-the-qualified-universe, rebalanced
    to equal weight at every weekly signal date, using the same executable
    holding-period return as the IC calc -- the benchmark the Top-K selector
    has to beat to prove the *selection* (not just being in this universe,
    in this market) is adding value. "Gross" = no transaction costs;
    dict key is deliberately suffixed so an `_implementable` (cost-aware)
    variant can be added later without renaming this one. Same
    `[{date, nav}]` shape as `forward_ledger.chain_segments` expects, so a
    caller stitching multiple universe-version segments together can reuse
    that function directly -- this only computes ONE segment's curve, PIT
    versioning across universe changes is the caller's job (see
    `forward_ledger.py`), never retroactive here."""

    working = _qualified(frame_with_returns, qualified_only=qualified_only)
    nav = initial_nav
    series = []
    for date in sorted(working["Date"].unique()):
        series.append({"date": str(pd.Timestamp(date).date()), "nav": nav})
        returns = working.loc[working["Date"].eq(date) & working["Status"].eq("FINAL"), "ForwardReturn"].dropna()
        if len(returns):
            nav *= 1.0 + float(returns.mean())
    return series


def selection_rate(k: int, universe_size: int) -> float:
    return k / universe_size if universe_size else 0.0


def alpha_vs_universe_ew_series(model_curve: list[dict], ew_curve: list[dict]) -> list[dict]:
    """Cumulative-return spread between the model's own NAV curve and the
    Universe EW benchmark, both rebased to their own first shared date --
    shows *when* an aggregate alpha number was actually earned instead of
    collapsing the whole period into one endpoint total. Both curves must
    already share a date grid (e.g. both sampled at weekly signal dates);
    dates present in only one are dropped rather than guessed at."""

    if not model_curve or not ew_curve:
        return []
    model_by_date = {row["date"]: row["nav"] for row in model_curve}
    dates = [row["date"] for row in ew_curve if row["date"] in model_by_date]
    if not dates:
        return []
    model_base = model_by_date[dates[0]]
    ew_by_date = {row["date"]: row["nav"] for row in ew_curve}
    ew_base = ew_by_date[dates[0]]
    series = []
    for date in dates:
        model_return = (model_by_date[date] / model_base - 1.0) if model_base else None
        ew_return = (ew_by_date[date] / ew_base - 1.0) if ew_base else None
        alpha = (model_return - ew_return) if (model_return is not None and ew_return is not None) else None
        series.append({"date": date, "model_return": model_return, "universe_ew_return": ew_return, "alpha": alpha})
    return series


def build_signal_health(scored: pd.DataFrame, factored: pd.DataFrame, *, k: int, windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS) -> dict[str, object]:
    """One segment's worth of signal-health diagnostics -- callers chain
    multiple segments together (concatenate the IC/spread series in date
    order -- FINAL entries never change, only the trailing PENDING one
    matures next call; run `forward_ledger.chain_segments` on the EW curves)
    the same way `forward_ledger.build_forward_ledger` chains NAV segments."""

    frame = next_holding_period_returns(scored, factored)
    universe_size = int(scored["Ticker"].nunique())

    portfolio_ic = rank_ic_series(frame, SCORE_COLUMN)
    factor_ic = {
        factor: rank_ic_series(frame, factor)
        for factor in FACTOR_COLUMNS
        if factor in frame.columns
    }
    spread = top_k_spread_series(frame, k=k)
    completed_weeks = sum(1 for row in portfolio_ic if row["status"] == "FINAL")

    return {
        "universe_size": universe_size,
        "top_k": k,
        "selection_rate": selection_rate(k, universe_size),
        "completed_weeks": completed_weeks,
        "portfolio_ic": portfolio_ic,
        "portfolio_ic_rolling": rolling_average(portfolio_ic, "ic", windows),
        "factor_ic": {
            factor: {"series": series, "rolling": rolling_average(series, "ic", windows)}
            for factor, series in factor_ic.items()
        },
        "top_k_spread": spread,
        "top_k_spread_rolling": rolling_average(spread, "spread", windows),
        "universe_equal_weight_curve_gross": universe_equal_weight_curve_gross(frame),
    }
