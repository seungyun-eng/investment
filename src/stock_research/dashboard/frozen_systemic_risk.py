"""Frozen Forward Shadow copy of the registered systemic-risk feature engine.

The formula is copied from the sealed champion source so cloud operation does
not depend on untracked research packages. The 2018-12-31 anchor uses the
existing current-roster-conditioned eligibility
cache. It is not a survivor-free historical market index. Pre-anchor values
are retrospective warm-up only. Covariance/normalization history always uses
the SAME subset selected from data available on the feature date.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits


def canonical_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

AR_WINDOW = 500
AR_HALF_LIFE = 250.0
AR_SMOOTH = 15
BASELINE_WINDOW = 252
TURB_WINDOW = 252
VALID_WINDOW = AR_WINDOW + BASELINE_WINDOW
MIN_NAMES = 50
ANCHOR_CUTOFF = pd.Timestamp("2018-12-31")
EXCLUDED = frozenset({"SPY", "QQQ"})


def _anchor(data):
    frame = data.frame
    needed = {"Date", "Ticker", "eligible"}
    if needed.difference(frame):
        raise ValueError("weekly Date/Ticker/eligible anchor data required")
    if frame.duplicated(["Date", "Ticker"]).any():
        raise ValueError("duplicate weekly anchor keys")
    eligible_dates = pd.to_datetime(frame.Date)
    past = frame.loc[eligible_dates.le(ANCHOR_CUTOFF)].copy()
    if past.empty:
        raise ValueError("no weekly observation at or before fixed 2018 anchor")
    date = pd.to_datetime(past.Date).max()
    selected = past.loc[pd.to_datetime(past.Date).eq(date) & past.eligible.eq(True), "Ticker"]
    names = tuple(sorted(set(selected.astype(str)).difference(EXCLUDED)))
    absent = set(names).difference(data.closes.columns)
    if absent:
        raise ValueError(f"anchor ticker missing from price matrix: {sorted(absent)}")
    if len(names) < MIN_NAMES:
        raise ValueError(f"anchor has {len(names)} names; need at least {MIN_NAMES}")
    return date, names


def _absorption(window):
    """500-close-return EW covariance; top rounded fifth, no standardization."""
    x = np.asarray(window, float)
    if len(x) != AR_WINDOW or not np.isfinite(x).all():
        return np.nan, np.nan
    weights = np.exp2(-np.arange(len(x)-1, -1, -1, dtype=float)/AR_HALF_LIFE)
    weights /= weights.sum()
    center = x - weights @ x
    covariance = (center*weights[:, None]).T @ center
    total = float(np.trace(covariance))
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    k = max(1, round(x.shape[1]/5))
    eigenvalues = np.linalg.eigvalsh(covariance)
    value = float(eigenvalues[-k:].sum()/total)
    if value < -1e-10 or value > 1+1e-10:
        raise RuntimeError("invalid covariance absorption ratio")
    return float(np.clip(value, 0, 1)), float(np.diag(covariance).max()/total)


def _turbulence(history, current):
    """Today against prior252 Ledoit-Wolf covariance, normalized by N."""
    x, now = np.asarray(history, float), np.asarray(current, float)
    if len(x) != TURB_WINDOW or not np.isfinite(x).all() or not np.isfinite(now).all():
        return np.nan
    if x.shape[1] == 0:
        return np.nan
    estimator = LedoitWolf(assume_centered=False, store_precision=False).fit(x)
    if np.trace(estimator.covariance_) <= 0:
        return np.nan
    innovation = now - estimator.location_
    try:
        factor = cho_factor(estimator.covariance_, check_finite=False)
        result = float(innovation @ cho_solve(factor, innovation, check_finite=False)/x.shape[1])
    except np.linalg.LinAlgError:
        # Missing numerical support is unknown, not zero risk or a fitted floor.
        return np.nan
    return max(result, 0.0) if np.isfinite(result) else np.nan


def _standardized(current, history):
    prior = np.asarray(history, float)
    if len(prior) != BASELINE_WINDOW or not np.isfinite(prior).all() or not np.isfinite(current):
        return np.nan
    scale = float(prior.std(ddof=1))
    return float((current-prior.mean())/scale) if scale > 0 else np.nan


class _SamePanel:
    """Lazy same-panel history cache; each day's primitive sees its past only."""

    def __init__(self, returns, dollar_volume):
        self.returns = np.asarray(returns, float)
        dollar = np.asarray(dollar_volume, float)
        usable = np.isfinite(self.returns) & np.isfinite(dollar) & (dollar > 0)
        ratios = np.full_like(self.returns, np.nan)
        np.divide(np.abs(self.returns), dollar, out=ratios, where=usable)
        self.illiq_n = usable.sum(axis=1)
        self.illiq = np.full(len(returns), np.nan)
        for i in np.flatnonzero(self.illiq_n >= MIN_NAMES):
            self.illiq[i] = np.median(ratios[i, usable[i]])
        self.ar = np.full(len(returns), np.nan)
        self.largest_variance_share = np.full(len(returns), np.nan)
        self.turbulence = np.full(len(returns), np.nan)
        self.computed = np.zeros(len(returns), dtype=bool)

    def at(self, i):
        # Both historical distributions are recalculated on this date's panel.
        # No previously emitted value from a differently sized panel is reused.
        for j in range(max(0, i-BASELINE_WINDOW), i+1):
            if self.computed[j]:
                continue
            self.ar[j], self.largest_variance_share[j] = _absorption(
                self.returns[max(0, j-AR_WINDOW+1):j+1]
            )
            self.turbulence[j] = _turbulence(self.returns[max(0, j-TURB_WINDOW):j], self.returns[j])
            self.computed[j] = True
        smooth = self.ar[max(0, i-AR_SMOOTH+1):i+1]
        smooth_mean = float(smooth.mean()) if len(smooth) == AR_SMOOTH and np.isfinite(smooth).all() else np.nan
        prior = self.turbulence[max(0, i-BASELINE_WINDOW):i]
        current = self.turbulence[i]
        percentile = np.nan
        if len(prior) == BASELINE_WINDOW and np.isfinite(prior).all() and np.isfinite(current):
            # Empirical mid-rank handles repeated values without forced extremes.
            percentile = float((np.sum(prior < current)+.5*np.sum(prior == current))/len(prior))
        return {
            "ar": self.ar[i],
            "ar_shift": _standardized(smooth_mean, self.ar[max(0, i-BASELINE_WINDOW):i]),
            "largest_variance_share": self.largest_variance_share[i],
            "turbulence": current,
            "turbulence_pct": percentile,
            "dispersion": float(self.returns[i].std(ddof=1)),
            "illiq": self.illiq[i],
            "illiq_shock": _standardized(self.illiq[i], self.illiq[max(0, i-BASELINE_WINDOW):i]),
            "illiq_n": int(self.illiq_n[i]),
        }


def _vix_slope(vix, dates):
    required = {"Date", "VIX", "VIX3M"}
    if required.difference(vix):
        raise ValueError("same-date Cboe Date/VIX/VIX3M closes required")
    source = vix.copy()
    source["Date"] = pd.to_datetime(source.Date)
    if source.Date.isna().any() or source.Date.duplicated().any():
        raise ValueError("unique nonmissing VIX dates required")
    aligned = source.set_index("Date")[["VIX", "VIX3M"]].reindex(dates)
    aligned = aligned.apply(pd.to_numeric, errors="raise")
    valid = np.isfinite(aligned).all(axis=1) & aligned.gt(0).all(axis=1)
    slope = np.log(aligned.VIX.where(valid)/aligned.VIX3M.where(valid))
    # Cboe close occurs after the regular stock close. Shift one actual stock
    # session, never use today's close and never carry over a missing session.
    return slope.shift(1), pd.Series(dates, index=dates).where(valid).shift(1)


def build_market_features(data, raw, vix):
    """Return all stock sessions with unknown warm-up/missing values as NaN.

    raw.Close must be the unadjusted cash quote, Volume_numeric actual shares;
    return signals use data.closes adjusted prices. vix is NOT pre-lagged: this
    function applies the sole one-stock-session reporting lag.
    """
    dates = pd.DatetimeIndex(data.closes.index)
    if dates.has_duplicates or dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError("unique chronological stock sessions required")
    if not dates.equals(pd.DatetimeIndex(data.sessions)):
        raise ValueError("StudyData close/session calendar mismatch")
    if data.closes.columns.has_duplicates:
        raise ValueError("duplicate close ticker columns")
    anchor_date, names = _anchor(data)
    prices = data.closes.loc[:, names].where(lambda x: np.isfinite(x) & x.gt(0))
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    valid = returns.notna().rolling(VALID_WINDOW, min_periods=VALID_WINDOW).sum().eq(VALID_WINDOW)
    fields = {"Date", "Ticker", "Close", "Volume_numeric"}
    if fields.difference(raw):
        raise ValueError("raw cash close and actual share volume required")
    quotes = raw.loc[raw.Ticker.isin(names), ["Date", "Ticker", "Close", "Volume_numeric"]].copy()
    quotes["Date"] = pd.to_datetime(quotes.Date)
    if quotes.duplicated(["Date", "Ticker"]).any() or quotes.Date.isna().any():
        raise ValueError("duplicate/missing raw quote keys")
    quotes["Close"] = pd.to_numeric(quotes.Close, errors="raise")
    quotes["Volume_numeric"] = pd.to_numeric(quotes.Volume_numeric, errors="raise")
    quotes["dollar_volume"] = quotes.Close * quotes.Volume_numeric
    good_raw = quotes.Close.gt(0) & quotes.Volume_numeric.gt(0)
    quotes.loc[~good_raw, "dollar_volume"] = np.nan
    dollar = quotes.pivot(index="Date", columns="Ticker", values="dollar_volume").reindex(index=dates, columns=names)
    values, cash_volume = returns.to_numpy(float), dollar.to_numpy(float)
    rows, panels, panel_registry = [], {}, {}
    with threadpool_limits(limits=1):
        for i, mask in enumerate(valid.to_numpy(bool)):
            ids = tuple(np.flatnonzero(mask).tolist())
            members = [names[j] for j in ids]
            subset_hash = canonical_hash(members)
            panel_registry.setdefault(subset_hash, members)
            row = {"Date": dates[i], "indicator_n": len(ids),
                   "indicator_k": max(1, round(len(ids)/5)) if ids else 0,
                   "indicator_subset_sha256": subset_hash,
                   "anchor_known": dates[i] >= anchor_date}
            if len(ids) >= MIN_NAMES:
                if ids not in panels:
                    panels[ids] = _SamePanel(values[:, ids], cash_volume[:, ids])
                row.update(panels[ids].at(i))
            rows.append(row)
    columns = ["Date", "indicator_n", "indicator_k", "indicator_subset_sha256",
               "anchor_known", "ar", "ar_shift", "largest_variance_share",
               "turbulence", "turbulence_pct", "dispersion", "illiq", "illiq_shock", "illiq_n"]
    result = pd.DataFrame(rows).reindex(columns=columns).set_index("Date", drop=False)
    result["vix_slope"], result["vix_source_date"] = _vix_slope(vix, dates)
    result.attrs.update({
        "anchor_date": str(anchor_date.date()), "anchor_cutoff": str(ANCHOR_CUTOFF.date()),
        "anchor_names": list(names), "anchor_count": len(names), "unique_valid_panels": len(panels),
        "indicator_panel_registry": panel_registry,
        "anchor_condition": "2018_ELIGIBLE_CURRENT368_ROSTER_BACKCAST_NOT_PIT_MEMBERSHIP",
        "pre_anchor_values": "RETROSPECTIVE_WARMUP_ONLY_NOT_TRADABLE_BEFORE_ANCHOR",
        "ar_window": AR_WINDOW, "ar_half_life": AR_HALF_LIFE, "ar_top_k": "max(1,round(N/5))",
        "ar_history_same_panel": True, "turbulence_window": TURB_WINDOW,
        "turbulence_estimator": "LedoitWolf prior252, solve covariance, divide N",
        "turbulence_percentile": "prior252 empirical midrank, same current panel, range0..1",
        "complete_return_window": VALID_WINDOW, "min_names": MIN_NAMES,
        "illiq_definition": "median abs(adjusted return)/(raw cash close*actual share volume)",
        "vix_lag_stock_sessions": 1, "missing_forward_fill": False,
    })
    return result
