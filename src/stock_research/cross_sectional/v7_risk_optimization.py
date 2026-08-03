from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths

from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .pit_validation import (
    apply_membership_to_panel,
    causal_signal_day_panel,
)
from .portfolio import PreparedMarket, prepare_market
from .signals import generate_rebalance_targets, score_panel
from .v7_capital_overlay import (
    SignedBacktestResult,
    run_signed_overlay_backtest,
)
from .v7_pit_evaluation import (
    build_v7_source_panel,
    load_ready_tickers,
    normalize_change_membership,
)
from .v7_slot_sweep import spy_buy_and_hold
from .v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
    slot5_params,
)


@dataclass(frozen=True)
class RiskCandidate:
    candidate_id: int
    name: str
    allocation_profile: str
    risk_profile: str
    score_strength: float
    inverse_volatility_strength: float
    apply_concentration_caps: bool
    fixed_long_gross: float | None
    target_portfolio_volatility: float | None
    minimum_long_gross: float
    max_long_gross: float
    risk_off_gross_cap: float


@dataclass(frozen=True)
class RiskOptimizationArtifacts:
    output_dir: Path
    data_audit_csv: Path
    candidate_summary_csv: Path
    period_summary_csv: Path
    stress_summary_csv: Path
    allocations_csv: Path
    allocation_diagnostics_csv: Path
    equity_csv: Path
    executions_csv: Path
    manifest_json: Path


@dataclass(frozen=True)
class _PeriodContext:
    scored: pd.DataFrame
    v7_targets: pd.DataFrame
    market: PreparedMarket


def generate_risk_candidates(
    config: dict[str, Any],
) -> list[RiskCandidate]:
    allocation_profiles = list(config["allocation_profiles"])
    risk_profiles = list(config["risk_profiles"])
    candidates: list[RiskCandidate] = []
    candidate_id = 0
    for allocation in allocation_profiles:
        for risk in risk_profiles:
            name = f"{allocation['name']}__{risk['name']}"
            candidates.append(
                RiskCandidate(
                    candidate_id=candidate_id,
                    name=name,
                    allocation_profile=str(allocation["name"]),
                    risk_profile=str(risk["name"]),
                    score_strength=float(allocation["score_strength"]),
                    inverse_volatility_strength=float(
                        allocation["inverse_volatility_strength"]
                    ),
                    apply_concentration_caps=bool(
                        allocation["apply_concentration_caps"]
                    ),
                    fixed_long_gross=(
                        None
                        if risk.get("fixed_long_gross") is None
                        else float(risk["fixed_long_gross"])
                    ),
                    target_portfolio_volatility=(
                        None
                        if risk.get("target_portfolio_volatility") is None
                        else float(risk["target_portfolio_volatility"])
                    ),
                    minimum_long_gross=float(
                        risk.get("minimum_long_gross", 0.0)
                    ),
                    max_long_gross=float(risk["max_long_gross"]),
                    risk_off_gross_cap=float(risk["risk_off_gross_cap"]),
                )
            )
            candidate_id += 1
    if not candidates:
        raise ValueError("Risk optimization config produced no candidates")
    if candidates[0].name != "EQUAL_UNCAPPED__UNLEVERED":
        raise ValueError(
            "The first risk candidate must be EQUAL_UNCAPPED__UNLEVERED"
        )
    for candidate in candidates:
        if (
            candidate.fixed_long_gross is not None
            and candidate.target_portfolio_volatility is not None
        ):
            raise ValueError(
                f"{candidate.name} cannot set both fixed gross and target vol"
            )
        if not 0 <= candidate.minimum_long_gross <= candidate.max_long_gross:
            raise ValueError(
                f"{candidate.name} has infeasible minimum/maximum gross"
            )
        if (
            candidate.fixed_long_gross is not None
            and not candidate.minimum_long_gross
            <= candidate.fixed_long_gross
            <= candidate.max_long_gross
        ):
            raise ValueError(f"{candidate.name} fixed gross is out of bounds")
    return candidates


def add_causal_asset_volatility(
    panel: pd.DataFrame,
    *,
    window: int,
    minimum_observations: int,
) -> pd.DataFrame:
    if window < 2:
        raise ValueError("window must be at least two sessions")
    if minimum_observations < 2 or minimum_observations > window:
        raise ValueError(
            "minimum_observations must be between two and window"
        )
    frame = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    frame["AssetReturn"] = frame.groupby(
        "Ticker",
        sort=False,
    )["Close"].pct_change(fill_method=None)
    frame["AssetVolatility"] = frame.groupby(
        "Ticker",
        sort=False,
    )["AssetReturn"].transform(
        lambda values: (
            values.rolling(
                window,
                min_periods=minimum_observations,
            ).std(ddof=1)
            * np.sqrt(252)
        )
    )
    return frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def load_sector_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    if "Ticker" not in frame:
        raise ValueError("Sector map requires Ticker")
    sector_column = next(
        (
            column
            for column in ("GicsSector", "GICSSector", "Sector")
            if column in frame
        ),
        None,
    )
    if sector_column is None:
        raise ValueError("Sector map requires GicsSector or Sector")
    ticker = frame["Ticker"].astype(str).str.upper().str.strip()
    sector = frame[sector_column].astype(str).str.strip()
    valid = ticker.ne("") & sector.ne("") & sector.str.lower().ne("nan")
    return dict(zip(ticker.loc[valid], sector.loc[valid], strict=True))


def validate_membership_history(
    membership: pd.DataFrame,
    *,
    minimum_snapshot_count: int,
) -> int:
    required = {"AsOfDate", "Ticker"}
    missing = sorted(required - set(membership.columns))
    if missing:
        raise ValueError(
            "Membership history is missing columns: " + ", ".join(missing)
        )
    snapshot_count = int(
        pd.to_datetime(
            membership["AsOfDate"],
            errors="coerce",
        ).nunique()
    )
    if snapshot_count < minimum_snapshot_count:
        raise ValueError(
            "Membership history is too coarse for V7-3: "
            f"found {snapshot_count} snapshots, require at least "
            f"{minimum_snapshot_count}. Use sp500_membership_changes.csv, "
            "not the annual sp500_membership.csv snapshot file."
        )
    return snapshot_count


def assert_baseline_parity(
    result: SignedBacktestResult,
    expected: dict[str, Any],
) -> None:
    summary = result.summary
    tolerance = float(expected.get("tolerance", 1e-6))
    date_checks = {
        "start_date": str(summary.start_date.date()),
        "end_date": str(summary.end_date.date()),
    }
    metric_checks = {
        "roi": summary.roi_percent,
        "cagr": summary.cagr_percent,
        "max_drawdown": summary.max_drawdown_percent,
        "sharpe": summary.sharpe_ratio,
    }
    failures: list[str] = []
    for key, actual in date_checks.items():
        wanted = str(expected[key])
        if actual != wanted:
            failures.append(f"{key}: expected {wanted}, got {actual}")
    for key, actual in metric_checks.items():
        wanted = float(expected[key])
        if not np.isclose(actual, wanted, atol=tolerance, rtol=0.0):
            failures.append(
                f"{key}: expected {wanted:.12f}, got {actual:.12f}"
            )
    if failures:
        raise RuntimeError(
            "V7-3 baseline parity failed; optimization aborted. "
            + "; ".join(failures)
        )


def prepare_spy_regime(
    spy: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    *,
    sma_window: int,
) -> pd.Series:
    if sma_window < 2:
        raise ValueError("sma_window must be at least two sessions")
    frame = (
        spy[["Date", "Close"]]
        .dropna()
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .copy()
    )
    frame["SMA"] = frame["Close"].rolling(
        sma_window,
        min_periods=sma_window,
    ).mean()
    frame["RiskOn"] = frame["Close"].ge(frame["SMA"])
    requested = pd.DataFrame(
        {"Date": pd.DatetimeIndex(signal_dates).sort_values().unique()}
    )
    merged = pd.merge_asof(
        requested,
        frame[["Date", "RiskOn"]],
        on="Date",
        direction="backward",
    )
    merged["RiskOn"] = merged["RiskOn"].fillna(True).astype(bool)
    return merged.set_index("Date")["RiskOn"]


def constrained_composition(
    raw_weights: pd.Series,
    sectors: pd.Series,
    *,
    max_ticker_weight: float,
    max_sector_weight: float,
) -> tuple[pd.Series, float, float]:
    raw = pd.to_numeric(raw_weights, errors="coerce").fillna(0.0).clip(lower=0)
    if raw.empty or float(raw.sum()) <= 0:
        raise ValueError("raw_weights must contain positive weight")
    raw = raw / float(raw.sum())
    groups = sectors.reindex(raw.index).fillna("").astype(str)
    groups = pd.Series(
        [
            value if value and value.lower() != "nan" else f"UNMAPPED:{ticker}"
            for ticker, value in zip(raw.index, groups, strict=True)
        ],
        index=raw.index,
        dtype=object,
    )
    ticker_cap = max(float(max_ticker_weight), 1.0 / len(raw))
    group_count = max(int(groups.nunique()), 1)
    sector_cap = max(float(max_sector_weight), 1.0 / group_count)
    raw_values = raw.to_numpy(dtype=float)
    group_indexes = [
        np.flatnonzero(groups.to_numpy() == group)
        for group in sorted(groups.unique())
    ]
    group_ticker_capacities = np.asarray(
        [len(indexes) * ticker_cap for indexes in group_indexes],
        dtype=float,
    )
    if float(np.minimum(sector_cap, group_ticker_capacities).sum()) < (
        1.0 - 1e-12
    ):
        lower = sector_cap
        upper = 1.0
        for _ in range(80):
            midpoint = (lower + upper) / 2.0
            capacity = float(
                np.minimum(midpoint, group_ticker_capacities).sum()
            )
            if capacity >= 1.0:
                upper = midpoint
            else:
                lower = midpoint
        sector_cap = upper
    group_capacities = np.minimum(
        sector_cap,
        group_ticker_capacities,
    )
    group_raw = np.asarray(
        [raw_values[indexes].sum() for indexes in group_indexes],
        dtype=float,
    )
    group_totals = _project_to_capped_simplex(
        group_raw,
        total=1.0,
        upper_bounds=group_capacities,
    )
    initial = np.zeros(len(raw), dtype=float)
    for indexes, group_total in zip(
        group_indexes,
        group_totals,
        strict=True,
    ):
        initial[indexes] = _project_to_capped_simplex(
            raw_values[indexes],
            total=float(group_total),
            upper_bounds=np.repeat(ticker_cap, len(indexes)),
        )
    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda values: float(values.sum() - 1.0)}
    ]
    for indexes in group_indexes:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values, positions=indexes: float(
                    sector_cap - values[positions].sum()
                ),
            }
        )
    result = minimize(
        lambda values: float(np.square(values - raw_values).sum()),
        initial,
        method="SLSQP",
        bounds=[(0.0, ticker_cap)] * len(raw),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    candidate_values = np.asarray(result.x, dtype=float)
    valid_result = (
        result.success
        and np.isfinite(candidate_values).all()
        and abs(float(candidate_values.sum()) - 1.0) <= 1e-8
        and float(candidate_values.min()) >= -1e-8
        and float(candidate_values.max()) <= ticker_cap + 1e-8
        and all(
            float(candidate_values[indexes].sum()) <= sector_cap + 1e-8
            for indexes in group_indexes
        )
    )
    values = candidate_values if valid_result else initial
    values = np.where(np.abs(values) < 1e-12, 0.0, values)
    values = values / float(values.sum())
    return (
        pd.Series(values, index=raw.index, dtype=float),
        ticker_cap,
        sector_cap,
    )


def _project_to_capped_simplex(
    values: np.ndarray,
    *,
    total: float,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    if values.shape != upper_bounds.shape or values.ndim != 1:
        raise ValueError("values and upper_bounds must be matching vectors")
    if total < -1e-12 or total > float(upper_bounds.sum()) + 1e-12:
        raise ValueError("Requested simplex total is infeasible")
    if total <= 1e-12:
        return np.zeros_like(values)
    lower = float(np.min(values - upper_bounds))
    upper = float(np.max(values))
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        projected = np.clip(values - midpoint, 0.0, upper_bounds)
        if float(projected.sum()) > total:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - upper, 0.0, upper_bounds)
    residual = total - float(projected.sum())
    if residual > 0:
        for index in range(len(projected)):
            addition = min(
                residual,
                float(upper_bounds[index] - projected[index]),
            )
            projected[index] += addition
            residual -= addition
            if residual <= 1e-12:
                break
    elif residual < 0:
        for index in range(len(projected)):
            reduction = min(-residual, float(projected[index]))
            projected[index] -= reduction
            residual += reduction
            if residual >= -1e-12:
                break
    return projected


def forecast_portfolio_volatility(
    returns_wide: pd.DataFrame,
    *,
    date: pd.Timestamp,
    weights: pd.Series,
    asset_volatility: pd.Series,
    window: int,
    minimum_observations: int,
    shrinkage: float,
    volatility_floor: float,
) -> float:
    tickers = list(weights.index)
    history = returns_wide.loc[
        returns_wide.index <= pd.Timestamp(date),
        tickers,
    ].tail(window)
    sample = history.cov(min_periods=minimum_observations).reindex(
        index=tickers,
        columns=tickers,
    )
    fallback_daily_variance = np.square(
        pd.to_numeric(
            asset_volatility.reindex(tickers),
            errors="coerce",
        )
        .fillna(volatility_floor)
        .clip(lower=volatility_floor)
        .to_numpy(dtype=float)
        / np.sqrt(252)
    )
    covariance = sample.to_numpy(dtype=float)
    covariance = np.where(np.isfinite(covariance), covariance, 0.0)
    diagonal = np.diag(covariance).copy()
    diagonal = np.where(diagonal > 0, diagonal, fallback_daily_variance)
    np.fill_diagonal(covariance, diagonal)
    diagonal_matrix = np.diag(diagonal)
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * diagonal_matrix
    )
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = eigenvectors @ np.diag(
        np.clip(eigenvalues, 1e-12, None)
    ) @ eigenvectors.T
    values = weights.reindex(tickers).to_numpy(dtype=float)
    annual_variance = float(values @ covariance @ values * 252)
    return float(np.sqrt(max(annual_variance, 0.0)))


def build_risk_adjusted_targets(
    v7_targets: pd.DataFrame,
    returns_wide: pd.DataFrame,
    risk_on: pd.Series,
    candidate: RiskCandidate,
    *,
    sector_map: dict[str, str],
    volatility_window: int,
    minimum_volatility_observations: int,
    covariance_shrinkage: float,
    asset_volatility_floor: float,
    max_ticker_weight: float,
    max_sector_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for date, group in v7_targets.groupby("Date", sort=True):
        date = pd.Timestamp(date)
        selected = (
            group.loc[group["ModelSelected"].fillna(False)]
            .sort_values(["Rank", "Ticker"])
            .copy()
        )
        if selected.empty:
            target_rows.append(
                {
                    "Date": date,
                    "Ticker": str(group.iloc[0]["Ticker"]),
                    "TargetWeight": 0.0,
                    "BaseCompositionWeight": 0.0,
                    "AlphaScore": np.nan,
                    "AssetVolatility": np.nan,
                    "Sector": "",
                }
            )
            diagnostics.append(
                {
                    "Date": date,
                    "RiskOn": bool(risk_on.get(date, True)),
                    "SelectedCount": 0,
                    "ForecastVolatilityAtOneX": 0.0,
                    "DesiredLongGross": 0.0,
                    "LongGross": 0.0,
                    "MaximumTickerComposition": 0.0,
                    "MaximumSectorComposition": 0.0,
                    "MappedSectorCount": 0,
                    "EffectiveTickerCap": max_ticker_weight,
                    "EffectiveSectorCap": max_sector_weight,
                }
            )
            continue
        selected = selected.set_index("Ticker", drop=False)
        alpha = pd.to_numeric(
            selected["AlphaScore"],
            errors="coerce",
        ).fillna(0.0)
        alpha_std = float(alpha.std(ddof=0))
        alpha_z = (
            (alpha - float(alpha.mean())) / alpha_std
            if alpha_std > 0
            else pd.Series(0.0, index=alpha.index)
        )
        volatility = pd.to_numeric(
            selected["AssetVolatility"],
            errors="coerce",
        )
        fallback_volatility = (
            float(volatility.dropna().median())
            if not volatility.dropna().empty
            else asset_volatility_floor
        )
        volatility = (
            volatility.fillna(fallback_volatility)
            .clip(lower=asset_volatility_floor)
        )
        raw = np.exp(
            np.clip(candidate.score_strength * alpha_z, -5.0, 5.0)
        )
        if candidate.inverse_volatility_strength > 0:
            raw = raw * np.power(
                volatility,
                -candidate.inverse_volatility_strength,
            )
        raw = pd.Series(raw, index=selected.index, dtype=float)
        sectors = pd.Series(
            {
                ticker: sector_map.get(ticker, f"UNMAPPED:{ticker}")
                for ticker in selected.index
            },
            dtype=object,
        )
        if candidate.apply_concentration_caps:
            composition, ticker_cap, sector_cap = constrained_composition(
                raw,
                sectors,
                max_ticker_weight=max_ticker_weight,
                max_sector_weight=max_sector_weight,
            )
        else:
            composition = raw / float(raw.sum())
            ticker_cap = 1.0
            sector_cap = 1.0
        forecast_volatility = forecast_portfolio_volatility(
            returns_wide,
            date=date,
            weights=composition,
            asset_volatility=volatility,
            window=volatility_window,
            minimum_observations=minimum_volatility_observations,
            shrinkage=covariance_shrinkage,
            volatility_floor=asset_volatility_floor,
        )
        if candidate.fixed_long_gross is not None:
            desired_gross = candidate.fixed_long_gross
        elif candidate.target_portfolio_volatility is not None:
            desired_gross = (
                candidate.target_portfolio_volatility
                / max(forecast_volatility, 1e-12)
            )
        else:
            desired_gross = 1.0
        desired_gross = min(
            max(desired_gross, candidate.minimum_long_gross),
            candidate.max_long_gross,
        )
        date_risk_on = bool(risk_on.get(date, True))
        long_gross = (
            desired_gross
            if date_risk_on
            else min(desired_gross, candidate.risk_off_gross_cap)
        )
        weights = composition * long_gross
        sector_exposure = weights.groupby(sectors).sum()
        for ticker, weight in weights.items():
            target_rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "TargetWeight": float(weight),
                    "BaseCompositionWeight": float(composition.loc[ticker]),
                    "AlphaScore": float(alpha.loc[ticker]),
                    "AssetVolatility": float(volatility.loc[ticker]),
                    "Sector": sectors.loc[ticker],
                }
            )
        mapped_sector_count = int(
            sum(not value.startswith("UNMAPPED:") for value in sectors)
        )
        diagnostics.append(
            {
                "Date": date,
                "RiskOn": date_risk_on,
                "SelectedCount": len(selected),
                "ForecastVolatilityAtOneX": forecast_volatility,
                "DesiredLongGross": desired_gross,
                "LongGross": float(weights.sum()),
                "MaximumTickerComposition": float(composition.max()),
                "MaximumSectorComposition": float(
                    composition.groupby(sectors).sum().max()
                ),
                "MaximumTickerTargetWeight": float(weights.max()),
                "MaximumSectorTargetWeight": float(
                    sector_exposure.max()
                ),
                "MappedSectorCount": mapped_sector_count,
                "EffectiveTickerCap": ticker_cap,
                "EffectiveSectorCap": sector_cap,
            }
        )
    return pd.DataFrame(target_rows), pd.DataFrame(diagnostics)


def run_v7_risk_optimization(
    paths: ProjectPaths,
    settings: ResearchSettings,
    base_params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    frozen_strategy_path: str | Path,
    spy_path: str | Path,
    optimization_config: dict[str, Any],
    sector_map_path: str | Path | None = None,
    expected_ready_count: int = 575,
    output_dir: str | Path | None = None,
) -> RiskOptimizationArtifacts:
    candidates = generate_risk_candidates(optimization_config)
    ready_tickers = load_ready_tickers(
        backfill_status_path,
        expected_count=expected_ready_count,
    )
    members, _ = discover_universe(paths, ticker_config_path)
    discoverable = {member.ticker for member in members}
    missing = sorted(ready_tickers - discoverable)
    if missing:
        raise ValueError(
            "Ready tickers are not discoverable: " + ", ".join(missing)
        )
    warmup_start = str(
        (
            pd.Timestamp(settings.train_start) - pd.DateOffset(years=2)
        ).date()
    )
    base_panel, data_audit = build_v7_source_panel(
        paths,
        members,
        settings,
        ready_tickers=ready_tickers,
        warmup_start=warmup_start,
    )
    observed_panel = add_v7_technical_observations(base_panel)
    raw_membership = pd.read_csv(membership_path)
    membership_snapshot_count = validate_membership_history(
        raw_membership,
        minimum_snapshot_count=int(
            optimization_config["minimum_membership_snapshot_count"]
        ),
    )
    membership = normalize_change_membership(raw_membership)
    pit_panel = apply_membership_to_panel(
        observed_panel,
        membership,
        settings,
    )
    technical_panel = add_v7_technical_factors(pit_panel, settings)
    technical_panel = add_causal_asset_volatility(
        technical_panel,
        window=int(optimization_config["volatility_window"]),
        minimum_observations=int(
            optimization_config["minimum_volatility_observations"]
        ),
    )
    v7_variant = next(
        variant
        for variant in TECHNICAL_VARIANTS
        if variant.name == "V7_3_MA_MACD_OBV_SLOT5"
    )
    scoring_panel = scoring_panel_for_variant(
        technical_panel,
        v7_variant,
    )
    latest_end = str(pd.Timestamp(scoring_panel["Date"].max()).date())
    periods: dict[str, tuple[str, str]] = {
        "TRAIN_2020_2024": (
            settings.train_start,
            min(settings.train_end, latest_end),
        )
    }
    for index, (start, end) in enumerate(settings.training_folds, start=1):
        periods[f"TRAIN_FOLD_{index}"] = (start, min(end, latest_end))
    for label, (start, end) in settings.validation_periods.items():
        if start <= latest_end:
            periods[label] = (start, min(end, latest_end))
    periods["FULL_2020_2026"] = (settings.train_start, latest_end)
    reference_dates = pd.DatetimeIndex(
        scoring_panel["Date"].drop_duplicates().sort_values()
    )
    params = slot5_params(base_params)
    contexts = {
        label: _build_period_context(
            scoring_panel,
            params,
            settings,
            start=start,
            end=end,
            reference_dates=reference_dates,
        )
        for label, (start, end) in periods.items()
    }
    returns_wide = scoring_panel.pivot(
        index="Date",
        columns="Ticker",
        values="AssetReturn",
    ).sort_index()
    spy = load_sp500_proxy(spy_path)
    signal_dates = pd.DatetimeIndex(
        pd.concat(
            [context.v7_targets["Date"] for context in contexts.values()],
            ignore_index=True,
        ).drop_duplicates()
    )
    risk_on = prepare_spy_regime(
        spy,
        signal_dates,
        sma_window=int(optimization_config["spy_sma_window"]),
    )
    sector_map = load_sector_map(sector_map_path)
    full_context = contexts["FULL_2020_2026"]
    full_start, full_end = periods["FULL_2020_2026"]
    baseline_full_targets, baseline_full_diagnostics = _candidate_targets(
        full_context,
        returns_wide,
        risk_on,
        candidates[0],
        sector_map,
        optimization_config,
    )
    baseline_full_result = run_signed_overlay_backtest(
        scoring_panel,
        baseline_full_targets,
        start=full_start,
        end=full_end,
        initial_capital=settings.initial_capital,
        transaction_cost_bps=settings.transaction_cost_bps,
        funding_annual_rate=float(optimization_config["funding_annual_rate"]),
        short_borrow_annual_rate=0.0,
        prepared_market=full_context.market,
    )
    assert_baseline_parity(
        baseline_full_result,
        optimization_config["baseline_parity"],
    )
    print(
        "BASELINE PARITY PASS "
        f"ROI={baseline_full_result.summary.roi_percent:.6f}% "
        f"CAGR={baseline_full_result.summary.cagr_percent:.6f}%"
    )
    training_labels = [
        label for label in periods if label.startswith("TRAIN_FOLD_")
    ]
    full_target_cagr = float(optimization_config["full_target_cagr"])
    candidate_rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        metrics: dict[str, SignedBacktestResult] = {}
        diagnostics_by_label: dict[str, pd.DataFrame] = {}
        for label in ["TRAIN_2020_2024", *training_labels]:
            context = contexts[label]
            targets, diagnostics = _candidate_targets(
                context,
                returns_wide,
                risk_on,
                candidate,
                sector_map,
                optimization_config,
            )
            start, end = periods[label]
            metrics[label] = run_signed_overlay_backtest(
                scoring_panel,
                targets,
                start=start,
                end=end,
                initial_capital=settings.initial_capital,
                transaction_cost_bps=settings.transaction_cost_bps,
                funding_annual_rate=float(
                    optimization_config["funding_annual_rate"]
                ),
                short_borrow_annual_rate=0.0,
                prepared_market=context.market,
            )
            diagnostics_by_label[label] = diagnostics
        if candidate.candidate_id == 0:
            full_result = baseline_full_result
            full_diagnostics = baseline_full_diagnostics
        else:
            full_targets, full_diagnostics = _candidate_targets(
                full_context,
                returns_wide,
                risk_on,
                candidate,
                sector_map,
                optimization_config,
            )
            full_result = run_signed_overlay_backtest(
                scoring_panel,
                full_targets,
                start=full_start,
                end=full_end,
                initial_capital=settings.initial_capital,
                transaction_cost_bps=settings.transaction_cost_bps,
                funding_annual_rate=float(
                    optimization_config["funding_annual_rate"]
                ),
                short_borrow_annual_rate=0.0,
                prepared_market=full_context.market,
            )
        train = metrics["TRAIN_2020_2024"]
        fold_rois = [
            metrics[label].summary.roi_percent
            for label in training_labels
        ]
        fold_cagrs = [
            metrics[label].summary.cagr_percent
            for label in training_labels
        ]
        positive_folds = int(sum(value > 0 for value in fold_rois))
        pass_constraints = bool(
            not train.ruined
            and train.summary.max_drawdown_percent
            >= float(optimization_config["minimum_train_mdd"])
            and positive_folds
            >= int(optimization_config["minimum_positive_folds"])
            and min(fold_rois)
            >= float(optimization_config["minimum_worst_fold_roi"])
        )
        fold_cagr_std = float(np.std(fold_cagrs, ddof=0))
        median_fold_cagr = float(np.median(fold_cagrs))
        robust_score = _robust_objective(
            train,
            median_fold_cagr=median_fold_cagr,
            fold_cagr_std=fold_cagr_std,
            config=optimization_config["objective"],
        )
        train_diagnostics = diagnostics_by_label["TRAIN_2020_2024"]
        candidate_rows.append(
            {
                **asdict(candidate),
                "TrainROI": train.summary.roi_percent,
                "TrainCAGR": train.summary.cagr_percent,
                "TrainSharpe": train.summary.sharpe_ratio,
                "TrainMaxDrawdown": train.summary.max_drawdown_percent,
                "TrainAnnualizedTurnover": (
                    train.summary.annualized_turnover
                ),
                "TrainFundingCost": train.total_funding_cost,
                "MedianFoldROI": float(np.median(fold_rois)),
                "WorstFoldROI": float(min(fold_rois)),
                "MedianFoldCAGR": median_fold_cagr,
                "FoldCAGRStd": fold_cagr_std,
                "PositiveFolds": positive_folds,
                "AverageLongGross": float(
                    train_diagnostics["LongGross"].mean()
                ),
                "RiskOffSignals": int(
                    (~train_diagnostics["RiskOn"]).sum()
                ),
                "FullROIReportOnly": full_result.summary.roi_percent,
                "FullCAGRReportOnly": full_result.summary.cagr_percent,
                "FullSharpeReportOnly": full_result.summary.sharpe_ratio,
                "FullMaxDrawdownReportOnly": (
                    full_result.summary.max_drawdown_percent
                ),
                "FullAnnualizedTurnoverReportOnly": (
                    full_result.summary.annualized_turnover
                ),
                "FullFundingCostReportOnly": (
                    full_result.total_funding_cost
                ),
                "FullAverageLongGrossReportOnly": float(
                    full_diagnostics["LongGross"].mean()
                ),
                "MeetsFullTargetReportOnly": bool(
                    full_result.summary.cagr_percent >= full_target_cagr
                ),
                "RobustScore": robust_score,
                "Ruined": train.ruined,
                "PassConstraints": pass_constraints,
            }
        )
        if candidate_index % 5 == 0 or candidate_index == len(candidates):
            print(
                f"RISK {candidate_index}/{len(candidates)} "
                "candidates evaluated"
            )
    candidate_summary = pd.DataFrame(candidate_rows)
    valid = candidate_summary.loc[
        candidate_summary["PassConstraints"]
    ].copy()
    if valid.empty:
        raise RuntimeError("No risk candidate passed the training constraints")
    selected_id = int(
        valid.sort_values(
            ["RobustScore", "MedianFoldCAGR", "TrainCAGR"],
            ascending=[False, False, False],
        ).iloc[0]["candidate_id"]
    )
    max_train_id = int(
        candidate_summary.sort_values(
            ["TrainCAGR", "MedianFoldCAGR"],
            ascending=[False, False],
        ).iloc[0]["candidate_id"]
    )
    target_pool = candidate_summary.loc[
        candidate_summary["PassConstraints"]
        & candidate_summary["MeetsFullTargetReportOnly"]
        & ~candidate_summary["Ruined"]
    ].copy()
    if target_pool.empty:
        post_hoc_target_label = "MAX_FULL_CAGR_POST_HOC"
        post_hoc_target_id = int(
            valid.sort_values(
                [
                    "FullCAGRReportOnly",
                    "FullMaxDrawdownReportOnly",
                    "FullSharpeReportOnly",
                ],
                ascending=[False, False, False],
            ).iloc[0]["candidate_id"]
        )
    else:
        post_hoc_target_label = "TARGET_35_MIN_DRAWDOWN_POST_HOC"
        post_hoc_target_id = int(
            target_pool.sort_values(
                [
                    "FullMaxDrawdownReportOnly",
                    "FullSharpeReportOnly",
                    "FullAnnualizedTurnoverReportOnly",
                ],
                ascending=[False, False, True],
            ).iloc[0]["candidate_id"]
        )
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    selected_candidate = candidate_by_id[selected_id]
    post_hoc_target_candidate = candidate_by_id[post_hoc_target_id]
    report_ids: dict[str, int] = {
        "V7_3_BASELINE": 0,
        "V7_3_RISK_SELECTED": selected_id,
    }
    if max_train_id not in report_ids.values():
        report_ids["MAX_TRAIN_CAGR"] = max_train_id
    if post_hoc_target_id not in report_ids.values():
        report_ids[post_hoc_target_label] = post_hoc_target_id
    period_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    execution_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    period_report_labels = [
        "TRAIN_2020_2024",
        *[
            label
            for label in settings.validation_periods
            if label in periods
        ],
        "FULL_2020_2026",
    ]
    for series, candidate_id in report_ids.items():
        candidate = candidate_by_id[candidate_id]
        for label in period_report_labels:
            context = contexts[label]
            targets, diagnostics = _candidate_targets(
                context,
                returns_wide,
                risk_on,
                candidate,
                sector_map,
                optimization_config,
            )
            start, end = periods[label]
            result = run_signed_overlay_backtest(
                scoring_panel,
                targets,
                start=start,
                end=end,
                initial_capital=settings.initial_capital,
                transaction_cost_bps=settings.transaction_cost_bps,
                funding_annual_rate=float(
                    optimization_config["funding_annual_rate"]
                ),
                short_borrow_annual_rate=0.0,
                prepared_market=context.market,
            )
            spy_summary, _ = spy_buy_and_hold(
                spy,
                start=start,
                end=end,
                initial_capital=settings.initial_capital,
                transaction_cost_bps=settings.transaction_cost_bps,
            )
            summary = result.summary
            period_rows.append(
                {
                    "Series": series,
                    "CandidateID": candidate_id,
                    "CandidateName": candidate.name,
                    "Period": label,
                    "StartDate": summary.start_date,
                    "EndDate": summary.end_date,
                    "FinalValue": summary.final_value,
                    "ROI": summary.roi_percent,
                    "CAGR": summary.cagr_percent,
                    "Sharpe": summary.sharpe_ratio,
                    "MaxDrawdown": summary.max_drawdown_percent,
                    "AnnualizedTurnover": summary.annualized_turnover,
                    "FundingCost": result.total_funding_cost,
                    "AverageLongGross": float(
                        diagnostics["LongGross"].mean()
                    ),
                    "MaximumLongGross": float(
                        diagnostics["LongGross"].max()
                    ),
                    "Ruined": result.ruined,
                    "SPYROI": spy_summary["ROI"],
                    "SPYCAGR": spy_summary["CAGR"],
                    "SPYExcessROI": (
                        summary.roi_percent - float(spy_summary["ROI"])
                    ),
                }
            )
            equity = result.daily.copy()
            equity.insert(0, "Series", series)
            equity.insert(1, "CandidateID", candidate_id)
            equity.insert(2, "Period", label)
            equity_frames.append(equity)
            executions = result.executions.copy()
            executions.insert(0, "Series", series)
            executions.insert(1, "CandidateID", candidate_id)
            executions.insert(2, "Period", label)
            execution_frames.append(executions)
            targets.insert(0, "Series", series)
            targets.insert(1, "CandidateID", candidate_id)
            targets.insert(2, "Period", label)
            allocation_frames.append(targets)
            diagnostics.insert(0, "Series", series)
            diagnostics.insert(1, "CandidateID", candidate_id)
            diagnostics.insert(2, "Period", label)
            diagnostic_frames.append(diagnostics)
    stress_rows: list[dict[str, object]] = []
    stress_candidates = [("TRAIN_SELECTED", selected_id)]
    if post_hoc_target_id != selected_id:
        stress_candidates.append(
            ("FULL_TARGET_POST_HOC", post_hoc_target_id)
        )
    for selection_basis, candidate_id in stress_candidates:
        candidate = candidate_by_id[candidate_id]
        stress_targets, _ = _candidate_targets(
            full_context,
            returns_wide,
            risk_on,
            candidate,
            sector_map,
            optimization_config,
        )
        for cost_bps in optimization_config["stress_transaction_cost_bps"]:
            for funding_rate in optimization_config[
                "stress_funding_annual_rates"
            ]:
                result = run_signed_overlay_backtest(
                    scoring_panel,
                    stress_targets,
                    start=full_start,
                    end=full_end,
                    initial_capital=settings.initial_capital,
                    transaction_cost_bps=float(cost_bps),
                    funding_annual_rate=float(funding_rate),
                    short_borrow_annual_rate=0.0,
                    prepared_market=full_context.market,
                )
                stress_rows.append(
                    {
                        "SelectionBasis": selection_basis,
                        "CandidateID": candidate_id,
                        "CandidateName": candidate.name,
                        "TransactionCostBps": float(cost_bps),
                        "FundingAnnualRate": float(funding_rate),
                        "ROI": result.summary.roi_percent,
                        "CAGR": result.summary.cagr_percent,
                        "Sharpe": result.summary.sharpe_ratio,
                        "MaxDrawdown": (
                            result.summary.max_drawdown_percent
                        ),
                        "AnnualizedTurnover": (
                            result.summary.annualized_turnover
                        ),
                        "FundingCost": result.total_funding_cost,
                        "Ruined": result.ruined,
                    }
                )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v7_risk_optimization"
            / f"{timestamp}_v7_3_risk_weighted"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "data_audit_csv": destination / "data_audit.csv",
        "candidate_summary_csv": destination / "candidate_summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "stress_summary_csv": destination / "stress_summary.csv",
        "allocations_csv": destination / "allocations.csv",
        "allocation_diagnostics_csv": (
            destination / "allocation_diagnostics.csv"
        ),
        "equity_csv": destination / "equity.csv",
        "executions_csv": destination / "executions.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    atomic_to_csv(
        candidate_summary,
        outputs["candidate_summary_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.DataFrame(period_rows),
        outputs["period_summary_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.DataFrame(stress_rows),
        outputs["stress_summary_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(allocation_frames, ignore_index=True),
        outputs["allocations_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(diagnostic_frames, ignore_index=True),
        outputs["allocation_diagnostics_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(equity_frames, ignore_index=True),
        outputs["equity_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(execution_frames, ignore_index=True),
        outputs["executions_csv"],
        index=False,
    )
    frozen_payload = json.loads(
        Path(frozen_strategy_path).read_text(encoding="utf-8")
    )
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "V7-3 risk-adjusted allocation and volatility targeting",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "v7_3_stock_selection_unchanged": True,
            "variant": v7_variant.name,
            "technical_components": list(v7_variant.components),
            "top_k": params.top_k,
            "exit_rank": params.exit_rank,
            "selected_candidate_id": selected_id,
            "selected_candidate": asdict(selected_candidate),
            "maximum_train_cagr_candidate_id": max_train_id,
            "full_target_cagr_report_only": full_target_cagr,
            "post_hoc_target_label": post_hoc_target_label,
            "post_hoc_target_candidate_id": post_hoc_target_id,
            "post_hoc_target_candidate": asdict(
                post_hoc_target_candidate
            ),
            "candidate_count": len(candidates),
            "selection_rule": (
                "maximum predeclared robust training score among candidates "
                "passing drawdown, fold, and no-ruin constraints"
            ),
            "optimization_config": optimization_config,
            "sector_map": (
                str(Path(sector_map_path).resolve())
                if sector_map_path is not None
                else None
            ),
            "sector_map_ticker_count": len(sector_map),
            "weight_formula": (
                "exp(score_strength * selected-name AlphaScore z-score) "
                "* annualized_asset_volatility ** "
                "(-inverse_volatility_strength)"
            ),
            "volatility_formula": (
                "63-session covariance forecast with configured diagonal "
                "shrinkage, using information through signal close"
            ),
            "signal_rule": (
                "unchanged V7-3 final exchange session of each W-FRI week"
            ),
            "execution_rule": "next available trading session open",
            "funding_rule": (
                "negative cash charged by calendar day at configured rate"
            ),
            "transaction_cost_bps": settings.transaction_cost_bps,
            "frozen_v6_candidate": frozen_payload.get("selected_candidate"),
            "frozen_strategy_sha256": _sha256(
                Path(frozen_strategy_path)
            ),
            "membership_sha256": _sha256(Path(membership_path)),
            "membership_row_count": len(raw_membership),
            "membership_snapshot_count": membership_snapshot_count,
            "baseline_parity": {
                "expected": optimization_config["baseline_parity"],
                "actual": {
                    "roi": (
                        baseline_full_result.summary.roi_percent
                    ),
                    "cagr": (
                        baseline_full_result.summary.cagr_percent
                    ),
                    "max_drawdown": (
                        baseline_full_result.summary.max_drawdown_percent
                    ),
                    "sharpe": (
                        baseline_full_result.summary.sharpe_ratio
                    ),
                },
                "passed": True,
            },
            "ready_ticker_count": len(ready_tickers),
            "financial_point_in_time": False,
            "caveats": [
                "Macrotrends financials are current restated history.",
                "The 575-name panel excludes unavailable acquired/delisted names.",
                "2025 and 2026 were already observed and are not fresh OOS.",
                "The current-sector map is not historical point-in-time metadata.",
                "This optimization adds another multiple-testing layer.",
                "Leverage is modeled as notional exposure with borrowing cost.",
                (
                    "The target-CAGR candidate uses observed full-period "
                    "performance and is explicitly post-hoc, not selected OOS."
                ),
            ],
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return RiskOptimizationArtifacts(output_dir=destination, **outputs)


def _candidate_targets(
    context: _PeriodContext,
    returns_wide: pd.DataFrame,
    risk_on: pd.Series,
    candidate: RiskCandidate,
    sector_map: dict[str, str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_risk_adjusted_targets(
        context.v7_targets,
        returns_wide,
        risk_on,
        candidate,
        sector_map=sector_map,
        volatility_window=int(config["volatility_window"]),
        minimum_volatility_observations=int(
            config["minimum_volatility_observations"]
        ),
        covariance_shrinkage=float(config["covariance_shrinkage"]),
        asset_volatility_floor=float(config["asset_volatility_floor"]),
        max_ticker_weight=float(config["max_ticker_weight"]),
        max_sector_weight=float(config["max_sector_weight"]),
    )


def _build_period_context(
    panel: pd.DataFrame,
    params: StrategyParams,
    settings: ResearchSettings,
    *,
    start: str,
    end: str,
    reference_dates: pd.DatetimeIndex,
) -> _PeriodContext:
    signal_days = causal_signal_day_panel(
        panel,
        start=start,
        end=end,
        reference_market_dates=reference_dates,
        rebalance_weekday=settings.rebalance_weekday,
    )
    scored = score_panel(signal_days, params)
    return _PeriodContext(
        scored=scored,
        v7_targets=generate_rebalance_targets(scored, params),
        market=prepare_market(panel, start=start, end=end),
    )


def _robust_objective(
    train: SignedBacktestResult,
    *,
    median_fold_cagr: float,
    fold_cagr_std: float,
    config: dict[str, Any],
) -> float:
    summary = train.summary
    return float(
        float(config["train_cagr_weight"]) * summary.cagr_percent
        + float(config["median_fold_cagr_weight"]) * median_fold_cagr
        - float(config["fold_cagr_std_penalty"]) * fold_cagr_std
        + float(config["train_sharpe_weight"]) * summary.sharpe_ratio
        + float(config["train_mdd_weight"])
        * summary.max_drawdown_percent
        - float(config["turnover_penalty"]) * summary.annualized_turnover
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
