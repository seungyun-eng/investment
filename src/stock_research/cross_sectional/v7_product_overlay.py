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

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths


@dataclass(frozen=True)
class ProductOverlayCandidate:
    candidate_id: int
    name: str
    product_type: str
    sma_window: int
    sma_buffer: float
    sma_slope_lookback: int
    vix_ceiling: float
    base_drawdown_floor: float
    risk_on_product_weight: float
    risk_off_product_weight: float
    risk_off_cash_weight: float


@dataclass(frozen=True)
class ProductOverlayArtifacts:
    output_dir: Path
    candidate_summary_csv: Path
    period_summary_csv: Path
    stress_summary_csv: Path
    calendar_returns_csv: Path
    daily_csv: Path
    regime_diagnostics_csv: Path
    manifest_json: Path


def generate_product_candidates(
    config: dict[str, Any],
) -> list[ProductOverlayCandidate]:
    candidates = [
        ProductOverlayCandidate(
            candidate_id=0,
            name="BASE_V7_1X",
            product_type="BASE_V7_1X",
            sma_window=0,
            sma_buffer=0.0,
            sma_slope_lookback=0,
            vix_ceiling=float("inf"),
            base_drawdown_floor=-1.0,
            risk_on_product_weight=0.0,
            risk_off_product_weight=0.0,
            risk_off_cash_weight=0.0,
        )
    ]
    candidate_id = 1
    grid = config["candidate_grid"]
    for product_type in grid["product_types"]:
        for sma_window in grid["sma_windows"]:
            for sma_buffer in grid.get("sma_buffers", [0.0]):
                for slope_lookback in grid["sma_slope_lookbacks"]:
                    for vix_ceiling in grid["vix_ceilings"]:
                        for drawdown_floor in grid["base_drawdown_floors"]:
                            for risk_on_weight in grid[
                                "risk_on_product_weights"
                            ]:
                                for risk_off_weight in grid[
                                    "risk_off_product_weights"
                                ]:
                                    for risk_off_cash_weight in grid[
                                        "risk_off_cash_weights"
                                    ]:
                                        if risk_off_weight > risk_on_weight:
                                            continue
                                        if (
                                            risk_off_weight
                                            + risk_off_cash_weight
                                            > 1.0
                                        ):
                                            continue
                                        name = (
                                            f"{product_type}__SMA{sma_window}"
                                            f"_BUFFER{float(sma_buffer):.3f}"
                                            f"_SLOPE{slope_lookback}"
                                            f"_VIX{float(vix_ceiling):g}"
                                            f"_DD{abs(float(drawdown_floor)):.2f}"
                                            f"_ON{float(risk_on_weight):.3f}"
                                            f"_OFF{float(risk_off_weight):.3f}"
                                            f"_CASH{float(risk_off_cash_weight):.3f}"
                                        )
                                        candidates.append(
                                            ProductOverlayCandidate(
                                                candidate_id=candidate_id,
                                                name=name,
                                                product_type=str(product_type),
                                                sma_window=int(sma_window),
                                                sma_buffer=float(sma_buffer),
                                                sma_slope_lookback=int(
                                                    slope_lookback
                                                ),
                                                vix_ceiling=float(vix_ceiling),
                                                base_drawdown_floor=float(
                                                    drawdown_floor
                                                ),
                                                risk_on_product_weight=float(
                                                    risk_on_weight
                                                ),
                                                risk_off_product_weight=float(
                                                    risk_off_weight
                                                ),
                                                risk_off_cash_weight=float(
                                                    risk_off_cash_weight
                                                ),
                                            )
                                        )
                                        candidate_id += 1
    return candidates


def build_causal_risk_on(
    frame: pd.DataFrame,
    candidate: ProductOverlayCandidate,
) -> pd.Series:
    if candidate.product_type == "BASE_V7_1X":
        return pd.Series(False, index=frame.index, dtype=bool)
    trend = frame["SPYClose"].gt(
        frame[f"SPYSMA{candidate.sma_window}"]
        * (1.0 + candidate.sma_buffer)
    )
    if candidate.sma_slope_lookback > 0:
        trend &= frame[
            f"SPYSMA{candidate.sma_window}Slope"
            f"{candidate.sma_slope_lookback}"
        ].gt(0.0)
    observed_risk_on = (
        trend
        & frame["VIX"].le(candidate.vix_ceiling)
        & frame["BaseDrawdown"].ge(candidate.base_drawdown_floor)
    )
    return observed_risk_on.shift(1, fill_value=False).astype(bool)


def _cash_returns(frame: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(frame["Date"]).reset_index(drop=True)
    annual_rates = (
        pd.to_numeric(frame["CashRate"], errors="raise")
        .to_numpy(dtype=float)
        / 100.0
    )
    returns = np.zeros(len(frame), dtype=float)
    if len(frame) < 2:
        return returns
    elapsed_days = dates.diff().dt.days.fillna(0).to_numpy(dtype=float)
    prior_rates = np.maximum(annual_rates[:-1], -0.99)
    returns[1:] = (1.0 + prior_rates) ** (
        elapsed_days[1:] / 365.25
    ) - 1.0
    return returns


def simulate_product_overlay(
    frame: pd.DataFrame,
    candidate: ProductOverlayCandidate,
    *,
    initial_capital: float,
    annual_expense_ratio: float,
    annual_financing_spread: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    base_returns = frame["BaseReturn"].to_numpy(dtype=float)
    risk_on = build_causal_risk_on(frame, candidate)
    sleeve_weight = np.where(
        risk_on.to_numpy(dtype=bool),
        candidate.risk_on_product_weight,
        candidate.risk_off_product_weight,
    ).astype(float)
    cash_weight = np.where(
        risk_on.to_numpy(dtype=bool),
        0.0,
        candidate.risk_off_cash_weight,
    ).astype(float)
    if candidate.product_type == "BASE_V7_1X":
        sleeve_weight[:] = 0.0
        cash_weight[:] = 0.0
        product_returns = base_returns
    else:
        if candidate.product_type not in {
            "SYNTHETIC_V7_2X",
            "SYNTHETIC_SPY_2X",
        }:
            raise ValueError(
                f"Unsupported product type: {candidate.product_type}"
            )
        underlying_returns = (
            base_returns
            if candidate.product_type == "SYNTHETIC_V7_2X"
            else frame["SPYReturn"].to_numpy(dtype=float)
        )
        annual_cost = (
            annual_expense_ratio
            + np.maximum(frame["CashRate"].to_numpy(dtype=float), 0.0)
            / 100.0
            + annual_financing_spread
        )
        product_returns = 2.0 * underlying_returns - annual_cost / 252.0
    sleeve_weight[0] = 0.0
    cash_weight[0] = 0.0
    base_weight = 1.0 - sleeve_weight - cash_weight
    if (base_weight < -1e-12).any():
        raise ValueError("Product and cash weights cannot exceed capital")
    cash_returns = _cash_returns(frame)
    prior_base_weight = np.r_[1.0, base_weight[:-1]]
    prior_product_weight = np.r_[0.0, sleeve_weight[:-1]]
    prior_cash_weight = np.r_[0.0, cash_weight[:-1]]
    traded_fraction = 0.5 * (
        np.abs(base_weight - prior_base_weight)
        + np.abs(sleeve_weight - prior_product_weight)
        + np.abs(cash_weight - prior_cash_weight)
    )
    switch_cost = traded_fraction * transaction_cost_bps / 10_000.0
    adjusted_returns = (
        base_weight * base_returns
        + sleeve_weight * product_returns
        + cash_weight * cash_returns
        - switch_cost
    )
    adjusted_returns = np.maximum(adjusted_returns, -0.999)
    equity = initial_capital * np.cumprod(1.0 + adjusted_returns)
    output = frame[["Date"]].copy()
    output["BaseReturn"] = base_returns
    output["ProductReturn"] = product_returns
    output["AdjustedReturn"] = adjusted_returns
    output["BaseWeight"] = base_weight
    output["SleeveWeight"] = sleeve_weight
    output["CashWeight"] = cash_weight
    output["EffectiveExposure"] = base_weight + 2.0 * sleeve_weight
    output["RiskOn"] = risk_on.to_numpy(dtype=bool)
    output["TradedFraction"] = traded_fraction
    output["SwitchCost"] = switch_cost
    output["Equity"] = equity
    return output


def summarize_period(
    daily: pd.DataFrame,
    *,
    start: str,
    end: str,
    initial_capital: float,
) -> dict[str, object]:
    period = daily.loc[daily["Date"].between(start, end)].copy()
    if period.empty:
        raise ValueError(f"No overlay observations between {start} and {end}")
    returns = period["AdjustedReturn"].to_numpy(dtype=float).copy()
    returns[0] = 0.0
    equity = initial_capital * np.cumprod(1.0 + returns)
    start_date = pd.Timestamp(period.iloc[0]["Date"])
    end_date = pd.Timestamp(period.iloc[-1]["Date"])
    years = max((end_date - start_date).days / 365.25, 1.0 / 252.0)
    final_value = float(equity[-1])
    roi = (final_value / initial_capital - 1.0) * 100.0
    cagr = (
        (final_value / initial_capital) ** (1.0 / years) - 1.0
    ) * 100.0
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    active_returns = returns[1:]
    volatility = (
        float(np.std(active_returns, ddof=1))
        if len(active_returns) >= 2
        else 0.0
    )
    sharpe = (
        float(np.mean(active_returns) / volatility * np.sqrt(252.0))
        if volatility > 0
        else 0.0
    )
    sleeve = period["SleeveWeight"].to_numpy(dtype=float)
    cash = period["CashWeight"].to_numpy(dtype=float)
    overlay_turnover = float(
        period["TradedFraction"].sum() / years
    )
    return {
        "StartDate": start_date,
        "EndDate": end_date,
        "FinalValue": final_value,
        "ROI": float(roi),
        "CAGR": float(cagr),
        "Sharpe": sharpe,
        "MaxDrawdown": float(drawdown.min() * 100.0),
        "AverageProductWeight": float(sleeve.mean()),
        "MaximumProductWeight": float(sleeve.max()),
        "AverageCashWeight": float(cash.mean()),
        "MaximumCashWeight": float(cash.max()),
        "AverageEffectiveExposure": float((1.0 + sleeve).mean()),
        "RiskOnDays": int(period["RiskOn"].sum()),
        "OverlayTurnover": overlay_turnover,
        "TotalSwitchCostPct": float(period["SwitchCost"].sum() * 100.0),
    }


def summarize_calendar_consistency(
    daily: pd.DataFrame,
    *,
    start: str,
    end: str,
    target_floor: float,
    target_ceiling: float,
    rolling_sessions: int = 252,
) -> dict[str, object]:
    """Summarize calendar-year and rolling-return consistency.

    Calendar returns remain continuous with the full simulation.  The method
    therefore does not reset holdings or equity at each January boundary.
    """
    if target_ceiling < target_floor:
        raise ValueError("target_ceiling cannot be below target_floor")
    if rolling_sessions < 2:
        raise ValueError("rolling_sessions must be at least two")
    period_mask = daily["Date"].between(start, end)
    period = daily.loc[period_mask].copy()
    if period.empty:
        raise ValueError(
            f"No consistency observations between {start} and {end}"
        )
    annual_returns = (
        (1.0 + period["AdjustedReturn"])
        .groupby(period["Date"].dt.year)
        .prod()
        .sub(1.0)
        .mul(100.0)
    )
    values = annual_returns.to_numpy(dtype=float)
    downside = np.maximum(target_floor - values, 0.0)
    upside = np.maximum(values - target_ceiling, 0.0)
    equity = pd.to_numeric(daily["Equity"], errors="raise")
    rolling = (equity / equity.shift(rolling_sessions) - 1.0) * 100.0
    rolling_period = rolling.loc[period_mask & rolling.notna()]
    return {
        "CalendarYearCount": len(values),
        "CalendarWorstReturn": float(values.min()),
        "CalendarMeanReturn": float(values.mean()),
        "CalendarReturnStd": float(np.std(values, ddof=0)),
        "CalendarTargetShortfallRMS": float(
            np.sqrt(np.mean(np.square(downside)))
        ),
        "CalendarTargetExcessRMS": float(
            np.sqrt(np.mean(np.square(upside)))
        ),
        "CalendarYearsAtOrAboveFloor": int(
            np.count_nonzero(values >= target_floor)
        ),
        "CalendarYearsWithinTarget": int(
            np.count_nonzero(
                (values >= target_floor) & (values <= target_ceiling)
            )
        ),
        "CalendarFloorMetEveryYear": bool(
            np.all(values >= target_floor)
        ),
        "RollingSessionCount": int(rolling_period.notna().sum()),
        "RollingWorstReturn": (
            float(rolling_period.min())
            if not rolling_period.empty
            else float("nan")
        ),
    }


def summarize_benchmark_consistency(
    daily: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    start: str,
    end: str,
    rolling_sessions: int = 252,
) -> dict[str, object]:
    """Summarize calendar and rolling percentage-point benchmark alpha."""
    if rolling_sessions < 2:
        raise ValueError("rolling_sessions must be at least two")
    if len(daily) != len(benchmark_returns):
        raise ValueError("daily and benchmark_returns must have equal length")
    benchmark = pd.to_numeric(
        benchmark_returns.reset_index(drop=True),
        errors="raise",
    ).copy()
    if benchmark.isna().any():
        raise ValueError("benchmark_returns cannot contain missing values")
    benchmark.iloc[0] = 0.0
    dates = pd.to_datetime(daily["Date"], errors="raise").reset_index(
        drop=True
    )
    strategy_returns = pd.to_numeric(
        daily["AdjustedReturn"].reset_index(drop=True),
        errors="raise",
    )
    period_mask = dates.between(start, end)
    if not period_mask.any():
        raise ValueError(
            f"No benchmark observations between {start} and {end}"
        )
    strategy_annual = (
        (1.0 + strategy_returns.loc[period_mask])
        .groupby(dates.loc[period_mask].dt.year)
        .prod()
        .sub(1.0)
        .mul(100.0)
    )
    benchmark_annual = (
        (1.0 + benchmark.loc[period_mask])
        .groupby(dates.loc[period_mask].dt.year)
        .prod()
        .sub(1.0)
        .mul(100.0)
    )
    alpha = strategy_annual - benchmark_annual
    downside = np.maximum(-alpha.to_numpy(dtype=float), 0.0)
    benchmark_period_equity = (
        1.0 + benchmark.loc[period_mask]
    ).cumprod()
    benchmark_period_drawdown = (
        benchmark_period_equity
        / benchmark_period_equity.cummax()
        - 1.0
    )
    strategy_equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + benchmark).cumprod()
    strategy_rolling = (
        strategy_equity / strategy_equity.shift(rolling_sessions) - 1.0
    ) * 100.0
    benchmark_rolling = (
        benchmark_equity / benchmark_equity.shift(rolling_sessions) - 1.0
    ) * 100.0
    rolling_alpha = (strategy_rolling - benchmark_rolling).loc[
        period_mask
        & strategy_rolling.notna()
        & benchmark_rolling.notna()
    ]
    output: dict[str, object] = {
        "CalendarWorstReturn": float(strategy_annual.min()),
        "BenchmarkCalendarYearCount": len(alpha),
        "BenchmarkWorstAlpha": float(alpha.min()),
        "BenchmarkMeanAlpha": float(alpha.mean()),
        "BenchmarkAlphaStd": float(np.std(alpha.to_numpy(), ddof=0)),
        "BenchmarkAlphaShortfallRMS": float(
            np.sqrt(np.mean(np.square(downside)))
        ),
        "BenchmarkMaxDrawdown": float(
            benchmark_period_drawdown.min() * 100.0
        ),
        "BenchmarkYearsOutperformed": int(np.count_nonzero(alpha.gt(0.0))),
        "BenchmarkOutperformedEveryYear": bool(alpha.gt(0.0).all()),
        "BenchmarkRollingSessionCount": int(rolling_alpha.notna().sum()),
        "BenchmarkRollingWorstAlpha": (
            float(rolling_alpha.min())
            if not rolling_alpha.empty
            else float("nan")
        ),
    }
    for year in alpha.index:
        output[f"CalendarReturn{int(year)}"] = float(
            strategy_annual.loc[year]
        )
        output[f"BenchmarkReturn{int(year)}"] = float(
            benchmark_annual.loc[year]
        )
        output[f"CalendarAlpha{int(year)}"] = float(alpha.loc[year])
    return output


def run_v7_product_overlay_optimization(
    paths: ProjectPaths,
    *,
    base_equity_path: str | Path,
    base_series: str,
    base_period: str,
    spy_path: str | Path,
    vix_path: str | Path,
    fed_funds_path: str | Path,
    optimization_config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> ProductOverlayArtifacts:
    candidates = generate_product_candidates(optimization_config)
    frame, input_audit = _prepare_inputs(
        base_equity_path,
        base_series=base_series,
        base_period=base_period,
        spy_path=spy_path,
        vix_path=vix_path,
        fed_funds_path=fed_funds_path,
        sma_windows=optimization_config["candidate_grid"]["sma_windows"],
        slope_lookbacks=optimization_config["candidate_grid"][
            "sma_slope_lookbacks"
        ],
    )
    initial_capital = float(optimization_config["initial_capital"])
    expense_ratio = float(optimization_config["annual_expense_ratio"])
    financing_spread = float(
        optimization_config["annual_financing_spread"]
    )
    transaction_cost_bps = float(
        optimization_config["transaction_cost_bps"]
    )
    periods = {
        key: tuple(value)
        for key, value in optimization_config["periods"].items()
    }
    training_folds = {
        key: tuple(value)
        for key, value in optimization_config["training_folds"].items()
    }
    consistency_config = optimization_config.get("annual_consistency")
    benchmark_config = optimization_config.get("benchmark_consistency")
    baseline_daily = simulate_product_overlay(
        frame,
        candidates[0],
        initial_capital=initial_capital,
        annual_expense_ratio=expense_ratio,
        annual_financing_spread=financing_spread,
        transaction_cost_bps=transaction_cost_bps,
    )
    baseline_full = summarize_period(
        baseline_daily,
        start=periods["FULL"][0],
        end=periods["FULL"][1],
        initial_capital=initial_capital,
    )
    _assert_baseline_parity(
        baseline_full,
        optimization_config["baseline_parity"],
    )
    print(
        "PRODUCT BASELINE PARITY PASS "
        f"ROI={baseline_full['ROI']:.6f}% "
        f"CAGR={baseline_full['CAGR']:.6f}%",
        flush=True,
    )

    candidate_rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        daily = simulate_product_overlay(
            frame,
            candidate,
            initial_capital=initial_capital,
            annual_expense_ratio=expense_ratio,
            annual_financing_spread=financing_spread,
            transaction_cost_bps=transaction_cost_bps,
        )
        train = summarize_period(
            daily,
            start=periods["TRAIN"][0],
            end=periods["TRAIN"][1],
            initial_capital=initial_capital,
        )
        folds = [
            summarize_period(
                daily,
                start=start,
                end=end,
                initial_capital=initial_capital,
            )
            for start, end in training_folds.values()
        ]
        full = summarize_period(
            daily,
            start=periods["FULL"][0],
            end=periods["FULL"][1],
            initial_capital=initial_capital,
        )
        fold_rois = [float(metrics["ROI"]) for metrics in folds]
        fold_cagrs = [float(metrics["CAGR"]) for metrics in folds]
        median_fold_cagr = float(np.median(fold_cagrs))
        fold_cagr_std = float(np.std(fold_cagrs, ddof=0))
        positive_folds = int(sum(value > 0 for value in fold_rois))
        pass_constraints = bool(
            float(train["MaxDrawdown"])
            >= float(optimization_config["minimum_train_mdd"])
            and positive_folds
            >= int(optimization_config["minimum_positive_folds"])
            and min(fold_rois)
            >= float(optimization_config["minimum_worst_fold_roi"])
        )
        row: dict[str, object] = {
                **asdict(candidate),
                "TrainROI": train["ROI"],
                "TrainCAGR": train["CAGR"],
                "TrainSharpe": train["Sharpe"],
                "TrainMaxDrawdown": train["MaxDrawdown"],
                "TrainAverageProductWeight": train[
                    "AverageProductWeight"
                ],
                "TrainAverageCashWeight": train["AverageCashWeight"],
                "TrainOverlayTurnover": train["OverlayTurnover"],
                "MedianFoldROI": float(np.median(fold_rois)),
                "WorstFoldROI": float(min(fold_rois)),
                "MedianFoldCAGR": median_fold_cagr,
                "FoldCAGRStd": fold_cagr_std,
                "PositiveFolds": positive_folds,
                "RobustScore": _robust_score(
                    train,
                    median_fold_cagr=median_fold_cagr,
                    fold_cagr_std=fold_cagr_std,
                    config=optimization_config["objective"],
                ),
                "PassConstraints": pass_constraints,
                "FullROIReportOnly": full["ROI"],
                "FullCAGRReportOnly": full["CAGR"],
                "FullSharpeReportOnly": full["Sharpe"],
                "FullMaxDrawdownReportOnly": full["MaxDrawdown"],
                "FullAverageProductWeightReportOnly": full[
                    "AverageProductWeight"
                ],
                "FullAverageCashWeightReportOnly": full[
                    "AverageCashWeight"
                ],
                "FullAverageEffectiveExposureReportOnly": full[
                    "AverageEffectiveExposure"
                ],
                "FullOverlayTurnoverReportOnly": full["OverlayTurnover"],
                "FullTotalSwitchCostPctReportOnly": full[
                    "TotalSwitchCostPct"
                ],
            }
        if consistency_config is not None:
            consistency = summarize_calendar_consistency(
                daily,
                start=periods["TRAIN"][0],
                end=periods["TRAIN"][1],
                target_floor=float(consistency_config["target_floor"]),
                target_ceiling=float(consistency_config["target_ceiling"]),
                rolling_sessions=int(
                    consistency_config.get("rolling_sessions", 252)
                ),
            )
            row.update(
                {
                    f"Train{key}": value
                    for key, value in consistency.items()
                }
            )
            row["AnnualConsistencyScore"] = _calendar_consistency_score(
                consistency,
                consistency_config["objective"],
            )
            full_calendar = (
                (1.0 + daily["AdjustedReturn"])
                .groupby(daily["Date"].dt.year)
                .prod()
                .sub(1.0)
                .mul(100.0)
            )
            row["Calendar2025ReturnReportOnly"] = float(
                full_calendar.get(2025, float("nan"))
            )
            row["Calendar2026PartialReturnReportOnly"] = float(
                full_calendar.get(2026, float("nan"))
            )
        if benchmark_config is not None:
            benchmark_consistency = summarize_benchmark_consistency(
                daily,
                frame["SPYReturn"],
                start=periods["TRAIN"][0],
                end=periods["TRAIN"][1],
                rolling_sessions=int(
                    benchmark_config.get("rolling_sessions", 252)
                ),
            )
            row.update(
                {
                    f"Train{key}": value
                    for key, value in benchmark_consistency.items()
                }
            )
            row["BenchmarkConsistencyScore"] = (
                _benchmark_consistency_score(
                    benchmark_consistency,
                    benchmark_config["objective"],
                )
            )
            report_benchmark = summarize_benchmark_consistency(
                daily,
                frame["SPYReturn"],
                start=periods["FULL"][0],
                end=periods["FULL"][1],
                rolling_sessions=int(
                    benchmark_config.get("rolling_sessions", 252)
                ),
            )
            row["Calendar2025AlphaReportOnly"] = report_benchmark.get(
                "CalendarAlpha2025",
                float("nan"),
            )
            row["Calendar2026PartialAlphaReportOnly"] = (
                report_benchmark.get(
                    "CalendarAlpha2026",
                    float("nan"),
                )
            )
        candidate_rows.append(row)
        if index % 500 == 0 or index == len(candidates):
            print(
                f"PRODUCT {index}/{len(candidates)} candidates evaluated",
                flush=True,
            )

    candidate_summary = pd.DataFrame(candidate_rows)
    valid = candidate_summary.loc[
        candidate_summary["PassConstraints"]
    ].copy()
    if valid.empty:
        raise RuntimeError("No product overlay candidate passed constraints")
    selected_id = int(
        valid.sort_values(
            ["RobustScore", "MedianFoldCAGR", "TrainCAGR"],
            ascending=[False, False, False],
        ).iloc[0]["candidate_id"]
    )
    annual_consistency_id: int | None = None
    annual_target_feasible: bool | None = None
    if consistency_config is not None:
        annual_target_feasible = bool(
            valid["TrainCalendarFloorMetEveryYear"].any()
        )
        annual_consistency_id = int(
            valid.sort_values(
                [
                    "TrainCalendarFloorMetEveryYear",
                    "TrainCalendarWorstReturn",
                    "TrainCalendarTargetShortfallRMS",
                    "TrainCalendarReturnStd",
                    "AnnualConsistencyScore",
                ],
                ascending=[False, False, True, True, False],
            ).iloc[0]["candidate_id"]
        )
    benchmark_consistency_id: int | None = None
    benchmark_target_feasible: bool | None = None
    if benchmark_config is not None:
        minimum_mdd_advantage = float(
            benchmark_config.get("minimum_mdd_advantage", 0.0)
        )
        benchmark_risk_pool = valid.loc[
            valid["TrainMaxDrawdown"]
            >= (
                valid["TrainBenchmarkMaxDrawdown"]
                + minimum_mdd_advantage
            )
        ].copy()
        if benchmark_risk_pool.empty:
            raise RuntimeError(
                "No candidate met the configured benchmark MDD advantage"
            )
        benchmark_joint_target_count = int(
            benchmark_risk_pool[
                "TrainBenchmarkOutperformedEveryYear"
            ].sum()
        )
        benchmark_target_feasible = benchmark_joint_target_count > 0
        benchmark_consistency_id = int(
            benchmark_risk_pool.sort_values(
                [
                    "TrainBenchmarkOutperformedEveryYear",
                    "TrainBenchmarkWorstAlpha",
                    "TrainCalendarWorstReturn",
                    "TrainBenchmarkAlphaShortfallRMS",
                    "TrainMaxDrawdown",
                    "TrainBenchmarkRollingWorstAlpha",
                    "TrainBenchmarkAlphaStd",
                    "BenchmarkConsistencyScore",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                ],
            ).iloc[0]["candidate_id"]
        )
    spy_valid = valid.loc[
        valid["product_type"].eq("SYNTHETIC_SPY_2X")
    ]
    best_spy_id = (
        int(
            spy_valid.sort_values("RobustScore", ascending=False).iloc[0][
                "candidate_id"
            ]
        )
        if not spy_valid.empty
        else None
    )
    target_low = float(optimization_config["target_full_cagr_band"][0])
    target_high = float(optimization_config["target_full_cagr_band"][1])
    target_pool = valid.loc[
        valid["FullCAGRReportOnly"].between(target_low, target_high)
        & valid["FullMaxDrawdownReportOnly"].ge(
            float(optimization_config["target_full_mdd_floor"])
        )
    ].copy()
    if target_pool.empty:
        above_target = valid.loc[
            valid["FullCAGRReportOnly"].ge(target_low)
        ].copy()
        if above_target.empty:
            nearest = valid.copy()
            nearest["TargetGap"] = (
                nearest["FullCAGRReportOnly"] - target_low
            ).abs()
            target_id = int(
                nearest.sort_values(
                    ["TargetGap", "FullMaxDrawdownReportOnly"],
                    ascending=[True, False],
                ).iloc[0]["candidate_id"]
            )
        else:
            target_id = int(
                above_target.sort_values(
                    [
                        "FullMaxDrawdownReportOnly",
                        "FullSharpeReportOnly",
                    ],
                    ascending=[False, False],
                ).iloc[0]["candidate_id"]
            )
    else:
        target_id = int(
            target_pool.sort_values(
                ["FullMaxDrawdownReportOnly", "FullSharpeReportOnly"],
                ascending=[False, False],
            ).iloc[0]["candidate_id"]
        )
    mdd_pool = valid.loc[
        valid["FullCAGRReportOnly"].ge(
            float(optimization_config["mdd_search_minimum_full_cagr"])
        )
    ].copy()
    if mdd_pool.empty:
        mdd_pool = valid.copy()
    mdd_id = int(
        mdd_pool.sort_values(
            ["FullMaxDrawdownReportOnly", "FullCAGRReportOnly"],
            ascending=[False, False],
        ).iloc[0]["candidate_id"]
    )
    by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    requested_report_ids = [
        ("BASE_V7_1X", 0),
        ("TRAIN_SELECTED", selected_id),
        ("TARGET_40_45_POST_HOC", target_id),
        ("BEST_MDD_AT_35_POST_HOC", mdd_id),
    ]
    if best_spy_id is not None:
        requested_report_ids.append(("BEST_SPY_2X_PROXY", best_spy_id))
    report_ids = _unique_report_ids(requested_report_ids)
    if annual_consistency_id is not None:
        report_ids.append(
            ("ANNUAL_CONSISTENCY_TRAIN_SELECTED", annual_consistency_id)
        )
    if benchmark_consistency_id is not None:
        report_ids.append(
            ("SPY_RELATIVE_TRAIN_SELECTED", benchmark_consistency_id)
        )
    period_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    calendar_rows: list[dict[str, object]] = []
    for series, candidate_id in report_ids:
        candidate = by_id[candidate_id]
        daily = simulate_product_overlay(
            frame,
            candidate,
            initial_capital=initial_capital,
            annual_expense_ratio=expense_ratio,
            annual_financing_spread=financing_spread,
            transaction_cost_bps=transaction_cost_bps,
        )
        for label, (start, end) in periods.items():
            metrics = summarize_period(
                daily,
                start=start,
                end=end,
                initial_capital=initial_capital,
            )
            period_rows.append(
                {
                    "Series": series,
                    "CandidateID": candidate_id,
                    "CandidateName": candidate.name,
                    "ProductType": candidate.product_type,
                    "Period": label,
                    **metrics,
                }
            )
        full_daily = daily.loc[
            daily["Date"].between(*periods["FULL"])
        ].copy()
        full_daily.insert(0, "Series", series)
        full_daily.insert(1, "CandidateID", candidate_id)
        daily_frames.append(full_daily)
        calendar_rows.extend(
            _calendar_returns(
                full_daily,
                series=series,
                candidate_id=candidate_id,
                initial_capital=initial_capital,
            )
        )

    stress_rows: list[dict[str, object]] = []
    stress_ids = _unique_report_ids(
        [
            ("TARGET_40_45_POST_HOC", target_id),
            ("BEST_MDD_AT_35_POST_HOC", mdd_id),
            ("TRAIN_SELECTED", selected_id),
            *(
                [
                    (
                        "ANNUAL_CONSISTENCY_TRAIN_SELECTED",
                        annual_consistency_id,
                    )
                ]
                if annual_consistency_id is not None
                else []
            ),
            *(
                [
                    (
                        "SPY_RELATIVE_TRAIN_SELECTED",
                        benchmark_consistency_id,
                    )
                ]
                if benchmark_consistency_id is not None
                else []
            ),
        ]
    )
    for series, candidate_id in stress_ids:
        candidate = by_id[candidate_id]
        for cost_bps in optimization_config[
            "stress_transaction_cost_bps"
        ]:
            for stressed_expense in optimization_config[
                "stress_annual_expense_ratios"
            ]:
                for stressed_spread in optimization_config[
                    "stress_annual_financing_spreads"
                ]:
                    daily = simulate_product_overlay(
                        frame,
                        candidate,
                        initial_capital=initial_capital,
                        annual_expense_ratio=float(stressed_expense),
                        annual_financing_spread=float(stressed_spread),
                        transaction_cost_bps=float(cost_bps),
                    )
                    metrics = summarize_period(
                        daily,
                        start=periods["FULL"][0],
                        end=periods["FULL"][1],
                        initial_capital=initial_capital,
                    )
                    stress_rows.append(
                        {
                            "Series": series,
                            "CandidateID": candidate_id,
                            "CandidateName": candidate.name,
                            "TransactionCostBps": float(cost_bps),
                            "AnnualExpenseRatio": float(stressed_expense),
                            "AnnualFinancingSpread": float(stressed_spread),
                            **metrics,
                        }
                    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v7_product_overlay"
            / f"{timestamp}_v7_3_daily_reset_2x"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_summary_csv": destination / "candidate_summary.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "stress_summary_csv": destination / "stress_summary.csv",
        "calendar_returns_csv": destination / "calendar_returns.csv",
        "daily_csv": destination / "daily.csv",
        "regime_diagnostics_csv": destination / "regime_diagnostics.csv",
        "manifest_json": destination / "manifest.json",
    }
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
        pd.DataFrame(calendar_rows),
        outputs["calendar_returns_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(daily_frames, ignore_index=True),
        outputs["daily_csv"],
        index=False,
    )
    diagnostic_columns = [
        column
        for column in frame.columns
        if column != "BaseEquity"
    ]
    atomic_to_csv(
        frame[diagnostic_columns],
        outputs["regime_diagnostics_csv"],
        index=False,
    )
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "V7-3 cash-funded daily-reset 2x product overlay",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "base_series": base_series,
            "base_period": base_period,
            "candidate_count": len(candidates),
            "selected_candidate_id": selected_id,
            "selected_candidate": asdict(by_id[selected_id]),
            "annual_consistency_candidate_id": annual_consistency_id,
            "annual_consistency_candidate": (
                asdict(by_id[annual_consistency_id])
                if annual_consistency_id is not None
                else None
            ),
            "annual_target_feasible_on_training_candidates": (
                annual_target_feasible
            ),
            "annual_consistency_selection_rule": (
                "training-only lexicographic order: every-year floor met, "
                "worst calendar return, target shortfall, calendar-return "
                "standard deviation, then configured score"
                if consistency_config is not None
                else None
            ),
            "benchmark_consistency_candidate_id": (
                benchmark_consistency_id
            ),
            "benchmark_consistency_candidate": (
                asdict(by_id[benchmark_consistency_id])
                if benchmark_consistency_id is not None
                else None
            ),
            "benchmark_target_feasible_on_training_candidates": (
                benchmark_target_feasible
            ),
            "benchmark_minimum_mdd_advantage": (
                float(benchmark_config.get("minimum_mdd_advantage", 0.0))
                if benchmark_config is not None
                else None
            ),
            "benchmark_consistency_selection_rule": (
                "training-only filter: candidate MDD must beat benchmark MDD "
                "by the configured minimum advantage; lexicographic order: "
                "outperform SPY every calendar year, worst annual "
                "percentage-point alpha, worst absolute calendar return, "
                "alpha shortfall, MDD, rolling 252-day alpha, alpha standard "
                "deviation, then configured score"
                if benchmark_config is not None
                else None
            ),
            "target_40_45_post_hoc_candidate_id": target_id,
            "target_40_45_post_hoc_candidate": asdict(by_id[target_id]),
            "best_mdd_at_35_post_hoc_candidate_id": mdd_id,
            "best_mdd_at_35_post_hoc_candidate": asdict(by_id[mdd_id]),
            "best_spy_2x_proxy_candidate_id": best_spy_id,
            "best_spy_2x_proxy_candidate": (
                asdict(by_id[best_spy_id])
                if best_spy_id is not None
                else None
            ),
            "baseline_parity": {
                "expected": optimization_config["baseline_parity"],
                "actual": baseline_full,
                "passed": True,
            },
            "product_model": {
                "daily_reset": True,
                "cash_funded_mix": (
                    "(1 - sleeve_weight) * V7_1X + sleeve_weight * 2X"
                ),
                "negative_cash_or_margin_loan": False,
                "annual_expense_ratio": expense_ratio,
                "annual_financing_cost": (
                    "point-in-time Fed funds rate + configured spread"
                ),
                "execution_timing": (
                    "prior-close regime determines next session return sleeve"
                ),
            },
            "input_audit": input_audit,
            "optimization_config": optimization_config,
            "caveats": [
                (
                    "SYNTHETIC_V7_2X is a hypothetical daily-reset wrapper; "
                    "one exchange-traded product covering the changing V7 "
                    "basket is not assumed to exist."
                ),
                (
                    "SYNTHETIC_SPY_2X is a fee-and-financing proxy, not an "
                    "actual leveraged ETF price series or execution test."
                ),
                (
                    "Daily reset can cause long-horizon returns to differ "
                    "materially from two times the underlying return."
                ),
                (
                    "2025 and 2026 are already observed; all full-period "
                    "target and MDD labels are explicitly post-hoc."
                ),
                (
                    "The underlying V7 financial history remains restated "
                    "rather than true point-in-time data."
                ),
            ],
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return ProductOverlayArtifacts(output_dir=destination, **outputs)


def _prepare_inputs(
    base_equity_path: str | Path,
    *,
    base_series: str,
    base_period: str,
    spy_path: str | Path,
    vix_path: str | Path,
    fed_funds_path: str | Path,
    sma_windows: list[int],
    slope_lookbacks: list[int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_path = Path(base_equity_path).expanduser().resolve()
    base = pd.read_csv(base_path)
    base = base.loc[
        base["Series"].eq(base_series) & base["Period"].eq(base_period),
        ["Date", "Equity"],
    ].copy()
    base["Date"] = pd.to_datetime(base["Date"], errors="raise")
    base = base.sort_values("Date").drop_duplicates("Date", keep="last")
    if base.empty:
        raise ValueError("Requested base series and period are not available")
    base = base.rename(columns={"Equity": "BaseEquity"})
    base["BaseReturn"] = base["BaseEquity"].pct_change().fillna(0.0)
    base["BaseDrawdown"] = (
        base["BaseEquity"] / base["BaseEquity"].cummax() - 1.0
    )

    spy = load_sp500_proxy(spy_path)[["Date", "Close"]].copy()
    spy = spy.sort_values("Date").drop_duplicates("Date", keep="last")
    spy = spy.rename(columns={"Close": "SPYClose"})
    spy["SPYReturn"] = spy["SPYClose"].pct_change().fillna(0.0)
    for window in sma_windows:
        sma_column = f"SPYSMA{int(window)}"
        spy[sma_column] = spy["SPYClose"].rolling(
            int(window),
            min_periods=int(window),
        ).mean()
        for lookback in slope_lookbacks:
            if int(lookback) <= 0:
                continue
            spy[f"{sma_column}Slope{int(lookback)}"] = (
                spy[sma_column].pct_change(
                    int(lookback),
                    fill_method=None,
                )
            )
    frame = base.merge(spy, on="Date", how="left", validate="one_to_one")
    if frame[["SPYClose", "SPYReturn"]].isna().any().any():
        raise ValueError("SPY is missing one or more V7 trading dates")

    vix = _load_value_series(vix_path, "VIX")
    vix = vix.rename(columns={"Date": "VIXSourceDate"})
    frame = pd.merge_asof(
        frame.sort_values("Date"),
        vix.sort_values("VIXSourceDate"),
        left_on="Date",
        right_on="VIXSourceDate",
        direction="backward",
    )
    frame["VIXAgeDays"] = (
        frame["Date"] - frame["VIXSourceDate"]
    ).dt.days
    if frame["VIX"].isna().any():
        raise ValueError("VIX history does not cover the V7 period")

    fed = _load_value_series(fed_funds_path, "CashRate")
    fed = fed.rename(columns={"Date": "CashRateSourceDate"})
    frame = pd.merge_asof(
        frame.sort_values("Date"),
        fed.sort_values("CashRateSourceDate"),
        left_on="Date",
        right_on="CashRateSourceDate",
        direction="backward",
    )
    if frame["CashRate"].isna().any():
        raise ValueError("Fed funds history does not cover the V7 period")
    return (
        frame.reset_index(drop=True),
        {
            "base_equity_path": str(base_path),
            "base_equity_sha256": _sha256(base_path),
            "base_rows": len(base),
            "start_date": str(base.iloc[0]["Date"].date()),
            "end_date": str(base.iloc[-1]["Date"].date()),
            "maximum_vix_age_days": int(frame["VIXAgeDays"].max()),
            "spy_path": str(Path(spy_path).resolve()),
            "spy_sha256": _sha256(Path(spy_path)),
            "vix_path": str(Path(vix_path).resolve()),
            "vix_sha256": _sha256(Path(vix_path)),
            "fed_funds_path": str(Path(fed_funds_path).resolve()),
            "fed_funds_sha256": _sha256(Path(fed_funds_path)),
        },
    )


def _load_value_series(path: str | Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"Date", "Value"} <= set(frame.columns):
        raise ValueError(f"{name} input requires Date and Value")
    frame = frame[["Date", "Value"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame[name] = pd.to_numeric(frame["Value"], errors="raise")
    return (
        frame[["Date", name]]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )


def _robust_score(
    train: dict[str, object],
    *,
    median_fold_cagr: float,
    fold_cagr_std: float,
    config: dict[str, Any],
) -> float:
    return float(
        float(config["train_cagr_weight"]) * float(train["CAGR"])
        + float(config["median_fold_cagr_weight"]) * median_fold_cagr
        - float(config["fold_cagr_std_penalty"]) * fold_cagr_std
        + float(config["train_sharpe_weight"]) * float(train["Sharpe"])
        + float(config["train_mdd_weight"]) * float(train["MaxDrawdown"])
        - float(config["overlay_turnover_penalty"])
        * float(train["OverlayTurnover"])
    )


def _calendar_consistency_score(
    metrics: dict[str, object],
    config: dict[str, Any],
) -> float:
    rolling_worst = float(metrics["RollingWorstReturn"])
    if not np.isfinite(rolling_worst):
        rolling_worst = 0.0
    return float(
        float(config["worst_return_weight"])
        * float(metrics["CalendarWorstReturn"])
        + float(config["mean_return_weight"])
        * float(metrics["CalendarMeanReturn"])
        - float(config["return_std_penalty"])
        * float(metrics["CalendarReturnStd"])
        - float(config["target_shortfall_penalty"])
        * float(metrics["CalendarTargetShortfallRMS"])
        - float(config["target_excess_penalty"])
        * float(metrics["CalendarTargetExcessRMS"])
        + float(config["rolling_worst_weight"]) * rolling_worst
    )


def _benchmark_consistency_score(
    metrics: dict[str, object],
    config: dict[str, Any],
) -> float:
    rolling_worst = float(metrics["BenchmarkRollingWorstAlpha"])
    if not np.isfinite(rolling_worst):
        rolling_worst = 0.0
    return float(
        float(config["worst_alpha_weight"])
        * float(metrics["BenchmarkWorstAlpha"])
        + float(config["mean_alpha_weight"])
        * float(metrics["BenchmarkMeanAlpha"])
        - float(config["alpha_std_penalty"])
        * float(metrics["BenchmarkAlphaStd"])
        - float(config["alpha_shortfall_penalty"])
        * float(metrics["BenchmarkAlphaShortfallRMS"])
        + float(config["rolling_worst_alpha_weight"])
        * rolling_worst
        + float(config["absolute_worst_return_weight"])
        * float(metrics["CalendarWorstReturn"])
    )


def _assert_baseline_parity(
    actual: dict[str, object],
    expected: dict[str, Any],
) -> None:
    tolerance = float(expected.get("tolerance", 1e-6))
    failures = []
    for key in ("ROI", "CAGR", "MaxDrawdown", "Sharpe"):
        wanted = float(expected[key])
        observed = float(actual[key])
        if not np.isclose(observed, wanted, atol=tolerance, rtol=0.0):
            failures.append(
                f"{key}: expected {wanted:.12f}, got {observed:.12f}"
            )
    if failures:
        raise RuntimeError(
            "V7 product-overlay baseline parity failed; aborted. "
            + "; ".join(failures)
        )


def _unique_report_ids(
    rows: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    output: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, candidate_id in rows:
        if candidate_id not in seen:
            output.append((label, candidate_id))
            seen.add(candidate_id)
    return output


def _calendar_returns(
    daily: pd.DataFrame,
    *,
    series: str,
    candidate_id: int,
    initial_capital: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous = initial_capital
    for year, group in daily.groupby(daily["Date"].dt.year, sort=True):
        ending = float(group.iloc[-1]["Equity"])
        rows.append(
            {
                "Series": series,
                "CandidateID": candidate_id,
                "Year": int(year),
                "StartEquity": previous,
                "EndEquity": ending,
                "Return": (ending / previous - 1.0) * 100.0,
            }
        )
        previous = ending
    return rows


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
