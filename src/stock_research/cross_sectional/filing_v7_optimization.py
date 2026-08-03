from __future__ import annotations

import itertools
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.data import load_sp500_proxy
from stock_research.paths import ProjectPaths

from .big_tech_10 import (
    build_single_asset_equity,
    calendar_return_rows,
    summarize_equity_curve,
)
from .config import ResearchSettings, StrategyParams
from .features import build_panel
from .filing_signals import (
    add_filing_factors,
    merge_filing_features,
)
from .data import UniverseMember
from .fixed_universe import members_from_config
from .live_top10_watchlist import apply_full_cash_market_gate
from .pit_validation import apply_membership_to_panel
from .portfolio import PortfolioResult, prepare_market, run_portfolio_backtest
from .signals import generate_rebalance_targets, score_panel, signal_day_panel
from .v7_pit_evaluation import normalize_change_membership
from .v7_technical import (
    TECHNICAL_VARIANTS,
    add_v7_technical_factors,
    add_v7_technical_observations,
    scoring_panel_for_variant,
)


@dataclass(frozen=True)
class FilingV7Policy:
    filing_weight: float
    minimum_filing_coverage: int
    filing_quality_floor: float | None
    red_flag_veto: bool
    top_k: int
    exit_rank_buffer: int
    hard_stop_return: float
    minimum_hold_rebalances: int
    replacement_score_advantage: float

    def __post_init__(self) -> None:
        if not 0 <= self.filing_weight <= 1:
            raise ValueError("filing_weight must be between zero and one")
        if self.minimum_filing_coverage < 0:
            raise ValueError("minimum_filing_coverage must be non-negative")
        if self.top_k < 1 or self.exit_rank_buffer < 0:
            raise ValueError("top_k must be positive and the buffer non-negative")
        if not -1 < self.hard_stop_return < 0:
            raise ValueError("hard_stop_return must be between -1 and zero")
        if self.minimum_hold_rebalances < 1:
            raise ValueError("minimum_hold_rebalances must be positive")
        if self.replacement_score_advantage < 0:
            raise ValueError("replacement_score_advantage must be non-negative")

    @property
    def exit_rank(self) -> int:
        return self.top_k + self.exit_rank_buffer

    @property
    def uses_filing_quality_gate(self) -> bool:
        return (
            self.minimum_filing_coverage > 0
            or self.filing_quality_floor is not None
        )

    def strategy_params(self, base: StrategyParams) -> StrategyParams:
        return replace(
            base,
            top_k=self.top_k,
            exit_rank=self.exit_rank,
            conviction_exit_rank=max(base.conviction_exit_rank, self.exit_rank),
            profit_rotation_exit_rank=max(
                base.profit_rotation_exit_rank or self.exit_rank,
                self.exit_rank,
            ),
            hard_stop_return=self.hard_stop_return,
            minimum_hold_rebalances=self.minimum_hold_rebalances,
            replacement_score_advantage=self.replacement_score_advantage,
        )


@dataclass(frozen=True)
class FilingV7Artifacts:
    output_dir: Path
    candidates_csv: Path
    period_summary_csv: Path
    calendar_returns_csv: Path
    equity_csv: Path
    selected_signals_csv: Path
    executions_csv: Path
    latest_scores_csv: Path
    data_audit_csv: Path
    selection_json: Path
    manifest_json: Path


def add_filing_v7_scores(
    signal_days: pd.DataFrame,
    base_params: StrategyParams,
    policy: FilingV7Policy,
) -> pd.DataFrame:
    """Use one causal score function for both optimization and simulation."""

    frame = score_panel(signal_days, base_params).rename(
        columns={"AlphaScore": "BaseV7Score"}
    )
    filing_score = pd.to_numeric(
        frame.get("FilingFundamentalFactor"), errors="coerce"
    )
    base_score = pd.to_numeric(frame["BaseV7Score"], errors="coerce")
    frame["FilingScore"] = filing_score
    frame["AlphaScore"] = (
        (1.0 - policy.filing_weight) * base_score
        + policy.filing_weight * filing_score.fillna(0.0)
    )
    frame["FilingCriticalFlag"] = _critical_filing_flag(frame)
    available = frame.get(
        "AvailableDate", pd.Series(pd.NaT, index=frame.index)
    ).notna()
    coverage = pd.to_numeric(
        frame.get("FilingCoverageCount", 0), errors="coerce"
    ).fillna(0)
    coverage_pass = available & coverage.ge(policy.minimum_filing_coverage)
    if policy.minimum_filing_coverage == 0:
        coverage_pass = pd.Series(True, index=frame.index)
    quality_pass = pd.Series(True, index=frame.index)
    if policy.filing_quality_floor is not None:
        quality_pass = filing_score.ge(policy.filing_quality_floor)
    filing_gate = coverage_pass & quality_pass
    if not policy.uses_filing_quality_gate:
        filing_gate = pd.Series(True, index=frame.index)
    veto_pass = ~frame["FilingCriticalFlag"] if policy.red_flag_veto else True
    frame["FilingCoveragePass"] = coverage_pass
    frame["FilingQualityPass"] = quality_pass
    frame["FilingVetoPass"] = veto_pass
    frame["Qualified"] = (
        frame["Eligible"].fillna(False)
        & pd.to_numeric(frame["Trend200"], errors="coerce").ge(
            base_params.trend_floor
        )
        & pd.to_numeric(frame["Return126"], errors="coerce").ge(
            base_params.momentum_floor
        )
        & filing_gate
        & veto_pass
    )
    frame["Rank"] = (
        frame["AlphaScore"]
        .where(frame["Qualified"])
        .groupby(frame["Date"])
        .rank(ascending=False, method="first")
    )
    frame["EntryBlocked"] = frame.get(
        "EntryBlocked", pd.Series(False, index=frame.index)
    ).fillna(False)
    frame["EntryBlockReason"] = frame.get(
        "EntryBlockReason", pd.Series(None, index=frame.index)
    )

    # The shared V7 generator already implements an immediate force-exit path
    # for non-members.  In this fixed universe only an explicit filing red flag
    # can temporarily mark a row as non-member.
    original_membership = frame.get(
        "UniverseMember", pd.Series(True, index=frame.index)
    ).fillna(True)
    frame["UniverseMember"] = (
        original_membership & ~frame["FilingCriticalFlag"]
        if policy.red_flag_veto
        else original_membership
    )
    return frame


def generate_filing_v7_targets(
    signal_days: pd.DataFrame,
    base_params: StrategyParams,
    policy: FilingV7Policy,
    *,
    compact: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = add_filing_v7_scores(signal_days, base_params, policy)
    targets = generate_rebalance_targets(
        scored,
        policy.strategy_params(base_params),
        compact=compact,
        force_universe_exit=policy.red_flag_veto,
    )
    return scored, targets


def run_filing_v7_optimization(
    paths: ProjectPaths,
    settings: ResearchSettings,
    *,
    universe_config: dict[str, Any],
    optimization_config: dict[str, Any],
    filing_features_path: str | Path,
    spy_path: str | Path,
    universe_members: list[UniverseMember] | None = None,
    membership: pd.DataFrame | None = None,
    market_cash_gate: dict[str, object] | None = None,
    output_dir: str | Path | None = None,
) -> FilingV7Artifacts:
    members = (
        universe_members
        if universe_members is not None
        else members_from_config(paths, universe_config["universe"])
    )
    if len(members) < 2:
        raise ValueError("Filing V7 requires at least two model-ready tickers")
    panel, data_audit = build_panel(members, settings)
    observed = add_v7_technical_observations(panel)
    if membership is not None:
        observed = apply_membership_to_panel(
            observed,
            normalize_change_membership(membership),
            settings,
        )
    technical = add_v7_technical_factors(observed, settings)
    variant_name = str(
        universe_config.get("technical_variant", "V7_3_MA_MACD_OBV_SLOT5")
    )
    variant = next(item for item in TECHNICAL_VARIANTS if item.name == variant_name)
    v7_panel = scoring_panel_for_variant(technical, variant)
    filing_features = pd.read_csv(filing_features_path)
    merged = merge_filing_features(v7_panel, filing_features)
    factored = add_filing_factors(
        merged,
        minimum_cross_section_size=min(
            settings.minimum_cross_section_size,
            max(4, len(members) // 2),
        ),
    )
    spy_prices = load_sp500_proxy(spy_path)
    if market_cash_gate and bool(market_cash_gate.get("enabled", True)):
        factored = apply_full_cash_market_gate(
            factored,
            spy_prices,
            slow_sessions=int(market_cash_gate.get("slow_sessions", 200)),
            fast_sessions=int(market_cash_gate.get("fast_sessions", 50)),
            band=float(market_cash_gate.get("band", 0.01)),
        )
    full_end = str(pd.Timestamp(factored["Date"].max()).date())
    train_signals = signal_day_panel(
        factored,
        settings.train_start,
        settings.train_end,
        settings.rebalance_weekday,
    )
    full_signals = signal_day_panel(
        factored,
        settings.train_start,
        full_end,
        settings.rebalance_weekday,
    )
    base_params = StrategyParams.from_dict(
        dict(optimization_config["base_v7_params"])
    )
    spy_curve = build_single_asset_equity(
        spy_prices,
        start=settings.train_start,
        end=full_end,
        initial_capital=settings.initial_capital,
    )
    train_market = prepare_market(
        factored, start=settings.train_start, end=settings.train_end
    )
    spy_train = summarize_equity_curve(
        spy_curve, start=settings.train_start, end=settings.train_end
    )
    spy_annual = _annual_return_map(
        spy_curve, start=settings.train_start, end=settings.train_end
    )

    policies = _candidate_policies(optimization_config)
    candidate_rows: list[dict[str, object]] = []
    print(f"Filing V7 optimization candidates: {len(policies)}", flush=True)
    for index, policy in enumerate(policies, start=1):
        _, targets = generate_filing_v7_targets(
            train_signals, base_params, policy, compact=True
        )
        result = run_portfolio_backtest(
            pd.DataFrame(),
            targets,
            start=settings.train_start,
            end=settings.train_end,
            initial_capital=settings.initial_capital,
            transaction_cost_bps=settings.transaction_cost_bps,
            prepared_market=train_market,
        )
        candidate_rows.append(
            _candidate_metrics(
                index,
                policy,
                result,
                spy_train=spy_train,
                spy_annual=spy_annual,
                constraints=dict(optimization_config["selection_constraints"]),
            )
        )
        if index % 50 == 0 or index == len(policies):
            print(f"Filing V7 optimization progress: {index}/{len(policies)}", flush=True)

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["PassStrict", "ConstraintPenalty", "Objective"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    selected_row = candidates.iloc[0]
    selected_policy = policies[int(selected_row["Candidate"]) - 1]
    selection_mode = (
        "STRICT_TRAIN_PASS"
        if bool(selected_row["PassStrict"])
        else "BEST_TRAIN_ONLY_NO_STRICT_PASS"
    )
    pure_candidates = candidates.loc[
        candidates["filing_weight"].eq(1.0)
        & candidates["minimum_filing_coverage"].ge(6)
        & candidates["filing_quality_floor"].ge(0.0)
    ]
    if pure_candidates.empty:
        raise RuntimeError("No 100% SEC good-company candidate was generated")
    pure_row = pure_candidates.iloc[0]
    pure_policy = policies[int(pure_row["Candidate"]) - 1]

    comparison_policies = _comparison_policies(base_params)
    comparison_policies["V7_SEC_100_OPTIMIZED"] = pure_policy
    comparison_policies["V7_SEC_COMBINED_OPTIMIZED"] = selected_policy
    full_market = prepare_market(
        factored, start=settings.train_start, end=full_end
    )
    curves: dict[str, pd.DataFrame] = {"SPY_BUY_HOLD": spy_curve}
    results: dict[str, PortfolioResult] = {}
    scored_by_series: dict[str, pd.DataFrame] = {}
    targets_by_series: dict[str, pd.DataFrame] = {}
    for name, policy in comparison_policies.items():
        scored, targets = generate_filing_v7_targets(
            full_signals, base_params, policy
        )
        result = run_portfolio_backtest(
            pd.DataFrame(),
            targets,
            start=settings.train_start,
            end=full_end,
            initial_capital=settings.initial_capital,
            transaction_cost_bps=settings.transaction_cost_bps,
            prepared_market=full_market,
            record_attribution=True,
        )
        curves[name] = result.daily[["Date", "Equity"]]
        results[name] = result
        scored_by_series[name] = scored
        targets_by_series[name] = targets

    periods = {
        "TRAIN_2020_2024": (settings.train_start, settings.train_end),
        **{
            f"{label}_REPORT_ONLY": (start, min(end, full_end))
            for label, (start, end) in settings.validation_periods.items()
            if start <= full_end
        },
        "FULL_2020_2026": (settings.train_start, full_end),
    }
    period_rows: list[dict[str, object]] = []
    calendar_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    for name, curve in curves.items():
        clean = curve.sort_values("Date").drop_duplicates("Date", keep="last")
        output = clean.copy()
        output.insert(0, "Series", name)
        equity_frames.append(output)
        for period, (start, end) in periods.items():
            metrics = summarize_equity_curve(clean, start=start, end=end)
            if name in results and period == "FULL_2020_2026":
                metrics["AnnualizedTurnover"] = results[
                    name
                ].summary.annualized_turnover
                metrics["TickerTrades"] = results[name].summary.ticker_trades
            period_rows.append({"Series": name, "Period": period, **metrics})
        calendar_rows.extend(calendar_return_rows(clean, series=name))

    generated_at = datetime.now(UTC)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else paths.results
        / "Cross_Sectional"
        / "filing_v7_optimization"
        / (
            f"{timestamp}_"
            f"{str(universe_config.get('output_label', 'filing_v7')).strip()}"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidates_csv": destination / "candidates.csv",
        "period_summary_csv": destination / "period_summary.csv",
        "calendar_returns_csv": destination / "calendar_returns.csv",
        "equity_csv": destination / "equity.csv",
        "selected_signals_csv": destination / "selected_signals.csv",
        "executions_csv": destination / "executions.csv",
        "latest_scores_csv": destination / "latest_scores.csv",
        "data_audit_csv": destination / "data_audit.csv",
        "selection_json": destination / "selection.json",
        "manifest_json": destination / "manifest.json",
    }
    selected_series = ("V7_SEC_100_OPTIMIZED", "V7_SEC_COMBINED_OPTIMIZED")
    signal_frames: list[pd.DataFrame] = []
    execution_frames: list[pd.DataFrame] = []
    latest_frames: list[pd.DataFrame] = []
    latest_columns = [
        "Date",
        "Ticker",
        "Company",
        "BaseV7Score",
        "FilingScore",
        "AlphaScore",
        "Rank",
        "Qualified",
        "FilingCoverageCount",
        "FilingCriticalFlag",
        "AvailableDate",
        "AccessionNumber",
        "Form",
        "MarketRiskOn",
        "SPYTrendRegime",
        "MarketGateReason",
    ]
    for name in selected_series:
        targets = targets_by_series[name].copy()
        targets.insert(0, "Series", name)
        signal_frames.append(targets)
        executions = results[name].executions.copy()
        executions.insert(0, "Series", name)
        execution_frames.append(executions)
        scored = scored_by_series[name]
        latest = scored.loc[scored["Date"].eq(scored["Date"].max())].copy()
        latest.insert(0, "Series", name)
        latest_frames.append(
            latest[["Series", *[c for c in latest_columns if c in latest]]]
        )

    atomic_to_csv(candidates, outputs["candidates_csv"], index=False)
    atomic_to_csv(pd.DataFrame(period_rows), outputs["period_summary_csv"], index=False)
    atomic_to_csv(pd.DataFrame(calendar_rows), outputs["calendar_returns_csv"], index=False)
    atomic_to_csv(pd.concat(equity_frames, ignore_index=True), outputs["equity_csv"], index=False)
    atomic_to_csv(pd.concat(signal_frames, ignore_index=True), outputs["selected_signals_csv"], index=False)
    atomic_to_csv(pd.concat(execution_frames, ignore_index=True), outputs["executions_csv"], index=False)
    atomic_to_csv(pd.concat(latest_frames, ignore_index=True), outputs["latest_scores_csv"], index=False)
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    selection_payload = {
        "selection_mode": selection_mode,
        "training_period": [settings.train_start, settings.train_end],
        "report_only_period": ["2025-01-02", full_end],
        "selected_combined": {
            **selected_row.to_dict(),
            **asdict(selected_policy),
        },
        "selected_pure_sec": {**pure_row.to_dict(), **asdict(pure_policy)},
    }
    _atomic_json(outputs["selection_json"], selection_payload)
    _atomic_json(
        outputs["manifest_json"],
        {
            "generated_at": generated_at.isoformat(),
            "task": "V7-3 SEC filing score, red-flag veto, and pure-SEC optimization",
            "model_status": "POST_HOC_TRAIN_ONLY_RESEARCH",
            "validation_is_fresh": False,
            "selection_mode": selection_mode,
            "candidate_count": len(candidates),
            "technical_variant": variant_name,
            "base_v7_params": base_params.as_dict(),
            "selected_combined_policy": asdict(selected_policy),
            "selected_pure_sec_policy": asdict(pure_policy),
            "comparison_policies": {
                name: asdict(policy)
                for name, policy in comparison_policies.items()
            },
            "score_definition": (
                "(1-filing_weight)*frozen V7-3 score + "
                "filing_weight*SEC filing fundamental score"
            ),
            "pure_sec_definition": (
                "filing_weight=1.0; rank uses only accession-specific SEC "
                "durability, cash-flow/margin, leverage, dilution, and disclosure evidence"
            ),
            "signal_timing": "weekly close signal; next session open execution",
            "filing_timing": "eligible the calendar day after SEC acceptance",
            "transaction_cost_bps": settings.transaction_cost_bps,
            "roi_formula": "(final_value / total_injected - 1) * 100",
            "selection_data": "2020-2024 only",
            "report_only_data": "2025-2026; never used to select parameters",
            "point_in_time_membership": membership is not None,
            "market_cash_gate": market_cash_gate or {"enabled": False},
            "caveats": [
                "User watchlist eligibility dates are explicit research assumptions.",
                "The period has already been observed, so validation is not fresh.",
                "Pure SEC ranking is filing-native, but V7 price filters and stops remain technical.",
                "Industry growth and management guidance accuracy are not yet structured features.",
                "No result is a guarantee of future annual returns.",
            ],
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    return FilingV7Artifacts(output_dir=destination, **outputs)


def _candidate_policies(config: dict[str, Any]) -> list[FilingV7Policy]:
    grid = dict(config["optimization_grid"])
    policies = [
        FilingV7Policy(
            filing_weight=float(values[0]),
            minimum_filing_coverage=int(values[1]),
            filing_quality_floor=(
                float(values[2]) if values[2] is not None else None
            ),
            red_flag_veto=True,
            top_k=int(values[3]),
            exit_rank_buffer=int(values[4]),
            hard_stop_return=float(values[5]),
            minimum_hold_rebalances=int(values[6]),
            replacement_score_advantage=float(values[7]),
        )
        for values in itertools.product(
            grid["filing_weights"],
            grid["minimum_filing_coverage"],
            grid["filing_quality_floors"],
            grid["top_k"],
            grid["exit_rank_buffers"],
            grid["hard_stop_returns"],
            grid["minimum_hold_rebalances"],
            grid["replacement_score_advantages"],
        )
    ]
    return policies


def _comparison_policies(
    base: StrategyParams,
) -> dict[str, FilingV7Policy]:
    common = {
        "minimum_filing_coverage": 0,
        "filing_quality_floor": None,
        "top_k": base.top_k,
        "exit_rank_buffer": base.exit_rank - base.top_k,
        "hard_stop_return": base.hard_stop_return,
        "minimum_hold_rebalances": base.minimum_hold_rebalances,
        "replacement_score_advantage": base.replacement_score_advantage,
    }
    return {
        "V7_BASELINE_SAME_ENGINE": FilingV7Policy(
            filing_weight=0.0, red_flag_veto=False, **common
        ),
        "V7_SEC_SCORE_05": FilingV7Policy(
            filing_weight=0.05, red_flag_veto=False, **common
        ),
        "V7_SEC_SCORE_10": FilingV7Policy(
            filing_weight=0.10, red_flag_veto=False, **common
        ),
        "V7_SEC_SCORE_15": FilingV7Policy(
            filing_weight=0.15, red_flag_veto=False, **common
        ),
        "V7_SEC_RED_FLAG_VETO_ONLY": FilingV7Policy(
            filing_weight=0.0, red_flag_veto=True, **common
        ),
    }


def _candidate_metrics(
    index: int,
    policy: FilingV7Policy,
    result: PortfolioResult,
    *,
    spy_train: dict[str, object],
    spy_annual: dict[int, float],
    constraints: dict[str, Any],
) -> dict[str, object]:
    annual = _annual_return_map(
        result.daily,
        start=str(result.summary.start_date.date()),
        end=str(result.summary.end_date.date()),
    )
    comparable_years = sorted(set(annual) & set(spy_annual))
    values = [annual[year] for year in comparable_years]
    beat_years = sum(annual[year] > spy_annual[year] for year in comparable_years)
    worst = min(values) if values else np.nan
    median = float(np.median(values)) if values else np.nan
    standard_deviation = float(np.std(values)) if values else np.nan
    summary = result.summary
    pass_strict = bool(
        summary.cagr_percent >= float(constraints["minimum_cagr"])
        and summary.max_drawdown_percent
        >= float(constraints["minimum_max_drawdown"])
        and worst >= float(constraints["minimum_calendar_return"])
        and beat_years >= int(constraints["minimum_spy_beat_years"])
    )
    constraint_penalty = (
        max(0.0, float(constraints["minimum_cagr"]) - summary.cagr_percent)
        + max(
            0.0,
            float(constraints["minimum_max_drawdown"])
            - summary.max_drawdown_percent,
        )
        + max(0.0, float(constraints["minimum_calendar_return"]) - worst)
        + 10.0
        * max(0, int(constraints["minimum_spy_beat_years"]) - beat_years)
    )
    objective = (
        summary.cagr_percent
        + 0.35 * median
        + 0.50 * worst
        - 0.25 * standard_deviation
        + 0.20 * summary.max_drawdown_percent
        + 2.0 * beat_years
        - 0.10 * summary.annualized_turnover
    )
    return {
        "Candidate": index,
        **asdict(policy),
        "exit_rank": policy.exit_rank,
        "TrainROI": summary.roi_percent,
        "TrainCAGR": summary.cagr_percent,
        "TrainMaxDrawdown": summary.max_drawdown_percent,
        "TrainSharpe": summary.sharpe_ratio,
        "TrainAnnualizedTurnover": summary.annualized_turnover,
        "TrainWorstCalendarReturn": worst,
        "TrainMedianCalendarReturn": median,
        "TrainCalendarReturnStd": standard_deviation,
        "TrainBeatSPYYears": beat_years,
        "TrainSPYCAGR": float(spy_train["CAGR"]),
        "TrainExcessCAGR": summary.cagr_percent - float(spy_train["CAGR"]),
        "PassStrict": pass_strict,
        "ConstraintPenalty": constraint_penalty,
        "Objective": objective,
    }


def _annual_return_map(
    curve: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> dict[int, float]:
    period = curve.loc[curve["Date"].between(start, end)].copy()
    rows = calendar_return_rows(period, series="SERIES")
    return {int(row["Year"]): float(row["Return"]) for row in rows}


def _critical_filing_flag(frame: pd.DataFrame) -> pd.Series:
    columns = (
        "FiledGoingConcernFlag",
        "FiledMaterialWeaknessFlag",
        "FiledRestatementFlag",
    )
    flags = [
        frame.get(column, pd.Series(False, index=frame.index)).eq(True)
        for column in columns
    ]
    return flags[0] | flags[1] | flags[2]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=f"{path.stem}_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                _json_ready(payload),
                ensure_ascii=False,
                indent=2,
                default=str,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
