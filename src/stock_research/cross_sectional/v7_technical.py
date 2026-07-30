from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.io_utils import atomic_to_csv
from stock_research.paths import ProjectPaths

from .config import ResearchSettings, StrategyParams
from .data import discover_universe
from .pit_validation import apply_membership_to_panel
from .v7_pit_evaluation import (
    build_v7_source_panel,
    evaluate_period,
    load_ready_tickers,
    normalize_change_membership,
)
from .winner_attribution import summarize_ticker_contributions


@dataclass(frozen=True)
class TechnicalVariant:
    name: str
    components: tuple[str, ...]
    description: str


TECHNICAL_VARIANTS = (
    TechnicalVariant(
        name="V7_1_MA_SLOT5_CONTROL",
        components=("MAFactor",),
        description="Five-stock control using the existing MA trend factor",
    ),
    TechnicalVariant(
        name="V7_2_MA_MACD_SLOT5",
        components=("MAFactor", "MACDFactor"),
        description="Existing MA trend plus normalized MACD line/histogram",
    ),
    TechnicalVariant(
        name="V7_3_MA_MACD_OBV_SLOT5",
        components=("MAFactor", "MACDFactor", "OBVFactor"),
        description="MA and MACD plus normalized 21/63-session OBV flow",
    ),
    TechnicalVariant(
        name="V7_4_MA_MACD_OBV_RSI_BOLLINGER_SLOT5",
        components=(
            "MAFactor",
            "MACDFactor",
            "OBVFactor",
            "RSIFactor",
            "BollingerFactor",
        ),
        description=(
            "MA, MACD, OBV, balanced RSI strength, and Bollinger percent-B"
        ),
    ),
)


@dataclass(frozen=True)
class V7TechnicalArtifacts:
    output_dir: Path
    data_audit_csv: Path
    feature_coverage_csv: Path
    variant_summary_csv: Path
    concentration_summary_csv: Path
    ticker_contributions_csv: Path
    executions_csv: Path
    selected_signals_csv: Path
    equity_csv: Path
    manifest_json: Path


def add_v7_technical_observations(panel: pd.DataFrame) -> pd.DataFrame:
    """Add causal price/volume observations without changing V6 columns."""

    frame = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    output_columns = (
        "MACDLineNormalized",
        "MACDHistogramNormalized",
        "OBVFlow21",
        "OBVFlow63",
        "RSI14",
        "BollingerPercentB",
    )
    outputs = {
        column: pd.Series(np.nan, index=frame.index, dtype=float)
        for column in output_columns
    }
    for _, indexes in frame.groupby("Ticker", sort=False).groups.items():
        positions = np.asarray(indexes, dtype=int)
        close = pd.to_numeric(
            frame.loc[positions, "Close"], errors="coerce"
        ).reset_index(drop=True)
        volume = pd.to_numeric(
            frame.loc[positions, "Volume"], errors="coerce"
        ).reset_index(drop=True)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        history_count = pd.Series(
            np.arange(1, len(close) + 1), index=close.index
        )
        outputs["MACDLineNormalized"].iloc[positions] = (
            macd / close.replace(0, np.nan)
        ).where(history_count.ge(26))
        outputs["MACDHistogramNormalized"].iloc[positions] = (
            (macd - macd_signal) / close.replace(0, np.nan)
        ).where(history_count.ge(34))

        direction = np.sign(close.diff())
        signed_volume = direction * volume
        for window, minimum in ((21, 15), (63, 42)):
            denominator = volume.rolling(
                window, min_periods=minimum
            ).sum()
            flow = (
                signed_volume.rolling(window, min_periods=minimum).sum()
                / denominator.replace(0, np.nan)
            )
            outputs[f"OBVFlow{window}"].iloc[positions] = flow

        delta = close.diff()
        average_gain = delta.clip(lower=0).rolling(
            14, min_periods=14
        ).mean()
        average_loss = (-delta.clip(upper=0)).rolling(
            14, min_periods=14
        ).mean()
        relative_strength = average_gain / average_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + relative_strength)
        rsi = rsi.where(average_loss.ne(0), 100.0)
        outputs["RSI14"].iloc[positions] = rsi

        middle = close.rolling(20, min_periods=20).mean()
        deviation = close.rolling(20, min_periods=20).std()
        lower = middle - 2 * deviation
        width = 4 * deviation
        outputs["BollingerPercentB"].iloc[positions] = (
            (close - lower) / width.replace(0, np.nan)
        )

    for column, values in outputs.items():
        frame[column] = values
    return frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def add_v7_technical_factors(
    panel: pd.DataFrame,
    settings: ResearchSettings,
) -> pd.DataFrame:
    frame = panel.copy()
    eligible = frame["Eligible"].fillna(False)
    macd_line = _centered_rank(
        frame,
        "MACDLineNormalized",
        eligible,
        settings.minimum_cross_section_size,
    )
    macd_histogram = _centered_rank(
        frame,
        "MACDHistogramNormalized",
        eligible,
        settings.minimum_cross_section_size,
    )
    obv21 = _centered_rank(
        frame,
        "OBVFlow21",
        eligible,
        settings.minimum_cross_section_size,
    )
    obv63 = _centered_rank(
        frame,
        "OBVFlow63",
        eligible,
        settings.minimum_cross_section_size,
    )
    rsi = pd.to_numeric(frame["RSI14"], errors="coerce")
    balanced_rsi = pd.Series(
        np.where(rsi.le(75), rsi, 150 - rsi),
        index=frame.index,
        dtype=float,
    )
    frame["RSIBalancedStrength"] = balanced_rsi
    rsi_factor = _centered_rank(
        frame,
        "RSIBalancedStrength",
        eligible,
        settings.minimum_cross_section_size,
    )
    frame["BollingerTrendStrength"] = pd.to_numeric(
        frame["BollingerPercentB"], errors="coerce"
    ).clip(lower=-0.25, upper=1.0)
    bollinger_factor = _centered_rank(
        frame,
        "BollingerTrendStrength",
        eligible,
        settings.minimum_cross_section_size,
    )
    frame["MAFactor"] = frame["TrendFactor"]
    frame["MACDFactor"] = pd.concat(
        [macd_line, macd_histogram], axis=1
    ).mean(axis=1)
    frame["OBVFactor"] = pd.concat([obv21, obv63], axis=1).mean(axis=1)
    frame["RSIFactor"] = rsi_factor
    frame["BollingerFactor"] = bollinger_factor
    factor_columns = [
        "MAFactor",
        "MACDFactor",
        "OBVFactor",
        "RSIFactor",
        "BollingerFactor",
    ]
    frame.loc[~eligible, factor_columns] = np.nan
    return frame


def scoring_panel_for_variant(
    panel: pd.DataFrame,
    variant: TechnicalVariant,
) -> pd.DataFrame:
    missing = sorted(set(variant.components) - set(panel.columns))
    if missing:
        raise ValueError(
            f"Variant {variant.name} is missing factors: {missing}"
        )
    frame = panel.copy()
    frame["V6TrendFactor"] = frame["TrendFactor"]
    frame["TrendFactor"] = frame.loc[:, list(variant.components)].mean(
        axis=1
    )
    frame["TechnicalVariant"] = variant.name
    frame["TechnicalComponentCount"] = len(variant.components)
    return frame


def slot5_params(base: StrategyParams) -> StrategyParams:
    """Change only the portfolio slot count; preserve every other V6-B rule."""

    return replace(base, top_k=5)


def run_v7_technical_variants(
    paths: ProjectPaths,
    settings: ResearchSettings,
    base_params: StrategyParams,
    *,
    ticker_config_path: str | Path,
    backfill_status_path: str | Path,
    membership_path: str | Path,
    frozen_strategy_path: str | Path,
    expected_ready_count: int = 575,
    output_dir: str | Path | None = None,
) -> V7TechnicalArtifacts:
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
    membership = normalize_change_membership(
        pd.read_csv(membership_path)
    )
    pit_panel = apply_membership_to_panel(
        observed_panel,
        membership,
        settings,
    )
    technical_panel = add_v7_technical_factors(pit_panel, settings)
    latest_end = str(pd.Timestamp(technical_panel["Date"].max()).date())
    periods = {
        "TRAIN_2020_2024": (
            settings.train_start,
            min(settings.train_end, latest_end),
        ),
        **{
            label: (start, min(end, latest_end))
            for label, (start, end) in settings.validation_periods.items()
            if start <= latest_end
        },
    }
    reference_dates = pd.DatetimeIndex(
        technical_panel["Date"].drop_duplicates().sort_values()
    )
    params = slot5_params(base_params)
    summary_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    contribution_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    execution_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    feature_coverage_frames: list[pd.DataFrame] = []
    for variant in TECHNICAL_VARIANTS:
        variant_panel = scoring_panel_for_variant(
            technical_panel, variant
        )
        feature_coverage_frames.append(
            _feature_coverage(variant_panel, variant, periods)
        )
        for period, (start, end) in periods.items():
            result = evaluate_period(
                variant_panel,
                params,
                settings,
                start=start,
                end=end,
                reference_dates=reference_dates,
            )
            strategy = result.strategy.summary
            benchmark = result.benchmark.summary
            summary_rows.append(
                {
                    "Variant": variant.name,
                    "Components": ",".join(variant.components),
                    "Period": period,
                    "StartDate": strategy.start_date,
                    "EndDate": strategy.end_date,
                    "TopK": params.top_k,
                    "ExitRank": params.exit_rank,
                    "FinancialWeight": params.financial_weight,
                    "StrategyROI": strategy.roi_percent,
                    "BenchmarkROI": benchmark.roi_percent,
                    "ExcessROI": (
                        strategy.roi_percent - benchmark.roi_percent
                    ),
                    "CAGR": strategy.cagr_percent,
                    "MaxDrawdown": strategy.max_drawdown_percent,
                    "Sharpe": strategy.sharpe_ratio,
                    "AnnualizedTurnover": strategy.annualized_turnover,
                    "TickerTrades": strategy.ticker_trades,
                    "Rebalances": strategy.rebalance_count,
                }
            )
            contributions = summarize_ticker_contributions(
                result.strategy,
                {period: (start, end)},
            )
            contributions.insert(0, "Variant", variant.name)
            contribution_frames.append(contributions)
            concentration_rows.append(
                _concentration_row(
                    variant,
                    period,
                    result.strategy,
                    contributions,
                )
            )
            equity = result.strategy.daily.copy()
            equity.insert(0, "Variant", variant.name)
            equity.insert(1, "Period", period)
            equity_frames.append(equity)
            executions = result.strategy.executions.copy()
            executions.insert(0, "Variant", variant.name)
            executions.insert(1, "Period", period)
            execution_frames.append(executions)
            selected = result.targets.loc[
                result.targets["ModelSelected"].fillna(False)
                | result.targets["TradeAction"].isin(["BUY", "SELL"])
            ].copy()
            selected.insert(0, "Variant", variant.name)
            selected.insert(1, "Period", period)
            signal_frames.append(selected)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            paths.results
            / "Cross_Sectional"
            / "v7_technical"
            / f"{timestamp}_slot5_fixed_weights"
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "data_audit_csv": destination / "data_audit.csv",
        "feature_coverage_csv": destination / "feature_coverage.csv",
        "variant_summary_csv": destination / "variant_summary.csv",
        "concentration_summary_csv": (
            destination / "concentration_summary.csv"
        ),
        "ticker_contributions_csv": (
            destination / "ticker_contributions.csv"
        ),
        "executions_csv": destination / "executions.csv",
        "selected_signals_csv": destination / "selected_signals.csv",
        "equity_csv": destination / "equity.csv",
        "manifest_json": destination / "manifest.json",
    }
    atomic_to_csv(data_audit, outputs["data_audit_csv"], index=False)
    atomic_to_csv(
        pd.concat(feature_coverage_frames, ignore_index=True),
        outputs["feature_coverage_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.DataFrame(summary_rows),
        outputs["variant_summary_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.DataFrame(concentration_rows),
        outputs["concentration_summary_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(contribution_frames, ignore_index=True),
        outputs["ticker_contributions_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(execution_frames, ignore_index=True),
        outputs["executions_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(signal_frames, ignore_index=True),
        outputs["selected_signals_csv"],
        index=False,
    )
    atomic_to_csv(
        pd.concat(equity_frames, ignore_index=True),
        outputs["equity_csv"],
        index=False,
    )
    frozen_payload = json.loads(
        Path(frozen_strategy_path).read_text(encoding="utf-8")
    )
    _atomic_json(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "task": "V7 technical variants with five equal-weight slots",
            "model_status": "POST_HOC_EXPERIMENT",
            "validation_is_fresh": False,
            "v6_b_unchanged": True,
            "frozen_v6_candidate": frozen_payload.get(
                "selected_candidate"
            ),
            "frozen_strategy_sha256": _sha256(
                Path(frozen_strategy_path)
            ),
            "membership_sha256": _sha256(Path(membership_path)),
            "ready_ticker_count": len(ready_tickers),
            "panel_ticker_count": int(
                technical_panel["Ticker"].nunique()
            ),
            "base_params": base_params.as_dict(),
            "v7_params": params.as_dict(),
            "only_parameter_change": (
                "top_k 3 -> 5; exit_rank and all replacement/loss rules "
                "remain frozen"
            ),
            "technical_weight_allocation": (
                "the frozen 36.0767508581% TrendFactor weight is split "
                "equally across each variant's listed components"
            ),
            "financial_weight": params.financial_weight,
            "technical_formulas": {
                "MAFactor": (
                    "existing centered ranks of price/SMA50 and "
                    "price/SMA200"
                ),
                "MACDFactor": (
                    "mean centered rank of (EMA12-EMA26)/close and "
                    "MACD histogram/close"
                ),
                "OBVFactor": (
                    "mean centered rank of 21/63-session signed-volume "
                    "sum divided by total volume"
                ),
                "RSIFactor": (
                    "centered rank of RSI14 when <=75, else 150-RSI14; "
                    "maximum strength at RSI 75"
                ),
                "BollingerFactor": (
                    "centered rank of 20-session percent-B clipped to "
                    "[-0.25, 1.0]"
                ),
            },
            "variants": [
                {
                    "name": variant.name,
                    "components": variant.components,
                    "description": variant.description,
                }
                for variant in TECHNICAL_VARIANTS
            ],
            "signal_rule": (
                "known final exchange session of each W-FRI week close"
            ),
            "execution_rule": "next available trading session open",
            "transaction_cost_bps": settings.transaction_cost_bps,
            "financial_point_in_time": False,
            "caveats": [
                "Macrotrends financials are current restated history.",
                "The 575-name panel excludes unavailable acquired/delisted names.",
                "V7 variants were specified after observing V6, 2025, and 2026.",
                "No random weight search was performed for these variants.",
            ],
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        outputs["manifest_json"],
    )
    return V7TechnicalArtifacts(
        output_dir=destination,
        **outputs,
    )


def _feature_coverage(
    panel: pd.DataFrame,
    variant: TechnicalVariant,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        frame = panel.loc[
            panel["Date"].between(start, end) & panel["Eligible"]
        ]
        for component in variant.components:
            available = frame[component].notna()
            rows.append(
                {
                    "Variant": variant.name,
                    "Period": period,
                    "Component": component,
                    "EligibleRows": len(frame),
                    "AvailableRows": int(available.sum()),
                    "CoveragePct": (
                        available.mean() * 100 if len(frame) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _concentration_row(
    variant: TechnicalVariant,
    period: str,
    strategy,
    contributions: pd.DataFrame,
) -> dict[str, object]:
    ordered = contributions.reindex(
        contributions["NetPnL"].abs().sort_values(ascending=False).index
    )
    absolute_total = float(ordered["NetPnL"].abs().sum())
    top = ordered.iloc[0] if not ordered.empty else None
    return {
        "Variant": variant.name,
        "Period": period,
        "DistinctHeldTickers": int(contributions["Ticker"].nunique()),
        "MeanSelectedCount": float(
            strategy.daily["SelectedCount"].mean()
        ),
        "MaximumSelectedCount": int(
            strategy.daily["SelectedCount"].max()
        ),
        "LargestAbsoluteContributor": (
            str(top["Ticker"]) if top is not None else ""
        ),
        "LargestAbsoluteContributionPct": (
            float(abs(top["NetPnL"]) / absolute_total * 100)
            if top is not None and absolute_total > 0
            else np.nan
        ),
        "Top3AbsoluteContributionPct": (
            float(ordered.head(3)["NetPnL"].abs().sum())
            / absolute_total
            * 100
            if absolute_total > 0
            else np.nan
        ),
    }


def _centered_rank(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
    minimum_count: int,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").where(eligible)
    counts = values.notna().groupby(frame["Date"]).transform("sum")
    ranks = values.groupby(frame["Date"]).rank(
        pct=True,
        method="average",
    ) - 0.5
    return ranks.where(counts.ge(minimum_count))


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
