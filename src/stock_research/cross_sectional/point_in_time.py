from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

PitStatus = Literal["PIT_SAFE", "PIT_FIXABLE", "PIT_UNSAFE", "UNKNOWN"]

# Resolved this session by tracing every raw input of the Growth/Quality
# factors (see artifacts/pit_audit/feature_lineage.csv for the full table
# with sources). All of these come from a Macrotrends-scraped workbook that
# carries only a fiscal *period-end* date -- no filing/publication date, no
# revision history -- so every one starts PIT_UNSAFE. A field is promoted to
# PIT_FIXABLE only where every one of its raw inputs has a real SEC XBRL
# concept mapping in sec_filings.py::STANDARD_CONCEPTS (verified by grep,
# not assumed): Revenue, OperatingIncome, OperatingCashFlow,
# CapitalExpenditures, NetIncome, DilutedShares, Cash, Assets, Liabilities,
# DebtCurrent, DebtNoncurrent, SharesOutstanding all have real tags.
# EBITDA/D&A and a standalone per-share EPS concept do NOT (confirmed absent
# from STANDARD_CONCEPTS), so anything depending on EbitdaTtm or the raw
# "EPS - Earnings Per Share" field has no reconstruction path today and
# stays PIT_UNSAFE with no PIT_FIXABLE promotion.
_FEATURE_PIT_STATUS: dict[str, PitStatus] = {
    # Legacy-mode inputs
    "RevenueGrowthYoY": "PIT_FIXABLE",  # Revenue tag exists
    "EpsGrowthYoY": "PIT_UNSAFE",  # raw per-share EPS field, no XBRL tag
    "OperatingMarginChangeYoY": "PIT_FIXABLE",  # diff of OperatingMargin
    "OperatingMargin": "PIT_FIXABLE",  # Revenue + OperatingIncome both tagged
    "FreeCashFlowMargin": "PIT_FIXABLE",  # OperatingCashFlow + CapEx + Revenue tagged
    "ReturnOnInvestment": "PIT_UNSAFE",  # Macrotrends ratio column, no XBRL equivalent
    "NetCashToAssets": "PIT_FIXABLE",  # Cash + Debt(Current/Noncurrent) + Assets tagged
    # ttm_value_momentum-mode inputs
    "EpsTtm": "PIT_FIXABLE",  # NetIncome + DilutedShares tagged
    "EpsTtmGrowthYoY": "PIT_FIXABLE",
    "EpsTtmGrowthAcceleration": "PIT_FIXABLE",
    "EbitdaTtm": "PIT_UNSAFE",  # no EBITDA/D&A XBRL tag mapped
    "EbitdaTtmGrowthYoY": "PIT_UNSAFE",
    "EbitdaTtmGrowthAcceleration": "PIT_UNSAFE",
    "DcfPrice": "PIT_UNSAFE",  # depends on EBIT*0.75 + D&A, no D&A tag
    "DcfPriceGrowthYoY": "PIT_UNSAFE",
    "DcfUpside": "PIT_UNSAFE",
    "PeTtm": "PIT_FIXABLE",  # Close (always safe) / EpsTtm
    "EvEbitdaTtm": "PIT_UNSAFE",  # blocked by EbitdaTtm
    "GrowthAdjustedPe": "PIT_FIXABLE",  # inherits EpsTtmGrowthYoY/PeTtm
    "GrowthAdjustedEvEbitda": "PIT_UNSAFE",  # inherits EbitdaTtmGrowthYoY/EvEbitdaTtm
}

# PIT_FIXABLE fields are still unsafe as currently sourced (Macrotrends).
# "Fixable" only means a reconstruction path exists via SEC XBRL for the
# subset of tickers with usable companyfacts.json coverage (~32 of 636 in
# this universe as of this audit) -- see pit_audit_report.md. Strict mode
# disables them universe-wide same as PIT_UNSAFE unless a caller supplies an
# already-reconstructed substitute (see Version C in run_pit_audit.py /
# the Phase 2 comparison script); this module does not silently promote them.
DISABLE_IN_STRICT_MODE: frozenset[str] = frozenset(
    name
    for name, status in _FEATURE_PIT_STATUS.items()
    if status in ("PIT_FIXABLE", "PIT_UNSAFE")
)


def classify_feature_pit_status(feature_name: str) -> PitStatus:
    """Classify a Growth/Quality raw or derived feature's PIT safety.

    Returns "UNKNOWN" for any feature not in the audited table above --
    callers must not assume UNKNOWN means safe.
    """

    return _FEATURE_PIT_STATUS.get(feature_name, "UNKNOWN")


@dataclass(frozen=True)
class PointInTimeConfig:
    """Governs how point_in_time_asof_join() and PIT-unsafe factor gating
    behave. Standalone (not a field on ResearchSettings) because it's
    specific to the audit/join layer, following the same
    standalone-frozen-dataclass pattern as FilingHybridConfig.
    """

    strict: bool = True
    unsafe_feature_policy: Literal["raise", "warn", "drop"] = "raise"

    def __post_init__(self) -> None:
        if self.unsafe_feature_policy not in ("raise", "warn", "drop"):
            raise ValueError(
                "unsafe_feature_policy must be 'raise', 'warn', or 'drop'"
            )


def point_in_time_asof_join(
    market_panel: pd.DataFrame,
    financial_data: pd.DataFrame,
    *,
    entity_col: str = "Ticker",
    signal_date_col: str = "Date",
    available_date_col: str = "AvailableFromDate",
    revision_version_col: str | None = None,
) -> pd.DataFrame:
    """Generic backward as-of join of point-in-time financial data onto a
    daily market panel, one entity at a time.

    No shared as-of-join helper existed anywhere in this repo before this
    function -- every module (filing_signals.py::merge_filing_features,
    tsla_integrated/features.py, macro_sp500/data.py, etc.) hand-rolled its
    own. This generalizes the proven pattern from
    filing_signals.py::merge_filing_features (per-ticker merge_asof +
    age-days leakage guard) so new PIT joins don't have to re-derive it.

    Guarantees:
    - Sorted by (entity_col, signal_date_col) before matching; each entity
      is joined independently (a merge_asof groupby loop), so one entity's
      financial history can never match another entity's market rows.
    - Duplicate (entity, available_date_col) rows are resolved by keeping
      the highest `revision_version_col` if supplied, else the last row
      after a stable sort -- documented here, not left implicit.
    - A market row with no financial row whose available date is on or
      before it stays NaN. Never back-filled or filled from a later,
      unrelated row.
    - Any match where the financial row's available date is strictly after
      the market signal date raises RuntimeError immediately, naming the
      entity, signal date, and source (available) date -- this can only
      happen if a caller's `available_date_col` values are wrong, since
      merge_asof(direction="backward") cannot itself produce a future match;
      the check exists as a hard backstop against that class of bug.
    """

    required_panel = {entity_col, signal_date_col}
    missing_panel = required_panel - set(market_panel.columns)
    if missing_panel:
        raise ValueError(f"market_panel missing columns: {sorted(missing_panel)}")
    required_financial = {entity_col, available_date_col}
    missing_financial = required_financial - set(financial_data.columns)
    if missing_financial:
        raise ValueError(
            f"financial_data missing columns: {sorted(missing_financial)}"
        )

    left = market_panel.copy()
    left[entity_col] = left[entity_col].astype(str).str.upper()
    left[signal_date_col] = pd.to_datetime(
        left[signal_date_col], errors="raise"
    ).dt.tz_localize(None)

    right = financial_data.copy()
    right[entity_col] = right[entity_col].astype(str).str.upper()
    right[available_date_col] = pd.to_datetime(
        right[available_date_col], errors="coerce"
    ).dt.tz_localize(None)
    right = right.dropna(subset=[available_date_col])
    sort_columns = [entity_col, available_date_col]
    if revision_version_col is not None and revision_version_col in right:
        sort_columns.append(revision_version_col)
    right = right.sort_values(sort_columns)
    right = right.drop_duplicates([entity_col, available_date_col], keep="last")

    outputs: list[pd.DataFrame] = []
    financial_columns = [
        column for column in right.columns if column != entity_col
    ]
    for entity, entity_panel in left.groupby(entity_col, sort=False):
        entity_financials = right.loc[right[entity_col].eq(entity), financial_columns]
        ordered = entity_panel.sort_values(signal_date_col)
        if entity_financials.empty:
            outputs.append(ordered)
            continue
        merged = pd.merge_asof(
            ordered,
            entity_financials.sort_values(available_date_col),
            left_on=signal_date_col,
            right_on=available_date_col,
            direction="backward",
            allow_exact_matches=True,
        )
        outputs.append(merged)
    result = pd.concat(outputs, ignore_index=True)
    _assert_no_leakage(result, entity_col, signal_date_col, available_date_col)
    return result.sort_values([signal_date_col, entity_col]).reset_index(drop=True)


def _assert_no_leakage(
    merged: pd.DataFrame,
    entity_col: str,
    signal_date_col: str,
    available_date_col: str,
) -> None:
    """Raise if any matched row's available date is after its signal date.

    A correct backward merge_asof cannot produce this on its own -- this is
    a defense-in-depth backstop against a future regression (e.g. someone
    changing direction="backward" to "nearest", or a caller passing
    pre-corrupted available dates). Extracted as its own function so the
    guard itself is directly unit-testable without needing to defeat
    merge_asof's own correctness to exercise it.
    """

    age_days = (merged[signal_date_col] - merged[available_date_col]).dt.days
    leaked = age_days.lt(0)
    if leaked.any():
        bad = merged.loc[leaked].iloc[0]
        raise RuntimeError(
            "Future financial record leaked into the market panel: "
            f"entity={bad[entity_col]!r}, "
            f"signal_date={bad[signal_date_col].date()}, "
            f"source_available_date={bad[available_date_col].date()}"
        )


def _centered_rank(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
    minimum_count: int,
) -> pd.Series:
    """Mirrors features.py::_centered_rank -- duplicated rather than
    imported to keep this module decoupled from features.py's private
    helpers; both compute the same causal, per-date percentile-rank-minus-
    0.5 transform."""

    values = pd.to_numeric(frame[column], errors="coerce").where(eligible)
    counts = values.notna().groupby(frame["Date"]).transform("sum")
    ranks = values.groupby(frame["Date"]).rank(pct=True, method="average") - 0.5
    return ranks.where(counts.ge(minimum_count))


def reconstruct_growth_quality_pit_safe(
    merged_panel: pd.DataFrame,
    *,
    minimum_cross_section_size: int = 8,
) -> pd.DataFrame:
    """Version C (diagnostic, per the PIT-audit plan): substitute the
    already point-in-time-safe `Filed*` columns -- produced by
    filing_signals.py::merge_filing_features, joined on real SEC
    acceptance dates -- for GrowthFactor/QualityFactor, in place of the
    Macrotrends-derived values.

    Requires `merged_panel` to already have gone through
    merge_filing_features() (so FiledRevenueGrowthYoY/FiledOperatingMargin/
    FiledFreeCashFlowMargin exist) and, if a V7-3-style technical variant
    is in use, scoring_panel_for_variant() (so its TrendFactor blend is
    already in place). Only GrowthFactor/QualityFactor are rewritten --
    Momentum/Trend/RiskControl are left exactly as computed upstream.

    Rows/dates with no filing coverage get NaN, never a silent fallback to
    the unsafe legacy value -- the same rule strict mode applies. This is
    intentionally a diagnostic comparison path, not wired into the
    production run_filing_v7_optimization pipeline by default, because SEC
    filing coverage is far from universal outside a specifically curated
    research universe (see artifacts/pit_audit/pit_coverage_by_ticker.csv).
    """

    required = {"FiledRevenueGrowthYoY", "FiledOperatingMargin", "FiledFreeCashFlowMargin"}
    missing = required - set(merged_panel.columns)
    if missing:
        raise ValueError(
            "merged_panel is missing filing columns "
            f"{sorted(missing)} -- did you run merge_filing_features() first?"
        )
    frame = merged_panel.copy()
    eligible = frame.get("Eligible", pd.Series(True, index=frame.index)).fillna(False)
    growth_rank = _centered_rank(
        frame, "FiledRevenueGrowthYoY", eligible, minimum_cross_section_size
    )
    operating_rank = _centered_rank(
        frame, "FiledOperatingMargin", eligible, minimum_cross_section_size
    )
    fcf_rank = _centered_rank(
        frame, "FiledFreeCashFlowMargin", eligible, minimum_cross_section_size
    )
    frame["GrowthFactor"] = growth_rank
    frame["QualityFactor"] = pd.concat([operating_rank, fcf_rank], axis=1).mean(axis=1)
    return frame
