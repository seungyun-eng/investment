from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def chained_return_metrics(annual_roi_percent: Sequence[float]) -> dict[str, float]:
    """Return compounded growth statistics for sequential annual ROIs."""

    values = np.asarray(list(annual_roi_percent), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Annual ROI values must be finite and non-empty.")
    factors = 1.0 + values / 100.0
    if np.any(factors <= 0):
        return {
            "GrowthMultiple": 0.0,
            "CumulativeROI": -100.0,
            "CAGR": -100.0,
            "ArithmeticMeanROI": float(values.mean()),
        }
    growth = float(np.prod(factors))
    return {
        "GrowthMultiple": growth,
        "CumulativeROI": (growth - 1.0) * 100.0,
        "CAGR": (growth ** (1.0 / values.size) - 1.0) * 100.0,
        "ArithmeticMeanROI": float(values.mean()),
    }


def rank_cagr_candidates(
    candidates: pd.DataFrame,
    development_years: Sequence[int],
) -> pd.DataFrame:
    """Rank by development CAGR for a deliberate overfit-control comparison."""

    if not development_years:
        raise ValueError("At least one development year is required.")
    roi_columns = [f"ROI_{year}" for year in development_years]
    missing = [column for column in roi_columns if column not in candidates]
    if missing:
        raise ValueError(f"Missing annual metric columns: {missing}")
    ranked = candidates.copy()
    roi = ranked[roi_columns].apply(pd.to_numeric, errors="coerce")
    factors = 1.0 + roi / 100.0
    valid = np.isfinite(roi).all(axis=1) & factors.gt(0).all(axis=1)
    growth = factors.prod(axis=1)
    ranked["DevelopmentCAGR"] = -100.0
    ranked.loc[valid, "DevelopmentCAGR"] = (
        growth.loc[valid] ** (1.0 / len(development_years)) - 1.0
    ) * 100.0
    ranked["WorstDevelopmentYearROI"] = roi.min(axis=1)
    ranked = ranked.sort_values(
        ["DevelopmentCAGR", "WorstDevelopmentYearROI"],
        ascending=False,
    ).reset_index(drop=True)
    ranked["CAGRSelectionRank"] = np.arange(1, len(ranked) + 1)
    return ranked


def rank_annual_floor_candidates(
    candidates: pd.DataFrame,
    development_years: Sequence[int],
    *,
    annual_roi_floor_percent: float = 40.0,
    maximum_drawdown_percent: float = -40.0,
) -> pd.DataFrame:
    """Rank candidates without using any later evaluation-year result.

    A hard-floor candidate must clear both constraints in every development
    calendar year.  When none qualifies, the fallback maximizes the worst
    development-year ROI first; this makes failure explicit instead of
    silently reverting to full-period compounded ROI.
    """

    if not development_years:
        raise ValueError("At least one development year is required.")
    roi_columns = [f"ROI_{year}" for year in development_years]
    mdd_columns = [f"MDD_{year}" for year in development_years]
    missing = [
        column
        for column in (*roi_columns, *mdd_columns)
        if column not in candidates.columns
    ]
    if missing:
        raise ValueError(f"Missing annual metric columns: {missing}")

    ranked = candidates.copy()
    roi = ranked[roi_columns].apply(pd.to_numeric, errors="coerce")
    mdd = ranked[mdd_columns].apply(pd.to_numeric, errors="coerce")
    ranked["WorstDevelopmentYearROI"] = roi.min(axis=1)
    ranked["MedianDevelopmentYearROI"] = roi.median(axis=1)
    ranked["MeanDevelopmentYearROI"] = roi.mean(axis=1)
    ranked["WorstDevelopmentYearMDD"] = mdd.min(axis=1)
    finite_roi = np.isfinite(roi).all(axis=1)
    finite_mdd = np.isfinite(mdd).all(axis=1)
    ranked["ROIFloorEligible"] = finite_roi & roi.ge(
        annual_roi_floor_percent
    ).all(axis=1)
    ranked["DrawdownEligible"] = finite_mdd & mdd.ge(
        maximum_drawdown_percent
    ).all(axis=1)
    ranked["AnnualFloorEligible"] = (
        ranked["ROIFloorEligible"] & ranked["DrawdownEligible"]
    )
    ranked = ranked.sort_values(
        [
            "ROIFloorEligible",
            "AnnualFloorEligible",
            "WorstDevelopmentYearROI",
            "MedianDevelopmentYearROI",
            "MeanDevelopmentYearROI",
            "WorstDevelopmentYearMDD",
        ],
        ascending=False,
    ).reset_index(drop=True)
    ranked["AnnualFloorSelectionRank"] = np.arange(1, len(ranked) + 1)
    return ranked
