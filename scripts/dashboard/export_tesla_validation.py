"""Export the V7.3+V2 head-to-head validation into the Tesla tab's payload.

The Tesla card shows today's target weight but says nothing about *why* that
rule replaced the old cycle strategy.  This turns the stored research output
under research/tsla_unlevered_v2_overlay_v1/ into the panel the app renders,
so the evidence ships with the signal instead of living only in a report.md.

Research/display only.  Nothing here feeds live_v73_v2_signal -- the card's
recommendation is computed independently and is not read or written here.

Derived, never hand-copied: re-running this script is the only way the panel
changes, so a stale panel is always traceable to a stale research output.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_OUTPUT = REPO_ROOT / "research" / "tsla_unlevered_v2_overlay_v1" / "output"
PAYLOAD_PATH = REPO_ROOT / "alpha-desk-cloud" / "public" / "data" / "tesla_validation.json"

# The overlay study restarts each calendar year, so its stored Equity column
# is per-year.  Compounding the daily return columns is what reproduces the
# combined 2022-2025 figures printed in report.md.
OVERLAY_CSV = RESEARCH_OUTPUT / "daily_overlay.csv"
CYCLE_CSV = RESEARCH_OUTPUT / "cycle_live_2022_2025_daily.csv"
COMPARISON_CSV = RESEARCH_OUTPUT / "cycle_vs_v2_comparison.csv"
COHORT_CSV = RESEARCH_OUTPUT / "cycle_vs_v2_start_year_cohorts.csv"

STRATEGY_KEYS = {
    "V7.3 + V2": "v73_v2",
    "V7.3 (TSLA only)": "v73",
    "Live cycle (SINGLE_70_SELL_50)": "cycle",
    "Live cycle (70/50)": "cycle",
}

NOTE = (
    "Research/Validation only. 저장된 백테스트 결과를 그대로 읽어 표시합니다. "
    "이 패널은 카드의 실시간 판단 계산에 관여하지 않습니다."
)

CAVEAT = (
    "채택 근거인 \"59개 월별 코호트 중 약 69% 승\"에서 지는 구간은 최근에 몰려 있습니다. "
    "위 시작연도 코호트 기준으로 2024·2025년 시작분에서는 V7.3+V2가 원금 아래로 내려갔고 "
    "옛 사이클 규칙이 이겼습니다. 낙폭 우위도 2020~2022년 시작분에서만 성립하고, "
    "2023년 이후 시작분에서는 사이클 규칙의 낙폭이 오히려 더 작습니다."
)


def _series(dates: pd.Series, values: pd.Series) -> list[dict[str, object]]:
    return [
        {"date": pd.Timestamp(d).date().isoformat(), "value": round(float(v), 4)}
        for d, v in zip(dates, values, strict=True)
    ]


def build_curves() -> tuple[dict[str, list[dict[str, object]]], dict[str, str]]:
    """Weekly marks of four growth-of-1 curves on one shared date grid."""

    overlay = pd.read_csv(OVERLAY_CSV, parse_dates=["Date"]).set_index("Date")
    cycle = pd.read_csv(CYCLE_CSV, parse_dates=["Date"]).set_index("Date")

    frame = pd.DataFrame(index=overlay.index)
    frame["v73"] = (1.0 + overlay["BaseReturn"]).cumprod()
    frame["v73_v2"] = (1.0 + overlay["SymmetricReturn"]).cumprod()
    frame["buy_hold"] = overlay["Close"] / overlay["Close"].iloc[0]
    # The cycle sim is a continuous live-account run, so its own equity ratio
    # is already directly comparable to the compounded overlay curves.
    frame["cycle"] = cycle["Equity"] / cycle["Equity"].iloc[0]

    weekly = frame.resample("W-FRI").last().dropna(how="any")
    curves = {key: _series(weekly.index, weekly[key]) for key in ("v73_v2", "v73", "cycle", "buy_hold")}
    period = {
        "start": pd.Timestamp(frame.index[0]).date().isoformat(),
        "end": pd.Timestamp(frame.index[-1]).date().isoformat(),
    }
    return curves, period


def build_headline() -> list[dict[str, object]]:
    comparison = pd.read_csv(COMPARISON_CSV)
    rows: list[dict[str, object]] = []
    for _, row in comparison.iterrows():
        rows.append(
            {
                "key": STRATEGY_KEYS.get(str(row["Strategy"]), str(row["Strategy"])),
                "label": str(row["Strategy"]),
                "roi": round(float(row["ROI"]), 2),
                "cagr": round(float(row["CAGR"]), 2),
                "mdd": round(float(row["MDD"]), 2),
                "sharpe": round(float(row["Sharpe"]), 2),
            }
        )
    return rows


def build_cohorts() -> list[dict[str, object]]:
    """Start-year cohorts as ending growth multiples.

    ROI percentages cross zero, which a log axis cannot draw and a linear one
    flattens; multiples keep every cohort on one readable scale and put the
    break-even line where a reader expects it.
    """

    cohorts = pd.read_csv(COHORT_CSV)
    rows: list[dict[str, object]] = []
    for start_year, group in cohorts.groupby("StartYear", sort=True):
        row: dict[str, object] = {"start_year": int(start_year)}
        for _, entry in group.iterrows():
            key = STRATEGY_KEYS.get(str(entry["Strategy"]))
            if key is None:
                continue
            row[key] = round(1.0 + float(entry["ROI"]) / 100.0, 3)
            row[f"{key}_mdd"] = round(float(entry["MDD"]), 2)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Tesla strategy validation panel.")
    parser.add_argument("--output", type=Path, default=PAYLOAD_PATH)
    args = parser.parse_args()

    curves, period = build_curves()
    payload = {
        "generated_on": date.today().isoformat(),
        "strategy": "V73_TSLA_PLUS_V2",
        "source": "research/tsla_unlevered_v2_overlay_v1/output",
        "period": period,
        "headline": build_headline(),
        "curves": curves,
        "cohorts": build_cohorts(),
        "note": NOTE,
        "caveat": CAVEAT,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "period": period,
                "weekly_marks": len(curves["v73_v2"]),
                "cohorts": len(payload["cohorts"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
