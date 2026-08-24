"""Export TSLA's current price / moving-average / RSI state for the Tesla tab.

Deliberately reads the *daily-refreshed* dashboard price file rather than the
legacy Processed Data/Tesla_지표포함.csv that live_v73_v2_signal still uses.
Those two disagree whenever the legacy file goes stale, so the panel stamps its
own as-of date and the app shows it next to the signal date -- a reader must be
able to see when the chart is newer than the recommendation, not silently mix
the two.

RSI comes from the same Wilder RSI function as the live TSLA V7.3 pipeline,
so the Tesla tab and its recommendation do not show conflicting RSI regimes.

Research/display only. Nothing here feeds the card's recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from stock_research.indicators import normalize_price_columns
from stock_research.io_utils import read_csv_fallback
from stock_research.paths import load_paths
from stock_research.tsla_integrated.features import wilder_rsi

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_PATH = REPO_ROOT / "alpha-desk-cloud" / "public" / "data" / "tesla_technicals.json"

# From 2020: the span the whole TSLA research line is argued over (the earliest
# start-year cohort is 2020), so the chart covers the same history as the
# validation panel below it instead of a rolling window that keeps cutting the
# 2020-2021 run-up off the left edge.  Indicators are still computed on the full
# series first, so SMA200 is already warm at the first plotted session.
HISTORY_START = "2020-01-01"
# 150 is the window the TSLA overlay research selected (TSLAMA150); 200 is the
# conventional trend line.  50 gives the short-term crossover a reader expects.
MOVING_AVERAGES = (50, 150, 200)

NOTE = (
    "Research/Display only. 일일 갱신되는 대시보드 가격 파일에서 계산합니다. "
    "이 패널은 카드의 판단 계산에 관여하지 않으며, 상단 신호 기준일과 날짜가 다를 수 있습니다."
)


def _clean(value: object, digits: int = 2) -> float | None:
    number = float(value) if value is not None else float("nan")
    return None if math.isnan(number) else round(number, digits)


def build_payload(price_path: Path) -> dict[str, object]:
    frame = normalize_price_columns(read_csv_fallback(price_path))
    close = pd.to_numeric(frame["종가"], errors="coerce")
    frame["RSI14"] = wilder_rsi(close)
    for window in MOVING_AVERAGES:
        frame[f"SMA{window}"] = close.rolling(window).mean()

    recent = frame.loc[frame["날짜"] >= pd.Timestamp(HISTORY_START)]
    rows = [
        {
            "date": pd.Timestamp(row["날짜"]).date().isoformat(),
            "close": _clean(row["종가"]),
            "rsi": _clean(row["RSI14"], 1),
            **{f"sma{window}": _clean(row[f"SMA{window}"]) for window in MOVING_AVERAGES},
        }
        for _, row in recent.iterrows()
    ]

    latest = frame.iloc[-1]
    latest_close = _clean(latest["종가"])
    summary: dict[str, object] = {
        "close": latest_close,
        "rsi": _clean(latest["RSI14"]),
        "change_pct": _clean((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0) if len(close) > 1 else None,
    }
    for window in MOVING_AVERAGES:
        value = _clean(latest[f"SMA{window}"])
        summary[f"sma{window}"] = value
        # "Above or below the line" is the whole reason these lines are drawn;
        # precomputing it keeps that judgement out of the rendering layer.
        summary[f"above_sma{window}"] = (
            None if value is None or latest_close is None else latest_close >= value
        )

    return {
        "generated_on": date.today().isoformat(),
        "ticker": "TSLA",
        "as_of": rows[-1]["date"] if rows else None,
        "start": rows[0]["date"] if rows else None,
        "source": str(price_path.name),
        "sessions": len(rows),
        "moving_averages": list(MOVING_AVERAGES),
        "summary": summary,
        "rows": rows,
        "note": NOTE,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Export the Tesla technical panel.")
    parser.add_argument("--stock-root")
    parser.add_argument("--output", type=Path, default=PAYLOAD_PATH)
    args = parser.parse_args()

    paths = load_paths(args.stock_root)
    price_path = paths.stock_root / "Dashboard Data" / "Prices" / "TSLA.csv"
    if not price_path.exists():
        raise SystemExit(f"{price_path} does not exist -- run the dashboard price refresh first.")

    payload = build_payload(price_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "start": payload["start"],
                "as_of": payload["as_of"],
                "sessions": payload["sessions"],
                "close": payload["summary"]["close"],
                "rsi": payload["summary"]["rsi"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
