"""Keep the dashboard's Remaining Upside monitor in step with the live universe.

The original study was run once against universe snapshot U001 (2026-08-12) and
its output was hand-copied into the dashboard. Two universe changes later
(U002 added COST/UBER/INTC and dropped JNJ/JPM/V, U003 added AEP/INSM/NTRA/VST)
the table still showed the U001 membership, so it listed three tickers that had
been removed and none of the seven that had been added.

This script closes that gap: it compares the newest universe snapshot against
the one the research outputs were built from, re-runs the research pipeline only
when they differ, and writes the dashboard payload either way.

Research-only. The frozen U001 artifacts under research/remaining_upside_v1 are
never written to, and nothing here feeds V7.3 ranking or AlphaScore.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research/remaining_upside_live"
OUT_DIR = RESEARCH_DIR / "output"
MONITOR_CSV = OUT_DIR / "latest_shadow_monitor.csv"
DECISION_JSON = OUT_DIR / "final_decision.json"
PAYLOAD_PATH = REPO_ROOT / "alpha-desk-cloud/public/data/remaining_upside_monitoring.json"

# Ordered: each stage consumes the previous stage's output.
STAGES = ("build_panel.py", "build_dataset.py", "run_walk_forward.py", "analyze_validation.py")

NOTE = "Research/Monitoring only. V7.3 랭킹·AlphaScore 계산에 전혀 관여하지 않습니다."


def _python() -> str:
    venv = REPO_ROOT / ".venv/Scripts/python.exe"
    return str(venv) if venv.exists() else sys.executable


def _stamp() -> dict[str, object]:
    sys.path.insert(0, str(RESEARCH_DIR))
    import build_panel  # noqa: PLC0415 -- research dir is only importable once added to sys.path

    return build_panel.current_stamp()


def _universe_is_current() -> bool:
    sys.path.insert(0, str(RESEARCH_DIR))
    import build_panel  # noqa: PLC0415

    return build_panel.is_current()


def run_pipeline() -> None:
    for stage in STAGES:
        print(f"--- {stage}", flush=True)
        subprocess.run(
            [_python(), stage],
            cwd=RESEARCH_DIR,
            check=True,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )


def write_payload() -> dict[str, object]:
    frame = pd.read_csv(MONITOR_CSV)
    decision = json.loads(DECISION_JSON.read_text(encoding="utf-8"))
    stamp = _stamp()

    rows = []
    for record in frame.to_dict("records"):
        rank = record.get("Rank")
        alpha = record.get("AlphaScore")
        rows.append(
            {
                "ticker": str(record["Ticker"]),
                "v7_rank": None if pd.isna(rank) else int(rank),
                "alpha_score": None if pd.isna(alpha) else float(alpha),
                "pred_p25": float(record["pred_p25"]),
                "pred_median": float(record["pred_median"]),
                "pred_p75": float(record["pred_p75"]),
                "prob_mfe_gt_10": float(record["prob_mfe_gt_10"]),
                "prob_mfe_gt_20": float(record["prob_mfe_gt_20"]),
                "exhaustion_0_100": float(record["move_exhaustion_score_0_100"]),
            }
        )
    rows.sort(key=lambda r: (r["v7_rank"] is None, r["v7_rank"] or 0, -r["pred_median"]))

    payload = {
        "as_of": str(frame["Date"].max())[:10],
        "model": "remaining_upside_live (Historical Analog N=50)",
        "verdict": str(decision["decision"]),
        "universe_snapshot": stamp["snapshot"],
        "note": NOTE,
        "rows": rows,
    }
    PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the research pipeline even when the universe has not changed.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report whether a re-run is needed and exit without doing any work.",
    )
    args = parser.parse_args()

    stamp = _stamp()
    current = _universe_is_current()

    if args.check:
        print(json.dumps({
            "snapshot": stamp["snapshot"],
            "universe_size": len(stamp["tickers"]),
            "rerun_needed": not current,
        }, ensure_ascii=False))
        return

    if current and not args.force:
        print(f"universe unchanged ({stamp['snapshot']}); refreshing payload only")
    else:
        reason = "forced" if args.force else "universe changed"
        print(f"re-running research pipeline ({reason}) for {stamp['snapshot']}")
        run_pipeline()

    payload = write_payload()
    print(json.dumps({
        "as_of": payload["as_of"],
        "universe_snapshot": payload["universe_snapshot"],
        "verdict": payload["verdict"],
        "rows": len(payload["rows"]),
        "written": str(PAYLOAD_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
