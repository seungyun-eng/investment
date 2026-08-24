from __future__ import annotations

"""Append-only log of the live TSLA (V7.3+V2) signal: one row per distinct
SignalAsOfClose date, never rewritten once written -- the same point-in-time
discipline as forward_ledger.py's registry segments, just for a plain
per-day signal record instead of a re-simulated NAV curve. This is what
lets the app show "what the model actually said, on the day it said it"
going forward, independent of the user's own trades.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from stock_research.paths import ProjectPaths

LEDGER_FILENAME = "tsla_signal_history.json"


def ledger_path(paths: ProjectPaths) -> Path:
    return paths.results / "mobile_investment_app" / LEDGER_FILENAME


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=path.stem + "_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_history(paths: ProjectPaths) -> list[dict[str, Any]]:
    path = ledger_path(paths)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8-sig"))


def append_signal(paths: ProjectPaths, tsla_card: dict[str, Any]) -> list[dict[str, Any]]:
    """Idempotent: reruns on the same SignalAsOfClose date are a no-op, so
    calling this on every publish (even same-day reruns) never duplicates
    or rewrites a row."""
    recommendation = tsla_card.get("recommendation") or {}
    signal_date = recommendation.get("SignalAsOfClose")
    if not signal_date:
        return load_history(paths)
    history = load_history(paths)
    if any(row.get("SignalAsOfClose") == signal_date for row in history):
        return history
    composition = recommendation.get("Composition") or {}
    explanation = recommendation.get("Explanation") or {}
    history.append({
        "SignalAsOfClose": signal_date,
        "Close": recommendation.get("ReferenceClose"),
        "BaseState": composition.get("BaseState"),
        "V2Exposure": composition.get("V2Exposure"),
        "TargetWeight": recommendation.get("TargetWeight"),
        "Action": recommendation.get("Action"),
        "Reason": recommendation.get("Reason"),
        "Headline": explanation.get("Headline"),
        "CompositeScore": explanation.get("CompositeScore"),
    })
    history.sort(key=lambda row: str(row.get("SignalAsOfClose")))
    _atomic_json(ledger_path(paths), history)
    return history
