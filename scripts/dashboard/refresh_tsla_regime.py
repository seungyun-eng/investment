# -*- coding: utf-8 -*-
"""Recompute just the TSLA card (now including regime context from the
existing pattern-recognition pipeline) and merge it into the already-
published latest_today.json, without re-running the full cross-sectional
rotation pipeline. Display-only addition -- live_cycle_signal's actual
recommendation logic is untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\LabPC\OneDrive\주식\investment\.venv\Lib\site-packages")
sys.path.insert(0, r"C:\Users\LabPC\OneDrive\주식\investment\src")

from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard import today
from stock_research.paths import load_paths

TARGET = Path(r"C:\Users\LabPC\OneDrive\주식\investment\alpha-desk-cloud\public\data\latest_today.json")


def main() -> None:
    paths = load_paths()
    state = dashboard_state.load_state(paths)
    tsla_card = today.build_tsla_card(paths, state)

    payload = json.loads(TARGET.read_text(encoding="utf-8"))
    payload["tsla"] = tsla_card
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    regime = tsla_card.get("regime")
    print(json.dumps({"tsla_regime": regime}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
