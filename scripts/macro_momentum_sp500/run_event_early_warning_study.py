"""Research-only 15% drawdown event warning with a strict zero-FP constraint.

The rule is selected using data through 2017-12-31. Data from 2018 onward is
reported as a frozen temporal holdout and never participates in selection.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.event_warning import (
    build_two_stage_signal,
    collapse_alert_episodes,
    detect_drawdown_events,
    evaluate_alert_episodes,
    warning_rule_candidates,
)
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths


OUTPUT_PATH = Path("artifacts/macro_event_early_warning/results.json")
CONFIG_PATH = "config/macro_momentum_sp500/research.json"
DISCOVERY_END = pd.Timestamp("2017-12-31")
HOLDOUT_START = pd.Timestamp("2018-01-01")
DRAWDOWN_THRESHOLD = -0.15
MINIMUM_LEAD = 7
MAXIMUM_LEAD = 63


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    config = load_research_config(CONFIG_PATH)
    data = load_research_data(load_paths().macro, config)
    features = build_features(data, config).reset_index(drop=True)
    events = detect_drawdown_events(features, DRAWDOWN_THRESHOLD)
    data_start = pd.Timestamp(features["Date"].min())
    data_end = pd.Timestamp(features["Date"].max())

    eligible: list[tuple[tuple[object, ...], object, dict[str, object]]] = []
    candidate_count = 0
    for rule in warning_rule_candidates():
        candidate_count += 1
        signal = build_two_stage_signal(features, rule)
        episodes = collapse_alert_episodes(
            signal,
            merge_gap=rule.episode_merge_gap,
            cooldown_days=rule.cooldown_days,
        )
        discovery = evaluate_alert_episodes(
            episodes,
            events,
            data_start,
            DISCOVERY_END,
            MINIMUM_LEAD,
            MAXIMUM_LEAD,
        )
        if discovery["false_positive_episodes"] != 0:
            continue
        if discovery["captured_events"] == 0:
            continue
        rank = (
            -int(discovery["captured_events"]),
            int(discovery["alert_episodes"]),
            -rule.minimum_trigger_components,
            -rule.trigger_confirmation_days,
            -rule.trigger_z,
            -rule.macro_threshold,
            -rule.breadth_threshold,
            -rule.maximum_pre_alert_drawdown,
        )
        eligible.append((rank, rule, discovery))

    if not eligible:
        payload = {
            "research_only": True,
            "generated_at": date.today().isoformat(),
            "status": "NO_NONTRIVIAL_ZERO_FP_RULE_FOUND",
            "candidate_count": candidate_count,
            "selection_constraint": "Discovery false-positive episodes == 0 and captured events >= 1",
        }
        _atomic_json(OUTPUT_PATH, payload)
        print(json.dumps(payload, indent=2))
        return

    _, selected_rule, discovery = min(eligible, key=lambda item: item[0])
    selected_signal = build_two_stage_signal(features, selected_rule)
    selected_episodes = collapse_alert_episodes(
        selected_signal,
        merge_gap=selected_rule.episode_merge_gap,
        cooldown_days=selected_rule.cooldown_days,
    )
    holdout = evaluate_alert_episodes(
        selected_episodes,
        events,
        HOLDOUT_START,
        data_end,
        MINIMUM_LEAD,
        MAXIMUM_LEAD,
    )
    full = evaluate_alert_episodes(
        selected_episodes,
        events,
        data_start,
        data_end,
        MINIMUM_LEAD,
        MAXIMUM_LEAD,
    )
    payload = {
        "research_only": True,
        "generated_at": date.today().isoformat(),
        "status": (
            "FROZEN_HOLDOUT_ZERO_FP"
            if holdout["false_positive_episodes"] == 0
            else "FROZEN_HOLDOUT_HAS_FALSE_POSITIVES"
        ),
        "important_caveat": (
            "Zero historical false positives cannot guarantee zero future false positives. "
            "Only the post-2017 period is a frozen temporal holdout; the historical event count is very small. "
            "Searching 5,346 candidates against only three discovery events creates substantial selection-overfit risk."
        ),
        "definition": {
            "event": "Disjoint adjusted-SPY peak-to-trough drawdown of at least 15%",
            "successful_alert": "Alert episode onset 63 to 7 market sessions before the event peak",
            "false_positive": "Alert episode onset outside every successful pre-peak window",
            "episode": "Alert dates separated by no more than 10 sessions are merged; 42-session cooldown",
        },
        "data_range": {
            "start": data_start.strftime("%Y-%m-%d"),
            "end": data_end.strftime("%Y-%m-%d"),
        },
        "selection": {
            "discovery_end": DISCOVERY_END.strftime("%Y-%m-%d"),
            "holdout_start": HOLDOUT_START.strftime("%Y-%m-%d"),
            "candidate_count": candidate_count,
            "eligible_zero_fp_candidates": len(eligible),
            "constraint": "Discovery false-positive episodes == 0 and captured events >= 1",
            "tie_break": "Most captured events, then fewest alerts, then stricter trigger",
        },
        "selected_rule": selected_rule.to_dict(),
        "discovery": discovery,
        "frozen_holdout": holdout,
        "full_period_descriptive_only": full,
    }
    _atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
